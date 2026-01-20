//! Jupyter kernel implementation using ZeroMQ
//!
//! This module implements the Jupyter messaging protocol over ZeroMQ sockets,
//! handling kernel_info, execute, and shutdown requests.

use crate::display::format_display_data;
use crate::executor::QueryExecutor;
use crate::message::{ConnectionInfo, JupyterMessage, MessageHeader};
use anyhow::Result;
use hmac::{Hmac, Mac};
use serde_json::{json, Value};
use sha2::Sha256;
use zeromq::{PubSocket, RepSocket, RouterSocket, Socket, SocketRecv, SocketSend};

type HmacSha256 = Hmac<Sha256>;

/// The ggsql Jupyter kernel server
pub struct KernelServer {
    shell: RouterSocket,
    iopub: PubSocket,
    control: RouterSocket,
    #[allow(dead_code)]
    stdin: RouterSocket,
    heartbeat: RepSocket,
    #[allow(dead_code)]
    connection: ConnectionInfo,
    executor: QueryExecutor,
    session: String,
    execution_count: u32,
    key: Vec<u8>,
    // Positron comm IDs
    variables_comm_id: Option<String>,
    ui_comm_id: Option<String>,
    plot_comm_id: Option<String>,
}

impl KernelServer {
    /// Create a new kernel server from connection info
    pub async fn new(connection: ConnectionInfo) -> Result<Self> {
        tracing::info!("Initializing kernel server");

        // Initialize sockets
        let mut shell = RouterSocket::new();
        let mut iopub = PubSocket::new();
        let mut control = RouterSocket::new();
        let mut stdin = RouterSocket::new();
        let mut heartbeat = RepSocket::new();

        // Bind sockets to ports
        let shell_addr = connection.socket_addr(connection.shell_port);
        let iopub_addr = connection.socket_addr(connection.iopub_port);
        let control_addr = connection.socket_addr(connection.control_port);
        let stdin_addr = connection.socket_addr(connection.stdin_port);
        let hb_addr = connection.socket_addr(connection.hb_port);

        tracing::info!("Binding shell socket to {}", shell_addr);
        shell.bind(&shell_addr).await?;

        tracing::info!("Binding iopub socket to {}", iopub_addr);
        iopub.bind(&iopub_addr).await?;

        tracing::info!("Binding control socket to {}", control_addr);
        control.bind(&control_addr).await?;

        tracing::info!("Binding stdin socket to {}", stdin_addr);
        stdin.bind(&stdin_addr).await?;

        tracing::info!("Binding heartbeat socket to {}", hb_addr);
        heartbeat.bind(&hb_addr).await?;

        // Create executor
        let executor = QueryExecutor::new()?;

        // Generate session ID
        let session = uuid::Uuid::new_v4().to_string();

        tracing::info!("Kernel initialized with session {}", session);

        let key = connection.key.as_bytes().to_vec();

        let mut kernel = Self {
            shell,
            iopub,
            control,
            stdin,
            heartbeat,
            connection,
            executor,
            session,
            execution_count: 0,
            key,
            variables_comm_id: None,
            ui_comm_id: None,
            plot_comm_id: None,
        };

        // Send initial "starting" status on IOPub
        // This is required by Jupyter protocol - exactly once at process startup
        kernel.send_status_initial("starting").await?;

        Ok(kernel)
    }

    /// Run the kernel event loop
    pub async fn run(&mut self) -> Result<()> {
        tracing::info!("Starting kernel event loop");

        loop {
            tokio::select! {
                msg = self.shell.recv() => {
                    if let Ok(msg) = msg {
                        self.handle_shell_message(msg).await?;
                    }
                }
                msg = self.control.recv() => {
                    if let Ok(msg) = msg {
                        if self.handle_control_message(msg).await? {
                            tracing::info!("Shutdown requested, exiting");
                            break;
                        }
                    }
                }
                msg = self.heartbeat.recv() => {
                    // Echo heartbeat
                    if let Ok(msg) = msg {
                        self.heartbeat.send(msg).await?;
                    }
                }
                _ = tokio::signal::ctrl_c() => {
                    tracing::warn!("Received SIGINT, shutting down gracefully");
                    // Send a final idle status before exiting
                    // Note: We don't have a parent message here, so we'll send without one
                    let msg = self.create_message(
                        "status",
                        json!({"execution_state": "idle"}),
                        None,
                    );
                    let zmq_msg = self.serialize_message_with_topic(&msg, "status")?;
                    let _ = self.iopub.send(zmq_msg).await;
                    break;
                }
            }
        }

        Ok(())
    }

    /// Handle messages on the shell channel
    async fn handle_shell_message(&mut self, msg: zeromq::ZmqMessage) -> Result<()> {
        let (identities, jupyter_msg) = self.parse_message(msg)?;
        let msg_type = &jupyter_msg.header.msg_type;

        tracing::info!(
            "Received shell message: {} (identities: {})",
            msg_type,
            identities.len()
        );

        match msg_type.as_str() {
            "kernel_info_request" => self.send_kernel_info(&jupyter_msg, &identities).await?,
            "execute_request" => self.execute(&jupyter_msg, &identities).await?,
            "is_complete_request" => self.is_complete(&jupyter_msg, &identities).await?,
            "comm_open" => self.handle_comm_open(&jupyter_msg, &identities).await?,
            "comm_msg" => self.handle_comm_msg(&jupyter_msg, &identities).await?,
            "comm_info_request" => self.handle_comm_info(&jupyter_msg, &identities).await?,
            "comm_close" => self.handle_comm_close(&jupyter_msg, &identities).await?,
            _ => {
                tracing::warn!("Unhandled message type: {}", msg_type);
            }
        }

        Ok(())
    }

    /// Handle messages on the control channel
    async fn handle_control_message(&mut self, msg: zeromq::ZmqMessage) -> Result<bool> {
        let (identities, jupyter_msg) = self.parse_message(msg)?;
        let msg_type = &jupyter_msg.header.msg_type;

        tracing::debug!("Received control message: {}", msg_type);

        match msg_type.as_str() {
            "kernel_info_request" => {
                // Handle kernel_info on control channel too
                self.send_kernel_info_control(&jupyter_msg, &identities)
                    .await?;
                Ok(false)
            }
            "shutdown_request" => {
                // Per Jupyter spec: send busy/idle for ALL requests
                self.send_status("busy", &jupyter_msg).await?;

                let content = &jupyter_msg.content;
                let restart = content["restart"].as_bool().unwrap_or(false);

                self.send_control_reply(
                    "shutdown_reply",
                    json!({"status": "ok", "restart": restart}),
                    &jupyter_msg,
                    &identities,
                )
                .await?;

                self.send_status("idle", &jupyter_msg).await?;

                Ok(true) // Signal shutdown
            }
            _ => {
                tracing::warn!("Unhandled control message: {}", msg_type);
                Ok(false)
            }
        }
    }

    /// Send kernel_info_reply on shell channel
    async fn send_kernel_info(
        &mut self,
        parent: &JupyterMessage,
        identities: &[Vec<u8>],
    ) -> Result<()> {
        tracing::info!(
            "Sending kernel_info_reply (identities: {})",
            identities.len()
        );

        // Per Jupyter spec: send busy/idle for ALL requests
        self.send_status("busy", parent).await?;

        let content = self.kernel_info_content();
        self.send_shell_reply("kernel_info_reply", content, parent, identities)
            .await?;

        self.send_status("idle", parent).await?;
        Ok(())
    }

    /// Send kernel_info_reply on control channel
    async fn send_kernel_info_control(
        &mut self,
        parent: &JupyterMessage,
        identities: &[Vec<u8>],
    ) -> Result<()> {
        // Per Jupyter spec: send busy/idle for ALL requests
        self.send_status("busy", parent).await?;

        let content = self.kernel_info_content();
        self.send_control_reply("kernel_info_reply", content, parent, identities)
            .await?;

        self.send_status("idle", parent).await?;
        Ok(())
    }

    /// Generate kernel info content
    fn kernel_info_content(&self) -> Value {
        json!({
            "status": "ok",
            "protocol_version": "5.3",
            "implementation": "ggsql-jupyter",
            "implementation_version": env!("CARGO_PKG_VERSION"),
            "language_info": {
                "name": "ggsql",
                "version": env!("CARGO_PKG_VERSION"),
                "mimetype": "text/x-ggsql",
                "file_extension": ".ggsql",
                // TODO: We will want our own highlighting syntax here, and for Quarto.
                "pygments_lexer": "sql",
                "codemirror_mode": "sql",
                "positron": {
                    "input_prompt": "ggsql> ",
                    "continuation_prompt": "... "
                }
            },
            "banner": format!("ggsql Jupyter Kernel v{}", env!("CARGO_PKG_VERSION")),
            "help_links": []
        })
    }

    /// Execute a ggsql query
    async fn execute(&mut self, parent: &JupyterMessage, identities: &[Vec<u8>]) -> Result<()> {
        let content = &parent.content;
        let code = content["code"].as_str().unwrap_or("");
        let silent = content["silent"].as_bool().unwrap_or(false);

        tracing::info!("Executing code ({} chars, silent={})", code.len(), silent);

        // Increment execution counter
        if !silent {
            self.execution_count += 1;
        }

        // Send status: busy
        self.send_status("busy", parent).await?;

        // Send execute_input
        if !silent {
            self.send_iopub(
                "execute_input",
                json!({
                    "code": code,
                    "execution_count": self.execution_count
                }),
                parent,
            )
            .await?;
        }

        // Execute the query
        let result = self.executor.execute(code);

        match result {
            Ok(exec_result) => {
                // Send execute_result (not display_data)
                // Per Jupyter spec: execute_result includes execution_count
                if !silent {
                    let display_data = format_display_data(exec_result);

                    // Build message content, including output_location if present
                    let mut content = json!({
                        "execution_count": self.execution_count,
                        "data": display_data["data"],
                        "metadata": display_data["metadata"]
                    });

                    // Add output_location for Positron routing (e.g., to Plots pane)
                    if let Some(location) = display_data.get("output_location") {
                        content["output_location"] = location.clone();
                        tracing::info!("Setting output_location: {}", location);
                    }

                    self.send_iopub("execute_result", content, parent).await?;
                }

                // Send execute_reply
                self.send_shell_reply(
                    "execute_reply",
                    json!({
                        "status": "ok",
                        "execution_count": self.execution_count,
                        "payload": [],
                        "user_expressions": {}
                    }),
                    parent,
                    identities,
                )
                .await?;
            }
            Err(err) => {
                tracing::error!("Execution error: {}", err);

                // Send error message
                let error_msg = format!("{:#}", err);
                self.send_iopub(
                    "error",
                    json!({
                        "ename": "ExecutionError",
                        "evalue": error_msg,
                        "traceback": [error_msg]
                    }),
                    parent,
                )
                .await?;

                // Send execute_reply with error status
                self.send_shell_reply(
                    "execute_reply",
                    json!({
                        "status": "error",
                        "execution_count": self.execution_count,
                        "ename": "ExecutionError",
                        "evalue": error_msg,
                        "traceback": [error_msg]
                    }),
                    parent,
                    identities,
                )
                .await?;
            }
        }

        // Send status: idle
        self.send_status("idle", parent).await?;

        Ok(())
    }

    /// Handle is_complete_request - check if code is a complete statement
    async fn is_complete(&mut self, parent: &JupyterMessage, identities: &[Vec<u8>]) -> Result<()> {
        let content = &parent.content;
        let code = content["code"].as_str().unwrap_or("");

        tracing::debug!("Checking if code is complete ({} chars)", code.len());

        // Send status: busy
        self.send_status("busy", parent).await?;

        // Determine if code is complete
        let status = if code.trim().is_empty() {
            "incomplete" // Empty code needs more input
        } else if is_code_complete(code) {
            "complete"
        } else {
            "incomplete"
        };

        tracing::debug!("Code completeness: {}", status);

        // Send is_complete_reply
        self.send_shell_reply(
            "is_complete_reply",
            json!({
                "status": status,
                "indent": ""
            }),
            parent,
            identities,
        )
        .await?;

        // Send status: idle
        self.send_status("idle", parent).await?;
        Ok(())
    }

    /// Handle comm_open - a client wants to open a comm channel
    async fn handle_comm_open(
        &mut self,
        parent: &JupyterMessage,
        _identities: &[Vec<u8>],
    ) -> Result<()> {
        let target_name = parent.content["target_name"].as_str().unwrap_or("");
        let comm_id = parent.content["comm_id"].as_str().unwrap_or("");
        let data = &parent.content["data"];

        tracing::info!(
            "COMM_OPEN: target_name={}, comm_id={}, data={}",
            target_name,
            comm_id,
            serde_json::to_string(data).unwrap_or_default()
        );

        self.send_status("busy", parent).await?;

        match target_name {
            "positron.variables" => {
                tracing::info!("Registering positron.variables comm: {}", comm_id);
                self.variables_comm_id = Some(comm_id.to_string());

                // Send initial refresh event with empty variables
                self.send_iopub(
                    "comm_msg",
                    json!({
                        "comm_id": comm_id,
                        "data": {
                            "jsonrpc": "2.0",
                            "method": "refresh",
                            "params": {
                                "variables": [],
                                "length": 0,
                                "version": 0
                            }
                        }
                    }),
                    parent,
                )
                .await?;
                tracing::info!("Sent initial variables refresh event");
            }
            "positron.ui" => {
                tracing::info!("Registering positron.ui comm: {}", comm_id);
                self.ui_comm_id = Some(comm_id.to_string());
            }
            _ => {
                tracing::warn!("Unknown comm target: {}", target_name);
            }
        }

        self.send_status("idle", parent).await?;
        Ok(())
    }

    /// Handle comm_msg - a message on an existing comm channel
    async fn handle_comm_msg(
        &mut self,
        parent: &JupyterMessage,
        identities: &[Vec<u8>],
    ) -> Result<()> {
        let comm_id = parent.content["comm_id"].as_str().unwrap_or("");
        let data = &parent.content["data"];

        tracing::info!(
            "COMM_MSG: comm_id={}, data={}",
            comm_id,
            serde_json::to_string(data).unwrap_or_default()
        );

        self.send_status("busy", parent).await?;

        // Check if it's a JSON-RPC request
        if let Some(method) = data["method"].as_str() {
            let rpc_id = &data["id"];

            tracing::info!("JSON-RPC request: method={}, id={}", method, rpc_id);

            // Handle positron.variables requests
            if Some(comm_id.to_string()) == self.variables_comm_id {
                match method {
                    "list" => {
                        tracing::info!("Handling variables.list request");
                        self.send_shell_reply(
                            "comm_msg",
                            json!({
                                "comm_id": comm_id,
                                "data": {
                                    "jsonrpc": "2.0",
                                    "id": rpc_id,
                                    "result": {
                                        "variables": [],
                                        "length": 0,
                                        "version": 0
                                    }
                                }
                            }),
                            parent,
                            identities,
                        )
                        .await?;
                    }
                    "clear" => {
                        tracing::info!("Handling variables.clear request (stub)");
                        self.send_shell_reply(
                            "comm_msg",
                            json!({
                                "comm_id": comm_id,
                                "data": {
                                    "jsonrpc": "2.0",
                                    "id": rpc_id,
                                    "result": {}
                                }
                            }),
                            parent,
                            identities,
                        )
                        .await?;
                    }
                    "delete" => {
                        tracing::info!("Handling variables.delete request (stub)");
                        self.send_shell_reply(
                            "comm_msg",
                            json!({
                                "comm_id": comm_id,
                                "data": {
                                    "jsonrpc": "2.0",
                                    "id": rpc_id,
                                    "result": []
                                }
                            }),
                            parent,
                            identities,
                        )
                        .await?;
                    }
                    "inspect" => {
                        tracing::info!("Handling variables.inspect request (stub)");
                        self.send_shell_reply(
                            "comm_msg",
                            json!({
                                "comm_id": comm_id,
                                "data": {
                                    "jsonrpc": "2.0",
                                    "id": rpc_id,
                                    "result": {
                                        "children": [],
                                        "length": 0
                                    }
                                }
                            }),
                            parent,
                            identities,
                        )
                        .await?;
                    }
                    _ => {
                        tracing::warn!("Unhandled variables method: {}", method);
                    }
                }
            }
            // Handle positron.ui requests
            else if Some(comm_id.to_string()) == self.ui_comm_id {
                tracing::info!("Received UI request: {} (ignoring)", method);
            }
            // Handle positron.plot requests
            else if Some(comm_id.to_string()) == self.plot_comm_id {
                tracing::info!("Received plot request: {} (ignoring)", method);
            }
            // Unknown comm
            else {
                tracing::warn!("Message for unknown comm_id: {}", comm_id);
            }
        }

        self.send_status("idle", parent).await?;
        Ok(())
    }

    /// Handle comm_info_request - list active comms
    async fn handle_comm_info(
        &mut self,
        parent: &JupyterMessage,
        identities: &[Vec<u8>],
    ) -> Result<()> {
        let target_name = parent.content["target_name"].as_str();

        tracing::info!("COMM_INFO_REQUEST: target_name={:?}", target_name);

        self.send_status("busy", parent).await?;

        let mut comms = json!({});

        // Add active comms to response
        if let Some(id) = &self.variables_comm_id {
            if target_name.is_none() || target_name == Some("positron.variables") {
                comms[id] = json!({"target_name": "positron.variables"});
            }
        }
        if let Some(id) = &self.ui_comm_id {
            if target_name.is_none() || target_name == Some("positron.ui") {
                comms[id] = json!({"target_name": "positron.ui"});
            }
        }
        if let Some(id) = &self.plot_comm_id {
            if target_name.is_none() || target_name == Some("positron.plot") {
                comms[id] = json!({"target_name": "positron.plot"});
            }
        }

        tracing::info!(
            "Returning comms: {}",
            serde_json::to_string(&comms).unwrap_or_default()
        );

        self.send_shell_reply(
            "comm_info_reply",
            json!({
                "status": "ok",
                "comms": comms
            }),
            parent,
            identities,
        )
        .await?;

        self.send_status("idle", parent).await?;
        Ok(())
    }

    /// Handle comm_close - a client is closing a comm channel
    async fn handle_comm_close(
        &mut self,
        parent: &JupyterMessage,
        _identities: &[Vec<u8>],
    ) -> Result<()> {
        let comm_id = parent.content["comm_id"].as_str().unwrap_or("");

        tracing::info!("COMM_CLOSE: comm_id={}", comm_id);

        self.send_status("busy", parent).await?;

        // Clear comm ID if it matches
        if Some(comm_id.to_string()) == self.variables_comm_id {
            tracing::info!("Closing positron.variables comm");
            self.variables_comm_id = None;
        } else if Some(comm_id.to_string()) == self.ui_comm_id {
            tracing::info!("Closing positron.ui comm");
            self.ui_comm_id = None;
        } else if Some(comm_id.to_string()) == self.plot_comm_id {
            tracing::info!("Closing positron.plot comm");
            self.plot_comm_id = None;
        } else {
            tracing::warn!("Close for unknown comm_id: {}", comm_id);
        }

        self.send_status("idle", parent).await?;
        Ok(())
    }

    /// Send a message on the IOPub channel
    async fn send_iopub(
        &mut self,
        msg_type: &str,
        content: Value,
        parent: &JupyterMessage,
    ) -> Result<()> {
        let msg = self.create_message(msg_type, content, Some(parent));
        let zmq_msg = self.serialize_message_with_topic(&msg, &msg.header.msg_type)?;
        self.iopub.send(zmq_msg).await?;
        Ok(())
    }

    /// Send a reply on the shell channel
    async fn send_shell_reply(
        &mut self,
        msg_type: &str,
        content: Value,
        parent: &JupyterMessage,
        identities: &[Vec<u8>],
    ) -> Result<()> {
        let msg = self.create_message(msg_type, content, Some(parent));
        let mut zmq_msg = self.serialize_message(&msg)?;

        // For router sockets, we need to prepend the identity frames
        // Prepend identities in REVERSE order (ROUTER sockets expect this)
        for identity in identities.iter().rev() {
            zmq_msg.push_front(bytes::Bytes::from(identity.clone()));
        }

        self.shell.send(zmq_msg).await?;
        Ok(())
    }

    /// Send a reply on the control channel
    async fn send_control_reply(
        &mut self,
        msg_type: &str,
        content: Value,
        parent: &JupyterMessage,
        identities: &[Vec<u8>],
    ) -> Result<()> {
        let msg = self.create_message(msg_type, content, Some(parent));
        let mut zmq_msg = self.serialize_message(&msg)?;

        // For router sockets, we need to prepend the identity frames
        // Prepend identities in REVERSE order (ROUTER sockets expect this)
        for identity in identities.iter().rev() {
            zmq_msg.push_front(bytes::Bytes::from(identity.clone()));
        }

        self.control.send(zmq_msg).await?;
        Ok(())
    }

    /// Send a status message
    async fn send_status(&mut self, state: &str, parent: &JupyterMessage) -> Result<()> {
        self.send_iopub("status", json!({"execution_state": state}), parent)
            .await
    }

    /// Send an initial status message without a parent (for kernel startup)
    async fn send_status_initial(&mut self, state: &str) -> Result<()> {
        let msg = self.create_message("status", json!({"execution_state": state}), None);
        let zmq_msg = self.serialize_message_with_topic(&msg, "status")?;
        self.iopub.send(zmq_msg).await?;
        Ok(())
    }

    /// Create a new Jupyter message
    fn create_message(
        &self,
        msg_type: &str,
        content: Value,
        parent: Option<&JupyterMessage>,
    ) -> JupyterMessage {
        JupyterMessage {
            header: MessageHeader {
                msg_id: uuid::Uuid::new_v4().to_string(),
                session: self.session.clone(),
                username: "ggsql".to_string(),
                date: chrono::Utc::now().to_rfc3339(),
                msg_type: msg_type.to_string(),
                version: "5.3".to_string(),
            },
            parent_header: parent
                .map(|p| serde_json::to_value(&p.header).unwrap())
                .unwrap_or(json!({})),
            metadata: json!({}),
            content,
            buffers: vec![],
        }
    }

    /// Parse a ZeroMQ message into a Jupyter message
    /// Returns (identities, jupyter_message)
    fn parse_message(&self, msg: zeromq::ZmqMessage) -> Result<(Vec<Vec<u8>>, JupyterMessage)> {
        // Jupyter wire protocol: [identity, ..., delimiter, hmac, header, parent, metadata, content]
        let frames: Vec<_> = msg.into_vec();

        if frames.len() < 6 {
            anyhow::bail!("Invalid message: too few frames (got {})", frames.len());
        }

        // Find delimiter
        let delim_pos = frames
            .iter()
            .position(|f| f.as_ref() == b"<IDS|MSG>")
            .ok_or_else(|| anyhow::anyhow!("Delimiter not found"))?;

        // Extract identity frames (everything before delimiter)
        let identities: Vec<Vec<u8>> = frames[..delim_pos].iter().map(|b| b.to_vec()).collect();

        let received_hmac = std::str::from_utf8(&frames[delim_pos + 1])?;
        let header_json = std::str::from_utf8(&frames[delim_pos + 2])?;
        let parent_json = std::str::from_utf8(&frames[delim_pos + 3])?;
        let metadata_json = std::str::from_utf8(&frames[delim_pos + 4])?;
        let content_json = std::str::from_utf8(&frames[delim_pos + 5])?;

        // SECURITY: Verify HMAC signature if key is set
        if !self.key.is_empty() {
            let expected_hmac =
                self.sign_message(header_json, parent_json, metadata_json, content_json);

            // Use constant-time comparison to prevent timing attacks
            if received_hmac != expected_hmac {
                anyhow::bail!("HMAC signature verification failed");
            }
        }

        let jupyter_msg = JupyterMessage {
            header: serde_json::from_str(header_json)?,
            parent_header: serde_json::from_str(parent_json)?,
            metadata: serde_json::from_str(metadata_json)?,
            content: serde_json::from_str(content_json)?,
            buffers: vec![],
        };

        Ok((identities, jupyter_msg))
    }

    /// Serialize a Jupyter message to ZeroMQ format
    fn serialize_message(&self, msg: &JupyterMessage) -> Result<zeromq::ZmqMessage> {
        let header = serde_json::to_string(&msg.header)?;
        let parent = serde_json::to_string(&msg.parent_header)?;
        let metadata = serde_json::to_string(&msg.metadata)?;
        let content = serde_json::to_string(&msg.content)?;

        // Calculate HMAC signature
        let signature = self.sign_message(&header, &parent, &metadata, &content);

        // Build ZeroMQ message: [delimiter, hmac, header, parent, metadata, content]
        let mut zmq_msg = zeromq::ZmqMessage::from(b"<IDS|MSG>".to_vec());
        zmq_msg.push_back(signature.into());
        zmq_msg.push_back(header.into());
        zmq_msg.push_back(parent.into());
        zmq_msg.push_back(metadata.into());
        zmq_msg.push_back(content.into());

        Ok(zmq_msg)
    }

    /// Serialize a Jupyter message to ZeroMQ format with topic (for IOPub)
    /// According to Jupyter protocol: "there should be just one prefix component, which is the topic"
    fn serialize_message_with_topic(
        &self,
        msg: &JupyterMessage,
        topic: &str,
    ) -> Result<zeromq::ZmqMessage> {
        let header = serde_json::to_string(&msg.header)?;
        let parent = serde_json::to_string(&msg.parent_header)?;
        let metadata = serde_json::to_string(&msg.metadata)?;
        let content = serde_json::to_string(&msg.content)?;

        // Calculate HMAC signature
        let signature = self.sign_message(&header, &parent, &metadata, &content);

        // Build ZeroMQ message with topic: [topic, delimiter, hmac, header, parent, metadata, content]
        let mut zmq_msg = zeromq::ZmqMessage::from(topic.as_bytes().to_vec());
        zmq_msg.push_back(b"<IDS|MSG>".to_vec().into());
        zmq_msg.push_back(signature.into());
        zmq_msg.push_back(header.into());
        zmq_msg.push_back(parent.into());
        zmq_msg.push_back(metadata.into());
        zmq_msg.push_back(content.into());

        Ok(zmq_msg)
    }

    /// Sign a message using HMAC-SHA256
    fn sign_message(&self, header: &str, parent: &str, metadata: &str, content: &str) -> String {
        if self.key.is_empty() {
            return String::new();
        }

        let mut mac = HmacSha256::new_from_slice(&self.key).expect("HMAC can take key of any size");
        mac.update(header.as_bytes());
        mac.update(parent.as_bytes());
        mac.update(metadata.as_bytes());
        mac.update(content.as_bytes());

        hex::encode(mac.finalize().into_bytes())
    }
}

/// Check if ggsql code is complete (balanced brackets, not in a string)
/// If the code contains VISUALISE, it must also contain at least one DRAW layer
fn is_code_complete(code: &str) -> bool {
    let trimmed = code.trim();

    // Empty or whitespace-only is incomplete
    if trimmed.is_empty() {
        return false;
    }

    // Check for balanced parentheses, brackets, and braces
    let mut paren_depth = 0i32;
    let mut bracket_depth = 0i32;
    let mut brace_depth = 0i32;
    let mut in_string = false;
    let mut string_char = ' ';

    for c in trimmed.chars() {
        if in_string {
            if c == string_char {
                in_string = false;
            }
        } else {
            match c {
                '\'' | '"' => {
                    in_string = true;
                    string_char = c;
                }
                '(' => paren_depth += 1,
                ')' => paren_depth -= 1,
                '[' => bracket_depth += 1,
                ']' => bracket_depth -= 1,
                '{' => brace_depth += 1,
                '}' => brace_depth -= 1,
                _ => {}
            }
        }
    }

    // Code is incomplete if brackets are unbalanced or we're in a string
    if in_string || paren_depth != 0 || bracket_depth != 0 || brace_depth != 0 {
        return false;
    }

    // If code contains VISUALISE/VISUALIZE, it must also contain DRAW
    let upper = trimmed.to_uppercase();
    if upper.contains("VISUALISE") || upper.contains("VISUALIZE") {
        return upper.contains("DRAW");
    }

    true
}
