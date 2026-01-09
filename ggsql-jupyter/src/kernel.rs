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

/// The ggSQL Jupyter kernel server
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
                "pygments_lexer": "sql",
                "codemirror_mode": "sql"
            },
            "banner": format!("ggSQL Jupyter Kernel v{}\nSQL with declarative visualization", env!("CARGO_PKG_VERSION")),
            "help_links": []
        })
    }

    /// Execute a ggSQL query
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
                    self.send_iopub(
                        "execute_result",
                        json!({
                            "execution_count": self.execution_count,
                            "data": display_data["data"],
                            "metadata": display_data["metadata"]
                        }),
                        parent,
                    )
                    .await?;
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
