//! Jupyter kernel implementation using ZeroMQ
//!
//! This module implements the Jupyter messaging protocol over ZeroMQ sockets,
//! handling kernel_info, execute, and shutdown requests.

use crate::connection;
use crate::data_explorer::{DataExplorerState, RpcResponse};
use crate::display::{format_display_data, Formatted, RenderHints, SessionMode};
use crate::executor::{self, ExecutionResult, QueryExecutor};
use crate::message::{ConnectionInfo, JupyterMessage, MessageHeader};
use crate::plot::comm::{PlotMetadata, RenderParams, RpcError};
use crate::plot::{PlotBackend, RenderOutcome, RenderTicket};
use anyhow::Result;
use ggsql::reader::Spec;
use hmac::{Hmac, Mac};
use serde_json::{json, Value};
use sha2::Sha256;
use std::collections::{HashMap, VecDeque};
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
    /// The render thread, and whether it found a GPU adapter.
    plots: PlotBackend,
    /// Finished comm renders, coming back from the render thread.
    ///
    /// Taken out of `self` before the event loop, because `select!` cannot
    /// borrow `self` mutably for this arm while the other arms do the same.
    render_outcomes: Option<tokio::sync::mpsc::UnboundedReceiver<RenderOutcome>>,
    /// Open plot comms, metadata only — each plot's `Spec` lives on the render
    /// thread, keeping the retained `DataFrame`s off the async task.
    plot_comms: HashMap<String, PlotMetadata>,
    /// Comm ids in the order they were opened, for oldest-first eviction.
    plot_order: VecDeque<String>,
    /// How many plots to keep. Positron imposes no cap of its own, so a long
    /// console session would otherwise retain every post-stat `DataFrame`.
    max_plots: usize,
    /// Counts plots this session, so an untitled one can be named.
    plot_seq: u32,
    /// The size, ratio and format the Plots pane last reported, from the ui
    /// comm's `did_change_plots_render_settings`.
    ///
    /// `None` until the pane has reported once, which is why the first plot of
    /// a session carries no pre-render.
    plot_render_settings: Option<RenderParams>,
    session: String,
    /// What the frontend declared this session to be, if it declared anything.
    /// `None` leaves classification to the session-id heuristic.
    session_mode: Option<SessionMode>,
    execution_count: u32,
    key: Vec<u8>,
    // Positron comm IDs
    variables_comm_id: Option<String>,
    ui_comm_id: Option<String>,
    connection_comm_id: Option<String>,
    data_explorer_comms: HashMap<String, DataExplorerState>,
}

impl KernelServer {
    /// Create a new kernel server from connection info
    pub async fn new(
        connection: ConnectionInfo,
        reader_uri: &str,
        session_mode: Option<SessionMode>,
        max_plots: usize,
    ) -> Result<Self> {
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

        // Create executor with the specified reader
        let executor = QueryExecutor::new_with_uri(reader_uri)?;

        // Generate session ID
        let session = uuid::Uuid::new_v4().to_string();

        tracing::info!("Kernel initialized with session {}", session);

        let key = connection.key.as_bytes().to_vec();

        let (render_tx, render_rx) = tokio::sync::mpsc::unbounded_channel();

        let mut kernel = Self {
            shell,
            iopub,
            control,
            stdin,
            heartbeat,
            connection,
            executor,
            plots: PlotBackend::spawn(render_tx),
            render_outcomes: Some(render_rx),
            plot_comms: HashMap::new(),
            plot_order: VecDeque::new(),
            max_plots: max_plots.max(1),
            plot_seq: 0,
            plot_render_settings: None,
            session,
            session_mode,
            execution_count: 0,
            key,
            variables_comm_id: None,
            ui_comm_id: None,
            connection_comm_id: None,
            data_explorer_comms: HashMap::new(),
        };

        // Send initial "starting" status on IOPub
        // This is required by Jupyter protocol - exactly once at process startup
        kernel.send_status_initial("starting").await?;

        // Only after `starting`: the probe can block for ten seconds, and a
        // kernel that has bound its sockets but announced nothing looks to a
        // supervisor like a failed launch.
        kernel.plots.finish_probe();

        // Open initial connection comm so the Connections pane shows the database
        kernel.open_connection_comm(reader_uri).await?;

        Ok(kernel)
    }

    /// Run the kernel event loop
    pub async fn run(&mut self) -> Result<()> {
        tracing::info!("Starting kernel event loop");

        // Out of `self`, so the outcome arm below can borrow it while the
        // other arms borrow `self` mutably.
        let mut render_outcomes = self
            .render_outcomes
            .take()
            .expect("the event loop runs once");

        // The render thread holds the only sender, so `None` means it is gone.
        // The arm is disabled rather than left to complete instantly on every
        // iteration and spin at 100% CPU; the session carries on serving SQL.
        let mut renders_alive = true;

        loop {
            tokio::select! {
                outcome = render_outcomes.recv(), if renders_alive => {
                    match outcome {
                        Some(outcome) => self.finish_render(outcome).await?,
                        None => {
                            tracing::error!(
                                "the render thread has stopped; plots will fail for the \
                                 rest of this session"
                            );
                            renders_alive = false;
                        }
                    }
                }
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
                    // With `interrupt_mode: "signal"` a SIGINT interrupts the
                    // running cell rather than stopping the kernel, so
                    // acknowledge it with a busy -> idle pair.
                    // TODO: cancel the in-flight request here once execution is async.
                    tracing::debug!("Received SIGINT; acknowledging.");
                    self.send_status_initial("busy").await?;
                    self.send_status_initial("idle").await?;
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
        let hints = RenderHints::from_request(&parent.header, content, self.session_mode);

        // `kind` decides where a plot goes, so log its two inputs: the declared
        // mode and the session id the heuristic falls back to. A plot in the
        // wrong slot is otherwise indistinguishable from one rendered wrongly.
        tracing::info!(
            "Executing code ({} chars, silent={}, kind={:?} from mode={:?} session={}, width_px={:?})",
            code.len(),
            silent,
            hints.kind,
            self.session_mode,
            parent.header.session,
            hints.output_width_px
        );

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

        // A cell may connect and then go on to query, so a connection change is
        // read from the executor's URI rather than from the result variant,
        // which only reports a cell that was nothing but meta-commands.
        let uri_before = self.executor.reader_uri().to_string();
        let executed = self.executor.execute(code);
        if self.executor.reader_uri() != uri_before {
            let uri = self.executor.reader_uri().to_string();
            self.open_connection_comm(&uri).await?;
        }

        // Formatting renders the plot, so it is fallible and its failures are
        // the user's. Both stages feed one `Result` and share the error path
        // below; propagating out of the loop would end the process and take
        // the session's database with it.
        let result = match executed {
            Ok(exec_result) => {
                // A bare connection change renders nothing.
                let is_connection_changed =
                    matches!(&exec_result, ExecutionResult::ConnectionChanged { .. });

                // Nothing to display for a silent cell, for a connection
                // change, or for DDL (which formats to `Nothing`).
                if silent || is_connection_changed {
                    Ok(Formatted::Nothing)
                } else {
                    format_display_data(exec_result, &hints, &self.plots)
                }
            }
            Err(err) => Err(err),
        };

        match result {
            Ok(formatted) => {
                // Send execute_result (not display_data)
                // Per Jupyter spec: execute_result includes execution_count
                match formatted {
                    // No `output_location`: it routes the output to the Plots
                    // pane as well as the cell, so the plot would arrive twice.
                    Formatted::Bundle(display_data) => {
                        let content = json!({
                            "execution_count": self.execution_count,
                            "data": display_data["data"],
                            "metadata": display_data["metadata"]
                        });
                        self.send_iopub("execute_result", content, parent).await?;
                    }
                    // No `execute_result`: the comm alone creates the pane
                    // entry, and an output message would duplicate it.
                    Formatted::PlotComm(spec) => {
                        self.open_plot_comm(spec, code, parent).await?;
                    }
                    Formatted::Nothing => {}
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
                // Both stages report here: whether the query or the rendering
                // of its result failed, the cell failed.
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
        #[allow(clippy::if_same_then_else)]
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
                // A notification carries no `id` and replying to one is a
                // protocol error, so check the shape before building a reply.
                if rpc_id.is_null() {
                    self.handle_ui_notification(method, &data["params"]);
                } else {
                    self.send_shell_reply(
                        "comm_msg",
                        json!({
                            "comm_id": comm_id,
                            "data": {
                                "jsonrpc": "2.0",
                                "id": rpc_id,
                                "result": null
                            }
                        }),
                        parent,
                        identities,
                    )
                    .await?;
                }
            }
            // Handle positron.plot requests
            else if self.plot_comms.contains_key(comm_id) {
                let rpc_id = rpc_id.clone();
                let deferred = self
                    .handle_plot_rpc(method, &rpc_id, comm_id, parent, identities)
                    .await?;
                // A render in flight has its `busy` closed by `finish_render`,
                // so return rather than close it here. The flag reports whether
                // one was actually dispatched, not whether `render` was asked
                // for, so a rejected request stays paired too.
                if deferred {
                    return Ok(());
                }
            }
            // Handle positron.connection requests
            else if Some(comm_id.to_string()) == self.connection_comm_id {
                self.handle_connection_rpc(method, rpc_id, comm_id, parent, identities)
                    .await?;
            }
            // Handle positron.dataExplorer requests
            else if self.data_explorer_comms.contains_key(comm_id) {
                self.handle_data_explorer_rpc(method, rpc_id, comm_id, parent, identities)
                    .await?;
            }
            // Unknown comm — still respond to avoid RPC timeouts
            else {
                tracing::warn!(
                    "JSON-RPC request for unknown comm_id: {}, method: {}",
                    comm_id,
                    method
                );
                self.send_shell_reply(
                    "comm_msg",
                    json!({
                        "comm_id": comm_id,
                        "data": {
                            "jsonrpc": "2.0",
                            "id": rpc_id,
                            "result": null
                        }
                    }),
                    parent,
                    identities,
                )
                .await?;
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
        // Positron calls `listClients(Plot)` on reconnect, so every open plot
        // has to be listed — not just one.
        if target_name.is_none() || target_name == Some("positron.plot") {
            for id in self.plot_comms.keys() {
                comms[id] = json!({"target_name": "positron.plot"});
            }
        }
        if let Some(id) = &self.connection_comm_id {
            if target_name.is_none() || target_name == Some("positron.connection") {
                comms[id] = json!({"target_name": "positron.connection"});
            }
        }
        for id in self.data_explorer_comms.keys() {
            if target_name.is_none() || target_name == Some("positron.dataExplorer") {
                comms[id] = json!({"target_name": "positron.dataExplorer"});
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
        } else if self.plot_comms.remove(comm_id).is_some() {
            tracing::info!("Closing positron.plot comm {}", comm_id);
            self.plot_order.retain(|id| id != comm_id);
            // Drop the retained plot too, or closing a comm would leak its
            // post-stat DataFrames for the rest of the session.
            self.plots.forget(comm_id);
        } else if Some(comm_id.to_string()) == self.connection_comm_id {
            tracing::info!("Closing positron.connection comm");
            self.connection_comm_id = None;
        } else if self.data_explorer_comms.remove(comm_id).is_some() {
            tracing::info!("Closing data explorer comm: {}", comm_id);
        } else {
            tracing::warn!("Close for unknown comm_id: {}", comm_id);
        }

        self.send_status("idle", parent).await?;
        Ok(())
    }

    /// Open (or replace) a `positron.connection` comm for the current reader.
    ///
    /// The kernel initiates this comm (backend-initiated). If an existing
    /// connection comm is open, it is closed first.
    async fn open_connection_comm(&mut self, uri: &str) -> Result<()> {
        // Close existing connection comm if any
        if let Some(old_id) = self.connection_comm_id.take() {
            tracing::info!("Closing old connection comm: {}", old_id);
            let close_msg = self.create_message("comm_close", json!({ "comm_id": old_id }), None);
            let zmq_msg = self.serialize_message_with_topic(&close_msg, "comm_close")?;
            self.iopub.send(zmq_msg).await?;
        }

        let comm_id = uuid::Uuid::new_v4().to_string();
        let display_name = executor::display_name_for_uri(uri);
        let type_name = executor::type_name_for_uri(uri);
        let host = executor::host_for_uri(uri);
        let meta_command = format!("-- @connect: {}", uri);

        tracing::info!(
            "Opening positron.connection comm: {} ({})",
            comm_id,
            display_name
        );

        let msg = self.create_message(
            "comm_open",
            json!({
                "comm_id": comm_id,
                "target_name": "positron.connection",
                "data": {
                    "name": display_name,
                    "language_id": "ggsql",
                    "host": host,
                    "type": type_name,
                    "code": meta_command
                }
            }),
            None,
        );
        let zmq_msg = self.serialize_message_with_topic(&msg, "comm_open")?;
        self.iopub.send(zmq_msg).await?;

        self.connection_comm_id = Some(comm_id);
        Ok(())
    }

    /// Open a `positron.plot` comm for a freshly executed plot.
    ///
    /// Backend-initiated, one comm per plot like `positron.dataExplorer`, since
    /// the pane shows a session's plots as a history.
    ///
    /// Must follow `execute_input`: Positron takes the plot's `code` metadata
    /// from the `_recentExecutions` map that message populates.
    async fn open_plot_comm(
        &mut self,
        spec: Box<Spec>,
        code: &str,
        parent: &JupyterMessage,
    ) -> Result<()> {
        let comm_id = uuid::Uuid::new_v4().to_string();
        self.plot_seq = self.plot_seq.saturating_add(1);

        let title = spec
            .plot()
            .labels
            .as_ref()
            .and_then(|labels| labels.labels.get("title"))
            .and_then(|title| title.as_deref());
        let metadata = PlotMetadata {
            name: crate::plot::comm::plot_name(title, self.plot_seq),
            execution_id: parent.header.msg_id.clone(),
            code: code.to_string(),
        };

        tracing::info!("Opening positron.plot comm {} ({})", comm_id, metadata.name);

        // The plot itself goes to the render thread, which owns it from here.
        self.plots.store(comm_id.clone(), spec);
        self.plot_comms.insert(comm_id.clone(), metadata);
        self.plot_order.push_back(comm_id.clone());

        // Rendered at the size the pane last reported; `None` for the first
        // plot of a session, before the pane has reported at all.
        let data = match self.pre_render(&comm_id) {
            Some(pre_render) => json!({ "pre_render": pre_render }),
            None => Value::Null,
        };

        // Parented to the `execute_request`, as every iopub message during an
        // execution must be — it ties the plot to the cell that produced it.
        let msg = self.create_message(
            "comm_open",
            json!({
                "comm_id": comm_id,
                "target_name": "positron.plot",
                "data": data,
            }),
            Some(parent),
        );
        let zmq_msg = self.serialize_message_with_topic(&msg, "comm_open")?;
        self.iopub.send(zmq_msg).await?;

        self.evict_old_plots(parent).await
    }

    /// Act on a notification pushed at us over the ui comm.
    ///
    /// Nothing is replied to — see the call site.
    fn handle_ui_notification(&mut self, method: &str, params: &Value) {
        match method {
            // The Plots pane reporting its size, ratio and preferred format on
            // resize. The only place a pane's size reaches us outside a render
            // request, and so the only thing that makes a pre-render possible.
            "did_change_plots_render_settings" => {
                match RenderParams::from_rpc(&params["settings"]) {
                    Ok(settings) => {
                        // The pane asks for `png`; a build with no raster
                        // writer stores the SVG it can actually pre-render.
                        let settings = settings.available(self.plots.raster());
                        tracing::debug!(
                            "plots pane render settings: {}x{} @ {}x",
                            settings.request.canvas.width,
                            settings.request.canvas.height,
                            settings.pixel_ratio
                        );
                        self.plot_render_settings = Some(settings);
                    }
                    Err(e) => tracing::warn!("unusable plot render settings: {:?}", e),
                }
            }
            other => tracing::debug!("ignoring ui notification '{other}'"),
        }
    }

    /// Render a plot at the pane's last known settings, for `comm_open`.
    ///
    /// `None` before the pane has reported, i.e. the first plot of a session:
    /// a guessed size would flash a wrong-sized picture and be replaced at
    /// once. The reference Python backend skips it for the same reason.
    ///
    /// Positron discards a pre-render without `settings`
    /// (`languageRuntimePlotClient.ts`), so this goes through `to_result`.
    fn pre_render(&self, comm_id: &str) -> Option<Value> {
        let settings = self.plot_render_settings?;
        // A pre-render is a nicety: if it fails or times out the comm still
        // opens, and the pane's own render request follows a moment later.
        match self.plots.render_stored(comm_id, settings.request) {
            Ok(bytes) => Some(settings.to_result(RenderParams::encode(&bytes))),
            Err(e) => {
                tracing::debug!("no pre-render for this plot: {e}");
                None
            }
        }
    }

    /// Close the oldest plot comms until the history is back within its cap.
    ///
    /// `comm_close` on iopub removes the plot from the pane, which is what "the
    /// kernel no longer keeps that plot" means; leaving it listed and failing
    /// its renders would be worse. Parented to the `execute_request` that
    /// pushed the history over its cap — a frontend may drop an unparented
    /// message mid-execution.
    async fn evict_old_plots(&mut self, parent: &JupyterMessage) -> Result<()> {
        while self.plot_order.len() > self.max_plots {
            let Some(comm_id) = self.plot_order.pop_front() else {
                break;
            };
            tracing::info!("Evicting plot comm {} (history is full)", comm_id);
            self.plot_comms.remove(&comm_id);
            self.plots.forget(&comm_id);

            let msg =
                self.create_message("comm_close", json!({ "comm_id": comm_id }), Some(parent));
            let zmq_msg = self.serialize_message_with_topic(&msg, "comm_close")?;
            self.iopub.send(zmq_msg).await?;
        }
        Ok(())
    }

    /// Serve a JSON-RPC request on a plot comm.
    ///
    /// `render` is dispatched to the render thread and answered later from
    /// [`Self::finish_render`]; the others are answered from state held here,
    /// because `get_metadata` gets Positron's default 5 s timeout where
    /// `render` gets 30 s and so must never queue behind one.
    ///
    /// Returns whether a render was dispatched. `true` means the reply — and
    /// the `idle` closing the request's `busy` — is [`Self::finish_render`]'s
    /// to send; every other outcome is answered here.
    async fn handle_plot_rpc(
        &mut self,
        method: &str,
        rpc_id: &Value,
        comm_id: &str,
        parent: &JupyterMessage,
        identities: &[Vec<u8>],
    ) -> Result<bool> {
        match method {
            "render" => {
                let params = match RenderParams::from_rpc(&parent.content["data"]["params"]) {
                    Ok(params) => params.available(self.plots.raster()),
                    Err(e) => {
                        self.reply_plot_error(&e, rpc_id, comm_id, parent, identities)
                            .await?;
                        return Ok(false);
                    }
                };
                let ticket = RenderTicket {
                    comm_id: comm_id.to_string(),
                    rpc_id: rpc_id.clone(),
                    params,
                    parent: parent.clone(),
                    identities: identities.to_vec(),
                };
                // Answered when the thread reports back — unless it is gone, in
                // which case no outcome will arrive and Positron would sit out
                // its 30 s timeout against a kernel that never returns to `idle`.
                match self.plots.request_render(comm_id, params.request, ticket) {
                    Ok(()) => Ok(true),
                    Err(e) => {
                        tracing::error!("could not dispatch a render: {e}");
                        let e = RpcError::Internal(format!("could not render this plot: {e}"));
                        self.reply_plot_error(&e, rpc_id, comm_id, parent, identities)
                            .await?;
                        Ok(false)
                    }
                }
            }
            // ggsql has no figure-size syntax, so there is no intrinsic size.
            // `null` makes Positron use its fill policy rather than offer an
            // "Intrinsic" option that would be a lie.
            "get_intrinsic_size" => {
                self.reply_plot_result(Value::Null, rpc_id, comm_id, parent, identities)
                    .await?;
                Ok(false)
            }
            "get_metadata" => {
                match self.plot_comms.get(comm_id) {
                    Some(metadata) => {
                        let result = metadata.to_result();
                        self.reply_plot_result(result, rpc_id, comm_id, parent, identities)
                            .await?;
                    }
                    None => {
                        let e = RpcError::Internal("this plot is no longer available".into());
                        self.reply_plot_error(&e, rpc_id, comm_id, parent, identities)
                            .await?;
                    }
                }
                Ok(false)
            }
            // An error rather than `result: null`, so a future Positron method
            // fails visibly instead of being satisfied with garbage. `show` and
            // `update` belong here: they mean "re-fetch this figure", but a
            // `Spec` is immutable per execution, so re-running a cell opens a
            // new comm as the R and matplotlib backends do. Do not add them.
            other => {
                let e = RpcError::MethodNotFound(format!("the plot comm has no '{other}' method"));
                self.reply_plot_error(&e, rpc_id, comm_id, parent, identities)
                    .await?;
                Ok(false)
            }
        }
    }

    /// Send a finished render back as its comm's RPC reply.
    async fn finish_render(&mut self, outcome: RenderOutcome) -> Result<()> {
        let RenderOutcome { ticket, result } = outcome;
        let RenderTicket {
            comm_id,
            rpc_id,
            params,
            parent,
            identities,
        } = *ticket;

        match result {
            Ok(bytes) => {
                // Always base64 here, SVG included — see `RenderParams::encode`.
                let encoded = RenderParams::encode(&bytes);
                self.reply_plot_result(
                    params.to_result(encoded),
                    &rpc_id,
                    &comm_id,
                    &parent,
                    &identities,
                )
                .await?;
            }
            Err(e) => {
                tracing::warn!("plot render failed: {e}");
                // Reply with an error rather than switching output kind
                // mid-comm; Positron shows a render error in the pane.
                let e = RpcError::Internal(format!("could not render this plot: {e}"));
                self.reply_plot_error(&e, &rpc_id, &comm_id, &parent, &identities)
                    .await?;
            }
        }

        // The `busy` that went out when the request arrived is now answered.
        self.send_status("idle", &parent).await
    }

    async fn reply_plot_result(
        &mut self,
        result: Value,
        rpc_id: &Value,
        comm_id: &str,
        parent: &JupyterMessage,
        identities: &[Vec<u8>],
    ) -> Result<()> {
        self.send_shell_reply(
            "comm_msg",
            json!({
                "comm_id": comm_id,
                "data": { "jsonrpc": "2.0", "id": rpc_id, "result": result }
            }),
            parent,
            identities,
        )
        .await
    }

    async fn reply_plot_error(
        &mut self,
        error: &RpcError,
        rpc_id: &Value,
        comm_id: &str,
        parent: &JupyterMessage,
        identities: &[Vec<u8>],
    ) -> Result<()> {
        self.send_shell_reply(
            "comm_msg",
            json!({
                "comm_id": comm_id,
                "data": { "jsonrpc": "2.0", "id": rpc_id, "error": error.to_json() }
            }),
            parent,
            identities,
        )
        .await
    }

    /// Handle JSON-RPC requests on the connection comm
    async fn handle_connection_rpc(
        &mut self,
        method: &str,
        rpc_id: &Value,
        comm_id: &str,
        parent: &JupyterMessage,
        identities: &[Vec<u8>],
    ) -> Result<()> {
        tracing::info!("Connection RPC: {}", method);

        let params = &parent.content["data"]["params"];

        let result = match method {
            "list_objects" => {
                let path: Vec<String> = params["path"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| {
                                v.get("name")
                                    .and_then(|n| n.as_str())
                                    .map(|s| s.to_string())
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                match connection::list_objects(self.executor.reader(), &path) {
                    Ok(objects) => json!(objects),
                    Err(e) => {
                        tracing::error!("list_objects failed: {}", e);
                        json!([])
                    }
                }
            }
            "list_fields" => {
                let path: Vec<String> = params["path"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| {
                                v.get("name")
                                    .and_then(|n| n.as_str())
                                    .map(|s| s.to_string())
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                match connection::list_fields(self.executor.reader(), &path) {
                    Ok(fields) => json!(fields),
                    Err(e) => {
                        tracing::error!("list_fields failed: {}", e);
                        json!([])
                    }
                }
            }
            "contains_data" => {
                let path: Vec<Value> = params["path"].as_array().cloned().unwrap_or_default();
                let has_data = connection::contains_data(&path);
                json!(has_data)
            }
            "get_icon" => json!(""),
            "preview_object" => {
                let path: Vec<String> = params["path"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| {
                                v.get("name")
                                    .and_then(|n| n.as_str())
                                    .map(|s| s.to_string())
                            })
                            .collect()
                    })
                    .unwrap_or_default();

                match DataExplorerState::open(self.executor.reader(), &path) {
                    Ok(state) => {
                        let de_comm_id = uuid::Uuid::new_v4().to_string();
                        let title = path.last().cloned().unwrap_or_default();

                        // Send comm_open on iopub to open the data viewer
                        let msg = self.create_message(
                            "comm_open",
                            json!({
                                "comm_id": de_comm_id,
                                "target_name": "positron.dataExplorer",
                                "data": {
                                    "title": title
                                }
                            }),
                            Some(parent),
                        );
                        let zmq_msg = self.serialize_message_with_topic(&msg, "comm_open")?;
                        self.iopub.send(zmq_msg).await?;

                        tracing::info!("Opened data explorer comm: {} for {}", de_comm_id, title);
                        self.data_explorer_comms.insert(de_comm_id, state);
                    }
                    Err(e) => {
                        tracing::error!("preview_object failed: {}", e);
                    }
                }
                json!(null)
            }
            "get_metadata" => {
                let uri = self.executor.reader_uri();
                json!({
                    "name": executor::display_name_for_uri(uri),
                    "language_id": "ggsql",
                    "host": executor::host_for_uri(uri),
                    "type": executor::type_name_for_uri(uri),
                    "code": format!("-- @connect: {}", uri)
                })
            }
            _ => {
                tracing::warn!("Unknown connection method: {}", method);
                json!(null)
            }
        };

        self.send_shell_reply(
            "comm_msg",
            json!({
                "comm_id": comm_id,
                "data": {
                    "jsonrpc": "2.0",
                    "id": rpc_id,
                    "result": result
                }
            }),
            parent,
            identities,
        )
        .await?;

        Ok(())
    }

    /// Handle JSON-RPC requests on a data explorer comm
    async fn handle_data_explorer_rpc(
        &mut self,
        method: &str,
        rpc_id: &Value,
        comm_id: &str,
        parent: &JupyterMessage,
        identities: &[Vec<u8>],
    ) -> Result<()> {
        tracing::info!("Data explorer RPC: {}", method);

        let params = &parent.content["data"]["params"];

        let RpcResponse { result, event } =
            if let Some(state) = self.data_explorer_comms.get(comm_id) {
                state.handle_rpc(method, params, self.executor.reader())
            } else {
                RpcResponse::reply(json!(null))
            };

        // Send the RPC reply
        self.send_shell_reply(
            "comm_msg",
            json!({
                "comm_id": comm_id,
                "data": {
                    "jsonrpc": "2.0",
                    "id": rpc_id,
                    "result": result
                }
            }),
            parent,
            identities,
        )
        .await?;

        // Send async event on iopub if present (e.g. return_column_profiles)
        if let Some(evt) = event {
            self.send_iopub(
                "comm_msg",
                json!({
                    "comm_id": comm_id,
                    "data": {
                        "jsonrpc": "2.0",
                        "method": evt.method,
                        "params": evt.params
                    }
                }),
                parent,
            )
            .await?;
        }

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
