//! Jupyter message structures and connection info
//!
//! This module defines the Jupyter messaging protocol structures used
//! for communication between the kernel and clients.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Connection information from Jupyter
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ConnectionInfo {
    pub ip: String,
    pub transport: String,
    pub signature_scheme: String,
    pub key: String,
    pub shell_port: u16,
    pub iopub_port: u16,
    pub stdin_port: u16,
    pub control_port: u16,
    pub hb_port: u16,
}

impl ConnectionInfo {
    /// Load connection info from a JSON file
    pub fn from_file(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let info: ConnectionInfo = serde_json::from_str(&content)?;
        Ok(info)
    }

    /// Get the socket address for a given port
    pub fn socket_addr(&self, port: u16) -> String {
        format!("{}://{}:{}", self.transport, self.ip, port)
    }
}

/// A Jupyter protocol message
#[derive(Debug, Serialize, Deserialize)]
pub struct JupyterMessage {
    pub header: MessageHeader,
    pub parent_header: Value,
    pub metadata: Value,
    pub content: Value,
    #[serde(default)]
    pub buffers: Vec<Vec<u8>>,
}

/// Message header
#[derive(Debug, Serialize, Deserialize)]
pub struct MessageHeader {
    pub msg_id: String,
    pub session: String,
    pub username: String,
    pub date: String,
    pub msg_type: String,
    pub version: String,
}
