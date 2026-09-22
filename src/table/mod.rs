//! Table types for ggsql specification
//!
//! This module will define the typed `Table` structure that represents parsed
//! `TABULATE` statements, parallel to how `plot` defines `Plot` for `VISUALISE`
//! statements. It is currently minimal: only `source` (from `TABULATE FROM`)
//! is populated so far.

use serde::{Deserialize, Serialize};

use crate::DataSource;

/// Complete ggsql table specification.
///
/// Parallel to [`crate::Plot`], but for `TABULATE` statements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Table {
    /// `FROM` source (CTE, table, or file path) from `TABULATE FROM`. Unlike
    /// `Plot`, there are no layers to hold a per-layer source override, so
    /// this is the only place a `TABULATE`'s data source can come from.
    pub source: Option<DataSource>,
}

impl Table {
    /// Create a new empty Table.
    pub fn new() -> Self {
        Self { source: None }
    }
}

impl Default for Table {
    fn default() -> Self {
        Self::new()
    }
}
