//! Table types for ggsql specification
//!
//! This module will define the typed `Table` structure that represents parsed
//! `TABULATE` statements, parallel to how `plot` defines `Plot` for `VISUALISE`
//! statements. It is currently a stub: no clauses parse into it yet.

use serde::{Deserialize, Serialize};

/// Complete ggsql table specification.
///
/// Parallel to [`crate::Plot`], but for `TABULATE` statements. No fields yet —
/// this exists so the rest of the pipeline (parser, [`crate::Spec`]) has a
/// concrete type to route through before any table-specific syntax lands.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Table {}
