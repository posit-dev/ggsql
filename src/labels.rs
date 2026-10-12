//! Text labels from `LABEL` clauses, shared by plots and tables.
//!
//! A `LABEL` clause maps a name — an aesthetic for `VISUALISE`, a column or
//! spanner id for `TABULATE` — to display text. `None` suppresses the label.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Text labels (from `LABEL` clause)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Labels {
    /// Label assignments (name → text, None = suppress)
    pub labels: HashMap<String, Option<String>>,
}

impl Labels {
    /// Merge another clause's labels into this one; later entries win.
    pub fn merge(&mut self, other: Labels) {
        self.labels.extend(other.labels);
    }

    /// Look up a name: `None` if there is no entry, `Some(None)` if the
    /// label is suppressed, `Some(Some(text))` for an explicit label.
    pub fn lookup(&self, name: &str) -> Option<Option<&str>> {
        self.labels.get(name).map(|v| v.as_deref())
    }

    /// Whether a name has an entry (set or suppressed).
    pub fn contains(&self, name: &str) -> bool {
        self.labels.contains_key(name)
    }
}
