//! Text titles from `TITLE` clauses, shared by plots and tables.
//!
//! A `TITLE` clause maps a name — an aesthetic for `VISUALISE`, a column or
//! spanner id for `TABULATE` — to display text. `None` suppresses the title.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Text titles (from `TITLE` clause)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Titles {
    /// Title assignments (name → text, None = suppress)
    pub titles: HashMap<String, Option<String>>,
}

impl Titles {
    /// Merge another clause's titles into this one; later entries win.
    pub fn merge(&mut self, other: Titles) {
        self.titles.extend(other.titles);
    }

    /// Whether a name has an entry (set or suppressed).
    pub fn contains(&self, name: &str) -> bool {
        self.titles.contains_key(name)
    }

    /// Look up a title's text.
    /// Both "not set" and suppressed (`TITLE <name> => NULL`) return `None`.
    pub fn get(&self, name: &str) -> Option<&str> {
        self.titles.get(name)?.as_deref()
    }

    /// Look up a title entry:
    /// * `None` means not set,
    /// * `Some(None)` means explicitly suppressed (`TITLE <name> => NULL`).
    /// * `Some(Some(String))` means typical title
    pub fn entry(&self, name: &str) -> Option<&Option<String>> {
        self.titles.get(name)
    }
}
