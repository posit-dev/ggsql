//! Table types for ggsql specification
//!
//! This module will define the typed `Table` structure that represents parsed
//! `TABULATE` statements, parallel to how `plot` defines `Plot` for `VISUALISE`
//! statements. Still minimal: `source` (from `TABULATE FROM`) and `labels`
//! (from `TABULATE LABEL`) are populated so far.

use serde::{Deserialize, Serialize};

use crate::plot::{
    validate_parameter, DefaultParamValue, Labels, ParamConstraint, ParamDefinition, Parameters,
};
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
    /// Column display labels (from `TABULATE LABEL`). Reuses `plot::Labels`
    /// as-is — the same "name → text, None = suppress" shape applies
    /// unchanged, just keyed by column name instead of aesthetic name. An
    /// empty `Labels` means no overrides at all.
    pub labels: Labels,
    /// Column spanners (from `TABULATE SPAN`), one per `SPAN` clause written
    /// — a query with several `SPAN` clauses produces one `Spanner` each,
    /// not one `Spanner` with several groups (`SPAN` itself never bundles
    /// more than one group per clause).
    pub spans: Vec<Spanner>,
}

impl Table {
    /// Create a new empty Table.
    pub fn new() -> Self {
        Self {
            source: None,
            labels: Labels::default(),
            spans: Vec::new(),
        }
    }
}

impl Default for Table {
    fn default() -> Self {
        Self::new()
    }
}

/// One `SPAN` clause: a named group of columns rendered as one spanner cell
/// above them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Spanner {
    /// Display text for the spanner cell. `None` is `SPAN NULL` — suppress
    /// the spanner cell but keep the column grouping (e.g. for `settings`
    /// that apply to the group regardless of whether it has a visible
    /// label). `Some(String::new())` is `SPAN ''` — a present but blank
    /// cell. Mirrors `label_assignment`'s own string/NULL value shape.
    pub label: Option<String>,
    /// The columns this spanner covers, in the order written.
    pub columns: Vec<String>,
    /// `SETTING` parameters for this spanner (e.g. `width => '40%'`).
    pub settings: Parameters,
}

/// `SETTING` parameters `SPAN` accepts — one static name/default/constraint
/// list, mirroring how a `GeomTrait` declares `default_params()`, so
/// `Spanner::validate_settings` doesn't hand-roll its own type checks per key.
const SPAN_PARAMS: &[ParamDefinition] = &[
    ParamDefinition {
        name: "gather",
        default: DefaultParamValue::Boolean(true),
        constraint: ParamConstraint::boolean(),
    },
    ParamDefinition {
        name: "level",
        // No default: absence means "assign automatically" (see
        // assign_spanner_levels), not "assume some fixed level".
        default: DefaultParamValue::Null,
        constraint: ParamConstraint::count(1.0),
    },
];

impl Spanner {
    /// Validate `settings` against `SPAN_PARAMS`, mirroring
    /// `Layer::validate_settings`.
    pub fn validate_settings(&self) -> Result<(), String> {
        let valid: Vec<&str> = SPAN_PARAMS.iter().map(|p| p.name).collect();

        for (name, value) in &self.settings {
            // Key isn't in SPAN_PARAMS at all.
            let Some(param) = SPAN_PARAMS.iter().find(|p| p.name == name.as_str()) else {
                return Err(format!(
                    "SPAN setting should be {}, not '{}'",
                    crate::or_list_quoted(&valid, '\''),
                    name
                ));
            };
            // Known key, but wrong shape (e.g. `gather` not a boolean).
            validate_parameter(name, value, &param.constraint)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::ParameterValue;

    fn spanner_with_settings(settings: Parameters) -> Spanner {
        Spanner {
            label: Some(String::new()),
            columns: vec!["a".to_string()],
            settings,
        }
    }

    #[test]
    fn validate_settings_accepts_valid_gather_and_level() {
        let mut settings = Parameters::new();
        settings.insert("gather".to_string(), ParameterValue::Boolean(false));
        settings.insert("level".to_string(), ParameterValue::Number(2.0));

        assert!(spanner_with_settings(settings).validate_settings().is_ok());
    }

    #[test]
    fn validate_settings_rejects_an_unknown_key() {
        let mut settings = Parameters::new();
        settings.insert(
            "width".to_string(),
            ParameterValue::String("40%".to_string()),
        );

        assert!(spanner_with_settings(settings).validate_settings().is_err());
    }
}
