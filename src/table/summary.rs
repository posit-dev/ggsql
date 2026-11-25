//! Row summaries: the AST for `TABULATE SUMMARISE` clauses.

use serde::{Deserialize, Serialize};

use crate::params::{DefaultParamValue, ParamConstraint, ParamDefinition, Parameters};

/// One `SUMMARISE` clause: an aggregate operation producing one summary row
/// per group — or a single row over all rows for `GRAND`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Summary {
    /// The aggregate operation, verbatim as written.
    pub operation: SummaryOperation,
    /// Which rows the operation aggregates over.
    pub scope: SummaryScope,
    /// The columns to summarise, in the order written.
    pub columns: Vec<String>,
    /// `SETTING` parameters for this SUMMARISE (e.g. `place => 'top'`,
    /// `rowlabel => 'Average'`).
    pub settings: Parameters,
}

/// What a `SUMMARISE` clause computes per column: either one named stat, or
/// a template whose `{stat}` placeholders are each evaluated and interpolated
/// into the cell text.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SummaryOperation {
    /// A bare stat name (e.g. `mean`) — it doubles as the default row label.
    /// Matching against the operation vocabulary is case-insensitive at
    /// execution time.
    Stat(String),
    /// A template literal (e.g. `{mean} +/- {sdev}`), placeholders intact.
    Template(String),
}

impl std::fmt::Display for SummaryOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SummaryOperation::Stat(name) => write!(f, "{name}"),
            SummaryOperation::Template(template) => write!(f, "'{template}'"),
        }
    }
}

/// Which rows a `SUMMARISE` clause aggregates over.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SummaryScope {
    /// One summary row per facet group. Without a `FACET`, all rows form a
    /// single group.
    #[default]
    Group,
    /// A single summary row over all rows.
    Grand,
}

/// Operations a `SUMMARISE` clause accepts: the plot aggregate vocabulary
/// (`AGG_NAMES`) minus the order-dependent positional ones (`first`, `last`,
/// `diff`), which need row-order machinery tables don't have.
pub const SUMMARY_OPERATIONS: &[&str] = &[
    "count", "sum", "prod", "min", "max", "range", "mid", "mean", "geomean", "harmean", "rms",
    "median", "sdev", "var", "iqr", "p05", "p10", "p25", "p50", "p75", "p90", "p95",
];

/// `SETTING` parameters `SUMMARISE` accepts.
const SUMMARY_PARAMS: &[ParamDefinition] = &[
    ParamDefinition {
        name: "place",
        // No default: an unset place resolves at execution time.
        default: DefaultParamValue::Null,
        constraint: ParamConstraint::string_option(&["top", "bottom"]),
    },
    ParamDefinition {
        name: "rowlabel",
        // No default: an unset rowlabel falls back to the operation verbatim.
        default: DefaultParamValue::Null,
        constraint: ParamConstraint::string(),
    },
    ParamDefinition {
        // Whether nulls make the summary null ('propagate') or are ignored
        // ('drop').
        name: "missing",
        default: DefaultParamValue::String("propagate"),
        constraint: ParamConstraint::string_option(&["propagate", "drop"]),
    },
];

impl Summary {
    /// Validate a `Stat` operation against `SUMMARY_OPERATIONS`,
    /// case-insensitively. A `Template` is not checked against the
    /// vocabulary here.
    pub fn validate_operation(&self) -> Result<(), String> {
        let SummaryOperation::Stat(name) = &self.operation else {
            return Ok(());
        };
        if SUMMARY_OPERATIONS
            .iter()
            .any(|op| name.eq_ignore_ascii_case(op))
        {
            Ok(())
        } else {
            Err(format!(
                "Unknown operation '{}'. Valid operations: {}",
                name,
                crate::or_list_quoted(SUMMARY_OPERATIONS, '\'')
            ))
        }
    }

    /// Validate `settings` against `SUMMARY_PARAMS` and resolve each to its
    /// effective value: the user's value if present, otherwise the
    /// definition's default.
    pub fn resolve_settings(&self) -> Result<Parameters, String> {
        crate::params::resolve_parameters(&self.settings, SUMMARY_PARAMS, "SUMMARISE setting")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn summary_with_settings(settings: Parameters) -> Summary {
        Summary {
            operation: SummaryOperation::Stat("mean".to_string()),
            scope: SummaryScope::Group,
            columns: vec!["a".to_string()],
            settings,
        }
    }

    #[test]
    fn summary_resolve_settings_accepts_place_and_rowlabel() {
        let settings = crate::params! { "place" => "top", "rowlabel" => "Average" };

        assert!(summary_with_settings(settings).resolve_settings().is_ok());
    }

    #[test]
    fn summary_resolve_settings_accepts_missing_options() {
        for value in ["propagate", "drop"] {
            let settings = crate::params! { "missing" => value };
            assert!(summary_with_settings(settings).resolve_settings().is_ok());
        }
        let settings = crate::params! { "missing" => "ignore" };
        assert!(summary_with_settings(settings).resolve_settings().is_err());
    }

    #[test]
    fn summary_resolve_settings_rejects_a_non_string_rowlabel() {
        let settings = crate::params! { "rowlabel" => 1.0 };

        assert!(summary_with_settings(settings).resolve_settings().is_err());
    }

    #[test]
    fn summary_resolve_settings_rejects_an_unknown_setting() {
        let settings = crate::params! { "colour" => "red" };

        assert!(summary_with_settings(settings).resolve_settings().is_err());
    }

    #[test]
    fn summary_validate_operation_accepts_case_insensitively() {
        assert!(summary_with_settings(Parameters::new())
            .validate_operation()
            .is_ok());
        let mut summary = summary_with_settings(Parameters::new());
        summary.operation = SummaryOperation::Stat("MEAN".to_string());
        assert!(summary.validate_operation().is_ok());
    }

    #[test]
    fn summary_validate_operation_rejects_unknown_and_positional() {
        let mut summary = summary_with_settings(Parameters::new());
        summary.operation = SummaryOperation::Stat("meen".to_string());
        assert!(summary.validate_operation().is_err());
        summary.operation = SummaryOperation::Stat("first".to_string());
        assert!(summary.validate_operation().is_err());
    }

    #[test]
    fn summary_validate_operation_passes_a_template() {
        let mut summary = summary_with_settings(Parameters::new());
        summary.operation = SummaryOperation::Template("{meen}".to_string());
        assert!(summary.validate_operation().is_ok());
    }
}
