//! Row summaries: the AST for `TABULATE SUMMARISE` clauses.

use serde::{Deserialize, Serialize};

use crate::params::{
    validate_parameter, DefaultParamValue, ParamConstraint, ParamDefinition, Parameters,
};

/// One `SUMMARISE` clause: an aggregate operation producing one summary row
/// per group — or a single row over all rows for `GRAND`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Summary {
    /// The aggregate operation (e.g. `mean`), verbatim as written — it doubles
    /// as the default row label. Matching against the operation vocabulary is
    /// case-insensitive at execution time.
    pub operation: String,
    /// Which rows the operation aggregates over.
    pub scope: SummaryScope,
    /// The columns to summarise, in the order written.
    pub columns: Vec<String>,
    /// `SETTING` parameters for this SUMMARISE (e.g. `place => 'top'`,
    /// `rowlabel => 'Average'`).
    pub settings: Parameters,
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
    /// Validate `operation` against `SUMMARY_OPERATIONS`, case-insensitively.
    pub fn validate_operation(&self) -> Result<(), String> {
        if SUMMARY_OPERATIONS
            .iter()
            .any(|op| self.operation.eq_ignore_ascii_case(op))
        {
            Ok(())
        } else {
            Err(format!(
                "Unknown operation '{}'. Valid operations: {}",
                self.operation,
                crate::or_list_quoted(SUMMARY_OPERATIONS, '\'')
            ))
        }
    }

    /// Validate `settings` against `SUMMARY_PARAMS`.
    pub fn validate_settings(&self) -> Result<(), String> {
        let valid: Vec<&str> = SUMMARY_PARAMS.iter().map(|p| p.name).collect();

        for (name, value) in &self.settings {
            let Some(param) = SUMMARY_PARAMS.iter().find(|p| p.name == name.as_str()) else {
                return Err(format!(
                    "SUMMARISE setting should be {}, not '{}'",
                    crate::or_list_quoted(&valid, '\''),
                    name
                ));
            };
            validate_parameter(name, value, &param.constraint)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::ParameterValue;

    fn summary_with_settings(settings: Parameters) -> Summary {
        Summary {
            operation: "mean".to_string(),
            scope: SummaryScope::Group,
            columns: vec!["a".to_string()],
            settings,
        }
    }

    #[test]
    fn summary_validate_settings_accepts_place_and_rowlabel() {
        let mut settings = Parameters::new();
        settings.insert(
            "place".to_string(),
            ParameterValue::String("top".to_string()),
        );
        settings.insert(
            "rowlabel".to_string(),
            ParameterValue::String("Average".to_string()),
        );

        assert!(summary_with_settings(settings).validate_settings().is_ok());
    }

    #[test]
    fn summary_validate_settings_accepts_missing_options() {
        for value in ["propagate", "drop"] {
            let mut settings = Parameters::new();
            settings.insert(
                "missing".to_string(),
                ParameterValue::String(value.to_string()),
            );
            assert!(summary_with_settings(settings).validate_settings().is_ok());
        }
        let mut settings = Parameters::new();
        settings.insert(
            "missing".to_string(),
            ParameterValue::String("ignore".to_string()),
        );
        assert!(summary_with_settings(settings).validate_settings().is_err());
    }

    #[test]
    fn summary_validate_settings_rejects_a_non_string_rowlabel() {
        let mut settings = Parameters::new();
        settings.insert("rowlabel".to_string(), ParameterValue::Number(1.0));

        assert!(summary_with_settings(settings).validate_settings().is_err());
    }

    #[test]
    fn summary_validate_settings_rejects_an_unknown_setting() {
        let mut settings = Parameters::new();
        settings.insert(
            "colour".to_string(),
            ParameterValue::String("red".to_string()),
        );

        assert!(summary_with_settings(settings).validate_settings().is_err());
    }

    #[test]
    fn summary_validate_operation_accepts_case_insensitively() {
        assert!(summary_with_settings(Parameters::new())
            .validate_operation()
            .is_ok());
        let mut summary = summary_with_settings(Parameters::new());
        summary.operation = "MEAN".to_string();
        assert!(summary.validate_operation().is_ok());
    }

    #[test]
    fn summary_validate_operation_rejects_unknown_and_positional() {
        let mut summary = summary_with_settings(Parameters::new());
        summary.operation = "meen".to_string();
        assert!(summary.validate_operation().is_err());
        summary.operation = "first".to_string();
        assert!(summary.validate_operation().is_err());
    }
}
