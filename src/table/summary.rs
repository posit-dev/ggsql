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

impl SummaryOperation {
    /// Validate against `SUMMARY_OPERATIONS`, case-insensitively: a `Stat`
    /// directly, a `Template`'s placeholders one by one.
    pub fn validate(&self) -> Result<(), String> {
        let unknown = |name: &str| {
            format!(
                "Unknown operation '{}'. Valid operations: {}",
                name,
                crate::or_list_quoted(SUMMARY_OPERATIONS, '\'')
            )
        };
        let known = |name: &str| {
            SUMMARY_OPERATIONS
                .iter()
                .any(|op| name.eq_ignore_ascii_case(op))
        };
        match self {
            SummaryOperation::Stat(name) if !known(name) => Err(unknown(name)),
            SummaryOperation::Stat(_) => Ok(()),
            SummaryOperation::Template(template) => {
                for segment in parse_template(template)? {
                    if let TemplateSegment::Stat { name, .. } = segment {
                        if !known(&name) {
                            return Err(unknown(&name));
                        }
                    }
                }
                Ok(())
            }
        }
    }
}

/// One piece of a `SUMMARISE` template: literal text, or a `{name:spec}`
/// placeholder naming a stat to evaluate and interpolate.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum TemplateSegment {
    /// Text copied to the cell verbatim.
    Literal(String),
    /// A `{name}` or `{name:spec}` placeholder. `spec` is the zygos label
    /// spec with the `%`-sugar already expanded (`{sd:%.1f}` stores
    /// `num %.1f`).
    Stat { name: String, spec: Option<String> },
}

/// Split a template literal into text and placeholder segments. `{` opens a
/// placeholder and must be terminated and non-empty; `}` outside a
/// placeholder is literal text. Matching against `SUMMARY_OPERATIONS` happens
/// in [`SummaryOperation::validate`], not here.
pub(crate) fn parse_template(template: &str) -> Result<Vec<TemplateSegment>, String> {
    let mut segments = Vec::new();
    let mut literal = String::new();
    let mut rest = template;
    while let Some(open) = rest.find('{') {
        literal.push_str(&rest[..open]);
        rest = &rest[open + 1..];
        let Some(close) = rest.find('}') else {
            return Err(format!("Missing '}}' in SUMMARISE template '{template}'"));
        };
        let placeholder = rest[..close].trim();
        rest = &rest[close + 1..];
        if !literal.is_empty() {
            segments.push(TemplateSegment::Literal(std::mem::take(&mut literal)));
        }
        if placeholder.is_empty() {
            return Err(format!("Empty '{{}}' in SUMMARISE template '{template}'"));
        }
        let (name, spec) = match placeholder.split_once(':') {
            Some((name, spec)) => (name.trim(), Some(spec.trim())),
            None => (placeholder, None),
        };
        if name.is_empty() {
            return Err(format!(
                "Missing operation name before ':' in SUMMARISE template '{template}'"
            ));
        }
        segments.push(TemplateSegment::Stat {
            name: name.to_string(),
            spec: spec.map(|spec| {
                // `{sd:%.1f}` is sugar for `{sd:num %.1f}`.
                if let Some(printf) = spec.strip_prefix('%') {
                    format!("num %{printf}")
                } else {
                    spec.to_string()
                }
            }),
        });
    }
    literal.push_str(rest);
    if !literal.is_empty() {
        segments.push(TemplateSegment::Literal(literal));
    }
    Ok(segments)
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
    fn summary_operation_validate_accepts_case_insensitively() {
        assert!(SummaryOperation::Stat("mean".to_string())
            .validate()
            .is_ok());
        assert!(SummaryOperation::Stat("MEAN".to_string())
            .validate()
            .is_ok());
    }

    #[test]
    fn summary_operation_validate_rejects_unknown_and_positional() {
        assert!(SummaryOperation::Stat("meen".to_string())
            .validate()
            .is_err());
        assert!(SummaryOperation::Stat("first".to_string())
            .validate()
            .is_err());
    }

    #[test]
    fn summary_operation_validate_checks_a_templates_placeholders() {
        assert!(SummaryOperation::Template("{mean} +/- {SDEV}".to_string())
            .validate()
            .is_ok());
        assert!(SummaryOperation::Template("{mean} +/- {meen}".to_string())
            .validate()
            .is_err());
    }

    #[test]
    fn parse_template_splits_literals_and_stats() {
        let segments = parse_template("{mean} +/- {sdev:num %.3f} (n={count})").unwrap();
        assert_eq!(
            segments,
            vec![
                TemplateSegment::Stat {
                    name: "mean".to_string(),
                    spec: None
                },
                TemplateSegment::Literal(" +/- ".to_string()),
                TemplateSegment::Stat {
                    name: "sdev".to_string(),
                    spec: Some("num %.3f".to_string())
                },
                TemplateSegment::Literal(" (n=".to_string()),
                TemplateSegment::Stat {
                    name: "count".to_string(),
                    spec: None
                },
                TemplateSegment::Literal(")".to_string()),
            ]
        );
    }

    #[test]
    fn parse_template_expands_the_printf_sugar() {
        let segments = parse_template("{sd:%.1f}").unwrap();
        assert_eq!(
            segments,
            vec![TemplateSegment::Stat {
                name: "sd".to_string(),
                spec: Some("num %.1f".to_string())
            }]
        );
    }

    #[test]
    fn parse_template_treats_a_stray_close_brace_as_literal() {
        let segments = parse_template("a } b").unwrap();
        assert_eq!(
            segments,
            vec![TemplateSegment::Literal("a } b".to_string())]
        );
    }

    #[test]
    fn parse_template_rejects_unterminated_and_empty_placeholders() {
        assert!(parse_template("{mean").is_err());
        assert!(parse_template("{}").is_err());
        assert!(parse_template("{:num}").is_err());
    }
}
