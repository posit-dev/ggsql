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
    /// The default row label: a `Stat`'s name; a `Template`'s literal text
    /// with each placeholder reduced to its bare operation name
    /// (`{mean:num %.2f} +/- {sdev}` becomes `mean +/- sdev`).
    pub(crate) fn as_label(&self) -> String {
        match self {
            SummaryOperation::Stat(name) => name.clone(),
            // A template that reaches here unparseable keeps its raw text;
            // execution parses (and errors) before building any label.
            SummaryOperation::Template(template) => match parse_template(template) {
                Ok(segments) => segments
                    .iter()
                    .map(|segment| match segment {
                        TemplateSegment::Literal(text) => text.as_str(),
                        TemplateSegment::Stat { name, .. } => name.as_str(),
                    })
                    .collect(),
                Err(_) => template.clone(),
            },
        }
    }

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
#[derive(Debug, Clone)]
pub(crate) enum TemplateSegment {
    /// Text copied to the cell verbatim.
    Literal(String),
    /// A `{name}` or `{name:spec}` placeholder, its zygos template
    /// pre-compiled: `{:spec}`, or `{}` without a spec. The `%`-sugar is
    /// already expanded (`{sd:%.1f}` compiles `{:num %.1f}`).
    Stat {
        name: String,
        template: zygos::Template,
    },
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
        // `{sd:%.1f}` is sugar for `{sd:num %.1f}`.
        let source = match spec {
            Some(spec) if spec.starts_with('%') => format!("{{:num {spec}}}"),
            Some(spec) => format!("{{:{spec}}}"),
            None => "{}".to_string(),
        };
        segments.push(TemplateSegment::Stat {
            name: name.to_string(),
            template: zygos::Template::parse(&source),
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
/// `diff`), which need row-order machinery tables don't have, plus `se` and
/// the type-free tallies `n_distinct` and `n_missing`.
pub const SUMMARY_OPERATIONS: &[&str] = &[
    "count",
    "n_distinct",
    "n_missing",
    "sum",
    "prod",
    "min",
    "max",
    "range",
    "mid",
    "mean",
    "geomean",
    "harmean",
    "rms",
    "median",
    "sdev",
    "se",
    "var",
    "iqr",
    "p05",
    "p10",
    "p25",
    "p50",
    "p75",
    "p90",
    "p95",
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
    fn summary_operation_as_label_strips_a_templates_specs() {
        assert_eq!(
            SummaryOperation::Template("{mean:num %.2f} +/- {sdev}".to_string()).as_label(),
            "mean +/- sdev"
        );
        assert_eq!(
            SummaryOperation::Stat("MEAN".to_string()).as_label(),
            "MEAN"
        );
    }

    #[test]
    fn parse_template_splits_literals_and_stats() {
        let segments = parse_template("{mean} +/- {sdev:num %.3f} (n={count})").unwrap();
        let stat = |index: usize| match &segments[index] {
            TemplateSegment::Stat { name, template } => (name.as_str(), template.source()),
            TemplateSegment::Literal(_) => panic!("segment {index} is a literal"),
        };
        let literal = |index: usize| match &segments[index] {
            TemplateSegment::Literal(text) => text.as_str(),
            TemplateSegment::Stat { .. } => panic!("segment {index} is a stat"),
        };

        assert_eq!(segments.len(), 6);
        assert_eq!(stat(0), ("mean", "{}"));
        assert_eq!(literal(1), " +/- ");
        assert_eq!(stat(2), ("sdev", "{:num %.3f}"));
        assert_eq!(literal(3), " (n=");
        assert_eq!(stat(4), ("count", "{}"));
        assert_eq!(literal(5), ")");
    }

    #[test]
    fn parse_template_expands_the_printf_sugar() {
        let segments = parse_template("{sd:%.1f}").unwrap();
        match &segments[0] {
            TemplateSegment::Stat { name, template } => {
                assert_eq!(name, "sd");
                assert_eq!(template.source(), "{:num %.1f}");
            }
            TemplateSegment::Literal(_) => panic!("expected a stat segment"),
        }
    }

    #[test]
    fn parse_template_treats_a_stray_close_brace_as_literal() {
        let segments = parse_template("a } b").unwrap();
        assert_eq!(segments.len(), 1);
        match &segments[0] {
            TemplateSegment::Literal(text) => assert_eq!(text, "a } b"),
            TemplateSegment::Stat { .. } => panic!("expected a literal segment"),
        }
    }

    #[test]
    fn parse_template_rejects_unterminated_and_empty_placeholders() {
        assert!(parse_template("{mean").is_err());
        assert!(parse_template("{}").is_err());
        assert!(parse_template("{:num}").is_err());
    }
}
