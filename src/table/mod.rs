//! Table types for ggsql specification
//!
//! Defines the typed `Table` structure that represents parsed `TABULATE`
//! statements, parallel to how `plot` defines `Plot` for `VISUALISE`
//! statements: `source` (from `TABULATE FROM`), `labels` (from `TABULATE
//! LABEL`), `spans` (from `TABULATE SPAN`), and `formats` (from `TABULATE
//! FORMAT`) are populated so far.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::plot::{
    validate_parameter, DefaultParamValue, Labels, NumberConstraint, ParamConstraint,
    ParamDefinition, Parameters,
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
    /// Cell formatting (from `TABULATE FORMAT`), one per `FORMAT` clause
    /// written — same one-clause-per-group model as `spans`.
    pub formats: Vec<Format>,
}

impl Table {
    /// Create a new empty Table.
    pub fn new() -> Self {
        Self {
            source: None,
            labels: Labels::default(),
            spans: Vec::new(),
            formats: Vec::new(),
        }
    }
}

impl Default for Table {
    fn default() -> Self {
        Self::new()
    }
}

impl Table {
    /// Expand a SPAN's `ACROSS` entry that names an earlier spanner's `id`
    /// into that spanner's own columns — `SPAN 'Y' ACROSS x_id, c` (after
    /// `SPAN 'X' ACROSS a, b SETTING id => 'x_id'`) resolves to columns a,
    /// b, c for `Y`. Only ids from earlier spans are recognised; an id
    /// declared later, or a genuine typo, is left as a literal string and
    /// caught downstream as an unknown column, the same as any other bad
    /// reference.
    ///
    /// Lives here rather than alongside its sibling spanner-resolution
    /// steps in `execute::table_spanner` because it operates on `Spanner`
    /// alone, with no `DataFrame` involved — `validate()` calls it directly
    /// to catch a duplicate id without depending on `execute`.
    pub fn resolve_spanner_ids(&self) -> Result<Vec<Spanner>, String> {
        let mut resolved: Vec<Spanner> = Vec::with_capacity(self.spans.len());
        let mut ids: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

        for span in &self.spans {
            let columns = span
                .columns
                .iter()
                .flat_map(|entry| match ids.get(entry) {
                    Some(&idx) => resolved[idx].columns.clone(),
                    None => vec![entry.clone()],
                })
                .collect();

            let mut resolved_span = span.clone();
            resolved_span.columns = columns;

            if let Some(id) = resolved_span.settings.get("id").and_then(|v| v.as_str()) {
                if ids.insert(id.to_string(), resolved.len()).is_some() {
                    return Err(format!("Duplicate SPAN id '{id}'"));
                }
            }

            resolved.push(resolved_span);
        }

        Ok(resolved)
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
    ParamDefinition {
        name: "id",
        // No default: absence means this spanner has no id and can't be
        // referenced by a later one's ACROSS list.
        default: DefaultParamValue::Null,
        constraint: ParamConstraint::string(),
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

/// One `FORMAT` clause: cell formatting for a group of columns.
///
/// `settings` is validated against `FORMAT_PARAMS` and resolved into each
/// covered column's `TableCell` properties — not yet consumed by any writer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Format {
    /// The columns this FORMAT applies to, in the order written.
    pub columns: Vec<String>,
    /// `SETTING` parameters for this FORMAT (e.g. `hjust => 'right'`).
    pub settings: Parameters,
    /// Value mappings for custom cell display (`RENAMING` clause). Maps a raw
    /// cell value to its display text; `None` suppresses the cell's text.
    /// Same shape as `Scale::label_mapping` — named `value_mapping` rather
    /// than `label_mapping` here because `Table::labels` already uses
    /// "label" for column headers, a different concept from a cell's value.
    #[serde(default)]
    pub value_mapping: Option<HashMap<String, Option<String>>>,
    /// Template for generating display text from cell values (e.g.
    /// `"{:num %.2f}"`), applied to values with no specific `value_mapping`
    /// entry. Default `"{}"` passes the value through unchanged. Same shape
    /// as `Scale::label_template`.
    #[serde(default = "crate::format::default_template")]
    pub value_template: String,
}

/// `SETTING` parameters `FORMAT` accepts.
const FORMAT_PARAMS: &[ParamDefinition] = &[ParamDefinition {
    name: "hjust",
    default: DefaultParamValue::Number(0.5),
    // Both spellings accepted here; resolve_column_properties standardises
    // "centre" to "center" when it builds a column's TableCell properties.
    constraint: ParamConstraint::string_option_or_number(
        &["left", "right", "centre", "center"],
        NumberConstraint::range(0.0, 1.0),
    ),
}];

impl Format {
    /// Validate `settings` against `FORMAT_PARAMS`.
    pub fn validate_settings(&self) -> Result<(), String> {
        let valid: Vec<&str> = FORMAT_PARAMS.iter().map(|p| p.name).collect();

        for (name, value) in &self.settings {
            let Some(param) = FORMAT_PARAMS.iter().find(|p| p.name == name.as_str()) else {
                return Err(format!(
                    "FORMAT setting should be {}, not '{}'",
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

    fn format_with_settings(settings: Parameters) -> Format {
        Format {
            columns: vec!["a".to_string()],
            settings,
            value_mapping: None,
            value_template: "{}".to_string(),
        }
    }

    #[test]
    fn format_validate_settings_accepts_a_string_hjust() {
        let mut settings = Parameters::new();
        settings.insert(
            "hjust".to_string(),
            ParameterValue::String("right".to_string()),
        );

        assert!(format_with_settings(settings).validate_settings().is_ok());
    }

    #[test]
    fn format_validate_settings_accepts_a_numeric_hjust() {
        let mut settings = Parameters::new();
        settings.insert("hjust".to_string(), ParameterValue::Number(0.25));

        assert!(format_with_settings(settings).validate_settings().is_ok());
    }

    #[test]
    fn format_validate_settings_rejects_an_unrecognized_string() {
        let mut settings = Parameters::new();
        settings.insert(
            "hjust".to_string(),
            ParameterValue::String("up".to_string()),
        );

        assert!(format_with_settings(settings).validate_settings().is_err());
    }
}
