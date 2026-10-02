//! Table types for ggsql specification
//!
//! Defines the typed `Table` structure that represents parsed `TABULATE`
//! statements, parallel to how `plot` defines `Plot` for `VISUALISE`
//! statements: `source` (from `TABULATE FROM`), `selection` (from
//! `TABULATE`'s own column list), `labels` (from `TABULATE LABEL`), `spans`
//! (from `TABULATE SPAN`), and `formats` (from `TABULATE FORMAT`) are
//! populated so far.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

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
    /// Column selection from `TABULATE`'s own column list, right after the
    /// keyword (e.g. `TABULATE bill_len, bill_dep AS Depth`) — mandatory in
    /// the grammar. Defaults to `*` for a `Table` built without the parser.
    pub selection: Vec<SelectionItem>,
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

/// One item in a TABULATE column selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SelectionItem {
    /// `*`.
    Wildcard,
    /// A column or a constant. `sql` is the verbatim source text (e.g.
    /// `bill_dep AS Depth`); `name` is the output column name (`Depth`).
    Column {
        sql: String,
        name: String,
        source: Option<String>,
    },
}

impl Table {
    /// Create a new empty Table.
    pub fn new() -> Self {
        Self {
            source: None,
            selection: vec![SelectionItem::Wildcard],
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
    /// Expand a SPAN's `OVER` entry that names an earlier spanner's `id`
    /// into that spanner's own columns — `SPAN y OVER x_id, c` (after
    /// `SPAN x_id OVER a, b`) resolves to columns a, b, c for `y`. Only
    /// ids from earlier spans are recognised; an id declared later, or a
    /// genuine typo, is left as a literal string and caught downstream as
    /// an unknown column, the same as any other bad reference.
    ///
    /// Lives here rather than alongside its sibling spanner-resolution
    /// steps in `execute::table::spanner` because it operates on `Spanner`
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

            if ids
                .insert(resolved_span.id.clone(), resolved.len())
                .is_some()
            {
                return Err(format!("Duplicate SPAN id '{}'", resolved_span.id));
            }

            resolved.push(resolved_span);
        }

        Ok(resolved)
    }

    /// Reject any SPAN naming a column that any FORMAT clause ever targeted
    /// at STUB — spanners over the stub aren't supported yet. No
    /// last-clause-wins resolution needed: a column that was ever declared
    /// STUB is off-limits to SPAN, regardless of whether a later FORMAT
    /// clause retargeted it to BODY. Takes already-`resolve_spanner_ids`-
    /// resolved spans, so a SPAN's `OVER` id reference is checked against
    /// the real columns it expands to, not the literal id text — and a
    /// STUB format's own column list gets the same treatment, expanding any
    /// entry that names a SPAN id into the columns it covers, so `FORMAT
    /// STUB <span_id>` is checked against real columns too rather than the
    /// literal id, which would never appear in a span's own `columns`.
    ///
    /// Needs no DataFrame: FORMAT/SPAN are both plain AST, so this runs from
    /// both `validate()` and real execution (`resolve_spanners`), mirroring
    /// `resolve_spanner_ids`.
    pub fn validate_span_stub_boundary(&self, resolved_spans: &[Spanner]) -> Result<(), String> {
        let stub_columns: HashSet<&str> = self
            .formats
            .iter()
            .filter(|format| format.target.is_stub())
            .flat_map(|format| &format.columns)
            .flat_map(
                |name| match resolved_spans.iter().find(|span| &span.id == name) {
                    Some(span) => span.columns.iter().map(String::as_str).collect::<Vec<_>>(),
                    None => vec![name.as_str()],
                },
            )
            .collect();

        for (idx, span) in resolved_spans.iter().enumerate() {
            span.validate_stub_boundary(&stub_columns)
                .map_err(|e| format!("SPAN {}: {}", idx + 1, e))?;
        }

        Ok(())
    }

    /// Fully resolve this table's spanners for execution: settings
    /// validated, `OVER` ids folded, STUB boundary checked, columns
    /// checked against the real schema, and each `label` overlaid with any
    /// `LABEL <id> => ...` entry.
    pub fn resolve_spanners(
        &self,
        labels: &Labels,
        column_names: Option<&[String]>,
    ) -> Result<Vec<Spanner>, String> {
        for (idx, spanner) in self.spans.iter().enumerate() {
            spanner
                .validate_settings()
                .map_err(|e| format!("SPAN {}: {}", idx + 1, e))?;
        }
        let spans = self.resolve_spanner_ids()?;
        self.validate_span_stub_boundary(&spans)?;
        // None (validate()'s case) skips this: no real schema to check.
        if let Some(names) = column_names {
            for span in &spans {
                span.validate_columns(names)?;
            }
        }

        Ok(spans
            .into_iter()
            .map(|mut span| {
                span.apply_labels(labels);
                span
            })
            .collect())
    }
}

/// One `SPAN` clause: a named group of columns rendered as one spanner cell
/// above them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Spanner {
    /// This spanner's identifier, unparsed (quotes included, if quoted).
    /// Referenced by a later `SPAN`'s `OVER` list or a `LABEL <id> =>
    /// ...` override. `SPAN NULL` still gets a real, internally generated
    /// id here — nothing else can reasonably reference it.
    pub id: String,
    /// Dequoted default display text. `None` for `SPAN NULL` — blank cell,
    /// columns still group. `LABEL <id> => ...` overrides either way.
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

    /// Check this spanner's columns against the table's STUB columns.
    pub fn validate_stub_boundary(&self, stub_columns: &HashSet<&str>) -> Result<(), String> {
        if let Some(name) = self
            .columns
            .iter()
            .find(|name| stub_columns.contains(name.as_str()))
        {
            return Err(format!("cannot include stub column '{name}'."));
        }

        Ok(())
    }

    /// Check this spanner's columns and `id` against the real query
    /// result's column names.
    pub fn validate_columns(&self, column_names: &[String]) -> Result<(), String> {
        for name in &self.columns {
            if !column_names.contains(name) {
                return Err(format!("SPAN references unknown column '{name}'"));
            }
        }

        if column_names.contains(&self.id) {
            return Err(format!(
                "SPAN id '{}' collides with an existing column name",
                self.id
            ));
        }

        Ok(())
    }

    /// Overlay a `LABEL <id> => ...` entry onto `label`, if the query has
    /// one for this spanner's id; otherwise leaves `label` as-is.
    pub fn apply_labels(&mut self, labels: &Labels) {
        if let Some(override_label) = labels.labels.get(&self.id) {
            self.label = override_label.clone();
        }
    }
}

/// Which section of the table a `FORMAT` clause's columns belong to.
///
/// `FORMAT STUB` is the only thing that assigns a column to the table's
/// stub; `FORMAT BODY`, or no target identifier at all, leaves columns in
/// the regular body — `Body` is this enum's default for that reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ColumnSection {
    #[default]
    Body,
    Stub,
}

impl ColumnSection {
    /// Whether this is the `STUB` section.
    pub fn is_stub(self) -> bool {
        self == ColumnSection::Stub
    }
}

/// One `FORMAT` clause: cell formatting for a group of columns.
///
/// `settings` is validated against `FORMAT_PARAMS` and resolved into each
/// covered column's `TableCell` properties.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Format {
    /// The columns this FORMAT applies to, in the order written.
    pub columns: Vec<String>,
    /// Which section of the table these columns belong to (`BODY`, the
    /// default, or `STUB`).
    #[serde(default)]
    pub target: ColumnSection,
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
const FORMAT_PARAMS: &[ParamDefinition] = &[
    ParamDefinition {
        name: "hjust",
        default: DefaultParamValue::Number(0.5),
        // Both spellings accepted here; resolve_column_properties standardises
        // "centre" to "center" when it builds a column's TableCell properties.
        constraint: ParamConstraint::string_option_or_number(
            &["left", "right", "centre", "center"],
            NumberConstraint::range(0.0, 1.0),
        ),
    },
    ParamDefinition {
        name: "width",
        // No default: an unset width leaves column sizing to the writer,
        // not to a value resolved here.
        default: DefaultParamValue::Null,
        constraint: ParamConstraint::string_numeric_with_unit(&["px", "%"]),
    },
];

impl Format {
    /// Merge `other` into `self`: `other`'s settings and value mappings
    /// override per-key, and its target replaces `self`'s; its template
    /// replaces `self`'s unless it is the default `"{}"`.
    pub fn merge(&mut self, other: &Format) {
        self.target = other.target;
        self.settings.extend(other.settings.clone());
        match (&mut self.value_mapping, &other.value_mapping) {
            (Some(mapping), Some(other)) => mapping.extend(other.clone()),
            (slot @ None, Some(other)) => *slot = Some(other.clone()),
            _ => {}
        }
        if other.value_template != "{}" {
            self.value_template = other.value_template.clone();
        }
    }

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
            id: "S".to_string(),
            label: Some("S".to_string()),
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
            target: ColumnSection::Body,
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

    #[test]
    fn format_validate_settings_accepts_a_percent_or_pixel_width() {
        let mut settings = Parameters::new();
        settings.insert(
            "width".to_string(),
            ParameterValue::String("20%".to_string()),
        );
        assert!(format_with_settings(settings).validate_settings().is_ok());

        let mut settings = Parameters::new();
        settings.insert(
            "width".to_string(),
            ParameterValue::String("240px".to_string()),
        );
        assert!(format_with_settings(settings).validate_settings().is_ok());
    }

    #[test]
    fn validate_span_stub_boundary_rejects_a_span_over_a_stub_column() {
        let table = Table {
            formats: vec![Format {
                columns: vec!["a".to_string()],
                target: ColumnSection::Stub,
                ..format_with_settings(Parameters::new())
            }],
            spans: vec![Spanner {
                id: "S".to_string(),
                label: Some("S".to_string()),
                columns: vec!["a".to_string(), "sales".to_string()],
                settings: Parameters::new(),
            }],
            ..Table::new()
        };

        let resolved = table.resolve_spanner_ids().unwrap();
        let error = table.validate_span_stub_boundary(&resolved).unwrap_err();
        assert!(error.contains("cannot include stub column 'a'"));
    }

    #[test]
    fn validate_span_stub_boundary_rejects_a_format_stub_naming_a_span_id() {
        // FORMAT STUB naming a SPAN id rather than its columns directly must
        // still be caught: `stub_columns` has to expand "G" into "a"/"b"
        // before comparing, or this collision goes undetected.
        let table = Table {
            formats: vec![Format {
                columns: vec!["G".to_string()],
                target: ColumnSection::Stub,
                ..format_with_settings(Parameters::new())
            }],
            spans: vec![Spanner {
                id: "G".to_string(),
                label: Some("G".to_string()),
                columns: vec!["a".to_string(), "b".to_string()],
                settings: Parameters::new(),
            }],
            ..Table::new()
        };

        let resolved = table.resolve_spanner_ids().unwrap();
        let error = table.validate_span_stub_boundary(&resolved).unwrap_err();
        assert!(error.contains("cannot include stub column"));
    }

    #[test]
    fn validate_columns_rejects_an_unknown_column() {
        let span = Spanner {
            id: "S".to_string(),
            label: Some("S".to_string()),
            columns: vec!["nope".to_string()],
            settings: Parameters::new(),
        };

        let error = span.validate_columns(&["a".to_string()]).unwrap_err();
        assert!(error.contains("SPAN references unknown column 'nope'"));
    }

    #[test]
    fn validate_columns_rejects_an_id_matching_a_real_column() {
        let span = Spanner {
            id: "a".to_string(),
            label: Some("a".to_string()),
            columns: vec!["b".to_string()],
            settings: Parameters::new(),
        };

        let error = span
            .validate_columns(&["a".to_string(), "b".to_string()])
            .unwrap_err();
        assert!(error.contains("SPAN id 'a' collides with an existing column name"));
    }
}
