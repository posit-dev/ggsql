//! Table types for ggsql specification
//!
//! Defines the typed `Table` structure that represents parsed `TABULATE`
//! statements, parallel to how `plot` defines `Plot` for `VISUALISE`
//! statements: `source` (from `TABULATE FROM`), `selection` (from
//! `TABULATE`'s own column list), `labels` (from `TABULATE LABEL`), `spans`
//! (from `TABULATE SPAN`), `formats` (from `TABULATE FORMAT`), and `facet`
//! (from `TABULATE FACET`) are populated so far.

mod facet;
mod layout;
mod spanner;

pub(crate) use facet::{groups_are_woven, setup_facet};
pub(crate) use layout::{
    compose_header, create_caption, create_column_labels, create_heading, extract_heading_labels,
    move_stub_columns, standardise_hjust, Slot,
};
pub use layout::{TableCell, TableCellKind, TableClass, TableColumn, TableLayout, TableRow};
pub use spanner::Spanner;
pub(crate) use spanner::{create_spanners, reorder_table_columns};

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::labels::Labels;
use crate::params::{
    validate_parameter, DefaultParamValue, NumberConstraint, ParamConstraint, ParamDefinition,
    Parameters,
};
use crate::plot::facet::Facet;
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
    /// Column display labels (from `TABULATE LABEL`). Reuses `labels::Labels`
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
    /// Row grouping (from `TABULATE FACET`). Only the wrap form is
    /// meaningful for tables; the builder rejects `BY`.
    pub facet: Option<Facet>,
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
            facet: None,
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

    /// Reject columns claimed by two table features at once: a SPAN over a
    /// STUB or FACET column, or a FORMAT STUB over a FACET column. Takes
    /// already-`resolve_spanner_ids`-resolved spans, so every check compares
    /// real column names, never a SPAN id. Needs no DataFrame, so this runs
    /// from both `validate()` and real execution (`resolve_spanners`).
    pub fn validate_boundaries(&self, resolved_spans: &[Spanner]) -> Result<(), String> {
        let stub_columns: Vec<String> = self
            .formats
            .iter()
            .filter(|format| format.target.is_stub())
            .flat_map(|format| &format.columns)
            .flat_map(
                |name| match resolved_spans.iter().find(|span| &span.id == name) {
                    Some(span) => span.columns.clone(),
                    None => vec![name.clone()],
                },
            )
            .collect();

        for (idx, span) in resolved_spans.iter().enumerate() {
            span.validate_boundary(&stub_columns, "stub")
                .map_err(|e| format!("SPAN {}: {}", idx + 1, e))?;
        }

        let Some(facet) = &self.facet else {
            return Ok(());
        };
        let facet_variables = facet.get_variables();
        for (idx, span) in resolved_spans.iter().enumerate() {
            span.validate_boundary(&facet_variables, "facet")
                .map_err(|e| format!("SPAN {}: {}", idx + 1, e))?;
        }
        if let Some(name) = facet_variables.iter().find(|v| stub_columns.contains(v)) {
            return Err(format!("FORMAT STUB cannot target FACET group '{name}'"));
        }

        Ok(())
    }

    /// Fully resolve this table's spanners for execution: settings
    /// validated, `OVER` ids folded, feature boundaries checked, columns
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
        self.validate_boundaries(&spans)?;
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

/// Which section of the table a column belongs to.
///
/// `FORMAT STUB` assigns a column to the table's stub and a `TABULATE
/// FACET` with `place => 'left'` assigns its variables to group stubs;
/// `FORMAT BODY`, or no target identifier at all, leaves columns in the
/// regular body — `Body` is this enum's default for that reason.
/// `GroupStub` is never reachable from a `FORMAT` target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ColumnSection {
    #[default]
    Body,
    Stub,
    /// A `TABULATE FACET` variable's column with `place => 'left'`; the
    /// payload is the variable's nesting depth (0 is outermost).
    GroupStub(usize),
}

impl ColumnSection {
    /// Whether this is a stub section (`STUB` or a facet group stub).
    pub fn is_stub(self) -> bool {
        matches!(self, ColumnSection::Stub | ColumnSection::GroupStub(_))
    }

    /// Section ordering for column placement: group stub columns first (by
    /// nesting depth), then STUB, then body.
    pub(crate) fn sort_order(self) -> (usize, usize) {
        match self {
            ColumnSection::GroupStub(depth) => (0, depth),
            ColumnSection::Stub => (1, 0),
            ColumnSection::Body => (2, 0),
        }
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
    fn validate_boundaries_rejects_a_span_over_a_stub_column() {
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
        let error = table.validate_boundaries(&resolved).unwrap_err();
        assert!(error.contains("cannot include stub column 'a'"));
    }

    #[test]
    fn validate_boundaries_rejects_a_format_stub_naming_a_span_id() {
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
        let error = table.validate_boundaries(&resolved).unwrap_err();
        assert!(error.contains("cannot include stub column"));
    }

    fn table_facet(variables: &[&str]) -> crate::plot::facet::Facet {
        crate::plot::facet::Facet {
            layout: crate::plot::facet::FacetLayout::Wrap {
                variables: variables.iter().map(|v| v.to_string()).collect(),
            },
            properties: Parameters::new(),
            resolved: false,
        }
    }

    #[test]
    fn validate_boundaries_rejects_a_span_over_a_facet_column() {
        let table = Table {
            facet: Some(table_facet(&["a"])),
            spans: vec![Spanner {
                id: "S".to_string(),
                label: Some("S".to_string()),
                columns: vec!["a".to_string(), "sales".to_string()],
                settings: Parameters::new(),
            }],
            ..Table::new()
        };

        let resolved = table.resolve_spanner_ids().unwrap();
        let error = table.validate_boundaries(&resolved).unwrap_err();
        assert!(error.contains("cannot include facet column 'a'"));
    }

    #[test]
    fn validate_boundaries_rejects_a_format_stub_targeting_a_facet_column() {
        let table = Table {
            facet: Some(table_facet(&["a"])),
            formats: vec![Format {
                columns: vec!["a".to_string()],
                target: ColumnSection::Stub,
                ..format_with_settings(Parameters::new())
            }],
            ..Table::new()
        };

        let resolved = table.resolve_spanner_ids().unwrap();
        let error = table.validate_boundaries(&resolved).unwrap_err();
        assert!(error.contains("FORMAT STUB cannot target FACET group 'a'"));
    }
}
