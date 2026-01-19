//! AST (Abstract Syntax Tree) types for ggsql specification
//!
//! This module defines the typed AST structures that represent parsed ggsql queries.
//! The AST is built from the tree-sitter CST (Concrete Syntax Tree) and provides
//! a more convenient, typed interface for working with ggsql specifications.
//!
//! # AST Structure
//!
//! ```text
//! VizSpec
//! ├─ global_mappings: GlobalMapping  (from VISUALISE clause mappings)
//! ├─ source: Option<DataSource>     (optional, from VISUALISE FROM clause)
//! ├─ layers: Vec<Layer>             (1+ LayerNode, one per DRAW clause)
//! ├─ scales: Vec<Scale>             (0+ ScaleNode, one per SCALE clause)
//! ├─ facet: Option<Facet>           (optional, from FACET clause)
//! ├─ coord: Option<Coord>           (optional, from COORD clause)
//! ├─ labels: Option<Labels>         (optional, merged from LABEL clauses)
//! ├─ guides: Vec<Guide>             (0+ GuideNode, one per GUIDE clause)
//! └─ theme: Option<Theme>           (optional, from THEME clause)
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::{DataFrame, GgsqlError, Result};

/// Column information from a data source schema
#[derive(Debug, Clone)]
pub struct ColumnInfo {
    /// Column name
    pub name: String,
    /// Whether this column is discrete (suitable for grouping)
    /// Discrete: String, Boolean, Categorical, Date
    /// Continuous: numeric types, Datetime, Time
    pub is_discrete: bool,
}

/// Schema of a data source - list of columns with type info
pub type Schema = Vec<ColumnInfo>;

/// Complete ggsql visualization specification
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VizSpec {
    /// Global aesthetic mappings (from VISUALISE clause)
    pub global_mappings: Mappings,
    /// FROM source (CTE, table, or file) when using VISUALISE FROM syntax
    pub source: Option<DataSource>,
    /// Visual layers (one per DRAW clause)
    pub layers: Vec<Layer>,
    /// Scale configurations (one per SCALE clause)
    pub scales: Vec<Scale>,
    /// Faceting specification (from FACET clause)
    pub facet: Option<Facet>,
    /// Coordinate system (from COORD clause)
    pub coord: Option<Coord>,
    /// Text labels (merged from all LABEL clauses)
    pub labels: Option<Labels>,
    /// Guide configurations (one per GUIDE clause)
    pub guides: Vec<Guide>,
    /// Theme styling (from THEME clause)
    pub theme: Option<Theme>,
}

/// Unified aesthetic mapping specification
///
/// Used for both global mappings (VISUALISE clause) and layer mappings (MAPPING clause).
/// Supports wildcards combined with explicit mappings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Mappings {
    /// Whether a wildcard (*) was specified
    pub wildcard: bool,
    /// Explicit aesthetic mappings (aesthetic → value)
    pub aesthetics: HashMap<String, AestheticValue>,
}

impl Mappings {
    /// Create a new empty Mappings
    pub fn new() -> Self {
        Self {
            wildcard: false,
            aesthetics: HashMap::new(),
        }
    }

    /// Create a new Mappings with wildcard flag set
    pub fn with_wildcard() -> Self {
        Self {
            wildcard: true,
            aesthetics: HashMap::new(),
        }
    }

    /// Check if the mappings are empty (no wildcard and no aesthetics)
    pub fn is_empty(&self) -> bool {
        !self.wildcard && self.aesthetics.is_empty()
    }

    /// Insert an aesthetic mapping
    pub fn insert(&mut self, aesthetic: impl Into<String>, value: AestheticValue) {
        self.aesthetics.insert(aesthetic.into(), value);
    }

    /// Get an aesthetic value by name
    pub fn get(&self, aesthetic: &str) -> Option<&AestheticValue> {
        self.aesthetics.get(aesthetic)
    }

    /// Check if an aesthetic is mapped
    pub fn contains_key(&self, aesthetic: &str) -> bool {
        self.aesthetics.contains_key(aesthetic)
    }

    /// Get the number of explicit aesthetic mappings
    pub fn len(&self) -> usize {
        self.aesthetics.len()
    }
}

/// Result of a statistical transformation
///
/// Stat transforms like histogram and bar count produce new columns with computed values.
/// This enum captures both the transformed query and the mappings from aesthetics to the
/// new column names.
#[derive(Debug, Clone, PartialEq)]
pub enum StatResult {
    /// No transformation needed - use original data as-is
    Identity,
    /// Transformation applied, with stat-computed columns
    Transformed {
        /// The transformed SQL query that produces the stat-computed columns
        query: String,
        /// Names of stat-computed columns (e.g., ["count", "bin", "x"])
        /// These are semantic names that will be prefixed with __ggsql_stat__
        /// and mapped to aesthetics via default_remappings or REMAPPING clause
        stat_columns: Vec<String>,
        /// Names of stat columns that are dummy/placeholder values
        /// (e.g., "x" when bar chart has no x mapped - produces a constant value)
        dummy_columns: Vec<String>,
        /// Names of aesthetics consumed by this stat transform
        /// These aesthetics were used as input to the stat and should be removed
        /// from the layer mappings after the transform completes
        consumed_aesthetics: Vec<String>,
    },
}

/// Data source for visualization or layer (from VISUALISE FROM or MAPPING ... FROM clause)
///
/// Allows specification of a data source - either a CTE/table name or a file path.
/// Used both for global `VISUALISE FROM` and layer-specific `MAPPING ... FROM`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DataSource {
    /// CTE or table name (unquoted identifier)
    Identifier(String),
    /// File path (quoted string like 'data.csv')
    FilePath(String),
}

impl DataSource {
    /// Returns the source as a string reference
    pub fn as_str(&self) -> &str {
        match self {
            DataSource::Identifier(s) => s,
            DataSource::FilePath(s) => s,
        }
    }

    /// Returns true if this is a file path source
    pub fn is_file(&self) -> bool {
        matches!(self, DataSource::FilePath(_))
    }
}

/// A single visualization layer (from DRAW clause)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Layer {
    /// Geometric object type
    pub geom: Geom,
    /// Aesthetic mappings (from MAPPING clause)
    pub mappings: Mappings,
    /// Stat remappings (from REMAPPING clause): stat_name → aesthetic
    /// Maps stat-computed columns (e.g., "count") to aesthetic channels (e.g., "y")
    pub remappings: Mappings,
    /// Geom parameters (not aesthetic mappings)
    pub parameters: HashMap<String, ParameterValue>,
    /// Optional data source for this layer (from MAPPING ... FROM)
    pub source: Option<DataSource>,
    /// Optional filter expression for this layer (from FILTER clause)
    pub filter: Option<SqlExpression>,
    /// Optional ORDER BY expression for this layer
    pub order_by: Option<SqlExpression>,
    /// Columns for grouping/partitioning (from PARTITION BY clause)
    pub partition_by: Vec<String>,
}

/// Raw SQL expression for layer-specific clauses (FILTER, ORDER BY)
///
/// This stores raw SQL text verbatim, which is passed directly to the database
/// backend. This allows any valid SQL expression to be used.
///
/// Example values:
/// - `"x > 10"` (filter)
/// - `"region = 'North' AND year >= 2020"` (filter)
/// - `"date ASC"` (order by)
/// - `"category, value DESC"` (order by)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SqlExpression(pub String);

impl SqlExpression {
    /// Create a new SQL expression from raw text
    pub fn new(sql: impl Into<String>) -> Self {
        Self(sql.into())
    }

    /// Get the raw SQL text
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consume and return the raw SQL text
    pub fn into_string(self) -> String {
        self.0
    }
}

/// Geometric object types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Geom {
    // Basic geoms
    Point,
    Line,
    Path,
    Bar,
    Area,
    Tile,
    Polygon,
    Ribbon,

    // Statistical geoms
    Histogram,
    Density,
    Smooth,
    Boxplot,
    Violin,

    // Annotation geoms
    Text,
    Label,
    Segment,
    Arrow,
    HLine,
    VLine,
    AbLine,
    ErrorBar,
}

/// Aesthetic information for a geom type
#[derive(Debug, Clone, Copy)]
pub struct GeomAesthetics {
    /// All aesthetics this geom type supports for user MAPPING
    pub supported: &'static [&'static str],
    /// Aesthetics required for this geom type to be valid
    pub required: &'static [&'static str],
    /// Hidden aesthetics (valid REMAPPING targets, not valid MAPPING targets)
    /// These are produced by stat transforms but shouldn't be manually mapped
    pub hidden: &'static [&'static str],
}

/// Default value for a layer parameter
#[derive(Debug, Clone)]
pub enum DefaultParamValue {
    String(&'static str),
    Number(f64),
    Boolean(bool),
    Null,
}

/// Layer parameter definition: name and default value
#[derive(Debug, Clone)]
pub struct DefaultParam {
    pub name: &'static str,
    pub default: DefaultParamValue,
}

impl std::fmt::Display for Geom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Geom::Point => "point",
            Geom::Line => "line",
            Geom::Path => "path",
            Geom::Bar => "bar",
            Geom::Area => "area",
            Geom::Tile => "tile",
            Geom::Polygon => "polygon",
            Geom::Ribbon => "ribbon",
            Geom::Histogram => "histogram",
            Geom::Density => "density",
            Geom::Smooth => "smooth",
            Geom::Boxplot => "boxplot",
            Geom::Violin => "violin",
            Geom::Text => "text",
            Geom::Label => "label",
            Geom::Segment => "segment",
            Geom::Arrow => "arrow",
            Geom::HLine => "hline",
            Geom::VLine => "vline",
            Geom::AbLine => "abline",
            Geom::ErrorBar => "errorbar",
        };
        write!(f, "{}", s)
    }
}

impl Geom {
    /// Returns aesthetic information for this geom type.
    /// Includes both supported aesthetics (for wildcard mapping) and
    /// required aesthetics (for validation).
    pub fn aesthetics(&self) -> GeomAesthetics {
        match self {
            // Position geoms
            Geom::Point => GeomAesthetics {
                supported: &[
                    "x", "y", "color", "colour", "fill", "size", "shape", "opacity",
                ],
                required: &["x", "y"],
                hidden: &[],
            },
            Geom::Line => GeomAesthetics {
                supported: &[
                    "x",
                    "y",
                    "color",
                    "colour",
                    "linetype",
                    "linewidth",
                    "opacity",
                ],
                required: &["x", "y"],
                hidden: &[],
            },
            Geom::Path => GeomAesthetics {
                supported: &[
                    "x",
                    "y",
                    "color",
                    "colour",
                    "linetype",
                    "linewidth",
                    "opacity",
                ],
                required: &["x", "y"],
                hidden: &[],
            },
            Geom::Bar => GeomAesthetics {
                // Bar supports optional x and y - stat decides aggregation
                // If x is missing: single bar showing total
                // If y is missing: stat computes COUNT or SUM(weight)
                // weight: optional, if mapped uses SUM(weight) instead of COUNT(*)
                supported: &[
                    "x", "y", "weight", "color", "colour", "fill", "width", "opacity",
                ],
                required: &[],
                hidden: &[],
            },
            Geom::Area => GeomAesthetics {
                supported: &["x", "y", "color", "colour", "fill", "opacity"],
                required: &["x", "y"],
                hidden: &[],
            },
            Geom::Tile => GeomAesthetics {
                supported: &[
                    "x", "y", "color", "colour", "fill", "width", "height", "opacity",
                ],
                required: &["x", "y"],
                hidden: &[],
            },
            Geom::Polygon => GeomAesthetics {
                supported: &["x", "y", "color", "colour", "fill", "opacity"],
                required: &["x", "y"],
                hidden: &[],
            },
            Geom::Ribbon => GeomAesthetics {
                supported: &["x", "ymin", "ymax", "color", "colour", "fill", "opacity"],
                required: &["x", "ymin", "ymax"],
                hidden: &[],
            },

            // Statistical geoms
            Geom::Histogram => GeomAesthetics {
                supported: &["x", "weight", "color", "colour", "fill", "opacity"],
                required: &["x"],
                // y and x2 are produced by stat_histogram but not valid for manual MAPPING
                hidden: &["y", "x2"],
            },
            Geom::Density => GeomAesthetics {
                supported: &["x", "color", "colour", "fill", "opacity"],
                required: &["x"],
                hidden: &[],
            },
            Geom::Smooth => GeomAesthetics {
                supported: &["x", "y", "color", "colour", "linetype", "opacity"],
                required: &["x", "y"],
                hidden: &[],
            },
            Geom::Boxplot => GeomAesthetics {
                supported: &["x", "y", "color", "colour", "fill", "opacity"],
                required: &["x", "y"],
                hidden: &[],
            },
            Geom::Violin => GeomAesthetics {
                supported: &["x", "y", "color", "colour", "fill", "opacity"],
                required: &["x", "y"],
                hidden: &[],
            },

            // Annotation geoms
            Geom::Text => GeomAesthetics {
                supported: &[
                    "x", "y", "label", "color", "colour", "size", "opacity", "family", "fontface",
                    "hjust", "vjust",
                ],
                required: &["x", "y"],
                hidden: &[],
            },
            Geom::Label => GeomAesthetics {
                supported: &[
                    "x", "y", "label", "color", "colour", "fill", "size", "opacity", "family",
                    "fontface", "hjust", "vjust",
                ],
                required: &["x", "y"],
                hidden: &[],
            },
            Geom::Segment => GeomAesthetics {
                supported: &[
                    "x",
                    "y",
                    "xend",
                    "yend",
                    "color",
                    "colour",
                    "linetype",
                    "linewidth",
                    "opacity",
                ],
                required: &["x", "y", "xend", "yend"],
                hidden: &[],
            },
            Geom::Arrow => GeomAesthetics {
                supported: &[
                    "x",
                    "y",
                    "xend",
                    "yend",
                    "color",
                    "colour",
                    "linetype",
                    "linewidth",
                    "opacity",
                ],
                required: &["x", "y", "xend", "yend"],
                hidden: &[],
            },
            Geom::HLine => GeomAesthetics {
                supported: &[
                    "yintercept",
                    "color",
                    "colour",
                    "linetype",
                    "linewidth",
                    "opacity",
                ],
                required: &["yintercept"],
                hidden: &[],
            },
            Geom::VLine => GeomAesthetics {
                supported: &[
                    "xintercept",
                    "color",
                    "colour",
                    "linetype",
                    "linewidth",
                    "opacity",
                ],
                required: &["xintercept"],
                hidden: &[],
            },
            Geom::AbLine => GeomAesthetics {
                supported: &[
                    "slope",
                    "intercept",
                    "color",
                    "colour",
                    "linetype",
                    "linewidth",
                    "opacity",
                ],
                required: &["slope", "intercept"],
                hidden: &[],
            },
            Geom::ErrorBar => GeomAesthetics {
                supported: &[
                    "x",
                    "y",
                    "ymin",
                    "ymax",
                    "xmin",
                    "xmax",
                    "color",
                    "colour",
                    "linewidth",
                    "opacity",
                ],
                required: &[],
                hidden: &[],
            },
        }
    }

    /// Returns default remappings for stat-computed columns to aesthetics.
    ///
    /// Each tuple is (stat_column_name, aesthetic_name).
    /// These defaults can be overridden by a REMAPPING clause.
    pub fn default_remappings(&self) -> &[(&str, &str)] {
        match self {
            Geom::Bar => &[("count", "y"), ("x", "x")],
            Geom::Histogram => &[("bin", "x"), ("bin_end", "x2"), ("count", "y")],
            // Other geoms don't have stat transforms yet
            _ => &[],
        }
    }

    /// Returns valid stat column names that can be used in REMAPPING.
    ///
    /// These are the columns produced by the geom's stat transform.
    /// REMAPPING can only map these columns to aesthetics.
    pub fn valid_stat_columns(&self) -> &[&str] {
        match self {
            Geom::Bar => &["count", "x", "proportion"],
            Geom::Histogram => &["bin", "bin_end", "count", "density"],
            // Other geoms don't have stat transforms
            _ => &[],
        }
    }

    /// Returns non-aesthetic parameters with their default values.
    ///
    /// These control stat behavior (e.g., bins for histogram).
    /// Defaults are applied to layer.parameters during execution via
    /// `Layer::apply_default_params()`.
    pub fn default_params(&self) -> &[DefaultParam] {
        match self {
            Geom::Histogram => &[
                DefaultParam {
                    name: "bins",
                    default: DefaultParamValue::Number(30.0),
                },
                DefaultParam {
                    name: "closed",
                    default: DefaultParamValue::String("right"),
                },
                DefaultParam {
                    name: "binwidth",
                    default: DefaultParamValue::Null,
                },
            ],
            Geom::Bar => &[DefaultParam {
                name: "width",
                default: DefaultParamValue::Number(0.9),
            }],
            // Future: Density might have bandwidth, Smooth might have span/method
            _ => &[],
        }
    }

    /// Returns aesthetics consumed as input by this geom's stat transform.
    ///
    /// Columns mapped to these aesthetics are used by the stat and don't need
    /// separate preservation in GROUP BY. All other aesthetic columns will be
    /// automatically preserved during stat transforms.
    pub fn stat_consumed_aesthetics(&self) -> &[&str] {
        match self {
            Geom::Histogram => &["x"],
            Geom::Bar => &["x", "y", "weight"],
            // Other geoms with stats would be added here
            _ => &[],
        }
    }

    /// Returns valid parameter names for SETTING clause.
    ///
    /// Combines supported aesthetics with non-aesthetic parameters from `default_params()`.
    /// Used for validation - invalid settings will produce an error.
    pub fn valid_settings(&self) -> Vec<&'static str> {
        let mut valid: Vec<&'static str> = self.aesthetics().supported.to_vec();
        for param in self.default_params() {
            valid.push(param.name);
        }
        valid
    }

    /// Check if this geom requires a statistical transformation
    ///
    /// Returns true if the geom needs data to be transformed before rendering.
    /// This is used to determine if a layer needs to query data even when
    /// it has no explicit source or filter.
    pub fn needs_stat_transform(&self, _aesthetics: &Mappings) -> bool {
        match self {
            Geom::Histogram => true,
            Geom::Bar => true, // Bar stat decides COUNT vs identity based on y mapping
            Geom::Density => true,
            Geom::Smooth => true,
            Geom::Boxplot => true,
            Geom::Violin => true,
            _ => false,
        }
    }

    /// Apply statistical transformation to the layer query.
    ///
    /// Some geoms require data transformations before rendering:
    /// - Histogram: bin continuous values and count
    /// - Bar (with only x mapped): count occurrences per category
    /// - Density: kernel density estimation (future)
    /// - Smooth: regression/smoothing (future)
    ///
    /// The default implementation returns the query unchanged.
    ///
    /// # Arguments
    /// * `query` - The base SQL query (with filter applied)
    /// * `aesthetics` - Layer aesthetic mappings (to find x, y columns)
    /// * `group_by` - Combined partition_by + facet variables for grouping
    /// * `execute_query` - Closure to execute SQL for data inspection
    ///
    /// Returns `StatResult::Identity` for no transformation (use original data),
    /// or `StatResult::Transformed` with the query and new aesthetic mappings.
    pub fn apply_stat_transform<F>(
        &self,
        query: &str,
        schema: &Schema,
        aesthetics: &Mappings,
        group_by: &[String],
        parameters: &HashMap<String, ParameterValue>,
        execute_query: &F,
    ) -> Result<StatResult>
    where
        F: Fn(&str) -> Result<DataFrame>,
    {
        match self {
            Geom::Histogram => {
                self.stat_histogram(query, aesthetics, group_by, parameters, execute_query)
            }
            Geom::Bar => self.stat_bar_count(query, schema, aesthetics, group_by),
            // Future: Geom::Density, Geom::Smooth, etc.
            _ => Ok(StatResult::Identity),
        }
    }

    /// Statistical transformation for histogram: bin continuous values and count
    fn stat_histogram<F>(
        &self,
        query: &str,
        aesthetics: &Mappings,
        group_by: &[String],
        parameters: &HashMap<String, ParameterValue>,
        execute_query: &F,
    ) -> Result<StatResult>
    where
        F: Fn(&str) -> Result<DataFrame>,
    {
        // Get x column name from aesthetics
        let x_col = get_column_name(aesthetics, "x").ok_or_else(|| {
            GgsqlError::ValidationError("Histogram requires 'x' aesthetic mapping".to_string())
        })?;

        // Get bins from parameters (default: 30)
        let bins = parameters
            .get("bins")
            .and_then(|p| match p {
                ParameterValue::Number(n) => Some(*n as usize),
                _ => None,
            })
            .expect("bins is not the correct format. Expected a number");

        // Get closed parameter (default: "right")
        let closed = parameters
            .get("closed")
            .and_then(|p| match p {
                ParameterValue::String(s) => Some(s.as_str()),
                _ => None,
            })
            .expect("closed is not the correct format. Expected a string");

        // Get binwidth from parameters (default: None - use bins to calculate)
        let explicit_binwidth = parameters.get("binwidth").and_then(|p| match p {
            ParameterValue::Number(n) => Some(*n),
            _ => None,
        });

        // Query min/max to compute bin width
        let stats_query = format!(
            "SELECT MIN({x}) as min_val, MAX({x}) as max_val FROM ({query})",
            x = x_col,
            query = query
        );
        let stats_df = execute_query(&stats_query)?;

        let (min_val, max_val) = extract_histogram_min_max(&stats_df)?;

        // Compute bin width: use explicit binwidth if provided, otherwise calculate from bins
        // Round to 10 decimal places to avoid SQL DECIMAL overflow issues
        let bin_width = if let Some(bw) = explicit_binwidth {
            bw
        } else if min_val >= max_val {
            1.0 // Fallback for edge case
        } else {
            ((max_val - min_val) / (bins - 1) as f64 * 1e10).round() / 1e10
        };
        let min_val = (min_val * 1e10).round() / 1e10;

        // Build the bin expression (bin start)
        let bin_expr = if closed == "left" {
            // Left-closed [a, b): use FLOOR
            format!(
                "(FLOOR(({x} - {min} + {w} * 0.5) / {w})) * {w} + {min} - {w} * 0.5",
                x = x_col,
                min = min_val,
                w = bin_width
            )
        } else {
            // Right-closed (a, b]: use CEIL - 1 with GREATEST for min value
            format!(
                "(GREATEST(CEIL(({x} - {min} + {w} * 0.5) / {w}) - 1, 0)) * {w} + {min} - {w} * 0.5",
                x = x_col,
                min = min_val,
                w = bin_width
            )
        };
        // Build the bin end expression (bin start + bin width)
        let bin_end_expr = format!("{expr} + {w}", expr = bin_expr, w = bin_width);

        // Build grouped columns (group_by includes partition_by + facet variables)
        let group_cols = if group_by.is_empty() {
            bin_expr.clone()
        } else {
            let mut cols: Vec<String> = group_by.to_vec();
            cols.push(bin_expr.clone());
            cols.join(", ")
        };

        // Determine aggregation expression based on weight aesthetic
        let agg_expr = if let Some(weight_value) = aesthetics.get("weight") {
            if weight_value.is_literal() {
                return Err(GgsqlError::ValidationError(
                    "Histogram weight aesthetic must be a column, not a literal".to_string(),
                ));
            }
            if let Some(weight_col) = weight_value.column_name() {
                format!("SUM({})", weight_col)
            } else {
                "COUNT(*)".to_string()
            }
        } else {
            "COUNT(*)".to_string()
        };

        // Use semantically meaningful column names with prefix to avoid conflicts
        // Include bin (start), bin_end (end), count/sum, and density
        // Use a two-stage query: first GROUP BY, then calculate density with window function
        let (binned_select, final_select) = if group_by.is_empty() {
            (
                format!(
                    "{} AS __ggsql_stat__bin, {} AS __ggsql_stat__bin_end, {} AS __ggsql_stat__count",
                    bin_expr, bin_end_expr, agg_expr
                ),
                "*, __ggsql_stat__count * 1.0 / SUM(__ggsql_stat__count) OVER () AS __ggsql_stat__density".to_string()
            )
        } else {
            let grp_cols = group_by.join(", ");
            (
                format!(
                    "{}, {} AS __ggsql_stat__bin, {} AS __ggsql_stat__bin_end, {} AS __ggsql_stat__count",
                    grp_cols, bin_expr, bin_end_expr, agg_expr
                ),
                format!(
                    "*, __ggsql_stat__count * 1.0 / SUM(__ggsql_stat__count) OVER (PARTITION BY {}) AS __ggsql_stat__density",
                    grp_cols
                )
            )
        };

        let transformed_query = format!(
            "WITH __stat_src__ AS ({query}), __binned__ AS (SELECT {binned} FROM __stat_src__ GROUP BY {group}) SELECT {final} FROM __binned__",
            query = query,
            binned = binned_select,
            group = group_cols,
            final = final_select
        );

        // Histogram always transforms - produces bin, bin_end, count, and density columns
        // Consumed aesthetics: x (transformed into bin/bin_end) and weight (used for weighted counts)
        Ok(StatResult::Transformed {
            query: transformed_query,
            stat_columns: vec![
                "bin".to_string(),
                "bin_end".to_string(),
                "count".to_string(),
                "density".to_string(),
            ],
            dummy_columns: vec![],
            consumed_aesthetics: vec!["x".to_string(), "weight".to_string()],
        })
    }

    /// Statistical transformation for bar: COUNT/SUM vs identity based on y and weight mappings
    ///
    /// Uses pre-fetched schema to check column existence (avoiding redundant queries).
    ///
    /// Decision logic for y:
    /// - y mapped to literal → identity (use original data)
    /// - y mapped to column that exists → identity (use original data)
    /// - y mapped to column that doesn't exist + from wildcard → aggregation
    /// - y mapped to column that doesn't exist + explicit → error
    /// - y not mapped → aggregation
    ///
    /// Decision logic for aggregation (when y triggers aggregation):
    /// - weight not mapped → COUNT(*)
    /// - weight mapped to literal → error (weight must be a column)
    /// - weight mapped to column that exists → SUM(weight_col)
    /// - weight mapped to column that doesn't exist + from wildcard → COUNT(*)
    /// - weight mapped to column that doesn't exist + explicit → error
    ///
    /// Returns `StatResult::Identity` for identity (no transformation),
    /// `StatResult::Transformed` for aggregation with new y mapping.
    fn stat_bar_count(
        &self,
        query: &str,
        schema: &Schema,
        aesthetics: &Mappings,
        group_by: &[String],
    ) -> Result<StatResult> {
        // x is now optional - if not mapped, we'll use a dummy constant
        let x_col = get_column_name(aesthetics, "x");
        let use_dummy_x = x_col.is_none();

        // Build column lookup set from pre-fetched schema
        let schema_columns: HashSet<&str> = schema.iter().map(|c| c.name.as_str()).collect();

        // Check if y is mapped
        // Note: With upfront validation, if y is mapped to a column, that column must exist
        if let Some(y_value) = aesthetics.get("y") {
            // y is a literal value - use identity (no transformation)
            if y_value.is_literal() {
                return Ok(StatResult::Identity);
            }

            // y is a column reference - if it exists in schema, use identity
            // (column existence validated upfront, but we still check schema for stat decision)
            if let Some(y_col) = y_value.column_name() {
                if schema_columns.contains(y_col) {
                    // y column exists - use identity (no transformation)
                    return Ok(StatResult::Identity);
                }
                // y mapped but column doesn't exist in schema - fall through to aggregation
                // (this shouldn't happen with upfront validation, but handle gracefully)
            }
        }

        // y not mapped - apply aggregation (COUNT or SUM)
        // Determine aggregation expression based on weight aesthetic
        // Note: stat column is always "count" for predictability, even when using SUM
        // Note: With upfront validation, if weight is mapped to a column, that column must exist
        let agg_expr = if let Some(weight_value) = aesthetics.get("weight") {
            // weight is mapped - check if it's valid
            if weight_value.is_literal() {
                return Err(GgsqlError::ValidationError(
                    "Bar weight aesthetic must be a column, not a literal".to_string(),
                ));
            }

            if let Some(weight_col) = weight_value.column_name() {
                if schema_columns.contains(weight_col) {
                    // weight column exists - use SUM (but still call it "count")
                    format!("SUM({}) AS __ggsql_stat__count", weight_col)
                } else {
                    // weight mapped but column doesn't exist - fall back to COUNT
                    // (this shouldn't happen with upfront validation, but handle gracefully)
                    "COUNT(*) AS __ggsql_stat__count".to_string()
                }
            } else {
                // Shouldn't happen (not literal, not column), fall back to COUNT
                "COUNT(*) AS __ggsql_stat__count".to_string()
            }
        } else {
            // weight not mapped - use COUNT
            "COUNT(*) AS __ggsql_stat__count".to_string()
        };

        // Build the query based on whether x is mapped or not
        // Use two-stage query: first GROUP BY, then calculate proportion with window function
        let (transformed_query, stat_columns, dummy_columns, consumed_aesthetics) = if use_dummy_x {
            // x is not mapped - use dummy constant, no GROUP BY on x
            let (grouped_select, final_select) = if group_by.is_empty() {
                (
                    format!(
                        "'__ggsql_stat__dummy__' AS __ggsql_stat__x, {agg}",
                        agg = agg_expr
                    ),
                    "*, __ggsql_stat__count * 1.0 / SUM(__ggsql_stat__count) OVER () AS __ggsql_stat__proportion".to_string()
                )
            } else {
                let grp_cols = group_by.join(", ");
                (
                    format!(
                        "{g}, '__ggsql_stat__dummy__' AS __ggsql_stat__x, {agg}",
                        g = grp_cols,
                        agg = agg_expr
                    ),
                    format!(
                        "*, __ggsql_stat__count * 1.0 / SUM(__ggsql_stat__count) OVER (PARTITION BY {}) AS __ggsql_stat__proportion",
                        grp_cols
                    )
                )
            };

            let query_str = if group_by.is_empty() {
                // No grouping at all - single aggregate
                format!(
                    "WITH __stat_src__ AS ({query}), __grouped__ AS (SELECT {grouped} FROM __stat_src__) SELECT {final} FROM __grouped__",
                    query = query,
                    grouped = grouped_select,
                    final = final_select
                )
            } else {
                // Group by partition/facet variables only
                let group_cols = group_by.join(", ");
                format!(
                    "WITH __stat_src__ AS ({query}), __grouped__ AS (SELECT {grouped} FROM __stat_src__ GROUP BY {group}) SELECT {final} FROM __grouped__",
                    query = query,
                    grouped = grouped_select,
                    group = group_cols,
                    final = final_select
                )
            };

            // Stat columns: x (dummy), count, and proportion - x is a dummy placeholder
            // Consumed: weight (used for weighted sums)
            (
                query_str,
                vec![
                    "x".to_string(),
                    "count".to_string(),
                    "proportion".to_string(),
                ],
                vec!["x".to_string()],
                vec!["weight".to_string()],
            )
        } else {
            // x is mapped - use existing logic with two-stage query
            let x_col = x_col.unwrap();

            // Build grouped columns (group_by includes partition_by + facet variables + x)
            let group_cols = if group_by.is_empty() {
                x_col.clone()
            } else {
                let mut cols = group_by.to_vec();
                cols.push(x_col.clone());
                cols.join(", ")
            };

            // Keep original x column name, only add the aggregated stat column
            let (grouped_select, final_select) = if group_by.is_empty() {
                (
                    format!("{x}, {agg}", x = x_col, agg = agg_expr),
                    "*, __ggsql_stat__count * 1.0 / SUM(__ggsql_stat__count) OVER () AS __ggsql_stat__proportion".to_string()
                )
            } else {
                let grp_cols = group_by.join(", ");
                (
                    format!("{g}, {x}, {agg}", g = grp_cols, x = x_col, agg = agg_expr),
                    format!(
                        "*, __ggsql_stat__count * 1.0 / SUM(__ggsql_stat__count) OVER (PARTITION BY {}) AS __ggsql_stat__proportion",
                        grp_cols
                    )
                )
            };

            let query_str = format!(
                "WITH __stat_src__ AS ({query}), __grouped__ AS (SELECT {grouped} FROM __stat_src__ GROUP BY {group}) SELECT {final} FROM __grouped__",
                query = query,
                grouped = grouped_select,
                group = group_cols,
                final = final_select
            );

            // count and proportion stat columns (x is preserved from original data), no dummies
            // Consumed: weight (used for weighted sums)
            (
                query_str,
                vec!["count".to_string(), "proportion".to_string()],
                vec![],
                vec!["weight".to_string()],
            )
        };

        // Return with stat column names and consumed aesthetics
        Ok(StatResult::Transformed {
            query: transformed_query,
            stat_columns,
            dummy_columns,
            consumed_aesthetics,
        })
    }
}

/// Helper to extract column name from aesthetic value
fn get_column_name(aesthetics: &Mappings, aesthetic: &str) -> Option<String> {
    aesthetics.get(aesthetic).and_then(|v| match v {
        AestheticValue::Column { name, .. } => Some(name.clone()),
        _ => None,
    })
}

/// Extract min and max from histogram stats DataFrame
fn extract_histogram_min_max(df: &DataFrame) -> Result<(f64, f64)> {
    if df.height() == 0 {
        return Err(GgsqlError::ValidationError(
            "No data for histogram statistics".to_string(),
        ));
    }

    let min_val = df
        .column("min_val")
        .ok()
        .and_then(|s| s.f64().ok())
        .and_then(|ca| ca.get(0))
        .ok_or_else(|| {
            GgsqlError::ValidationError("Could not extract min value for histogram".to_string())
        })?;

    let max_val = df
        .column("max_val")
        .ok()
        .and_then(|s| s.f64().ok())
        .and_then(|ca| ca.get(0))
        .ok_or_else(|| {
            GgsqlError::ValidationError("Could not extract max value for histogram".to_string())
        })?;

    Ok((min_val, max_val))
}

/// Value for aesthetic mappings
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AestheticValue {
    /// Column reference
    Column {
        name: String,
        /// Whether this is a dummy/placeholder column (e.g., for bar charts without x mapped)
        is_dummy: bool,
    },
    /// Literal value (quoted string, number, or boolean)
    Literal(LiteralValue),
}

impl AestheticValue {
    /// Create a column mapping
    pub fn standard_column(name: impl Into<String>) -> Self {
        Self::Column {
            name: name.into(),
            is_dummy: false,
        }
    }

    /// Create a dummy/placeholder column mapping (e.g., for bar charts without x mapped)
    pub fn dummy_column(name: impl Into<String>) -> Self {
        Self::Column {
            name: name.into(),
            is_dummy: true,
        }
    }

    /// Get column name if this is a column mapping
    pub fn column_name(&self) -> Option<&str> {
        match self {
            Self::Column { name, .. } => Some(name),
            _ => None,
        }
    }

    /// Check if this is a dummy/placeholder column
    pub fn is_dummy(&self) -> bool {
        match self {
            Self::Column { is_dummy, .. } => *is_dummy,
            _ => false,
        }
    }

    /// Check if this is a literal value (not a column mapping)
    pub fn is_literal(&self) -> bool {
        matches!(self, Self::Literal(_))
    }
}

/// Literal values in aesthetic mappings
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LiteralValue {
    String(String),
    Number(f64),
    Boolean(bool),
}

/// Value for geom parameters
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ParameterValue {
    String(String),
    Number(f64),
    Boolean(bool),
}

/// Scale configuration (from SCALE clause)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Scale {
    /// The aesthetic this scale applies to
    pub aesthetic: String,
    /// Scale type (optional, inferred if not specified)
    pub scale_type: Option<ScaleType>,
    /// Scale properties
    pub properties: HashMap<String, ScalePropertyValue>,
}

/// Scale types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ScaleType {
    // Continuous scales
    Linear,
    Log10,
    Log,
    Log2,
    Sqrt,
    Reverse,

    // Discrete scales
    Ordinal,
    Categorical,

    // Temporal scales
    Date,
    DateTime,
    Time,

    // Color palettes
    Viridis,
    Plasma,
    Magma,
    Inferno,
    Cividis,
    Diverging,
    Sequential,
}

/// Values for scale properties
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ScalePropertyValue {
    String(String),
    Number(f64),
    Boolean(bool),
    Array(Vec<ArrayElement>),
}

/// Elements in arrays
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ArrayElement {
    String(String),
    Number(f64),
    Boolean(bool),
}

/// Faceting specification (from FACET clause)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Facet {
    /// FACET WRAP variables
    Wrap {
        variables: Vec<String>,
        scales: FacetScales,
    },
    /// FACET rows BY cols
    Grid {
        rows: Vec<String>,
        cols: Vec<String>,
        scales: FacetScales,
    },
}

/// Scale sharing options for facets
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FacetScales {
    Fixed,
    Free,
    FreeX,
    FreeY,
}

impl Facet {
    /// Get all variables used for faceting
    ///
    /// Returns all column names that will be used to split the data into facets.
    /// For Wrap facets, returns the variables list.
    /// For Grid facets, returns combined rows and cols variables.
    pub fn get_variables(&self) -> Vec<String> {
        match self {
            Facet::Wrap { variables, .. } => variables.clone(),
            Facet::Grid { rows, cols, .. } => {
                let mut vars = rows.clone();
                vars.extend(cols.iter().cloned());
                vars
            }
        }
    }
}

/// Coordinate system (from COORD clause)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Coord {
    /// Coordinate system type
    pub coord_type: CoordType,
    /// Coordinate-specific options
    pub properties: HashMap<String, CoordPropertyValue>,
}

/// Coordinate system types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CoordType {
    Cartesian,
    Polar,
    Flip,
    Fixed,
    Trans,
    Map,
    QuickMap,
}

/// Values for coordinate properties
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CoordPropertyValue {
    String(String),
    Number(f64),
    Boolean(bool),
    Array(Vec<ArrayElement>),
}

/// Text labels (from LABELS clause)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Labels {
    /// Label assignments (label type → text)
    pub labels: HashMap<String, String>,
}

/// Guide configuration (from GUIDE clause)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Guide {
    /// The aesthetic this guide applies to
    pub aesthetic: String,
    /// Guide type
    pub guide_type: Option<GuideType>,
    /// Guide properties
    pub properties: HashMap<String, GuidePropertyValue>,
}

/// Guide types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GuideType {
    Legend,
    ColorBar,
    Axis,
    None,
}

/// Values for guide properties
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GuidePropertyValue {
    String(String),
    Number(f64),
    Boolean(bool),
}

/// Theme styling (from THEME clause)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Theme {
    /// Base theme style
    pub style: Option<String>,
    /// Theme property overrides
    pub properties: HashMap<String, ThemePropertyValue>,
}

/// Values for theme properties
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ThemePropertyValue {
    String(String),
    Number(f64),
    Boolean(bool),
}

impl VizSpec {
    /// Create a new empty VizSpec
    pub fn new() -> Self {
        Self {
            global_mappings: Mappings::new(),
            source: None,
            layers: Vec::new(),
            scales: Vec::new(),
            facet: None,
            coord: None,
            labels: None,
            guides: Vec::new(),
            theme: None,
        }
    }

    /// Create a new VizSpec with the given global mapping
    pub fn with_global_mappings(global_mappings: Mappings) -> Self {
        Self {
            global_mappings,
            source: None,
            layers: Vec::new(),
            scales: Vec::new(),
            facet: None,
            coord: None,
            labels: None,
            guides: Vec::new(),
            theme: None,
        }
    }

    /// Check if the spec has any layers
    pub fn has_layers(&self) -> bool {
        !self.layers.is_empty()
    }

    /// Get the number of layers
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Find a scale for a specific aesthetic
    pub fn find_scale(&self, aesthetic: &str) -> Option<&Scale> {
        self.scales
            .iter()
            .find(|scale| scale.aesthetic == aesthetic)
    }

    /// Find a guide for a specific aesthetic
    pub fn find_guide(&self, aesthetic: &str) -> Option<&Guide> {
        self.guides
            .iter()
            .find(|guide| guide.aesthetic == aesthetic)
    }

    /// Compute aesthetic labels for axes and legends.
    ///
    /// For each aesthetic used in any layer, determines the appropriate label:
    /// - If user specified a label via LABEL clause, use that
    /// - Otherwise, use the first non-synthetic column name mapped to that aesthetic
    /// - Falls back to the aesthetic name itself if only constants are mapped
    ///
    /// This ensures that synthetic constant columns (like `__ggsql_const_color_0__`)
    /// don't appear as axis/legend titles.
    pub fn compute_aesthetic_labels(&mut self) {
        // Ensure Labels struct exists
        if self.labels.is_none() {
            self.labels = Some(Labels {
                labels: HashMap::new(),
            });
        }
        let labels = self.labels.as_mut().unwrap();

        // Collect all aesthetics used across all layers
        let mut all_aesthetics: HashSet<String> = HashSet::new();
        for layer in &self.layers {
            for aesthetic in layer.mappings.aesthetics.keys() {
                all_aesthetics.insert(aesthetic.clone());
            }
        }

        // For each aesthetic, compute label if not already user-specified
        for aesthetic in all_aesthetics {
            // Skip secondary/interval aesthetics (x2, y2, xmin, etc.)
            // Only primary aesthetics (x, y, color, etc.) should get labels
            if matches!(
                aesthetic.as_str(),
                "x2" | "y2" | "xmin" | "xmax" | "ymin" | "ymax" | "xend" | "yend"
            ) {
                continue;
            }

            // Skip if user already specified this label
            if labels.labels.contains_key(&aesthetic) {
                continue;
            }

            // Find first non-constant column mapping
            let mut label = aesthetic.clone(); // Default to aesthetic name
            for layer in &self.layers {
                if let Some(AestheticValue::Column { name, .. }) = layer.mappings.get(&aesthetic) {
                    // Skip synthetic constant columns
                    if name.starts_with("__ggsql_const_") {
                        continue;
                    }
                    // Strip __ggsql_stat__ prefix for human-readable labels
                    if let Some(stat_name) = name.strip_prefix("__ggsql_stat__") {
                        label = stat_name.to_string();
                    } else {
                        label = name.clone();
                    }
                    break;
                }
            }

            labels.labels.insert(aesthetic, label);
        }
    }
}

impl Layer {
    /// Create a new layer with the given geom
    pub fn new(geom: Geom) -> Self {
        Self {
            geom,
            mappings: Mappings::new(),
            remappings: Mappings::new(),
            parameters: HashMap::new(),
            source: None,
            filter: None,
            order_by: None,
            partition_by: Vec::new(),
        }
    }

    /// Set the filter expression
    pub fn with_filter(mut self, filter: SqlExpression) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Set the ORDER BY expression
    pub fn with_order_by(mut self, order: SqlExpression) -> Self {
        self.order_by = Some(order);
        self
    }

    /// Set the data source for this layer
    pub fn with_source(mut self, source: DataSource) -> Self {
        self.source = Some(source);
        self
    }

    /// Add an aesthetic mapping
    pub fn with_aesthetic(mut self, aesthetic: impl Into<String>, value: AestheticValue) -> Self {
        self.mappings.insert(aesthetic, value);
        self
    }

    /// Set the wildcard flag
    pub fn with_wildcard(mut self) -> Self {
        self.mappings.wildcard = true;
        self
    }

    /// Add a parameter
    pub fn with_parameter(mut self, parameter: String, value: ParameterValue) -> Self {
        self.parameters.insert(parameter, value);
        self
    }

    /// Set the partition columns for grouping
    pub fn with_partition_by(mut self, columns: Vec<String>) -> Self {
        self.partition_by = columns;
        self
    }

    /// Get a column reference from an aesthetic, if it's mapped to a column
    pub fn get_column(&self, aesthetic: &str) -> Option<&str> {
        match self.mappings.get(aesthetic) {
            Some(AestheticValue::Column { name, .. }) => Some(name),
            _ => None,
        }
    }

    /// Get a literal value from an aesthetic, if it's mapped to a literal
    pub fn get_literal(&self, aesthetic: &str) -> Option<&LiteralValue> {
        match self.mappings.get(aesthetic) {
            Some(AestheticValue::Literal(lit)) => Some(lit),
            _ => None,
        }
    }

    /// Check if this layer has the required aesthetics for its geom
    pub fn validate_required_aesthetics(&self) -> std::result::Result<(), String> {
        for aesthetic in self.geom.aesthetics().required {
            if !self.mappings.contains_key(aesthetic) {
                return Err(format!(
                    "Geom '{}' requires aesthetic '{}' but it was not provided",
                    self.geom, aesthetic
                ));
            }
        }

        Ok(())
    }

    /// Apply default parameter values for any params not specified by user.
    ///
    /// Call this during execution to ensure all stat params have values.
    pub fn apply_default_params(&mut self) {
        for param in self.geom.default_params() {
            if !self.parameters.contains_key(param.name) {
                let value = match &param.default {
                    DefaultParamValue::String(s) => ParameterValue::String(s.to_string()),
                    DefaultParamValue::Number(n) => ParameterValue::Number(*n),
                    DefaultParamValue::Boolean(b) => ParameterValue::Boolean(*b),
                    DefaultParamValue::Null => continue, // Don't insert null defaults
                };
                self.parameters.insert(param.name.to_string(), value);
            }
        }
    }

    /// Validate that all SETTING parameters are valid for this layer's geom
    pub fn validate_settings(&self) -> std::result::Result<(), String> {
        let valid = self.geom.valid_settings();
        for param_name in self.parameters.keys() {
            if !valid.contains(&param_name.as_str()) {
                return Err(format!(
                    "Invalid setting '{}' for geom '{}'. Valid settings are: {}",
                    param_name,
                    self.geom,
                    valid.join(", ")
                ));
            }
        }
        Ok(())
    }
}

impl Default for VizSpec {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for AestheticValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AestheticValue::Column { name, .. } => write!(f, "{}", name),
            AestheticValue::Literal(lit) => write!(f, "{}", lit),
        }
    }
}

impl std::fmt::Display for LiteralValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LiteralValue::String(s) => write!(f, "'{}'", s),
            LiteralValue::Number(n) => write!(f, "{}", n),
            LiteralValue::Boolean(b) => write!(f, "{}", b),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_viz_spec_creation() {
        let spec = VizSpec::new();
        assert!(spec.global_mappings.is_empty());
        assert_eq!(spec.layers.len(), 0);
        assert!(!spec.has_layers());
        assert_eq!(spec.layer_count(), 0);
    }

    #[test]
    fn test_viz_spec_with_global_mappings() {
        let mut mapping = Mappings::new();
        mapping.insert("x", AestheticValue::standard_column("date"));
        mapping.insert("y", AestheticValue::standard_column("y"));
        let spec = VizSpec::with_global_mappings(mapping.clone());
        assert_eq!(spec.global_mappings.aesthetics.len(), 2);
        assert!(spec.global_mappings.aesthetics.contains_key("x"));
    }

    #[test]
    fn test_global_mappings_wildcard() {
        let mapping = Mappings::with_wildcard();
        let spec = VizSpec::with_global_mappings(mapping);
        assert!(spec.global_mappings.wildcard);
    }

    #[test]
    fn test_layer_creation() {
        let layer = Layer::new(Geom::Point)
            .with_aesthetic("x".to_string(), AestheticValue::standard_column("date"))
            .with_aesthetic("y".to_string(), AestheticValue::standard_column("revenue"))
            .with_aesthetic(
                "color".to_string(),
                AestheticValue::Literal(LiteralValue::String("blue".to_string())),
            );

        assert_eq!(layer.geom, Geom::Point);
        assert_eq!(layer.get_column("x"), Some("date"));
        assert_eq!(layer.get_column("y"), Some("revenue"));
        assert!(matches!(layer.get_literal("color"), Some(LiteralValue::String(s)) if s == "blue"));
        assert!(layer.filter.is_none());
    }

    #[test]
    fn test_layer_with_filter() {
        let filter = SqlExpression::new("year > 2020");
        let layer = Layer::new(Geom::Point).with_filter(filter);
        assert!(layer.filter.is_some());
        assert_eq!(layer.filter.as_ref().unwrap().as_str(), "year > 2020");
    }

    #[test]
    fn test_layer_validation() {
        let valid_point = Layer::new(Geom::Point)
            .with_aesthetic("x".to_string(), AestheticValue::standard_column("x"))
            .with_aesthetic("y".to_string(), AestheticValue::standard_column("y"));

        assert!(valid_point.validate_required_aesthetics().is_ok());

        let invalid_point = Layer::new(Geom::Point)
            .with_aesthetic("x".to_string(), AestheticValue::standard_column("x"));

        assert!(invalid_point.validate_required_aesthetics().is_err());

        let valid_ribbon = Layer::new(Geom::Ribbon)
            .with_aesthetic("x".to_string(), AestheticValue::standard_column("x"))
            .with_aesthetic("ymin".to_string(), AestheticValue::standard_column("ymin"))
            .with_aesthetic("ymax".to_string(), AestheticValue::standard_column("ymax"));

        assert!(valid_ribbon.validate_required_aesthetics().is_ok());
    }

    #[test]
    fn test_viz_spec_layer_operations() {
        let mut spec = VizSpec::new();

        let layer1 = Layer::new(Geom::Point);
        let layer2 = Layer::new(Geom::Line);

        spec.layers.push(layer1);
        spec.layers.push(layer2);

        assert!(spec.has_layers());
        assert_eq!(spec.layer_count(), 2);
        assert_eq!(spec.layers[0].geom, Geom::Point);
        assert_eq!(spec.layers[1].geom, Geom::Line);
    }

    #[test]
    fn test_aesthetic_value_display() {
        let column = AestheticValue::standard_column("sales");
        let string_lit = AestheticValue::Literal(LiteralValue::String("blue".to_string()));
        let number_lit = AestheticValue::Literal(LiteralValue::Number(3.53));
        let bool_lit = AestheticValue::Literal(LiteralValue::Boolean(true));

        assert_eq!(format!("{}", column), "sales");
        assert_eq!(format!("{}", string_lit), "'blue'");
        assert_eq!(format!("{}", number_lit), "3.53");
        assert_eq!(format!("{}", bool_lit), "true");
    }

    #[test]
    fn test_geom_display() {
        assert_eq!(format!("{}", Geom::Point), "point");
        assert_eq!(format!("{}", Geom::Histogram), "histogram");
        assert_eq!(format!("{}", Geom::ErrorBar), "errorbar");
    }

    // ========================================
    // Mappings Struct Tests
    // ========================================

    #[test]
    fn test_mappings_new() {
        let mappings = Mappings::new();
        assert!(!mappings.wildcard);
        assert!(mappings.aesthetics.is_empty());
        assert!(mappings.is_empty());
    }

    #[test]
    fn test_mappings_with_wildcard() {
        let mappings = Mappings::with_wildcard();
        assert!(mappings.wildcard);
        assert!(mappings.aesthetics.is_empty());
        assert!(!mappings.is_empty()); // wildcard counts as non-empty
    }

    #[test]
    fn test_mappings_insert_and_get() {
        let mut mappings = Mappings::new();
        mappings.insert("x", AestheticValue::standard_column("date"));
        mappings.insert("y", AestheticValue::standard_column("value"));

        assert_eq!(mappings.len(), 2);
        assert!(mappings.contains_key("x"));
        assert!(mappings.contains_key("y"));
        assert!(!mappings.contains_key("color"));

        let x_val = mappings.get("x").unwrap();
        assert_eq!(x_val.column_name(), Some("date"));
    }

    #[test]
    fn test_aesthetic_value_column_constructors() {
        let col = AestheticValue::standard_column("date");
        assert!(!col.is_dummy());
        assert_eq!(col.column_name(), Some("date"));

        let dummy_col = AestheticValue::dummy_column("x");
        assert!(dummy_col.is_dummy());
        assert_eq!(dummy_col.column_name(), Some("x"));
    }

    #[test]
    fn test_aesthetic_value_literal() {
        let lit = AestheticValue::Literal(LiteralValue::String("red".to_string()));
        assert!(!lit.is_dummy());
        assert_eq!(lit.column_name(), None);
    }

    #[test]
    fn test_layer_with_wildcard() {
        let layer = Layer::new(Geom::Point).with_wildcard();
        assert!(layer.mappings.wildcard);
    }

    #[test]
    fn test_geom_aesthetics() {
        // Point geom
        let point = Geom::Point.aesthetics();
        assert!(point.supported.contains(&"x"));
        assert!(point.supported.contains(&"size"));
        assert!(point.supported.contains(&"shape"));
        assert!(!point.supported.contains(&"linetype"));
        assert_eq!(point.required, &["x", "y"]);

        // Line geom
        let line = Geom::Line.aesthetics();
        assert!(line.supported.contains(&"linetype"));
        assert!(line.supported.contains(&"linewidth"));
        assert!(!line.supported.contains(&"size"));
        assert_eq!(line.required, &["x", "y"]);

        // Bar geom - optional x and y (stat decides aggregation)
        let bar = Geom::Bar.aesthetics();
        assert!(bar.supported.contains(&"fill"));
        assert!(bar.supported.contains(&"width"));
        assert!(bar.supported.contains(&"y")); // Bar accepts optional y
        assert!(bar.supported.contains(&"x")); // Bar accepts optional x
        assert_eq!(bar.required, &[] as &[&str]); // No required aesthetics

        // Text geom
        let text = Geom::Text.aesthetics();
        assert!(text.supported.contains(&"label"));
        assert!(text.supported.contains(&"family"));
        assert_eq!(text.required, &["x", "y"]);

        // Statistical geoms only require x
        assert_eq!(Geom::Histogram.aesthetics().required, &["x"]);
        assert_eq!(Geom::Density.aesthetics().required, &["x"]);

        // Ribbon requires ymin/ymax
        assert_eq!(Geom::Ribbon.aesthetics().required, &["x", "ymin", "ymax"]);

        // Segment/arrow require endpoints
        assert_eq!(
            Geom::Segment.aesthetics().required,
            &["x", "y", "xend", "yend"]
        );

        // Reference lines
        assert_eq!(Geom::HLine.aesthetics().required, &["yintercept"]);
        assert_eq!(Geom::VLine.aesthetics().required, &["xintercept"]);
        assert_eq!(Geom::AbLine.aesthetics().required, &["slope", "intercept"]);

        // ErrorBar has no strict requirements
        assert_eq!(Geom::ErrorBar.aesthetics().required, &[] as &[&str]);
    }
}
