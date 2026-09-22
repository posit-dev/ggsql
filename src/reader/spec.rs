//! Implementation of ResolvedPlot, ResolvedTable, and ResolvedSpec methods.

use std::collections::HashMap;

use crate::naming;
use crate::plot::Plot;
use crate::validate::ValidationWarning;
use crate::{DataFrame, Table};

use super::{Metadata, ResolvedPlot, ResolvedSpec, ResolvedTable};

impl ResolvedPlot {
    /// Create a new ResolvedPlot from PreparedData
    pub(crate) fn new(
        plot: Plot,
        data: HashMap<String, DataFrame>,
        sql: String,
        visual: String,
        layer_sql: Vec<Option<String>>,
        stat_sql: Vec<Option<String>>,
        warnings: Vec<ValidationWarning>,
    ) -> Self {
        // Compute metadata from data
        // Get rows from data, but columns from layer mappings (since scale-syntax renames columns)
        let rows = data
            .get(naming::GLOBAL_DATA_KEY)
            .or_else(|| data.get(&naming::layer_key(0)))
            .map(|df| df.height())
            .unwrap_or(0);

        // Get aesthetic names from mappings (these are what the user thinks of as columns)
        // This provides backwards-compatible column names like "x", "y" instead of internal names
        let columns: Vec<String> = if !plot.layers.is_empty() {
            plot.layers[0].mappings.aesthetics.keys().cloned().collect()
        } else {
            Vec::new()
        };

        let layer_count = plot.layers.len();
        let metadata = Metadata {
            rows,
            columns,
            layer_count,
        };

        Self {
            plot,
            data,
            metadata,
            sql,
            visual,
            layer_sql,
            stat_sql,
            warnings,
        }
    }

    /// Get the resolved plot specification.
    pub fn plot(&self) -> &Plot {
        &self.plot
    }

    /// Get visualization metadata.
    pub fn metadata(&self) -> &Metadata {
        &self.metadata
    }

    /// Number of layers.
    pub fn layer_count(&self) -> usize {
        self.plot.layers.len()
    }

    /// Get layer-specific data (from FILTER or FROM clause).
    pub fn layer_data(&self, layer_index: usize) -> Option<&DataFrame> {
        self.data.get(&naming::layer_key(layer_index))
    }

    /// Get stat transform data (e.g., histogram bins, density estimates).
    pub fn stat_data(&self, layer_index: usize) -> Option<&DataFrame> {
        self.layer_data(layer_index)
    }

    /// Get internal data map (all DataFrames by key).
    pub fn data(&self) -> &HashMap<String, DataFrame> {
        &self.data
    }

    /// The main SQL query that was executed.
    pub fn sql(&self) -> &str {
        &self.sql
    }

    /// The VISUALISE portion (raw text).
    pub fn visual(&self) -> &str {
        &self.visual
    }

    /// Layer filter/source query, or `None` if using global data.
    pub fn layer_sql(&self, layer_index: usize) -> Option<&str> {
        self.layer_sql.get(layer_index).and_then(|s| s.as_deref())
    }

    /// Stat transform query, or `None` if no stat transform.
    pub fn stat_sql(&self, layer_index: usize) -> Option<&str> {
        self.stat_sql.get(layer_index).and_then(|s| s.as_deref())
    }

    /// Validation warnings from preparation.
    pub fn warnings(&self) -> &[ValidationWarning] {
        &self.warnings
    }
}

impl ResolvedTable {
    /// Create a new ResolvedTable.
    pub(crate) fn new(
        table: Table,
        body: DataFrame,
        sql: String,
        warnings: Vec<ValidationWarning>,
    ) -> Self {
        Self {
            table,
            body,
            sql,
            warnings,
        }
    }

    /// Get the resolved table specification.
    pub fn table(&self) -> &Table {
        &self.table
    }

    /// Get the resolved body data. See the PROVISIONAL note on the `body`
    /// field in `reader::mod` — this accessor's return type will likely
    /// change once real table writers exist.
    pub fn body(&self) -> &DataFrame {
        &self.body
    }

    /// The SQL query that was executed to produce `body`.
    pub fn sql(&self) -> &str {
        &self.sql
    }

    /// Validation warnings from preparation.
    pub fn warnings(&self) -> &[ValidationWarning] {
        &self.warnings
    }
}

impl ResolvedSpec {
    /// Borrow the inner `ResolvedPlot`, or `None` if this is a `ResolvedTable`.
    pub fn as_plot(&self) -> Option<&ResolvedPlot> {
        match self {
            ResolvedSpec::Plot(plot) => Some(plot),
            ResolvedSpec::Table(_) => None,
        }
    }

    /// Borrow the inner `ResolvedTable`, or `None` if this is a `ResolvedPlot`.
    pub fn as_table(&self) -> Option<&ResolvedTable> {
        match self {
            ResolvedSpec::Plot(_) => None,
            ResolvedSpec::Table(table) => Some(table),
        }
    }

    /// Consume this `ResolvedSpec`, returning the inner `ResolvedPlot`, or
    /// `None` if it was a `ResolvedTable`.
    pub fn into_plot(self) -> Option<ResolvedPlot> {
        match self {
            ResolvedSpec::Plot(plot) => Some(*plot),
            ResolvedSpec::Table(_) => None,
        }
    }

    /// Consume this `ResolvedSpec`, returning the inner `ResolvedTable`, or
    /// `None` if it was a `ResolvedPlot`.
    pub fn into_table(self) -> Option<ResolvedTable> {
        match self {
            ResolvedSpec::Plot(_) => None,
            ResolvedSpec::Table(table) => Some(table),
        }
    }
}
