//! Implementation of ResolvedPlot, ResolvedTable, and ResolvedSpec methods.

use std::collections::HashMap;

use crate::execute::{count_cell_cols, count_cell_rows};
use crate::naming;
use crate::plot::Plot;
use crate::validate::ValidationWarning;
use crate::{DataFrame, TableCell, TableColumn, TableRow};

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
        cells: Vec<TableCell>,
        columns: Vec<TableColumn>,
        rows: Vec<TableRow>,
        sql: String,
        warnings: Vec<ValidationWarning>,
    ) -> Self {
        Self {
            cells,
            columns,
            rows,
            sql,
            warnings,
        }
    }

    /// Get the resolved layout: one cell per column label and per data value.
    pub fn cells(&self) -> &[TableCell] {
        &self.cells
    }

    /// Resolved per-column properties — a writer wanting a whole-column
    /// value (e.g. `width`) reads it here instead of the same value
    /// repeated across the column's cells.
    pub fn columns(&self) -> &[TableColumn] {
        &self.columns
    }

    /// Resolved per-row properties, symmetric with `columns`. `properties`
    /// has no row-wide `TABULATE` clause to populate it yet, but `classes`
    /// does — a `Title`/`Subtitle` cell's row carries `TableClass::Heading`
    /// here, for a writer to put on the enclosing `<tr>`.
    pub fn rows(&self) -> &[TableRow] {
        &self.rows
    }

    /// Total number of rows in the table's rendered grid — heading, spanner,
    /// and column-label rows included, not just data rows.
    pub fn nrow(&self) -> usize {
        count_cell_rows(&self.cells)
    }

    /// Number of columns, computed from `cells`.
    pub fn ncol(&self) -> usize {
        count_cell_cols(&self.cells)
    }

    /// The SQL query that was executed to produce `cells`.
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
            ResolvedSpec::Table(table) => Some(*table),
        }
    }
}
