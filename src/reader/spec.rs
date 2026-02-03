//! Implementation of Spec methods.

use std::collections::HashMap;

use crate::naming;
use crate::plot::Plot;
use crate::validate::ValidationWarning;
use crate::DataFrame;

use super::{Metadata, Spec};

impl Spec {
    /// Create a new Spec from PreparedData
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
        let (rows, columns) = if let Some(df) = data.get(naming::GLOBAL_DATA_KEY) {
            let cols: Vec<String> = df
                .get_column_names()
                .iter()
                .map(|s| s.to_string())
                .collect();
            (df.height(), cols)
        } else if let Some(df) = data.values().next() {
            let cols: Vec<String> = df
                .get_column_names()
                .iter()
                .map(|s| s.to_string())
                .collect();
            (df.height(), cols)
        } else {
            (0, Vec::new())
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

    /// Get global data (main query result).
    pub fn data(&self) -> Option<&DataFrame> {
        self.data.get(naming::GLOBAL_DATA_KEY)
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
    pub fn data_map(&self) -> &HashMap<String, DataFrame> {
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
