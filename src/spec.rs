//! The parse-time result of a single VISUALISE/TABULATE statement.

use serde::{Deserialize, Serialize};

use crate::{Plot, Table};

/// One parsed statement from a ggsql query: either a visualization or a table.
///
/// A query may contain several `VISUALISE`/`TABULATE` statements
/// (`parser::parse_query` returns one `Spec` per statement, in source order).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Spec {
    // Boxed: `Plot` is far larger than `Table`, and clippy flags the
    // resulting size gap (`large_enum_variant`) otherwise.
    Plot(Box<Plot>),
    Table(Table),
}

impl Spec {
    /// Borrow the inner `Plot`, or `None` if this is a `Table`.
    pub fn as_plot(&self) -> Option<&Plot> {
        match self {
            Spec::Plot(plot) => Some(plot),
            Spec::Table(_) => None,
        }
    }

    /// Borrow the inner `Table`, or `None` if this is a `Plot`.
    pub fn as_table(&self) -> Option<&Table> {
        match self {
            Spec::Plot(_) => None,
            Spec::Table(table) => Some(table),
        }
    }

    /// Consume this `Spec`, returning the inner `Plot`, or `None` if it was a `Table`.
    pub fn into_plot(self) -> Option<Plot> {
        match self {
            Spec::Plot(plot) => Some(*plot),
            Spec::Table(_) => None,
        }
    }

    /// Consume this `Spec`, returning the inner `Table`, or `None` if it was a `Plot`.
    pub fn into_table(self) -> Option<Table> {
        match self {
            Spec::Plot(_) => None,
            Spec::Table(table) => Some(table),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn as_plot_and_as_table_are_mutually_exclusive() {
        let plot = Spec::Plot(Box::default());
        assert!(plot.as_plot().is_some());
        assert!(plot.as_table().is_none());

        let table = Spec::Table(Table::default());
        assert!(table.as_plot().is_none());
        assert!(table.as_table().is_some());
    }

    #[test]
    fn into_plot_and_into_table_are_mutually_exclusive() {
        let plot = Spec::Plot(Box::default());
        assert!(plot.into_plot().is_some());

        let table = Spec::Table(Table::default());
        assert!(table.into_table().is_some());

        assert!(Spec::Plot(Box::default()).into_table().is_none());
        assert!(Spec::Table(Table::default()).into_plot().is_none());
    }
}
