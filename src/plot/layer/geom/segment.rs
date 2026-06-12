//! Segment geom implementation

use super::types::POSITION_VALUES;
use super::{
    densify_edges, needs_projection, project_position_columns, DefaultAesthetics,
    DefaultParamValue, GeomTrait, GeomType, ParamConstraint, ParamDefinition,
};
use crate::plot::projection::Projection;
use crate::plot::types::{DefaultAestheticValue, ParameterValue};
use crate::reader::SqlDialect;
use crate::{naming, Mappings, Result};

/// Segment geom - line segments between two points
#[derive(Debug, Clone, Copy)]
pub struct Segment;

impl GeomTrait for Segment {
    fn geom_type(&self) -> GeomType {
        GeomType::Segment
    }

    fn aesthetics(&self) -> DefaultAesthetics {
        DefaultAesthetics {
            defaults: &[
                ("pos1", DefaultAestheticValue::Required),
                ("pos2", DefaultAestheticValue::Required),
                ("pos1end", DefaultAestheticValue::Required),
                ("pos2end", DefaultAestheticValue::Required),
                ("stroke", DefaultAestheticValue::String("black")),
                ("linewidth", DefaultAestheticValue::Number(1.0)),
                ("opacity", DefaultAestheticValue::Number(1.0)),
                ("linetype", DefaultAestheticValue::String("solid")),
            ],
        }
    }

    fn default_params(&self) -> &'static [ParamDefinition] {
        const PARAMS: &[ParamDefinition] = &[
            ParamDefinition {
                name: "position",
                default: DefaultParamValue::String("identity"),
                constraint: ParamConstraint::string_option(POSITION_VALUES),
            },
            super::types::AGGREGATE_PARAM,
        ];
        PARAMS
    }

    fn aggregate_domain_aesthetics(&self) -> Option<&'static [&'static str]> {
        Some(&[])
    }

    fn apply_projection(
        &self,
        query: &str,
        projection: &Projection,
        dialect: &dyn SqlDialect,
        mappings: &mut Mappings,
        partition_by: &mut Vec<String>,
        parameters: &mut std::collections::HashMap<String, crate::plot::types::ParameterValue>,
    ) -> Result<String> {
        if !needs_projection(projection) {
            return Ok(query.to_string());
        }

        let columns = mappings.column_names();
        let (expanded, expanded_columns) = expand_segment_to_vertices(query, &columns);

        partition_by.push(naming::DENSIFY_ID_COLUMN.to_string());
        parameters.insert("densified".to_string(), ParameterValue::Boolean(true));

        let densified = densify_edges(
            &expanded,
            dialect,
            &expanded_columns,
            partition_by,
            Some("__ggsql_vertex__"),
            false,
            1.0,
            360,
        );
        let projected =
            project_position_columns(&densified, projection, dialect, &expanded_columns)?;

        mappings.insert_column("pos1end", "pos1");
        mappings.insert_column("pos2end", "pos2");

        Ok(projected)
    }
}

/// Expand each segment row into 2 vertex rows (start + end).
///
/// Input: one row per segment with pos1/pos2 (start) and pos1end/pos2end (end).
/// Output: two rows per segment with pos1/pos2 vertex positions and a
/// `DENSIFY_ID_COLUMN` grouping column. Material aesthetics pass through unchanged.
fn expand_segment_to_vertices(query: &str, columns: &[String]) -> (String, Vec<String>) {
    let pos1_col = naming::aesthetic_column("pos1");
    let pos2_col = naming::aesthetic_column("pos2");
    let pos1end_col = naming::aesthetic_column("pos1end");
    let pos2end_col = naming::aesthetic_column("pos2end");

    let passthrough_cols: Vec<&String> = columns
        .iter()
        .filter(|c| *c != &pos1_col && *c != &pos2_col && *c != &pos1end_col && *c != &pos2end_col)
        .collect();
    let passthrough: Vec<String> = passthrough_cols
        .iter()
        .map(|c| naming::quote_ident(c))
        .collect();

    let densify_id_q = naming::quote_ident(naming::DENSIFY_ID_COLUMN);

    let numbered = format!(
        "SELECT *, ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) \
         AS {densify_id_q} FROM ({query})"
    );

    let vertices_table = "(SELECT 0 AS \"__ggsql_vertex__\" UNION ALL SELECT 1)";

    let pos1_q = naming::quote_ident(&pos1_col);
    let pos2_q = naming::quote_ident(&pos2_col);
    let pos1end_q = naming::quote_ident(&pos1end_col);
    let pos2end_q = naming::quote_ident(&pos2end_col);

    let mut select_parts: Vec<String> = passthrough;
    select_parts.push(densify_id_q.to_string());
    select_parts.push("\"__ggsql_vertex__\"".to_string());
    select_parts.push(format!(
        "CASE \"__ggsql_vertex__\" WHEN 0 THEN {pos1_q} WHEN 1 THEN {pos1end_q} END AS {pos1_q}"
    ));
    select_parts.push(format!(
        "CASE \"__ggsql_vertex__\" WHEN 0 THEN {pos2_q} WHEN 1 THEN {pos2end_q} END AS {pos2_q}"
    ));

    let sql = format!(
        "SELECT {} FROM ({numbered}) \"__ggsql_seg__\" \
         CROSS JOIN {vertices_table} \"__ggsql_vertices__\"",
        select_parts.join(", ")
    );

    let mut out_columns: Vec<String> = passthrough_cols.into_iter().cloned().collect();
    out_columns.push(naming::DENSIFY_ID_COLUMN.to_string());
    out_columns.push(pos1_col);
    out_columns.push(pos2_col);

    (sql, out_columns)
}

impl std::fmt::Display for Segment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "segment")
    }
}

#[cfg(test)]
mod tests {
    use super::Segment;
    use crate::plot::layer::geom::GeomTrait;
    use crate::plot::projection::Projection;
    use crate::plot::types::ParameterValue;
    use crate::plot::{AestheticContext, AestheticValue, Geom, Layer};
    use crate::{naming, Mappings};

    fn create_segment_mappings() -> Mappings {
        let mut mappings = Mappings::new();
        for aes in &["pos1", "pos2", "pos1end", "pos2end"] {
            mappings.insert_column(aes, aes);
        }
        mappings
    }

    fn validate_segment(mappings: &[(&str, &str)]) -> Result<(), String> {
        let mut layer = Layer::new(Geom::segment());
        for (aesthetic, column) in mappings {
            layer.mappings.insert(
                aesthetic.to_string(),
                AestheticValue::standard_column(column.to_string()),
            );
        }
        let ctx = AestheticContext::from_static(&["x", "y"], &[]);
        layer.validate_mapping(&Some(ctx), false)
    }

    #[test]
    fn test_segment_requires_both_endpoints() {
        let result = validate_segment(&[("pos1", "x"), ("pos2", "y")]);
        assert!(result.is_err(), "Should fail when missing both endpoints");

        let result = validate_segment(&[("pos1", "x"), ("pos2", "y"), ("pos1end", "xend")]);
        assert!(result.is_err(), "Should fail when missing pos2end");

        let result = validate_segment(&[("pos1", "x"), ("pos2", "y"), ("pos2end", "yend")]);
        assert!(result.is_err(), "Should fail when missing pos1end");
    }

    #[test]
    fn test_segment_validates_with_both_endpoints() {
        let result = validate_segment(&[
            ("pos1", "x"),
            ("pos2", "y"),
            ("pos1end", "xend"),
            ("pos2end", "yend"),
        ]);
        assert!(
            result.is_ok(),
            "Expected validation to pass with both endpoints, got error: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_apply_projection_no_op_without_map() {
        let segment = Segment;
        let projection = Projection::cartesian();
        let mut mappings = create_segment_mappings();
        let mut partition_by = vec![];

        let result = segment
            .apply_projection(
                "SELECT * FROM t",
                &projection,
                &crate::reader::AnsiDialect,
                &mut mappings,
                &mut partition_by,
                &mut std::collections::HashMap::new(),
            )
            .unwrap();

        assert_eq!(result, "SELECT * FROM t");
        assert!(partition_by.is_empty());
    }

    #[test]
    fn test_apply_projection_expands_and_densifies() {
        let segment = Segment;
        let mut projection = Projection::map();
        projection.properties.insert(
            "source".to_string(),
            ParameterValue::String("EPSG:4326".to_string()),
        );
        projection.properties.insert(
            "target".to_string(),
            ParameterValue::String("+proj=merc".to_string()),
        );

        let mut mappings = create_segment_mappings();
        let mut partition_by = vec![];

        let result = segment
            .apply_projection(
                "SELECT * FROM t",
                &projection,
                &crate::reader::AnsiDialect,
                &mut mappings,
                &mut partition_by,
                &mut std::collections::HashMap::new(),
            )
            .unwrap();

        assert!(result.contains(naming::DENSIFY_ID_COLUMN));
        assert!(result.contains("CROSS JOIN"));
        assert!(result.contains("ST_Transform"));
        assert!(partition_by.contains(&naming::DENSIFY_ID_COLUMN.to_string()));
        // pos1end/pos2end remapped to pos1/pos2 columns
        assert!(mappings.contains_key("pos1end"));
        assert!(mappings.contains_key("pos2end"));
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_densified_segment_produces_intermediate_vertices() {
        use crate::reader::{DuckDBReader, Reader};
        use arrow::array::Array;

        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let dialect = reader.dialect();

        // A segment spanning 40° of longitude at 45°N
        let input = format!(
            "SELECT -80.0 AS \"{}\", 45.0 AS \"{}\", \
                    -40.0 AS \"{}\", 45.0 AS \"{}\"",
            naming::aesthetic_column("pos1"),
            naming::aesthetic_column("pos2"),
            naming::aesthetic_column("pos1end"),
            naming::aesthetic_column("pos2end"),
        );

        let segment = Segment;
        let mut projection = Projection::map();
        projection.properties.insert(
            "source".to_string(),
            ParameterValue::String("EPSG:4326".to_string()),
        );
        projection.properties.insert(
            "target".to_string(),
            ParameterValue::String("+proj=ortho +lat_0=45 +lon_0=-60".to_string()),
        );

        let mut mappings = create_segment_mappings();
        let mut partition_by = vec![];

        for stmt in dialect.sql_spatial_setup() {
            reader.execute_sql(&stmt).unwrap();
        }

        let projected_sql = segment
            .apply_projection(
                &input,
                &projection,
                dialect,
                &mut mappings,
                &mut partition_by,
                &mut std::collections::HashMap::new(),
            )
            .unwrap();

        let df = reader.execute_sql(&projected_sql).unwrap();
        let n = df.inner().num_rows();
        assert!(
            n > 2,
            "expected densified vertices (more than start+end), got {n}"
        );

        let pos1_col = df
            .inner()
            .column_by_name(&naming::aesthetic_column("pos1"))
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap();
        let pos2_col = df
            .inner()
            .column_by_name(&naming::aesthetic_column("pos2"))
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap();

        // Verify no NULLs in the projected positions
        assert_eq!(pos1_col.null_count(), 0);
        assert_eq!(pos2_col.null_count(), 0);

        // First and last vertex should differ (segment has distinct endpoints)
        let first_x = pos1_col.value(0);
        let last_x = pos1_col.value(n - 1);
        assert!(
            (first_x - last_x).abs() > 1e-6,
            "endpoints should differ after projection"
        );
    }
}
