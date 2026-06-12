//! Rule geom implementation

use super::{
    densify_edges, needs_projection, project_position_columns, DefaultAesthetics, GeomTrait,
    GeomType, ParamDefinition,
};
use crate::plot::projection::coord::map::clip_boundary_table;
use crate::plot::projection::coord::CoordKind;
use crate::plot::projection::Projection;
use crate::plot::types::{DefaultAestheticValue, ParameterValue};
use crate::reader::SqlDialect;
use crate::{naming, Mappings, Result};

/// Rule geom - horizontal and vertical reference lines
#[derive(Debug, Clone, Copy)]
pub struct Rule;

impl GeomTrait for Rule {
    fn geom_type(&self) -> GeomType {
        GeomType::Rule
    }

    fn aesthetics(&self) -> DefaultAesthetics {
        DefaultAesthetics {
            defaults: &[
                ("pos1", DefaultAestheticValue::Required),
                ("slope", DefaultAestheticValue::Number(0.0)),
                ("stroke", DefaultAestheticValue::String("black")),
                ("linewidth", DefaultAestheticValue::Number(1.0)),
                ("opacity", DefaultAestheticValue::Number(1.0)),
                ("linetype", DefaultAestheticValue::String("solid")),
            ],
        }
    }

    fn default_params(&self) -> &'static [ParamDefinition] {
        const PARAMS: &[ParamDefinition] = &[super::types::AGGREGATE_PARAM];
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

        // The rule input always has one position column named __ggsql_aes_pos1__
        // regardless of whether the aesthetic key is "pos1" or "pos2" — the executor
        // normalizes it to the pos1 slot.
        let columns = vec![naming::aesthetic_column("pos1")];

        let has_pos1 = mappings.contains_key("pos1");
        let bbox_expr = match projection.coord.coord_kind() {
            CoordKind::Map
                if matches!(
                    projection.properties.get("clip"),
                    Some(ParameterValue::Boolean(true))
                ) =>
            {
                let boundary_table = clip_boundary_table();
                dialect.sql_geometry_bbox("geom", &boundary_table)
            }
            // If we don't have a bbox, we cannot expand rule layer
            _ => return project_position_columns(query, projection, dialect, &columns),
        };
        let (expanded, expanded_columns) =
            expand_rule_to_segment(query, &columns, has_pos1, &bbox_expr);

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
        let clipped = match projection.coord.coord_kind() {
            CoordKind::Map => {
                let pos1_q = naming::quote_ident(&naming::aesthetic_column("pos1"));
                let pos2_q = naming::quote_ident(&naming::aesthetic_column("pos2"));
                let clip_table = clip_boundary_table();
                format!(
                    "SELECT * FROM ({densified}) WHERE ST_Contains(\
                     (SELECT geom FROM {clip_table}), ST_Point({pos1_q}, {pos2_q}))"
                )
            }
            _ => densified,
        };

        let projected = project_position_columns(&clipped, projection, dialect, &expanded_columns)?;

        // Both pos1 and pos2 columns exist in the output SQL (the spanning axis
        // was synthesized). Add the new axis to mappings so the writer encodes it.
        if !has_pos1 {
            mappings.aesthetics.remove("pos2");
            mappings.insert_column("pos1", "pos1");
            mappings.insert_column("pos2", "pos2");
        } else {
            mappings.insert_column("pos2", "pos2");
        }

        // After densification both axes are explicit — disable the orientation flip
        // that would otherwise swap the DataFrame columns.
        parameters.insert(
            "orientation".to_string(),
            ParameterValue::String("aligned".to_string()),
        );

        Ok(projected)
    }

    fn validate_aesthetics(
        &self,
        mappings: &crate::Mappings,
        aesthetic_ctx: &Option<crate::plot::aesthetic::AestheticContext>,
        parameters: &std::collections::HashMap<String, crate::plot::types::ParameterValue>,
    ) -> std::result::Result<(), String> {
        // Rule requires exactly one of pos1 or pos2 (XOR logic).
        // After densification both axes are present — skip the check.
        if matches!(
            parameters.get("densified"),
            Some(ParameterValue::Boolean(true))
        ) {
            return Ok(());
        }

        let has_pos1 = mappings.contains_key("pos1");
        let has_pos2 = mappings.contains_key("pos2");

        if has_pos1 && has_pos2 {
            let translate = |aes: &str| match aesthetic_ctx {
                Some(ctx) => ctx.map_internal_to_user(aes),
                None => aes.to_string(),
            };
            return Err(format!(
                "Layer 'rule' requires exactly one of `{}` or `{}`, not both.",
                translate("pos1"),
                translate("pos2")
            ));
        }

        Ok(())
    }

    fn setup_layer(
        &self,
        mappings: &mut crate::plot::layer::Mappings,
        parameters: &mut std::collections::HashMap<String, crate::plot::ParameterValue>,
    ) -> crate::Result<()> {
        use crate::plot::layer::AestheticValue;
        use crate::plot::ParameterValue;

        // For diagonal rules (slope present), convert position aesthetics to AnnotationColumn
        // so they don't participate in scale training. The position value is the intercept,
        // not the actual extent of the line.

        // Check if slope is present and non-zero (in either mappings or parameters)
        let has_diagonal_slope = mappings.get("slope").is_some_and(|mapping| {
            !matches!(mapping, AestheticValue::Literal(ParameterValue::Number(n)) if *n == 0.0)
        }) || parameters.get("slope").is_some_and(|param| {
            !matches!(param, ParameterValue::Number(n) if *n == 0.0)
        });

        if !has_diagonal_slope {
            return Ok(());
        }
        parameters.insert("diagonal".to_string(), ParameterValue::Boolean(true));

        // Determine orientation from which intercept is present.
        // We override the bidirectionality algorithm here since it uses
        // scales to determine orientation. We can't rely on that here,
        // because diagonal lines purposefully use AnnotationColumns to
        // avoid training the scale.
        let orientation = if mappings.contains_key("pos1") {
            // x-intercept → pos2 varies
            crate::plot::layer::orientation::TRANSPOSED
        } else {
            // y-intercept → pos1 varies (or default)
            crate::plot::layer::orientation::ALIGNED
        };

        parameters.insert(
            "orientation".to_string(),
            ParameterValue::String(orientation.to_string()),
        );

        // For diagonal rules, convert pos1/pos2 to AnnotationColumn so they don't participate in scale training
        // The position value is the intercept, not the actual extent of the line
        for aesthetic in ["pos1", "pos2"] {
            if let Some(mapping) = mappings.aesthetics.get_mut(aesthetic) {
                // Convert Column to AnnotationColumn
                if let AestheticValue::Column { name, .. } = &*mapping {
                    let name = name.clone();
                    *mapping = AestheticValue::AnnotationColumn { name };
                }
            }
        }

        Ok(())
    }
}

/// Expand each rule into 2 vertex rows (start + end of the spanning axis).
///
/// The input always has a single position column `__ggsql_aes_pos1__` (the
/// executor normalizes the rule's fixed axis into the pos1 slot). `has_pos1`
/// tells us the semantic meaning:
/// - true (vertical rule): pos1 is longitude (fixed), synthesize pos2 from bbox y.
/// - false (horizontal rule): pos1 is latitude (fixed), rename to pos2, synthesize
///   pos1 from bbox x.
///
/// `bbox_expr` is a SQL expression that yields xmin, ymin, xmax, ymax columns.
fn expand_rule_to_segment(
    query: &str,
    columns: &[String],
    has_pos1: bool,
    bbox_expr: &str,
) -> (String, Vec<String>) {
    let pos1_col = naming::aesthetic_column("pos1");
    let pos2_col = naming::aesthetic_column("pos2");
    let pos1_q = naming::quote_ident(&pos1_col);
    let pos2_q = naming::quote_ident(&pos2_col);

    // The input column is always __ggsql_aes_pos1__. Build the SELECT
    // expressions that produce both pos1 and pos2 in the output.
    let (fixed_expr, span_expr) = if has_pos1 {
        // Vertical rule: input pos1 = longitude (keep as pos1), synthesize pos2 from y-extent
        let fixed = pos1_q.clone();
        let span = format!(
            "CASE \"__ggsql_vertex__\" WHEN 0 THEN (SELECT ymin FROM ({bbox_expr})) \
             WHEN 1 THEN (SELECT ymax FROM ({bbox_expr})) END AS {pos2_q}"
        );
        (fixed, span)
    } else {
        // Horizontal rule: input pos1 = latitude (rename to pos2), synthesize pos1 from x-extent
        let fixed = format!("{pos1_q} AS {pos2_q}");
        let span = format!(
            "CASE \"__ggsql_vertex__\" WHEN 0 THEN (SELECT xmin FROM ({bbox_expr})) \
             WHEN 1 THEN (SELECT xmax FROM ({bbox_expr})) END AS {pos1_q}"
        );
        (fixed, span)
    };

    // Passthrough: columns minus the input pos1 column (we handle it explicitly above)
    let passthrough_cols: Vec<&String> = columns.iter().filter(|c| *c != &pos1_col).collect();
    let passthrough_quoted: Vec<String> = passthrough_cols
        .iter()
        .map(|c| naming::quote_ident(c))
        .collect();

    let densify_id_q = naming::quote_ident(naming::DENSIFY_ID_COLUMN);

    let numbered = format!(
        "SELECT *, ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) \
         AS {densify_id_q} FROM ({query})"
    );

    let vertices_table = "(SELECT 0 AS \"__ggsql_vertex__\" UNION ALL SELECT 1)";

    let mut select_parts: Vec<String> = passthrough_quoted;
    select_parts.push(densify_id_q.to_string());
    select_parts.push("\"__ggsql_vertex__\"".to_string());
    select_parts.push(fixed_expr);
    select_parts.push(span_expr);

    let sql = format!(
        "SELECT {} FROM ({numbered}) \"__ggsql_rule__\" \
         CROSS JOIN {vertices_table} \"__ggsql_vertices__\"",
        select_parts.join(", ")
    );

    let mut out_columns: Vec<String> = passthrough_cols.into_iter().cloned().collect();
    out_columns.push(naming::DENSIFY_ID_COLUMN.to_string());
    out_columns.push(pos1_col);
    out_columns.push(pos2_col);

    (sql, out_columns)
}

impl std::fmt::Display for Rule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "rule")
    }
}

#[cfg(test)]
mod tests {
    use super::{expand_rule_to_segment, Rule};
    use crate::plot::layer::geom::{densify_edges, GeomTrait};
    use crate::plot::projection::Projection;
    use crate::plot::types::ParameterValue;
    use crate::plot::{AestheticContext, AestheticValue, Geom, Layer};
    use crate::{naming, Mappings};

    fn validate_rule(mappings: &[(&str, &str)]) -> Result<(), String> {
        let mut layer = Layer::new(Geom::rule());
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
    fn test_rule_requires_exactly_one_position() {
        // Rule requires exactly one of pos1 or pos2 (XOR logic)

        // Missing both should fail
        let result = validate_rule(&[]);
        assert!(result.is_err(), "Should fail when missing both x and y");

        // Both present should fail
        let result = validate_rule(&[("pos1", "x"), ("pos2", "y")]);
        assert!(result.is_err(), "Should fail when both x and y are present");
    }

    #[test]
    fn test_rule_validates_with_x_only() {
        // Vertical rule with only x
        let result = validate_rule(&[("pos1", "x")]);
        assert!(
            result.is_ok(),
            "Expected validation to pass with only x, got error: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_rule_validates_with_y_only() {
        // Horizontal rule with only y
        let result = validate_rule(&[("pos2", "y")]);
        assert!(
            result.is_ok(),
            "Expected validation to pass with only y, got error: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_apply_projection_no_op_without_map() {
        let rule = Rule;
        let projection = Projection::cartesian();
        let mut mappings = Mappings::new();
        mappings.insert_column("pos1", "pos1");
        let mut partition_by = vec![];

        let result = rule
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
    fn test_apply_projection_without_clip_only_projects() {
        let rule = Rule;
        let mut projection = Projection::map();
        projection.properties.insert(
            "source".to_string(),
            ParameterValue::String("EPSG:4326".to_string()),
        );
        projection.properties.insert(
            "target".to_string(),
            ParameterValue::String("+proj=merc".to_string()),
        );
        projection
            .properties
            .insert("clip".to_string(), ParameterValue::Boolean(false));

        let mut mappings = Mappings::new();
        mappings.insert_column("pos1", "pos1");
        let mut partition_by = vec![];

        let result = rule
            .apply_projection(
                "SELECT * FROM t",
                &projection,
                &crate::reader::AnsiDialect,
                &mut mappings,
                &mut partition_by,
                &mut std::collections::HashMap::new(),
            )
            .unwrap();

        assert!(result.contains("ST_Transform"));
        assert!(!result.contains(naming::DENSIFY_ID_COLUMN));
        assert!(partition_by.is_empty());
    }

    #[test]
    fn test_apply_projection_with_clip_expands_and_densifies() {
        let rule = Rule;
        let mut projection = Projection::map();
        projection.properties.insert(
            "source".to_string(),
            ParameterValue::String("EPSG:4326".to_string()),
        );
        projection.properties.insert(
            "target".to_string(),
            ParameterValue::String("+proj=merc".to_string()),
        );
        projection
            .properties
            .insert("clip".to_string(), ParameterValue::Boolean(true));

        let mut mappings = Mappings::new();
        mappings.insert_column("pos1", "pos1");
        let mut partition_by = vec![];

        let result = rule
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
        // pos2 should be added to mappings (the spanning axis)
        assert!(mappings.contains_key("pos2"));
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_densified_rule_produces_intermediate_vertices() {
        use crate::plot::projection::coord::map::clip_boundary_table;
        use crate::reader::{DuckDBReader, Reader};
        use arrow::array::Array;

        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let dialect = reader.dialect();

        for stmt in dialect.sql_spatial_setup() {
            reader.execute_sql(&stmt).unwrap();
        }

        // Create a clip boundary table simulating an orthographic hemisphere
        let boundary_table = clip_boundary_table();
        let create_sql = format!(
            "CREATE TEMP TABLE \"{boundary_table}\" AS \
             SELECT ST_GeomFromText(\
                 'POLYGON ((-90 -60, 90 -60, 90 60, -90 60, -90 -60))'\
             ) AS geom"
        );
        reader.execute_sql(&create_sql).unwrap();

        // A vertical rule at lon = -30
        let input = format!("SELECT -30.0 AS \"{}\"", naming::aesthetic_column("pos1"),);

        let rule = Rule;
        let mut projection = Projection::map();
        projection.properties.insert(
            "source".to_string(),
            ParameterValue::String("EPSG:4326".to_string()),
        );
        projection.properties.insert(
            "target".to_string(),
            ParameterValue::String("+proj=ortho +lat_0=0 +lon_0=0".to_string()),
        );
        projection
            .properties
            .insert("clip".to_string(), ParameterValue::Boolean(true));

        let mut mappings = Mappings::new();
        mappings.insert_column("pos1", "pos1");
        let mut partition_by = vec![];

        let projected_sql = rule
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

        assert_eq!(pos1_col.null_count(), 0);
        assert_eq!(pos2_col.null_count(), 0);
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_densified_horizontal_rule_keeps_latitude_constant() {
        use crate::plot::projection::coord::map::clip_boundary_table;
        use crate::reader::{DuckDBReader, Reader};

        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let dialect = reader.dialect();

        for stmt in dialect.sql_spatial_setup() {
            reader.execute_sql(&stmt).unwrap();
        }

        // Create a clip boundary table (simple rectangle in EPSG:4326)
        let boundary_table = clip_boundary_table();
        let create_sql = format!(
            "CREATE TEMP TABLE \"{boundary_table}\" AS \
             SELECT ST_GeomFromText(\
                 'POLYGON ((-90 -60, 90 -60, 90 60, -90 60, -90 -60))'\
             ) AS geom"
        );
        reader.execute_sql(&create_sql).unwrap();

        // A horizontal rule at lat = 20 (pos2 mapped, not pos1)
        let input = format!("SELECT 20.0 AS \"{}\"", naming::aesthetic_column("pos1"),);

        let rule = Rule;
        let mut projection = Projection::map();
        projection.properties.insert(
            "source".to_string(),
            ParameterValue::String("EPSG:4326".to_string()),
        );
        projection.properties.insert(
            "target".to_string(),
            ParameterValue::String("+proj=ortho +lat_0=0 +lon_0=0".to_string()),
        );
        projection
            .properties
            .insert("clip".to_string(), ParameterValue::Boolean(true));

        // Horizontal rule: pos2 is mapped (latitude is the fixed axis)
        let mut mappings = Mappings::new();
        mappings.insert_column("pos2", "pos2");
        let mut partition_by = vec![];

        rule.apply_projection(
            &input,
            &projection,
            dialect,
            &mut mappings,
            &mut partition_by,
            &mut std::collections::HashMap::new(),
        )
        .unwrap();

        // Run expand + densify manually to verify latitude stays constant pre-projection.
        let columns = vec![naming::aesthetic_column("pos1")];
        let has_pos1 = false;
        let bbox_expr = dialect.sql_geometry_bbox("geom", &boundary_table);
        let (expanded, expanded_columns) =
            expand_rule_to_segment(&input, &columns, has_pos1, &bbox_expr);
        let densified = densify_edges(
            &expanded,
            dialect,
            &expanded_columns,
            &[naming::DENSIFY_ID_COLUMN.to_string()],
            Some("__ggsql_vertex__"),
            false,
            1.0,
            360,
        );

        let df = reader.execute_sql(&densified).unwrap();
        let n = df.inner().num_rows();
        assert!(n > 2, "expected densified vertices, got {n}");

        let pos2_col = df
            .inner()
            .column_by_name(&naming::aesthetic_column("pos2"))
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap();

        // All pos2 (latitude) values should be exactly 20.0
        for i in 0..n {
            let val = pos2_col.value(i);
            assert!(
                (val - 20.0).abs() < 1e-10,
                "row {i}: expected pos2=20.0, got {val}"
            );
        }
    }
}
