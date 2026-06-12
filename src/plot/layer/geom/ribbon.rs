//! Ribbon geom implementation

use super::stat_aggregate;
use super::types::{wrap_with_order_by, POSITION_VALUES};
use super::{
    densify_edges, has_aggregate_param, needs_projection, project_position_columns,
    DefaultAesthetics, GeomTrait, GeomType, StatResult,
};
use crate::plot::projection::Projection;
use crate::plot::types::{DefaultAestheticValue, ParameterValue};
use crate::plot::{DefaultParamValue, ParamConstraint, ParamDefinition};
use crate::reader::SqlDialect;
use crate::{naming, Mappings, Result};

/// Ribbon geom - confidence bands and ranges
#[derive(Debug, Clone, Copy)]
pub struct Ribbon;

impl GeomTrait for Ribbon {
    fn geom_type(&self) -> GeomType {
        GeomType::Ribbon
    }

    fn aesthetics(&self) -> DefaultAesthetics {
        DefaultAesthetics {
            defaults: &[
                ("pos1", DefaultAestheticValue::Required),
                ("pos2min", DefaultAestheticValue::Required),
                ("pos2max", DefaultAestheticValue::Required),
                ("fill", DefaultAestheticValue::String("black")),
                ("stroke", DefaultAestheticValue::String("black")),
                ("opacity", DefaultAestheticValue::Number(0.8)),
                ("linewidth", DefaultAestheticValue::Number(1.0)),
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
        Some(&["pos1"])
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
        let (expanded, expanded_columns) = expand_ribbon_to_polygon(query, &columns, partition_by);

        partition_by.push(naming::DENSIFY_ID_COLUMN.to_string());
        parameters.insert("densified".to_string(), ParameterValue::Boolean(true));

        let densified = densify_edges(
            &expanded,
            dialect,
            &expanded_columns,
            partition_by,
            Some("__ggsql_vertex__"),
            true,
            1.0,
            360,
        );
        let projected =
            project_position_columns(&densified, projection, dialect, &expanded_columns)?;

        mappings.insert_column("pos2", "pos2");
        mappings.insert_column("pos2min", "pos2");
        mappings.insert_column("pos2max", "pos2");

        Ok(projected)
    }

    fn apply_stat_transform(
        &self,
        query: &str,
        schema: &crate::plot::Schema,
        aesthetics: &Mappings,
        group_by: &[String],
        parameters: &std::collections::HashMap<String, crate::plot::ParameterValue>,
        _execute_query: &dyn Fn(&str) -> crate::Result<crate::DataFrame>,
        dialect: &dyn crate::reader::SqlDialect,
        aesthetic_ctx: &crate::plot::aesthetic::AestheticContext,
    ) -> crate::Result<StatResult> {
        let result = if has_aggregate_param(parameters) {
            stat_aggregate::apply(
                query,
                schema,
                aesthetics,
                group_by,
                parameters,
                dialect,
                aesthetic_ctx,
                self.aggregate_domain_aesthetics().unwrap_or(&[]),
            )?
        } else {
            StatResult::Identity
        };
        // Ribbon needs ordering by pos1 (domain axis) for proper rendering, in both
        // the Identity and Aggregate paths.
        Ok(wrap_with_order_by(query, result, "pos1"))
    }
}

/// Expand a ribbon (pos1, pos2min, pos2max) into a closed polygon outline.
///
/// The outline traces the upper edge forward (pos1 ascending, pos2max) then
/// the lower edge backward (pos1 descending, pos2min). Each row produces two
/// vertices; the vertex index (`__ggsql_vertex__`) orders upper edge first,
/// then lower edge reversed.
fn expand_ribbon_to_polygon(
    query: &str,
    columns: &[String],
    partition_by: &[String],
) -> (String, Vec<String>) {
    let pos1_col = naming::aesthetic_column("pos1");
    let pos2min_col = naming::aesthetic_column("pos2min");
    let pos2max_col = naming::aesthetic_column("pos2max");
    let pos2_col = naming::aesthetic_column("pos2");

    let passthrough_cols: Vec<&String> = columns
        .iter()
        .filter(|c| *c != &pos1_col && *c != &pos2min_col && *c != &pos2max_col)
        .collect();
    let passthrough_quoted: Vec<String> = passthrough_cols
        .iter()
        .map(|c| naming::quote_ident(c))
        .collect();

    let pos1_q = naming::quote_ident(&pos1_col);
    let pos2min_q = naming::quote_ident(&pos2min_col);
    let pos2max_q = naming::quote_ident(&pos2max_col);
    let pos2_q = naming::quote_ident(&pos2_col);

    // Number rows within each group by pos1 order and compute the group size.
    let partition_clause = if partition_by.is_empty() {
        String::new()
    } else {
        let parts: Vec<String> = partition_by
            .iter()
            .map(|c| naming::quote_ident(c))
            .collect();
        format!("PARTITION BY {} ", parts.join(", "))
    };

    // ribbon_id: unique per partition group (DENSE_RANK over partition columns).
    // When no partition columns exist, every row belongs to one ribbon → constant 1.
    let ribbon_id_expr = if partition_by.is_empty() {
        "1".to_string()
    } else {
        let parts: Vec<String> = partition_by
            .iter()
            .map(|c| naming::quote_ident(c))
            .collect();
        format!("DENSE_RANK() OVER (ORDER BY {})", parts.join(", "))
    };

    let densify_id_q = naming::quote_ident(naming::DENSIFY_ID_COLUMN);

    let numbered = format!(
        "SELECT *, \
         ROW_NUMBER() OVER ({partition_clause}ORDER BY {pos1_q}) AS \"__ggsql_row_idx__\", \
         COUNT(*) OVER ({partition_clause}) AS \"__ggsql_n_rows__\", \
         {ribbon_id_expr} AS {densify_id_q} \
         FROM ({query})"
    );

    // Build select list for each half
    let mut common_select: Vec<String> = passthrough_quoted.clone();
    common_select.push(densify_id_q.to_string());

    // Upper edge: vertex index = row_idx (1..n), pos2 = pos2max
    let mut upper_parts = common_select.clone();
    upper_parts.push("\"__ggsql_row_idx__\" AS \"__ggsql_vertex__\"".to_string());
    upper_parts.push(pos1_q.to_string());
    upper_parts.push(format!("{pos2max_q} AS {pos2_q}"));

    // Lower edge: vertex index = 2*n - row_idx + 1 (n+1..2n), pos2 = pos2min
    let mut lower_parts = common_select;
    lower_parts.push(
        "(2 * \"__ggsql_n_rows__\" - \"__ggsql_row_idx__\" + 1) AS \"__ggsql_vertex__\""
            .to_string(),
    );
    lower_parts.push(pos1_q.to_string());
    lower_parts.push(format!("{pos2min_q} AS {pos2_q}"));

    let sql = format!(
        "WITH \"__ggsql_r__\" AS ({numbered}) \
         SELECT {} FROM \"__ggsql_r__\" \
         UNION ALL \
         SELECT {} FROM \"__ggsql_r__\"",
        upper_parts.join(", "),
        lower_parts.join(", "),
    );

    let mut out_columns: Vec<String> = passthrough_cols.into_iter().cloned().collect();
    out_columns.push(naming::DENSIFY_ID_COLUMN.to_string());
    out_columns.push(pos1_col);
    out_columns.push(pos2_col);

    (sql, out_columns)
}

impl std::fmt::Display for Ribbon {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ribbon")
    }
}

#[cfg(test)]
mod tests {
    use super::Ribbon;
    use crate::plot::layer::geom::GeomTrait;
    use crate::plot::projection::Projection;
    use crate::plot::types::ParameterValue;
    use crate::{naming, Mappings};

    fn create_ribbon_mappings() -> Mappings {
        let mut mappings = Mappings::new();
        for aes in &["pos1", "pos2min", "pos2max"] {
            mappings.insert_column(aes, aes);
        }
        mappings
    }

    #[test]
    fn test_apply_projection_no_op_without_map() {
        let ribbon = Ribbon;
        let projection = Projection::cartesian();
        let mut mappings = create_ribbon_mappings();
        let mut partition_by = vec![];

        let result = ribbon
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
    fn test_apply_projection_expands_to_polygon() {
        let ribbon = Ribbon;
        let mut projection = Projection::map();
        projection.properties.insert(
            "source".to_string(),
            ParameterValue::String("EPSG:4326".to_string()),
        );
        projection.properties.insert(
            "target".to_string(),
            ParameterValue::String("+proj=merc".to_string()),
        );

        let mut mappings = create_ribbon_mappings();
        let mut partition_by = vec![];

        let result = ribbon
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
        assert!(result.contains("UNION ALL"));
        assert!(result.contains("ST_Transform"));
        assert!(partition_by.contains(&naming::DENSIFY_ID_COLUMN.to_string()));
        assert!(mappings.contains_key("pos2"));
    }

    #[cfg(feature = "duckdb")]
    #[test]
    fn test_densified_ribbon_produces_closed_polygon() {
        use crate::reader::{DuckDBReader, Reader};
        use arrow::array::Array;

        let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let dialect = reader.dialect();

        // A ribbon spanning 60° of longitude with a 10° band width
        let input = format!(
            "SELECT * FROM (VALUES \
             (-90.0, 40.0, 50.0), \
             (-60.0, 42.0, 52.0), \
             (-30.0, 38.0, 48.0)) \
             AS t(\"{}\", \"{}\", \"{}\")",
            naming::aesthetic_column("pos1"),
            naming::aesthetic_column("pos2min"),
            naming::aesthetic_column("pos2max"),
        );

        let ribbon = Ribbon;
        let mut projection = Projection::map();
        projection.properties.insert(
            "source".to_string(),
            ParameterValue::String("EPSG:4326".to_string()),
        );
        projection.properties.insert(
            "target".to_string(),
            ParameterValue::String("+proj=ortho +lat_0=45 +lon_0=-60".to_string()),
        );

        let mut mappings = create_ribbon_mappings();
        let mut partition_by = vec![];

        for stmt in dialect.sql_spatial_setup() {
            reader.execute_sql(&stmt).unwrap();
        }

        let projected_sql = ribbon
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
        // 3 input rows → 6 polygon vertices (upper + lower) before densification,
        // densification adds intermediate vertices along edges > 1°
        assert!(n > 6, "expected densified polygon vertices, got {n}");

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
}
