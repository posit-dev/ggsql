use super::{DefaultAesthetics, GeomTrait, GeomType, StatResult};
use crate::naming;
use crate::plot::projection::coord::map::clip_boundary_table;
use crate::plot::projection::coord::CoordKind;
use crate::plot::projection::Projection;
use crate::plot::types::DefaultAestheticValue;
use crate::plot::ParameterValue;
use crate::reader::SqlDialect;
use crate::Mappings;

fn apply_clip_boundary(
    query: &str,
    col: &str,
    source: &str,
    crs: &str,
    dialect: &dyn SqlDialect,
    columns: &[String],
) -> String {
    let clip_table = clip_boundary_table();
    let clip_geom = format!("(SELECT geom FROM {clip_table})");

    let clipped = format!("ST_Intersection({col}, {clip_geom})");
    let transformed = dialect.sql_st_transform(&clipped, source, crs);
    let geom_expr = format!("ST_MakeValid({transformed})");

    let filtered = format!("SELECT * FROM ({query}) WHERE ST_Intersects({col}, {clip_geom})");
    dialect.sql_select_replace(&geom_expr, col, &filtered, columns)
}

#[derive(Debug, Clone, Copy)]
pub struct Spatial;

impl GeomTrait for Spatial {
    fn geom_type(&self) -> GeomType {
        GeomType::Spatial
    }

    fn aesthetics(&self) -> DefaultAesthetics {
        DefaultAesthetics {
            defaults: &[
                ("geometry", DefaultAestheticValue::Required),
                ("fill", DefaultAestheticValue::String("#747474")),
                ("stroke", DefaultAestheticValue::String("black")),
                ("opacity", DefaultAestheticValue::Number(0.8)),
                ("linewidth", DefaultAestheticValue::Number(0.2)),
                ("linetype", DefaultAestheticValue::String("solid")),
            ],
        }
    }

    fn apply_stat_transform(
        &self,
        query: &str,
        _schema: &crate::plot::Schema,
        _aesthetics: &Mappings,
        _group_by: &[String],
        _parameters: &std::collections::HashMap<String, crate::plot::ParameterValue>,
        execute_query: &dyn Fn(&str) -> crate::Result<crate::DataFrame>,
        dialect: &dyn crate::reader::SqlDialect,
        _aesthetic_ctx: &crate::plot::aesthetic::AestheticContext,
    ) -> crate::Result<StatResult> {
        for stmt in dialect.sql_spatial_setup() {
            execute_query(&stmt)?;
        }

        Ok(StatResult::Transformed {
            query: query.to_string(),
            stat_columns: vec![],
            dummy_columns: vec![],
            consumed_aesthetics: vec![],
        })
    }

    fn apply_projection(
        &self,
        query: &str,
        projection: &Projection,
        dialect: &dyn SqlDialect,
        mappings: &mut Mappings,
        _partition_by: &mut Vec<String>,
        _parameters: &mut std::collections::HashMap<String, crate::plot::types::ParameterValue>,
    ) -> crate::Result<String> {
        let columns = mappings.column_names();
        let col = naming::quote_ident(&naming::aesthetic_column("geometry"));
        let is_map = projection.coord.coord_kind() == CoordKind::Map;
        let clip = matches!(
            projection.properties.get("clip"),
            Some(ParameterValue::Boolean(true))
        );

        // WORKAROUND(duckdb-rs#714): normalize column to GEOMETRY since it may
        // be WKB BLOB from the Arrow export workaround.
        let ensure_geom = dialect.sql_ensure_geometry(&col);
        let geom_query = dialect.sql_select_replace(&ensure_geom, &col, query, &columns);

        let geom_expr = if let (true, Some(ParameterValue::String(crs))) =
            (is_map, projection.properties.get("target"))
        {
            let source = match projection.properties.get("source") {
                Some(ParameterValue::String(s)) => s.as_str(),
                _ => "EPSG:4326",
            };

            if clip {
                return Ok(apply_clip_boundary(
                    &geom_query,
                    &col,
                    source,
                    crs,
                    dialect,
                    &columns,
                ));
            }

            dialect.sql_st_transform(&col, source, crs)
        } else if is_map {
            // Map coord without CRS — keep native geometry (WKB added later by framing)
            return Ok(geom_query);
        } else {
            // Non-map coord — convert to WKB directly
            let wkb_expr = dialect.sql_geometry_to_wkb(&col);
            return Ok(dialect.sql_select_replace(&wkb_expr, &col, &geom_query, &columns));
        };

        // Map coord with CRS — output native projected geometry (WKB added by framing)
        Ok(dialect.sql_select_replace(&geom_expr, &col, &geom_query, &columns))
    }
}

impl std::fmt::Display for Spatial {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "spatial")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::AnsiDialect;

    #[test]
    fn test_apply_projection_without_map_coord() {
        let spatial = Spatial;
        let projection = Projection::cartesian();
        let result = spatial
            .apply_projection(
                "SELECT * FROM t",
                &projection,
                &AnsiDialect,
                &mut Mappings::new(),
                &mut vec![],
                &mut std::collections::HashMap::new(),
            )
            .unwrap();

        assert!(result.contains("ST_AsBinary"));
        assert!(!result.contains("ST_Transform"));
    }

    #[test]
    fn test_apply_projection_map_without_crs() {
        let spatial = Spatial;
        let projection = Projection::map();
        let result = spatial
            .apply_projection(
                "SELECT * FROM t",
                &projection,
                &AnsiDialect,
                &mut Mappings::new(),
                &mut vec![],
                &mut std::collections::HashMap::new(),
            )
            .unwrap();

        // Map without CRS passes through (ensure_geometry is identity for AnsiDialect)
        assert!(result.contains("SELECT * FROM"));
        assert!(!result.contains("ST_Transform"));
    }

    #[test]
    fn test_apply_projection_map_with_crs_no_clip() {
        let spatial = Spatial;
        let mut projection = Projection::map();
        projection.properties.insert(
            "target".to_string(),
            ParameterValue::String("+proj=merc".to_string()),
        );
        let result = spatial
            .apply_projection(
                "SELECT * FROM t",
                &projection,
                &AnsiDialect,
                &mut Mappings::new(),
                &mut vec![],
                &mut std::collections::HashMap::new(),
            )
            .unwrap();

        // Without clip=true, just ST_Transform
        assert!(!result.contains("ST_AsBinary"));
        assert!(result.contains("ST_Transform"));
        assert!(result.contains("+proj=merc"));
        assert!(!result.contains("ST_Intersection"));
    }

    #[test]
    fn test_apply_projection_mercator_with_clip() {
        let spatial = Spatial;
        let mut projection = Projection::map();
        projection.properties.insert(
            "target".to_string(),
            ParameterValue::String("+proj=merc".to_string()),
        );
        projection
            .properties
            .insert("clip".to_string(), ParameterValue::Boolean(true));
        let result = spatial
            .apply_projection(
                "SELECT * FROM t",
                &projection,
                &AnsiDialect,
                &mut Mappings::new(),
                &mut vec![],
                &mut std::collections::HashMap::new(),
            )
            .unwrap();

        assert!(result.contains("ST_Intersection"));
        assert!(result.contains("ST_Intersects"));
        assert!(result.contains("__ggsql_clip_boundary_"));
    }

    #[test]
    fn test_orthographic_with_clip() {
        let spatial = Spatial;
        let mut projection = Projection::map();
        projection.properties.insert(
            "target".to_string(),
            ParameterValue::String("+proj=ortho +lat_0=45 +lon_0=10".to_string()),
        );
        projection
            .properties
            .insert("clip".to_string(), ParameterValue::Boolean(true));
        let result = spatial
            .apply_projection(
                "SELECT * FROM t",
                &projection,
                &AnsiDialect,
                &mut Mappings::new(),
                &mut vec![],
                &mut std::collections::HashMap::new(),
            )
            .unwrap();

        assert!(result.contains("ST_Transform"));
        assert!(result.contains("ST_MakeValid"));
        assert!(result.contains("ST_Intersection"));
        assert!(result.contains("ST_Intersects"));
        assert!(result.contains("__ggsql_clip_boundary_"));
    }

    #[test]
    fn test_gnomonic_with_clip() {
        let spatial = Spatial;
        let mut projection = Projection::map();
        projection.properties.insert(
            "target".to_string(),
            ParameterValue::String("+proj=gnom +lat_0=90 +lon_0=0".to_string()),
        );
        projection
            .properties
            .insert("clip".to_string(), ParameterValue::Boolean(true));
        let result = spatial
            .apply_projection(
                "SELECT * FROM t",
                &projection,
                &AnsiDialect,
                &mut Mappings::new(),
                &mut vec![],
                &mut std::collections::HashMap::new(),
            )
            .unwrap();

        assert!(result.contains("ST_MakeValid"));
        assert!(result.contains("ST_Intersection"));
        assert!(result.contains("ST_Intersects"));
        assert!(result.contains("__ggsql_clip_boundary_"));
    }
}
