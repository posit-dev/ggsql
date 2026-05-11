use super::{DefaultAesthetics, GeomTrait, GeomType};
use crate::naming;
use crate::plot::projection::coord::map::CLIP_BOUNDARY_KEY;
use crate::plot::projection::coord::CoordKind;
use crate::plot::projection::Projection;
use crate::plot::types::DefaultAestheticValue;
use crate::plot::ParameterValue;
use crate::reader::SqlDialect;

fn apply_clip_boundary(
    query: &str,
    col: &str,
    source: &str,
    crs: &str,
    clip_table: &str,
    dialect: &dyn SqlDialect,
) -> String {
    let clip_geom = format!("(SELECT geom FROM {clip_table})");
    let geom_expr = format!(
        "ST_MakeValid(ST_Transform(\
            ST_Intersection({col}, {clip_geom}),\
            '{source}', '{crs}', always_xy := true\
        ))",
        col = col,
        clip_geom = clip_geom,
        source = source.replace('\'', "''"),
        crs = crs.replace('\'', "''"),
    );
    let wkb_expr = dialect.sql_geometry_to_wkb(&geom_expr);
    format!(
        "SELECT * REPLACE ({wkb_expr} AS {col}) FROM ({query}) \
         WHERE ST_Intersects({col}, {clip_geom})",
        col = col,
        wkb_expr = wkb_expr,
        query = query,
        clip_geom = clip_geom,
    )
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

    fn apply_projection(
        &self,
        query: &str,
        projection: &Projection,
        dialect: &dyn SqlDialect,
    ) -> crate::Result<String> {
        let col = naming::quote_ident(&naming::aesthetic_column("geometry"));

        let geom_expr = if let (CoordKind::Map, Some(ParameterValue::String(crs))) = (
            projection.coord.coord_kind(),
            projection.properties.get("crs"),
        ) {
            let source = match projection.properties.get("source") {
                Some(ParameterValue::String(s)) => s.as_str(),
                _ => "EPSG:4326",
            };

            if let Some(ParameterValue::String(clip_table)) =
                projection.properties.get(CLIP_BOUNDARY_KEY)
            {
                return Ok(apply_clip_boundary(
                    query, &col, source, crs, clip_table, dialect,
                ));
            }

            dialect.sql_st_transform(&col, source, crs)
        } else {
            col.clone()
        };

        let wkb_expr = dialect.sql_geometry_to_wkb(&geom_expr);
        Ok(format!(
            "SELECT * REPLACE ({wkb_expr} AS {col}) FROM ({query})"
        ))
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
            .apply_projection("SELECT * FROM t", &projection, &AnsiDialect)
            .unwrap();

        assert!(result.contains("ST_AsBinary"));
        assert!(!result.contains("ST_Transform"));
    }

    #[test]
    fn test_apply_projection_map_without_crs() {
        let spatial = Spatial;
        let projection = Projection::map();
        let result = spatial
            .apply_projection("SELECT * FROM t", &projection, &AnsiDialect)
            .unwrap();

        assert!(result.contains("ST_AsBinary"));
        assert!(!result.contains("ST_Transform"));
    }

    #[test]
    fn test_apply_projection_map_with_crs() {
        let spatial = Spatial;
        let mut projection = Projection::map();
        projection.properties.insert(
            "crs".to_string(),
            ParameterValue::String("+proj=merc".to_string()),
        );
        let result = spatial
            .apply_projection("SELECT * FROM t", &projection, &AnsiDialect)
            .unwrap();

        assert!(result.contains("ST_AsBinary"));
        assert!(result.contains("ST_Transform"));
        assert!(result.contains("+proj=merc"));
        assert!(!result.contains("ST_Intersection"), "mercator should not clip");
    }

    #[test]
    fn test_orthographic_gets_clip_boundary() {
        let spatial = Spatial;
        let mut projection = Projection::map();
        projection.properties.insert(
            "crs".to_string(),
            ParameterValue::String("+proj=ortho +lat_0=45 +lon_0=10".to_string()),
        );
        projection.properties.insert(
            CLIP_BOUNDARY_KEY.to_string(),
            ParameterValue::String("__ggsql_clip_boundary__".to_string()),
        );
        let result = spatial
            .apply_projection("SELECT * FROM t", &projection, &AnsiDialect)
            .unwrap();

        assert!(result.contains("ST_Transform"));
        assert!(result.contains("ST_MakeValid"));
        assert!(result.contains("ST_Intersection"));
        assert!(result.contains("ST_Intersects"));
        assert!(result.contains("__ggsql_clip_boundary__"));
    }

    #[test]
    fn test_gnomonic_gets_clip_boundary() {
        let spatial = Spatial;
        let mut projection = Projection::map();
        projection.properties.insert(
            "crs".to_string(),
            ParameterValue::String("+proj=gnom +lat_0=90 +lon_0=0".to_string()),
        );
        projection.properties.insert(
            CLIP_BOUNDARY_KEY.to_string(),
            ParameterValue::String("__ggsql_clip_boundary__".to_string()),
        );
        let result = spatial
            .apply_projection("SELECT * FROM t", &projection, &AnsiDialect)
            .unwrap();

        assert!(result.contains("ST_MakeValid"));
        assert!(result.contains("ST_Intersection"));
        assert!(result.contains("ST_Intersects"));
        assert!(result.contains("__ggsql_clip_boundary__"));
    }
}
