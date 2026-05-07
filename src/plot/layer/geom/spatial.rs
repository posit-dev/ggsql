use super::{DefaultAesthetics, GeomTrait, GeomType};
use crate::naming;
use crate::plot::projection::coord::CoordKind;
use crate::plot::projection::Projection;
use crate::plot::types::DefaultAestheticValue;
use crate::plot::ParameterValue;
use crate::reader::SqlDialect;

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
    // Note: in AnsiDialect ST_AsBinary is the function to get WKB.

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
    }
}
