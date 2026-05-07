//! Map coordinate system implementation

use super::{CoordKind, CoordTrait};
use crate::naming;
use crate::plot::layer::geom::GeomType;
use crate::plot::types::{DefaultParamValue, ParamConstraint, ParamDefinition};
use crate::plot::{Layer, ParameterValue};
use crate::reader::SqlDialect;
use crate::DataFrame;

/// Map coordinate system - for geographic/cartographic projections
#[derive(Debug, Clone, Copy)]
pub struct Map;

impl CoordTrait for Map {
    fn coord_kind(&self) -> CoordKind {
        CoordKind::Map
    }

    fn name(&self) -> &'static str {
        "map"
    }

    fn position_aesthetic_names(&self) -> &'static [&'static str] {
        &["lon", "lat"]
    }

    fn default_properties(&self) -> &'static [ParamDefinition] {
        const PARAMS: &[ParamDefinition] = &[
            ParamDefinition {
                name: "crs",
                default: DefaultParamValue::Null,
                constraint: ParamConstraint::string(),
            },
            ParamDefinition {
                name: "source",
                default: DefaultParamValue::Null,
                constraint: ParamConstraint::string(),
            },
            ParamDefinition {
                name: "clip",
                default: DefaultParamValue::Boolean(true),
                constraint: ParamConstraint::boolean(),
            },
        ];
        PARAMS
    }

    fn apply_projection_transforms(
        &self,
        layers: &[Layer],
        layer_queries: &mut [String],
        projection: &mut super::super::Projection,
        dialect: &dyn SqlDialect,
        execute_query: &dyn Fn(&str) -> crate::Result<DataFrame>,
    ) -> crate::Result<()> {
        for stmt in dialect.sql_spatial_setup() {
            execute_query(&stmt)?;
        }

        // Detect source CRS from geometry columns if not explicitly set
        if !projection.properties.contains_key("source") {
            if let Some(srid) = detect_source_srid(layers, layer_queries, execute_query) {
                projection
                    .properties
                    .insert("source".to_string(), ParameterValue::String(srid));
            }
        }

        for (idx, layer) in layers.iter().enumerate() {
            layer_queries[idx] =
                layer.geom.apply_projection(&layer_queries[idx], projection, dialect)?;
        }
        Ok(())
    }
}

impl std::fmt::Display for Map {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

fn detect_source_srid(
    layers: &[Layer],
    layer_queries: &[String],
    execute_query: &dyn Fn(&str) -> crate::Result<DataFrame>,
) -> Option<String> {
    let geom_col = naming::quote_ident(&naming::aesthetic_column("geometry"));

    for (idx, layer) in layers.iter().enumerate() {
        if layer.geom.geom_type() != GeomType::Spatial {
            continue;
        }
        let sql = format!(
            "SELECT ST_SRID({geom_col}) AS srid FROM ({}) WHERE {geom_col} IS NOT NULL LIMIT 1",
            layer_queries[idx]
        );
        if let Ok(df) = execute_query(&sql) {
            let batch = df.inner();
            if batch.num_rows() == 0 {
                continue;
            }
            if let Some(arr) = batch
                .column(0)
                .as_any()
                .downcast_ref::<arrow::array::Int32Array>()
            {
                let srid = arr.value(0);
                if srid != 0 {
                    return Some(format!("EPSG:{srid}"));
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::ParameterValue;
    use std::collections::HashMap;

    #[test]
    fn test_map_properties() {
        let map = Map;
        assert_eq!(map.coord_kind(), CoordKind::Map);
        assert_eq!(map.name(), "map");
        assert_eq!(map.position_aesthetic_names(), &["lon", "lat"]);
    }

    #[test]
    fn test_map_default_properties() {
        let map = Map;
        let defaults = map.default_properties();
        let names: Vec<&str> = defaults.iter().map(|p| p.name).collect();
        assert!(names.contains(&"crs"));
        assert!(names.contains(&"source"));
        assert!(names.contains(&"clip"));
        assert_eq!(defaults.len(), 3);
    }

    #[test]
    fn test_map_accepts_crs_string() {
        let map = Map;
        let mut props = HashMap::new();
        props.insert(
            "crs".to_string(),
            ParameterValue::String("+proj=merc".to_string()),
        );

        let resolved = map.resolve_properties(&props);
        assert!(resolved.is_ok());
        let resolved = resolved.unwrap();
        assert_eq!(
            resolved.get("crs").unwrap(),
            &ParameterValue::String("+proj=merc".to_string())
        );
    }

    #[test]
    fn test_map_rejects_unknown_property() {
        let map = Map;
        let mut props = HashMap::new();
        props.insert(
            "unknown".to_string(),
            ParameterValue::String("value".to_string()),
        );

        let resolved = map.resolve_properties(&props);
        assert!(resolved.is_err());
        let err = resolved.unwrap_err();
        assert!(err.contains("not 'unknown'"));
    }
}
