//! Line geom implementation

use super::{
    DefaultAesthetics, DefaultParamValue, GeomTrait, GeomType, ParamConstraint, ParamDefinition,
    StatResult,
};
use crate::plot::layer::orientation::{ALIGNED, ORIENTATION_VALUES};
use crate::plot::types::DefaultAestheticValue;
use crate::{naming, Mappings};

/// Line geom - line charts with connected points
#[derive(Debug, Clone, Copy)]
pub struct Line;

impl GeomTrait for Line {
    fn geom_type(&self) -> GeomType {
        GeomType::Line
    }

    fn aesthetics(&self) -> DefaultAesthetics {
        DefaultAesthetics {
            defaults: &[
                ("pos1", DefaultAestheticValue::Required),
                ("pos2", DefaultAestheticValue::Required),
                ("stroke", DefaultAestheticValue::String("black")),
                ("linewidth", DefaultAestheticValue::Number(1.5)),
                ("opacity", DefaultAestheticValue::Number(1.0)),
                ("linetype", DefaultAestheticValue::String("solid")),
            ],
        }
    }

    fn default_params(&self) -> &'static [ParamDefinition] {
        const PARAMS: &[ParamDefinition] = &[ParamDefinition {
            name: "orientation",
            default: DefaultParamValue::String(ALIGNED),
            constraint: ParamConstraint::string_option(ORIENTATION_VALUES),
        }];
        PARAMS
    }

    fn needs_stat_transform(&self, _aesthetics: &Mappings) -> bool {
        true
    }

    fn apply_stat_transform(
        &self,
        query: &str,
        _schema: &crate::plot::Schema,
        _aesthetics: &Mappings,
        _group_by: &[String],
        _parameters: &std::collections::HashMap<String, crate::plot::ParameterValue>,
        _execute_query: &dyn Fn(&str) -> crate::Result<polars::prelude::DataFrame>,
        _dialect: &dyn crate::reader::SqlDialect,
    ) -> crate::Result<StatResult> {
        // Line geom needs ordering by pos1 (domain axis) for proper rendering
        let order_col = naming::aesthetic_column("pos1");
        Ok(StatResult::Transformed {
            query: format!("{} ORDER BY {}", query, naming::quote_ident(&order_col)),
            stat_columns: vec![],
            dummy_columns: vec![],
            consumed_aesthetics: vec![],
        })
    }
}

impl std::fmt::Display for Line {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "line")
    }
}
