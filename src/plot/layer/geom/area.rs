//! Area geom implementation

use crate::plot::layer::orientation::{ALIGNED, ORIENTATION_VALUES};
use crate::plot::types::DefaultAestheticValue;
use crate::plot::{DefaultParamValue, ParamDefinition};
use crate::{naming, Mappings};

use super::types::{ParamConstraint, POSITION_VALUES};
use super::{DefaultAesthetics, GeomTrait, GeomType, StatResult};

/// Area geom - filled area charts
#[derive(Debug, Clone, Copy)]
pub struct Area;

impl GeomTrait for Area {
    fn geom_type(&self) -> GeomType {
        GeomType::Area
    }

    fn aesthetics(&self) -> DefaultAesthetics {
        DefaultAesthetics {
            defaults: &[
                ("pos1", DefaultAestheticValue::Required),
                ("pos2", DefaultAestheticValue::Required),
                ("fill", DefaultAestheticValue::String("black")),
                ("stroke", DefaultAestheticValue::String("black")),
                ("opacity", DefaultAestheticValue::Number(0.8)),
                ("linewidth", DefaultAestheticValue::Number(1.0)),
                ("linetype", DefaultAestheticValue::String("solid")),
                ("pos2end", DefaultAestheticValue::Delayed),
            ],
        }
    }

    fn default_remappings(&self) -> DefaultAesthetics {
        DefaultAesthetics {
            defaults: &[("pos2end", DefaultAestheticValue::Number(0.0))],
        }
    }

    fn default_params(&self) -> &'static [ParamDefinition] {
        const PARAMS: &[ParamDefinition] = &[
            ParamDefinition {
                name: "position",
                default: DefaultParamValue::String("stack"),
                constraint: ParamConstraint::string_option(POSITION_VALUES),
            },
            ParamDefinition {
                name: "orientation",
                default: DefaultParamValue::String(ALIGNED),
                constraint: ParamConstraint::string_option(ORIENTATION_VALUES),
            },
        ];
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
        // Area geom needs ordering by pos1 (domain axis) for proper rendering
        let order_col = naming::aesthetic_column("pos1");
        Ok(StatResult::Transformed {
            query: format!("{} ORDER BY {}", query, naming::quote_ident(&order_col)),
            stat_columns: vec![],
            dummy_columns: vec![],
            consumed_aesthetics: vec![],
        })
    }
}

impl std::fmt::Display for Area {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "area")
    }
}
