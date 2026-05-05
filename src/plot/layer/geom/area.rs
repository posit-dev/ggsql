//! Area geom implementation

use crate::plot::layer::orientation::{ALIGNED, ORIENTATION_VALUES};
use crate::plot::types::DefaultAestheticValue;
use crate::plot::{DefaultParamValue, ParamDefinition};
use crate::Mappings;

use super::stat_aggregate;
use super::types::{wrap_with_order_by, ParamConstraint, POSITION_VALUES};
use super::{has_aggregate_param, DefaultAesthetics, GeomTrait, GeomType, StatResult};

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

    fn supports_aggregate(&self) -> bool {
        true
    }

    fn aggregate_domain_aesthetics(&self) -> &'static [&'static str] {
        &["pos1"]
    }

    fn needs_stat_transform(&self, _aesthetics: &Mappings) -> bool {
        true
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
                self.aggregate_domain_aesthetics(),
            )?
        } else {
            StatResult::Identity
        };
        // Area needs ordering by pos1 (domain axis) for proper rendering, in both
        // the Identity and Aggregate paths.
        Ok(wrap_with_order_by(query, result, "pos1"))
    }
}

impl std::fmt::Display for Area {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "area")
    }
}
