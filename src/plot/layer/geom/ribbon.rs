//! Ribbon geom implementation

use super::stat_aggregate;
use super::types::{wrap_with_order_by, POSITION_VALUES};
use super::{has_aggregate_param, DefaultAesthetics, GeomTrait, GeomType, StatResult};
use crate::plot::layer::orientation::ORIENTATION_VALUES;
use crate::plot::types::DefaultAestheticValue;
use crate::plot::{DefaultParamValue, ParamConstraint, ParamDefinition};
use crate::Mappings;

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
                // pos2 is the input column for the Aggregate stat in range mode
                // (`SETTING aggregate => (lower_func, upper_func)` consumes pos2
                // and produces pos2min/pos2max). Optional otherwise.
                ("pos2", DefaultAestheticValue::Null),
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
            // Default Null → resolve_orientation auto-detects from mappings/scales.
            // User can override with `SETTING orientation => 'transposed'`.
            ParamDefinition {
                name: "orientation",
                default: DefaultParamValue::Null,
                constraint: ParamConstraint::string_option(ORIENTATION_VALUES),
            },
        ];
        PARAMS
    }

    fn supports_aggregate(&self) -> bool {
        true
    }

    fn aggregate_range_pair(&self) -> Option<(&'static str, &'static str)> {
        Some(("pos2min", "pos2max"))
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
    ) -> crate::Result<StatResult> {
        let result = if has_aggregate_param(parameters) {
            stat_aggregate::apply(
                query,
                schema,
                aesthetics,
                group_by,
                parameters,
                dialect,
                self.aggregate_slots(),
                self.aggregate_range_pair(),
            )?
        } else {
            StatResult::Identity
        };
        // Ribbon needs ordering by pos1 (domain axis) for proper rendering, in both
        // the Identity and Aggregate paths.
        Ok(wrap_with_order_by(query, result, "pos1"))
    }
}

impl std::fmt::Display for Ribbon {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ribbon")
    }
}
