//! Text geom implementation

use super::types::POSITION_VALUES;
use super::{
    DefaultAesthetics, DefaultParamValue, GeomTrait, GeomType, ParamConstraint, ParamDefinition,
    ParameterValue,
};
use crate::plot::types::DefaultAestheticValue;
use crate::plot::{ArrayConstraint, NumberConstraint};
use crate::{naming, DataFrame, Result};
use std::collections::HashMap;

/// Text geom - text labels at positions
#[derive(Debug, Clone, Copy)]
pub struct Text;

impl GeomTrait for Text {
    fn geom_type(&self) -> GeomType {
        GeomType::Text
    }

    fn aesthetics(&self) -> DefaultAesthetics {
        DefaultAesthetics {
            defaults: &[
                ("pos1", DefaultAestheticValue::Required),
                ("pos2", DefaultAestheticValue::Required),
                ("label", DefaultAestheticValue::Required),
                ("stroke", DefaultAestheticValue::Null),
                ("fill", DefaultAestheticValue::String("black")),
                ("opacity", DefaultAestheticValue::Number(1.0)),
                ("typeface", DefaultAestheticValue::Null),
                ("fontsize", DefaultAestheticValue::Number(11.0)),
                ("fontweight", DefaultAestheticValue::String("normal")), // Accepts: CSS keywords or numeric values
                ("italic", DefaultAestheticValue::Boolean(false)),
                ("hjust", DefaultAestheticValue::Number(0.5)),
                ("vjust", DefaultAestheticValue::Number(0.5)),
                ("rotation", DefaultAestheticValue::Number(0.0)),
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
            ParamDefinition {
                name: "offset",
                default: DefaultParamValue::Null,
                constraint: ParamConstraint::number_or_numeric_array(
                    NumberConstraint::unconstrained(),
                    ArrayConstraint::of_numbers_len(NumberConstraint::unconstrained(), 2),
                ),
            },
            ParamDefinition {
                name: "format",
                default: DefaultParamValue::Null,
                constraint: ParamConstraint::string(),
            },
        ];
        PARAMS
    }

    fn supports_aggregate(&self) -> bool {
        true
    }

    fn post_process(
        &self,
        df: DataFrame,
        parameters: &HashMap<String, ParameterValue>,
    ) -> Result<DataFrame> {
        // Check if format parameter is specified
        let format_template = match parameters.get("format") {
            Some(ParameterValue::String(template)) => template,
            _ => return Ok(df), // No formatting, return original
        };

        // Use format.rs helper to do the formatting
        let label_col_name = naming::aesthetic_column("label");
        crate::format::format_dataframe_column(&df, &label_col_name, format_template)
            .map_err(crate::GgsqlError::ValidationError)
    }
}

impl std::fmt::Display for Text {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "text")
    }
}
