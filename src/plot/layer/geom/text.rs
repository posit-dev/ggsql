//! Text geom implementation

use super::types::POSITION_VALUES;
use super::{
    project_position_columns, DefaultAesthetics, DefaultParamValue, GeomTrait, GeomType,
    ParamConstraint, ParamDefinition,
};
use crate::plot::projection::Projection;
use crate::plot::types::{DefaultAestheticValue, ParameterValue, Parameters};
use crate::plot::{ArrayConstraint, NumberConstraint};
use crate::reader::SqlDialect;
use crate::{naming, DataFrame, Mappings, Result};

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
            ParamDefinition {
                name: "parse",
                default: DefaultParamValue::Boolean(true),
                constraint: ParamConstraint::boolean(),
            },
            super::types::AGGREGATE_PARAM,
        ];
        PARAMS
    }

    fn aggregate_domain_aesthetics(&self) -> Option<&'static [&'static str]> {
        Some(&[])
    }

    fn apply_projection(
        &self,
        query: &str,
        projection: &Projection,
        dialect: &dyn SqlDialect,
        mappings: &mut Mappings,
        partition_by: &mut Vec<String>,
        _parameters: &mut std::collections::HashMap<String, crate::plot::types::ParameterValue>,
    ) -> Result<String> {
        let columns = crate::util::set_union(mappings.column_names(), partition_by);
        project_position_columns(query, projection, dialect, &columns)
    }

    fn post_process(&self, df: DataFrame, parameters: &Parameters) -> Result<DataFrame> {
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

#[cfg(test)]
mod tests {
    use crate::plot::types::ParameterValue;
    use crate::plot::{Geom, Layer};

    /// `parse` is on unless the user says otherwise, so a label carrying markdown
    /// renders as rich text without asking.
    #[test]
    fn test_parse_defaults_to_true() {
        let mut layer = Layer::new(Geom::text());
        layer.apply_default_params();
        assert_eq!(
            layer.parameters.get("parse"),
            Some(&ParameterValue::Boolean(true))
        );
    }

    /// An explicit `SETTING parse => false` survives default application.
    #[test]
    fn test_parse_setting_is_kept() {
        let mut layer = Layer::new(Geom::text());
        layer
            .parameters
            .insert("parse".to_string(), ParameterValue::Boolean(false));
        layer.apply_default_params();
        assert_eq!(
            layer.parameters.get("parse"),
            Some(&ParameterValue::Boolean(false))
        );
    }

    /// `parse` is a boolean; anything else is a validation error rather than a
    /// value coerced into one.
    #[test]
    fn test_parse_rejects_non_boolean() {
        let mut layer = Layer::new(Geom::text());
        layer.parameters.insert(
            "parse".to_string(),
            ParameterValue::String("yes".to_string()),
        );
        assert!(layer.validate_settings().is_err());
    }
}
