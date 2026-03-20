//! Rule geom implementation

use super::{DefaultAesthetics, GeomTrait, GeomType};
use crate::plot::{
    orientation::{ALIGNED, ORIENTATION_VALUES},
    types::DefaultAestheticValue,
    DefaultParamValue, ParamConstraint, ParamDefinition,
};

/// Rule geom - horizontal and vertical reference lines
#[derive(Debug, Clone, Copy)]
pub struct Rule;

impl GeomTrait for Rule {
    fn geom_type(&self) -> GeomType {
        GeomType::Rule
    }

    fn aesthetics(&self) -> DefaultAesthetics {
        DefaultAesthetics {
            defaults: &[
                ("pos1", DefaultAestheticValue::Null),
                ("slope", DefaultAestheticValue::Null),
                ("intercept", DefaultAestheticValue::Null),
                ("stroke", DefaultAestheticValue::String("black")),
                ("linewidth", DefaultAestheticValue::Number(1.0)),
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

    fn post_process(
        &self,
        df: crate::DataFrame,
        parameters: &std::collections::HashMap<String, crate::plot::ParameterValue>,
    ) -> crate::Result<crate::DataFrame> {
        use crate::{naming, GgsqlError};
        use polars::prelude::{IntoColumn, NamedFrom, Series};

        let mut result = df;
        let row_count = result.height();

        // For diagonal rules (slope + intercept), add these as DataFrame columns
        // The Vega-Lite writer needs them as columns to create transform calculations
        for aesthetic in &["slope", "intercept"] {
            if let Some(value) = parameters.get(*aesthetic) {
                // Only accept numeric values for slope and intercept
                let n = match value {
                    crate::plot::ParameterValue::Number(n) => *n,
                    _ => {
                        return Err(GgsqlError::ValidationError(format!(
                            "Rule '{}' aesthetic must be a number, not {:?}.",
                            aesthetic, value
                        )))
                    }
                };

                // Create a column with the aesthetic's prefixed name
                let col_name = naming::aesthetic_column(aesthetic);
                let series = Series::new(col_name.clone().into(), vec![n; row_count]);

                // Add the column to the DataFrame
                result = result
                    .with_column(series.into_column())
                    .map_err(|e| {
                        GgsqlError::InternalError(format!(
                            "Failed to add {} column: {}",
                            aesthetic, e
                        ))
                    })?
                    .clone();
            }
        }

        Ok(result)
    }
}

impl std::fmt::Display for Rule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "rule")
    }
}
