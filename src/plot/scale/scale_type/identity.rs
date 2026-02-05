//! Identity scale type implementation

use std::collections::HashMap;

use polars::prelude::{Column, DataType};

use super::{CastTargetType, ScaleTypeKind, ScaleTypeTrait};
use crate::plot::{ArrayElement, ParameterValue};

/// Identity scale type - delegates to inferred type
#[derive(Debug, Clone, Copy)]
pub struct Identity;

impl ScaleTypeTrait for Identity {
    fn scale_type_kind(&self) -> ScaleTypeKind {
        ScaleTypeKind::Identity
    }

    fn name(&self) -> &'static str {
        "identity"
    }

    fn uses_discrete_input_range(&self) -> bool {
        true
    }

    fn allows_data_type(&self, _dtype: &DataType) -> bool {
        true // Identity accepts any type
    }

    fn resolve_input_range(
        &self,
        user_range: Option<&[ArrayElement]>,
        _columns: &[&Column],
        _properties: &HashMap<String, ParameterValue>,
    ) -> Result<Option<Vec<ArrayElement>>, String> {
        // Identity scales don't support input range or expansion
        match user_range {
            Some(_) => Err("Identity scale does not support input range specification".to_string()),
            None => Ok(None),
        }
    }

    fn default_output_range(
        &self,
        _aesthetic: &str,
        _scale: &super::super::Scale,
    ) -> Result<Option<Vec<ArrayElement>>, String> {
        Ok(None) // Identity scales use inferred defaults
    }

    /// Identity scales never require casting - they accept data as-is.
    fn required_cast_type(
        &self,
        _column_dtype: &DataType,
        _target_dtype: &DataType,
    ) -> Option<CastTargetType> {
        None
    }
}

impl std::fmt::Display for Identity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}
