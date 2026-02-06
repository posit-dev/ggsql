//! Identity scale type implementation

use polars::prelude::DataType;

use super::{CastTargetType, ScaleTypeKind, ScaleTypeTrait};
use crate::plot::ArrayElement;

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
