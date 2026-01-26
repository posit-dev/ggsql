//! Identity scale type implementation

use polars::prelude::{Column, DataType};

use super::{ScaleTypeKind, ScaleTypeTrait};
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

    fn is_discrete(&self) -> bool {
        false
    }

    fn allows_data_type(&self, _dtype: &DataType) -> bool {
        true // Identity accepts any type
    }

    fn resolve_input_range(
        &self,
        user_range: Option<&[ArrayElement]>,
        _columns: &[&Column],
    ) -> Result<Option<Vec<ArrayElement>>, String> {
        match user_range {
            Some(_) => Err("Identity scale does not support input range specification".to_string()),
            None => Ok(None),
        }
    }

    fn default_output_range(
        &self,
        _aesthetic: &str,
        _input_range: Option<&[ArrayElement]>,
    ) -> Result<Option<Vec<ArrayElement>>, String> {
        Ok(None) // Identity scales use inferred defaults
    }
}

impl std::fmt::Display for Identity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}
