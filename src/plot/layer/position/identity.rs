//! Identity position adjustment
//!
//! No position adjustment - elements are positioned at their exact data values.

use super::{Layer, PositionTrait, PositionType};
use crate::{DataFrame, Plot, Result};

/// Identity position - no adjustment applied
#[derive(Debug, Clone, Copy)]
pub struct Identity;

impl std::fmt::Display for Identity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "identity")
    }
}

impl PositionTrait for Identity {
    fn position_type(&self) -> PositionType {
        PositionType::Identity
    }

    fn apply_adjustment(
        &self,
        df: DataFrame,
        _layer: &Layer,
        _spec: &Plot,
    ) -> Result<(DataFrame, Option<f64>)> {
        // Identity returns data unchanged
        Ok((df, None))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::df;

    #[test]
    fn test_identity_no_change() {
        let identity = Identity;
        assert_eq!(identity.position_type(), PositionType::Identity);

        let df = df! {
            "x" => vec![1i32, 2, 3],
            "y" => vec![10i32, 20, 30],
        }
        .unwrap();

        let layer = Layer::new(crate::plot::layer::Geom::point());
        let spec = Plot::new();

        let (result, width) = identity.apply_adjustment(df, &layer, &spec).unwrap();

        assert_eq!(result.height(), 3);
        assert!(width.is_none());
    }
}
