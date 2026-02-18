//! Scale and guide types for ggsql visualization specifications
//!
//! This module defines scale and guide configuration for aesthetic mappings.

pub mod breaks;
pub mod colour;
pub mod linetype;
pub mod palettes;
mod scale_type;
pub mod shape;
pub mod transform;
mod types;

pub use crate::format::apply_label_template;
pub use crate::plot::types::{CastTargetType, SqlTypeNames};
pub use colour::{color_to_hex, gradient, interpolate_colors, is_color_aesthetic, ColorSpace};
pub use linetype::linetype_to_stroke_dash;
pub use scale_type::{
    coerce_dtypes, default_oob, dtype_to_cast_target, infer_transform_from_input_range, needs_cast,
    Binned, Continuous, Discrete, Identity, InputRange, ScaleDataContext, ScaleType, ScaleTypeKind,
    ScaleTypeTrait, TypeFamily, OOB_CENSOR, OOB_KEEP, OOB_SQUISH,
};

pub use shape::shape_to_svg_path;
pub use transform::{Transform, TransformKind, TransformTrait, ALL_TRANSFORM_NAMES};
pub use types::{OutputRange, Scale};

use crate::plot::{ArrayElement, ArrayElementType};

// =============================================================================
// Pure Logic Functions for Scale Handling
// =============================================================================

/// Check if an aesthetic gets a default scale (type inferred from data).
///
/// Returns true for aesthetics that benefit from scale resolution
/// (input range, output range, transforms, breaks).
/// Returns false for aesthetics that should use Identity scale.
///
/// This is used during automatic scale creation to determine whether
/// an unmapped aesthetic should get a scale with type inference (Continuous/Discrete)
/// or an Identity scale (pass-through, no transformation).
pub fn gets_default_scale(aesthetic: &str) -> bool {
    matches!(
        aesthetic,
        // Position aesthetics
        "x" | "y" | "xmin" | "xmax" | "ymin" | "ymax" | "xend" | "yend"
        // Color aesthetics (color/colour/col already split to fill/stroke)
        | "fill" | "stroke"
        // Size aesthetics
        | "size" | "linewidth"
        // Other visual aesthetics
        | "opacity" | "shape" | "linetype"
    )
}

/// Infer the target type for coercion based on scale kind.
///
/// Different scale kinds determine type differently:
/// - **Discrete/Ordinal**: Type from input range (e.g., `FROM [true, false]` → Boolean)
/// - **Continuous**: Type from transform (e.g., `VIA date` → Date, `VIA log10` → Number)
/// - **Binned**: No coercion (binning happens in SQL before DataFrame)
/// - **Identity**: No coercion
///
/// This is used to coerce DataFrame columns to the appropriate type before
/// scale resolution (e.g., coercing string "true"/"false" to boolean when
/// the scale has `FROM [true, false]`).
pub fn infer_scale_target_type(scale: &Scale) -> Option<ArrayElementType> {
    let scale_type = scale.scale_type.as_ref()?;

    match scale_type.scale_type_kind() {
        // Discrete/Ordinal: type from input range
        ScaleTypeKind::Discrete | ScaleTypeKind::Ordinal => scale
            .input_range
            .as_ref()
            .and_then(|r| ArrayElement::infer_type(r)),
        // Continuous: type from transform
        ScaleTypeKind::Continuous => scale.transform.as_ref().map(|t| t.target_type()),
        // Binned: no coercion (binning happens in SQL before DataFrame)
        ScaleTypeKind::Binned => None,
        // Identity: no coercion
        ScaleTypeKind::Identity => None,
    }
}
