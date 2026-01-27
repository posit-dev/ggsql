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

pub use colour::{color_to_hex, gradient, interpolate_colors, is_color_aesthetic, ColorSpace};
pub use linetype::linetype_to_stroke_dash;
pub use scale_type::{
    Binned, Continuous, Discrete, Identity, ScaleType, ScaleTypeKind, ScaleTypeTrait,
    OOB_CENSOR, OOB_KEEP, OOB_SQUISH,
};
pub use shape::shape_to_svg_path;
pub use transform::{Transform, TransformKind, TransformTrait, ALL_TRANSFORM_NAMES};
pub use types::{Guide, GuideType, OutputRange, Scale};
