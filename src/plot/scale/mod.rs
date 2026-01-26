//! Scale and guide types for ggsql visualization specifications
//!
//! This module defines scale and guide configuration for aesthetic mappings.

pub mod colour;
pub mod linetype;
pub mod palettes;
mod scale_type;
pub mod shape;
mod types;

pub use colour::{color_to_hex, gradient, interpolate_colors, is_color_aesthetic, ColorSpace};
pub use linetype::linetype_to_stroke_dash;
pub use scale_type::{
    Binned, Continuous, Date, DateTime, Discrete, Identity, ScaleType, ScaleTypeKind,
    ScaleTypeTrait, Time,
};
pub use shape::shape_to_svg_path;
pub use types::{Guide, GuideType, OutputRange, Scale};
