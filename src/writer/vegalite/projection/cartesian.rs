use super::ProjectionRenderer;

/// Cartesian projection — standard x/y coordinates.
pub(in crate::writer) struct CartesianProjection {
    pub is_faceted: bool,
}

impl ProjectionRenderer for CartesianProjection {
    fn is_faceted(&self) -> bool {
        self.is_faceted
    }

    fn position_channels(&self) -> (&'static str, &'static str) {
        ("x", "y")
    }

    fn offset_channels(&self) -> (&'static str, &'static str) {
        ("xOffset", "yOffset")
    }
}
