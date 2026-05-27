use super::ProjectionRenderer;

/// Cartesian projection — standard x/y coordinates.
pub(in crate::writer) struct CartesianProjection {
    is_faceted: bool,
}

impl CartesianProjection {
    pub(super) fn new(facet: Option<&crate::plot::Facet>) -> Self {
        Self {
            is_faceted: facet.is_some_and(|f| !f.get_variables().is_empty()),
        }
    }
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
