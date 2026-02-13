//! String transform implementation (for discrete scales)

use super::{TransformKind, TransformTrait};
use crate::plot::ArrayElement;

/// String transform - casts values to string for discrete scales
#[derive(Debug, Clone, Copy)]
pub struct String;

impl TransformTrait for String {
    fn transform_kind(&self) -> TransformKind {
        TransformKind::String
    }

    fn name(&self) -> &'static str {
        "string"
    }

    fn allowed_domain(&self) -> (f64, f64) {
        (f64::NEG_INFINITY, f64::INFINITY)
    }

    fn calculate_breaks(&self, _min: f64, _max: f64, _n: usize, _pretty: bool) -> Vec<f64> {
        // String transform is for discrete scales - no breaks calculation
        Vec::new()
    }

    fn calculate_minor_breaks(
        &self,
        _major_breaks: &[f64],
        _n: usize,
        _range: Option<(f64, f64)>,
    ) -> Vec<f64> {
        // String transform is for discrete scales - no minor breaks
        Vec::new()
    }

    fn transform(&self, value: f64) -> f64 {
        // Pass-through - string transform doesn't apply numeric transformations
        value
    }

    fn inverse(&self, value: f64) -> f64 {
        // Pass-through - string transform doesn't apply numeric transformations
        value
    }

    fn wrap_numeric(&self, value: f64) -> ArrayElement {
        // Convert numeric values to strings
        ArrayElement::String(value.to_string())
    }

    fn parse_value(&self, elem: &ArrayElement) -> ArrayElement {
        match elem {
            ArrayElement::String(_) => elem.clone(),
            ArrayElement::Number(n) => ArrayElement::String(n.to_string()),
            ArrayElement::Boolean(b) => ArrayElement::String(b.to_string()),
            ArrayElement::Date(d) => ArrayElement::String(ArrayElement::date_to_iso(*d)),
            ArrayElement::DateTime(dt) => ArrayElement::String(ArrayElement::datetime_to_iso(*dt)),
            ArrayElement::Time(t) => ArrayElement::String(ArrayElement::time_to_iso(*t)),
            ArrayElement::Null => ArrayElement::Null,
        }
    }
}

impl std::fmt::Display for String {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_transform_kind() {
        let t = String;
        assert_eq!(t.transform_kind(), TransformKind::String);
        assert_eq!(t.name(), "string");
    }

    #[test]
    fn test_string_domain() {
        let t = String;
        let (min, max) = t.allowed_domain();
        assert!(min.is_infinite() && min.is_sign_negative());
        assert!(max.is_infinite() && max.is_sign_positive());
    }

    #[test]
    fn test_string_transform_passthrough() {
        let t = String;
        assert_eq!(t.transform(1.0), 1.0);
        assert_eq!(t.transform(-5.0), -5.0);
        assert_eq!(t.inverse(100.0), 100.0);
    }

    #[test]
    fn test_string_wrap_numeric() {
        let t = String;
        assert_eq!(t.wrap_numeric(42.0), ArrayElement::String("42".to_string()));
        assert_eq!(
            t.wrap_numeric(-3.54),
            ArrayElement::String("-3.54".to_string())
        );
    }

    #[test]
    fn test_string_breaks_empty() {
        let t = String;
        // String transform doesn't calculate breaks
        assert!(t.calculate_breaks(0.0, 100.0, 5, true).is_empty());
        assert!(t
            .calculate_minor_breaks(&[0.0, 50.0, 100.0], 1, None)
            .is_empty());
    }

    #[test]
    fn test_string_parse_value_string() {
        use super::TransformTrait;
        let t = String;
        // String stays as string
        let input = ArrayElement::String("hello".to_owned());
        assert_eq!(
            t.parse_value(&input),
            ArrayElement::String("hello".to_owned())
        );
    }

    #[test]
    fn test_string_parse_value_number() {
        use super::TransformTrait;
        let t = String;
        // Number converts to string
        let input = ArrayElement::Number(42.0);
        assert_eq!(t.parse_value(&input), ArrayElement::String("42".to_owned()));
    }

    #[test]
    fn test_string_parse_value_boolean() {
        use super::TransformTrait;
        let t = String;
        // Boolean converts to string
        assert_eq!(
            t.parse_value(&ArrayElement::Boolean(true)),
            ArrayElement::String("true".to_owned())
        );
        assert_eq!(
            t.parse_value(&ArrayElement::Boolean(false)),
            ArrayElement::String("false".to_owned())
        );
    }

    #[test]
    fn test_string_parse_value_null() {
        use super::TransformTrait;
        let t = String;
        // Null stays null
        assert_eq!(t.parse_value(&ArrayElement::Null), ArrayElement::Null);
    }
}
