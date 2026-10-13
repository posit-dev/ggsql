//! `TABULATE FACET` resolution: validates the clause's `SETTING` parameters
//! against `TABLE_FACET_PARAMS`, applies defaults, and marks the facet
//! resolved. Unlike plot facet resolution, no data context is involved —
//! group extraction happens during table execution.

use crate::params::{validate_parameter, DefaultParamValue, ParamConstraint, ParamDefinition};
use crate::plot::facet::Facet;
use crate::plot::ParameterValue;

/// `SETTING` parameters a `TABULATE FACET` accepts.
const TABLE_FACET_PARAMS: &[ParamDefinition] = &[
    ParamDefinition {
        // Where group labels go: interleaved rows above each group, or a
        // stub column on the left.
        name: "place",
        default: DefaultParamValue::String("top"),
        constraint: ParamConstraint::string_option(&["left", "top"]),
    },
    ParamDefinition {
        name: "missing",
        default: DefaultParamValue::Null,
        constraint: ParamConstraint::string_option(&["repeat", "null"]),
    },
];

/// Validate a table facet's settings, apply defaults, and return the
/// resolved facet. Passes `None` straight through.
pub(crate) fn setup_facet(facet: &Option<Facet>) -> Result<Option<Facet>, String> {
    let Some(facet) = facet else {
        return Ok(None);
    };

    let valid: Vec<&str> = TABLE_FACET_PARAMS.iter().map(|p| p.name).collect();

    for (name, value) in &facet.properties {
        let Some(param) = TABLE_FACET_PARAMS.iter().find(|p| p.name == name.as_str()) else {
            return Err(format!(
                "FACET setting should be {}, not '{}'",
                crate::or_list_quoted(&valid, '\''),
                name
            ));
        };
        validate_parameter(name, value, &param.constraint)?;
    }

    let mut resolved = facet.clone();
    if !resolved.properties.contains_key("place") {
        resolved.properties.insert(
            "place".to_string(),
            ParameterValue::String("top".to_string()),
        );
    }
    resolved.resolved = true;
    Ok(Some(resolved))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::facet::FacetLayout;
    use crate::plot::Parameters;

    fn table_facet(settings: Parameters) -> Facet {
        Facet {
            layout: FacetLayout::Wrap {
                variables: vec!["region".to_string()],
            },
            properties: settings,
            resolved: false,
        }
    }

    #[test]
    fn setup_defaults_place_to_top() {
        let facet = table_facet(Parameters::new());

        let resolved = setup_facet(&Some(facet)).unwrap().unwrap();

        assert_eq!(
            resolved.properties.get("place"),
            Some(&ParameterValue::String("top".to_string()))
        );
        assert!(resolved.resolved);
    }

    #[test]
    fn setup_accepts_place_left() {
        let mut settings = Parameters::new();
        settings.insert(
            "place".to_string(),
            ParameterValue::String("left".to_string()),
        );
        let facet = table_facet(settings);

        let resolved = setup_facet(&Some(facet)).unwrap().unwrap();

        assert_eq!(
            resolved.properties.get("place"),
            Some(&ParameterValue::String("left".to_string()))
        );
    }

    #[test]
    fn setup_rejects_an_unknown_setting() {
        let mut settings = Parameters::new();
        settings.insert("ncol".to_string(), ParameterValue::Number(2.0));
        let facet = table_facet(settings);

        let err = setup_facet(&Some(facet)).unwrap_err();

        assert!(err.contains("ncol"));
    }

    #[test]
    fn setup_rejects_an_invalid_place_value() {
        let mut settings = Parameters::new();
        settings.insert(
            "place".to_string(),
            ParameterValue::String("right".to_string()),
        );
        let facet = table_facet(settings);

        assert!(setup_facet(&Some(facet)).is_err());
    }

    #[test]
    fn setup_accepts_missing_repeat_and_null() {
        for value in ["repeat", "null"] {
            let mut settings = Parameters::new();
            settings.insert(
                "missing".to_string(),
                ParameterValue::String(value.to_string()),
            );
            let facet = table_facet(settings);

            setup_facet(&Some(facet)).unwrap();
        }
    }

    #[test]
    fn setup_passes_none_through() {
        assert_eq!(setup_facet(&None).unwrap(), None);
    }
}
