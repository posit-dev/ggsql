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
        // Where rows with null facet values go: a group sorted first or
        // last, or dropped from the table.
        name: "missing",
        default: DefaultParamValue::String("last"),
        constraint: ParamConstraint::string_option(&["first", "last", "drop"]),
    },
    ParamDefinition {
        // Joins nested group labels into one, e.g. 'north - q1'. Unset
        // keeps one label row/column per nesting level.
        name: "separator",
        default: DefaultParamValue::Null,
        constraint: ParamConstraint::string(),
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
    if !resolved.properties.contains_key("missing") {
        resolved.properties.insert(
            "missing".to_string(),
            ParameterValue::String("last".to_string()),
        );
    }
    resolved.resolved = true;
    Ok(Some(resolved))
}

/// Whether a resolved table facet weaves its group labels in as interleaved
/// rows (`place => 'top'`) rather than group stub columns (`place =>
/// 'left'`).
pub(crate) fn groups_are_woven(facet: &Facet) -> bool {
    facet.properties.get("place") == Some(&ParameterValue::String("top".to_string()))
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
    fn setup_materializes_default_settings() {
        let facet = table_facet(Parameters::new());

        let resolved = setup_facet(&Some(facet)).unwrap().unwrap();

        assert_eq!(
            resolved.properties.get("place"),
            Some(&ParameterValue::String("top".to_string()))
        );
        assert_eq!(
            resolved.properties.get("missing"),
            Some(&ParameterValue::String("last".to_string()))
        );
        assert!(resolved.resolved);
    }

    #[test]
    fn setup_passes_none_through() {
        assert_eq!(setup_facet(&None).unwrap(), None);
    }

    #[test]
    fn groups_are_woven_only_when_place_is_top() {
        let woven = setup_facet(&Some(table_facet(Parameters::new())))
            .unwrap()
            .unwrap();
        assert!(groups_are_woven(&woven));

        let mut settings = Parameters::new();
        settings.insert(
            "place".to_string(),
            ParameterValue::String("left".to_string()),
        );
        let left = setup_facet(&Some(table_facet(settings))).unwrap().unwrap();
        assert!(!groups_are_woven(&left));
    }
}
