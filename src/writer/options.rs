//! Free-form key–value options for writers.
//!
//! A frontend collects `key=value` pairs from its user (`-D width=1600` on the
//! CLI) and hands them to
//! [`Writer::from_options`](super::Writer::from_options), so each writer
//! exposes its configuration without the frontend knowing its shape.

use std::collections::BTreeMap;

use crate::util::or_list_quoted;
use crate::{GgsqlError, Result};

/// Key–value configuration handed to a [`Writer`](super::Writer).
///
/// Keys are normalised — trimmed, lowercased, and `-` folded to `_` — so
/// `background-color`, `Background_Color`, and `background_color` are the same
/// key. Values are stored verbatim; the accessors below interpret them, and the
/// errors they produce name the offending option so a frontend can pass them
/// straight to the user.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WriterOptions {
    values: BTreeMap<String, String>,
}

impl WriterOptions {
    /// An empty set of options — every writer then uses its own defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse `key=value` strings, as a frontend collects them from a repeatable
    /// flag.
    ///
    /// One string may carry several options separated by `;`, so a caller can
    /// write out either form, or mix them:
    ///
    /// ```text
    /// ["width=1600", "height=1200"]      // one option per flag
    /// ["width=1600;height=1200"]         // collapsed into one
    /// ```
    ///
    /// `;` is the only separator; `,` is common inside a value
    /// (`background=rgba(0,0,0,0)`). The value runs from the first `=` to the
    /// next `;`, and a later occurrence of a key overrides an earlier one.
    ///
    /// # Errors
    ///
    /// Returns `GgsqlError::WriterError` if an entry has no `=` or an empty key.
    pub fn parse<I, S>(pairs: I) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut options = Self::new();
        for pair in pairs {
            // An empty segment is a trailing or doubled `;`, not a mistake worth
            // an error.
            for entry in pair.as_ref().split(';').filter(|e| !e.trim().is_empty()) {
                let Some((key, value)) = entry.split_once('=') else {
                    return Err(GgsqlError::WriterError(format!(
                        "invalid writer option '{}': expected 'key=value'",
                        entry.trim()
                    )));
                };
                if normalise_key(key).is_empty() {
                    return Err(GgsqlError::WriterError(format!(
                        "invalid writer option '{}': the key is empty",
                        entry.trim()
                    )));
                }
                options = options.set(key, value.trim());
            }
        }
        Ok(options)
    }

    /// Set one option, overriding any previous value for the same key.
    pub fn set(mut self, key: &str, value: impl Into<String>) -> Self {
        self.values.insert(normalise_key(key), value.into());
        self
    }

    /// True when no options were supplied.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// The raw value of `key`, or `None` when it was not supplied.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.values.get(&normalise_key(key)).map(String::as_str)
    }

    /// The value of `key` parsed as a finite number.
    ///
    /// # Errors
    ///
    /// Returns `GgsqlError::WriterError` if the value is not a finite number.
    pub fn number(&self, key: &str) -> Result<Option<f64>> {
        let Some(raw) = self.get(key) else {
            return Ok(None);
        };
        match raw.parse::<f64>() {
            Ok(value) if value.is_finite() => Ok(Some(value)),
            _ => Err(GgsqlError::WriterError(format!(
                "writer option '{}' expects a number, got '{raw}'",
                normalise_key(key)
            ))),
        }
    }

    /// The value of `key` parsed as a boolean.
    ///
    /// Accepts `true`/`false`, `yes`/`no`, `on`/`off` and `1`/`0`, ignoring case
    /// and surrounding whitespace. Unsupplied is `None`, so a flag defaulting to
    /// `true` can tell that apart from an explicit `false`.
    ///
    /// # Errors
    ///
    /// Returns `GgsqlError::WriterError` if the value is not one of those
    /// spellings.
    pub fn boolean(&self, key: &str) -> Result<Option<bool>> {
        let Some(raw) = self.get(key) else {
            return Ok(None);
        };
        match raw.trim().to_lowercase().as_str() {
            "true" | "yes" | "on" | "1" => Ok(Some(true)),
            "false" | "no" | "off" | "0" => Ok(Some(false)),
            _ => Err(GgsqlError::WriterError(format!(
                "writer option '{}' expects true or false, got '{raw}'",
                normalise_key(key)
            ))),
        }
    }

    /// The value of `key`, checked against a closed set of allowed values.
    ///
    /// Matching ignores case and surrounding whitespace, mirroring how keys are
    /// normalised.
    ///
    /// # Errors
    ///
    /// Returns `GgsqlError::WriterError` if the value is not in `allowed`.
    pub fn one_of<'a>(&self, key: &str, allowed: &[&'a str]) -> Result<Option<&'a str>> {
        let Some(raw) = self.get(key) else {
            return Ok(None);
        };
        let needle = raw.trim().to_lowercase();
        match allowed.iter().find(|value| **value == needle) {
            Some(value) => Ok(Some(value)),
            None => Err(GgsqlError::WriterError(format!(
                "writer option '{}' expects {}, got '{raw}'",
                normalise_key(key),
                or_list_quoted(allowed, '\'')
            ))),
        }
    }

    /// Reject any option the writer does not understand.
    ///
    /// Writers call this first so a mistyped key is an error rather than a
    /// silently ignored setting.
    ///
    /// # Errors
    ///
    /// Returns `GgsqlError::WriterError` naming the unknown keys and listing
    /// the supported ones.
    pub fn reject_unknown(&self, known: &[&str]) -> Result<()> {
        // The declared names are normalised too, so a writer may declare the
        // hyphenated spelling its docs use and still match either form. The
        // error lists them as declared.
        let canonical: Vec<String> = known.iter().map(|key| normalise_key(key)).collect();
        let unknown: Vec<&str> = self
            .values
            .keys()
            .map(String::as_str)
            .filter(|key| !canonical.iter().any(|k| k == key))
            .collect();
        if unknown.is_empty() {
            return Ok(());
        }
        let subject = if unknown.len() == 1 {
            "option"
        } else {
            "options"
        };
        let supported = if known.is_empty() {
            "this writer takes no options".to_string()
        } else {
            format!("supported options: {}", known.join(", "))
        };
        Err(GgsqlError::WriterError(format!(
            "unknown writer {subject} {} — {supported}",
            or_list_quoted(&unknown, '\'')
        )))
    }
}

/// Fold a key to its canonical form: trimmed, lowercased, `-` as `_`.
fn normalise_key(key: &str) -> String {
    key.trim().to_lowercase().replace('-', "_")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_reads_key_value_pairs() {
        let options = WriterOptions::parse(["width=1600", "height=1200"]).unwrap();
        assert_eq!(options.get("width"), Some("1600"));
        assert_eq!(options.number("height").unwrap(), Some(1200.0));
        assert_eq!(options.get("dpi"), None);
        assert_eq!(options.number("dpi").unwrap(), None);
    }

    #[test]
    fn parse_normalises_keys_and_trims_values() {
        let options = WriterOptions::parse([" Background-Color = red "]).unwrap();
        assert_eq!(options.get("background_color"), Some("red"));
        assert_eq!(options.get("BACKGROUND-COLOR"), Some("red"));
    }

    #[test]
    fn parse_collapses_several_options_into_one_entry() {
        let collapsed = WriterOptions::parse(["width=1600;height=1200;units=px"]).unwrap();
        let separate = WriterOptions::parse(["width=1600", "height=1200", "units=px"]).unwrap();
        assert_eq!(collapsed, separate);
        // The two forms mix, and a stray or trailing `;` is not an error.
        let mixed = WriterOptions::parse(["width=1600;height=1200;", "units=px"]).unwrap();
        assert_eq!(mixed, separate);
    }

    #[test]
    fn parse_keeps_commas_inside_a_value() {
        let options = WriterOptions::parse(["background=rgb(255, 0, 0);dpi=150"]).unwrap();
        assert_eq!(options.get("background"), Some("rgb(255, 0, 0)"));
        assert_eq!(options.number("dpi").unwrap(), Some(150.0));
    }

    #[test]
    fn parse_splits_on_the_first_equals_only() {
        let options = WriterOptions::parse(["background=rgba(0,0,0,0)", "title=a=b"]).unwrap();
        assert_eq!(options.get("background"), Some("rgba(0,0,0,0)"));
        assert_eq!(options.get("title"), Some("a=b"));
    }

    #[test]
    fn parse_lets_a_later_occurrence_win() {
        let options = WriterOptions::parse(["width=100", "width=200"]).unwrap();
        assert_eq!(options.get("width"), Some("200"));
    }

    #[test]
    fn parse_rejects_malformed_entries() {
        let err = WriterOptions::parse(["width"]).unwrap_err().to_string();
        assert!(err.contains("expected 'key=value'"), "{err}");
        let err = WriterOptions::parse(["=1600"]).unwrap_err().to_string();
        assert!(err.contains("the key is empty"), "{err}");
    }

    #[test]
    fn number_rejects_non_numeric_values() {
        let options = WriterOptions::parse(["width=wide"]).unwrap();
        let err = options.number("width").unwrap_err().to_string();
        assert!(
            err.contains("'width' expects a number, got 'wide'"),
            "{err}"
        );
        let options = WriterOptions::parse(["width=inf"]).unwrap();
        assert!(options.number("width").is_err());
    }

    #[test]
    fn boolean_accepts_the_usual_spellings() {
        for yes in ["true", "TRUE", " yes ", "on", "1"] {
            let options = WriterOptions::new().set("embed_fonts", yes);
            assert_eq!(options.boolean("embed_fonts").unwrap(), Some(true), "{yes}");
        }
        for no in ["false", "No", "off", "0"] {
            let options = WriterOptions::new().set("embed_fonts", no);
            assert_eq!(options.boolean("embed_fonts").unwrap(), Some(false), "{no}");
        }
        // Unsupplied stays distinct from an explicit `false`, so a writer whose
        // default is `true` can tell them apart.
        assert_eq!(WriterOptions::new().boolean("embed_fonts").unwrap(), None);
    }

    #[test]
    fn boolean_rejects_anything_else() {
        let options = WriterOptions::new().set("embed_fonts", "maybe");
        let err = options.boolean("embed_fonts").unwrap_err().to_string();
        assert!(
            err.contains("'embed_fonts' expects true or false, got 'maybe'"),
            "{err}"
        );
    }

    #[test]
    fn one_of_matches_case_insensitively() {
        let options = WriterOptions::parse(["units=CM"]).unwrap();
        assert_eq!(options.one_of("units", &["px", "cm"]).unwrap(), Some("cm"));
        assert_eq!(
            WriterOptions::new().one_of("units", &["px", "cm"]).unwrap(),
            None
        );
    }

    #[test]
    fn one_of_rejects_values_outside_the_set() {
        let options = WriterOptions::parse(["units=furlongs"]).unwrap();
        let err = options.one_of("units", &["px", "cm"]).unwrap_err();
        assert!(
            err.to_string()
                .contains("'units' expects 'px' or 'cm', got 'furlongs'"),
            "{err}"
        );
    }

    #[test]
    fn reject_unknown_names_the_bad_keys() {
        let options = WriterOptions::parse(["width=10", "hight=10", "colour=red"]).unwrap();
        let err = options
            .reject_unknown(&["width", "height"])
            .unwrap_err()
            .to_string();
        assert!(err.contains("unknown writer options"), "{err}");
        assert!(err.contains("'colour' or 'hight'"), "{err}");
        assert!(err.contains("supported options: width, height"), "{err}");
        assert!(options
            .reject_unknown(&["width", "height", "hight", "colour"])
            .is_ok());
    }

    #[test]
    fn reject_unknown_says_so_when_no_options_are_taken() {
        let err = WriterOptions::parse(["width=10"])
            .unwrap()
            .reject_unknown(&[])
            .unwrap_err()
            .to_string();
        assert!(err.contains("this writer takes no options"), "{err}");
        assert!(WriterOptions::new().reject_unknown(&[]).is_ok());
    }
}
