//! A data explorer selection as CSV, TSV or HTML, for the clipboard.

use crate::display::escape_html;

/// Format `rows` for export. A `header` is written first when given; a null
/// cell is empty.
pub fn format(
    header: Option<&[String]>,
    rows: &[Vec<String>],
    format: &str,
) -> Result<String, String> {
    let all = header
        .into_iter()
        .map(<[String]>::to_vec)
        .chain(rows.iter().cloned());
    match format {
        "csv" => Ok(delimited(all, ",", csv_field)),
        "tsv" => Ok(delimited(all, "\t", tsv_field)),
        "html" => Ok(html(header, rows)),
        other => Err(format!("Unsupported export format {other:?}")),
    }
}

fn delimited(
    rows: impl Iterator<Item = Vec<String>>,
    sep: &str,
    field: fn(&str) -> String,
) -> String {
    let lines: Vec<String> = rows
        .map(|row| row.iter().map(|v| field(v)).collect::<Vec<_>>().join(sep))
        .collect();
    lines.join("\n")
}

/// Quoted only when it has to be, with embedded quotes doubled (RFC 4180).
fn csv_field(v: &str) -> String {
    if v.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", v.replace('"', "\"\""))
    } else {
        v.to_string()
    }
}

/// TSV has no quoting, so the characters that would break a row become spaces.
fn tsv_field(v: &str) -> String {
    v.replace(['\t', '\n', '\r'], " ")
}

fn html(header: Option<&[String]>, rows: &[Vec<String>]) -> String {
    let mut out = String::from("<table>\n");
    if let Some(header) = header {
        out.push_str("<thead><tr>");
        for h in header {
            out.push_str(&format!("<th>{}</th>", escape_html(h)));
        }
        out.push_str("</tr></thead>\n");
    }
    out.push_str("<tbody>\n");
    for row in rows {
        out.push_str("<tr>");
        for v in row {
            out.push_str(&format!("<td>{}</td>", escape_html(v)));
        }
        out.push_str("</tr>\n");
    }
    out.push_str("</tbody>\n</table>");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows() -> Vec<Vec<String>> {
        vec![
            vec!["1".into(), "a, \"b\"".into()],
            vec!["2".into(), "tab\there".into()],
        ]
    }

    #[test]
    fn test_csv_quotes_only_what_needs_it() {
        let header = ["n".to_string(), "s".to_string()];
        assert_eq!(
            format(Some(&header), &rows(), "csv").unwrap(),
            "n,s\n1,\"a, \"\"b\"\"\"\n2,tab\there"
        );
    }

    #[test]
    fn test_tsv_flattens_tabs() {
        assert_eq!(
            format(None, &rows(), "tsv").unwrap(),
            "1\ta, \"b\"\n2\ttab here"
        );
    }

    #[test]
    fn test_html_escapes() {
        let out = format(None, &[vec!["<b>".into()]], "html").unwrap();
        assert!(out.contains("<td>&lt;b&gt;</td>"), "{out}");
    }

    #[test]
    fn test_unknown_format_is_an_error() {
        assert!(format(None, &rows(), "xlsx").is_err());
    }
}
