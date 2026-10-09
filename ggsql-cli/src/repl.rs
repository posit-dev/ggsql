//! The interactive REPL behind `ggsql repl`.
//!
//! The loop itself is a thin rustyline shell; everything with logic in it —
//! statement completeness, classification, dispatch — is a free function so it
//! can be tested without a terminal.

use ggsql::reader::{Reader, Spec};
use ggsql::validate::validate;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;

use crate::table;

const PROMPT: &str = "ggsql> ";
const CONTINUATION_PROMPT: &str = "  ...> ";
const MAX_TABLE_ROWS: usize = 100;

/// What one submitted input means.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Statement {
    /// Nothing but whitespace — ignore it.
    Empty,
    /// Leave the session.
    Quit,
    /// List the available commands.
    Help,
    /// A `:` command the REPL doesn't know.
    Unknown(String),
    /// A ggsql query, with the terminating `;` stripped.
    Query(String),
}

/// Where a finished plot goes. The window feature supplies the real
/// implementation; without it, and in tests, a stub explains the situation.
pub trait PlotDisplay {
    fn show(&self, spec: Spec) -> Result<(), String>;
}

/// Whether the accumulated input is ready to run.
///
/// Meta-commands are complete on one line. Queries are complete when brackets
/// balance, no string is open, and the statement ends with `;` — matching how
/// SQL clients decide, and avoiding guesses about where a VISUALISE clause
/// ends.
pub fn is_complete(input: &str) -> bool {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return false;
    }
    if trimmed.starts_with(':') {
        return true;
    }
    if !trimmed.ends_with(';') {
        return false;
    }

    let mut paren_depth = 0i32;
    let mut bracket_depth = 0i32;
    let mut in_string: Option<char> = None;
    let mut chars = trimmed.chars().peekable();

    while let Some(c) = chars.next() {
        match in_string {
            Some(quote) if c == quote => {
                // SQL escapes a quote inside a string by doubling it.
                if chars.peek() == Some(&quote) {
                    chars.next();
                } else {
                    in_string = None;
                }
            }
            Some(_) => {}
            None => match c {
                '\'' | '"' => in_string = Some(c),
                '(' => paren_depth += 1,
                ')' => paren_depth -= 1,
                '[' => bracket_depth += 1,
                ']' => bracket_depth -= 1,
                _ => {}
            },
        }
    }

    in_string.is_none() && paren_depth == 0 && bracket_depth == 0
}

/// Turn complete input into what to do with it.
pub fn classify(input: &str) -> Statement {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Statement::Empty;
    }
    if let Some(command) = trimmed.strip_prefix(':') {
        return match command {
            "quit" | "exit" | "q" => Statement::Quit,
            "help" | "h" | "?" => Statement::Help,
            other => Statement::Unknown(other.to_string()),
        };
    }
    // `is_complete` guarantees the trailing `;`; all that remains is to strip
    // it so the parser never sees it.
    let query = trimmed.trim_end_matches(';').trim().to_string();
    if query.is_empty() {
        Statement::Empty
    } else {
        Statement::Query(query)
    }
}

/// Run one query, printing the table or handing the plot to `plots`.
///
/// Errors are reported, not propagated: one bad statement must not end the
/// session.
pub fn execute_query(query: &str, reader: &dyn Reader, plots: &dyn PlotDisplay, verbose: bool) {
    let validated = match validate(query) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Failed to validate query: {e}");
            return;
        }
    };
    if !validated.valid() {
        eprintln!("Validation errors:");
        for err in validated.errors() {
            eprintln!("  - {}", err.message);
        }
        return;
    }

    if !validated.has_visual() {
        match table::format(query, reader, MAX_TABLE_ROWS) {
            Ok(table) => println!("{table}"),
            Err(e) => eprintln!("{e}"),
        }
        return;
    }

    let spec = match reader.execute(query) {
        Ok(spec) => spec,
        Err(e) => {
            eprintln!("Failed to execute query: {e}");
            return;
        }
    };
    if verbose {
        let metadata = spec.metadata();
        eprintln!("Rows: {}, layers: {}", metadata.rows, metadata.layer_count);
    }
    if let Err(e) = plots.show(spec) {
        eprintln!("{e}");
    }
}

fn print_help() {
    println!("Commands:");
    println!("  :quit, :exit   Leave the session (Ctrl-D works too)");
    println!("  :help          Show this text");
    println!();
    println!("Statements end with ';'. A plain SQL statement prints as a table;");
    println!("a query with a VISUALISE clause draws in the plot window.");
}

fn history_path() -> Option<std::path::PathBuf> {
    #[allow(deprecated)]
    std::env::home_dir().map(|home| home.join(".ggsql_history"))
}

/// Run the read-eval-print loop until the user leaves.
pub fn run(reader: &dyn Reader, plots: &dyn PlotDisplay, verbose: bool) -> rustyline::Result<()> {
    let mut editor = DefaultEditor::new()?;
    let history = history_path();
    if let Some(path) = &history {
        let _ = editor.load_history(path);
    }

    println!("ggsql REPL — end statements with ';'. :help for help, :quit to leave.");

    let mut buffer = String::new();
    loop {
        let prompt = if buffer.trim().is_empty() {
            PROMPT
        } else {
            CONTINUATION_PROMPT
        };
        match editor.readline(prompt) {
            Ok(line) => {
                buffer.push_str(&line);
                buffer.push('\n');
                if !is_complete(&buffer) {
                    continue;
                }
                if let Err(e) = editor.add_history_entry(buffer.trim()) {
                    eprintln!("warning: could not record history: {e}");
                }
                let statement = classify(&buffer);
                buffer.clear();
                match statement {
                    Statement::Empty => {}
                    Statement::Quit => break,
                    Statement::Help => print_help(),
                    Statement::Unknown(command) => {
                        eprintln!("Unknown command ':{command}'. :help lists the commands.")
                    }
                    Statement::Query(query) => execute_query(&query, reader, plots, verbose),
                }
            }
            // Ctrl-C abandons the current statement, not the session.
            Err(ReadlineError::Interrupted) => {
                buffer.clear();
                println!("^C");
            }
            Err(ReadlineError::Eof) => break,
            Err(e) => return Err(e),
        }
    }

    if let Some(path) = &history {
        let _ = editor.save_history(path);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_is_never_complete() {
        assert!(!is_complete(""));
        assert!(!is_complete("  \n  "));
        assert_eq!(classify("  \n "), Statement::Empty);
    }

    #[test]
    fn queries_need_a_terminating_semicolon() {
        assert!(!is_complete("SELECT 1"));
        assert!(is_complete("SELECT 1;"));
        assert!(!is_complete("SELECT 1; extra"));
    }

    #[test]
    fn brackets_and_strings_delay_completeness() {
        assert!(!is_complete("SELECT (1 + 2;"));
        assert!(is_complete("SELECT (1 + 2);"));
        assert!(!is_complete("SELECT 'oops;"));
        // A semicolon inside a string doesn't count as the terminator — but
        // since the string is still open the statement is incomplete anyway,
        // and once closed it needs a real `;`.
        assert!(!is_complete("SELECT ';'"));
        assert!(is_complete("SELECT ';';"));
    }

    #[test]
    fn doubled_quotes_stay_inside_the_string() {
        assert!(!is_complete("SELECT 'it''s"));
        assert!(is_complete("SELECT 'it''s';"));
    }

    #[test]
    fn meta_commands_are_complete_on_one_line() {
        assert!(is_complete(":quit"));
        assert_eq!(classify(":quit"), Statement::Quit);
        assert_eq!(classify(":q"), Statement::Quit);
        assert_eq!(classify(":exit"), Statement::Quit);
        assert_eq!(classify(":help"), Statement::Help);
        assert_eq!(classify(":h"), Statement::Help);
        assert_eq!(classify(":bogus"), Statement::Unknown("bogus".to_string()));
    }

    #[test]
    fn classify_strips_the_terminator() {
        assert_eq!(
            classify("SELECT 1;\n"),
            Statement::Query("SELECT 1".to_string())
        );
        assert_eq!(
            classify("SELECT * FROM t VISUALISE DRAW point USING x AS x, y AS y;"),
            Statement::Query(
                "SELECT * FROM t VISUALISE DRAW point USING x AS x, y AS y".to_string()
            )
        );
        assert_eq!(classify(";"), Statement::Empty);
    }
}
