//! Parser error types
//!
//! Provides detailed error information for parsing failures, including
//! location information and helpful error messages.

use std::fmt;

/// Detailed parse error with location information
#[derive(Debug, Clone)]
pub struct ParseError {
    /// Error message
    pub message: String,
    /// Line number where error occurred (0-based)
    pub line: usize,
    /// Column number where error occurred (0-based)
    pub column: usize,
    /// The parsing context where the error occurred
    pub context: String,
}

impl ParseError {
    /// Create a new parse error
    pub fn new(message: String, line: usize, column: usize, context: String) -> Self {
        Self {
            message,
            line,
            column,
            context,
        }
    }

    /// Create a simple parse error without location info
    pub fn simple(message: String) -> Self {
        Self {
            message,
            line: 0,
            column: 0,
            context: "unknown".to_string(),
        }
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.line > 0 || self.column > 0 {
            write!(
                f,
                "{} at line {}, column {} (in {})",
                self.message,
                self.line + 1,   // Display as 1-based
                self.column + 1, // Display as 1-based
                self.context
            )
        } else {
            write!(f, "{}", self.message)
        }
    }
}

impl std::error::Error for ParseError {}

impl From<ParseError> for crate::ggsqlError {
    fn from(err: ParseError) -> Self {
        crate::ggsqlError::ParseError(err.to_string())
    }
}
