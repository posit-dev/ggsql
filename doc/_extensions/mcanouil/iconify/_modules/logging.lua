--- MC Logging - Formatted log output for Quarto Lua filters and shortcodes
--- @module logging
--- @license MIT
--- @copyright 2026 Mickaël Canouil
--- @author Mickaël Canouil
--- @version 1.0.0

local M = {}

-- ============================================================================
-- LOGGING UTILITIES
-- ============================================================================

--- Format and log an error message with extension prefix.
--- Provides standardised error messages with consistent formatting across extensions.
--- Format: [extension-name] Message with details.
---
--- @param extension_name string The name of the extension (e.g., "external", "lua-env")
--- @param message string The error message to display
--- @usage M.log_error("external", "Could not open file 'example.md'.")
function M.log_error(extension_name, message)
  quarto.log.error('[' .. extension_name .. '] ' .. message)
end

--- Format and log a warning message with extension prefix.
--- Provides standardised warning messages with consistent formatting across extensions.
--- Format: [extension-name] Message with details.
---
--- @param extension_name string The name of the extension (e.g., "external", "lua-env")
--- @param message string The warning message to display
--- @usage M.log_warning("lua-env", "No variable name provided.")
function M.log_warning(extension_name, message)
  quarto.log.warning('[' .. extension_name .. '] ' .. message)
end

--- Format and log an output message with extension prefix.
--- Provides standardised informational messages with consistent formatting across extensions.
--- Format: [extension-name] Message with details.
---
--- @param extension_name string The name of the extension (e.g., "lua-env")
--- @param message string The informational message to display
--- @usage M.log_output("lua-env", "Exported metadata to: output.json")
function M.log_output(extension_name, message)
  quarto.log.output('[' .. extension_name .. '] ' .. message)
end

--- Format and log a debug message with extension prefix.
--- Provides standardised debug messages with consistent formatting across extensions.
--- Format: [extension-name] Message with details.
---
--- @param extension_name string The name of the extension (e.g., "lua-env")
--- @param message string The debug message to display
--- @usage M.log_debug("lua-env", "Variable 'x' has value: 42")
function M.log_debug(extension_name, message)
  quarto.log.debug('[' .. extension_name .. '] ' .. message)
end

-- ============================================================================
-- MODULE EXPORT
-- ============================================================================

return M
