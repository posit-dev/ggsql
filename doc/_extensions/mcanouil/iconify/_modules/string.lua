--- MC String - String manipulation and escaping for Quarto Lua filters and shortcodes
--- @module string
--- @license MIT
--- @copyright 2026 Mickaël Canouil
--- @author Mickaël Canouil
--- @version 1.0.0

local M = {}

-- ============================================================================
-- STRING UTILITIES
-- ============================================================================

--- Pandoc utility function for converting values to strings
--- @type function
M.stringify = pandoc.utils.stringify

--- Check if a string is empty or nil.
--- Utility function to determine if a value is empty or nil,
--- which is useful for parameter validation throughout the module.
--- @param s string|nil|table The value to check for emptiness
--- @return boolean True if the value is nil or empty, false otherwise
--- @usage local result = M.is_empty("") -- returns true
--- @usage local result = M.is_empty(nil) -- returns true
--- @usage local result = M.is_empty("hello") -- returns false
function M.is_empty(s)
  return s == nil or s == ''
end

--- Escape special pattern characters in a string for Lua pattern matching
--- @param s string The string to escape
--- @return string The escaped string
--- @usage local escaped = M.escape_pattern("user/repo#123")
function M.escape_pattern(s)
  local escaped = s:gsub('([%^%$%(%)%%%.%[%]%*%+%-%?])', '%%%1')
  return escaped
end

--- Split a string by a separator
--- @param str string The string to split
--- @param sep string The separator pattern
--- @return table Array of string fields
--- @usage local parts = M.split("a.b.c", ".")
function M.split(str, sep)
  local fields = {}
  local pattern = string.format('([^%s]+)', sep)
  str:gsub(pattern, function(c) fields[#fields + 1] = c end)
  return fields
end

--- Trim leading and trailing whitespace from a string
--- @param str string The string to trim
--- @return string The trimmed string
--- @usage local trimmed = M.trim("  hello world  ") -- returns "hello world"
function M.trim(str)
  if str == nil then return '' end
  return str:match('^%s*(.-)%s*$')
end

--- Convert any value to a string, handling Pandoc objects and empty values.
--- Returns nil for empty or nil values, otherwise returns a string representation.
--- @param val any The value to convert
--- @return string|nil The string value or nil if empty
--- @usage local str = M.to_string(kwargs.value)
function M.to_string(val)
  if not val then return nil end
  if type(val) == 'string' then
    return val ~= '' and val or nil
  end
  -- Handle Pandoc objects
  if pandoc and pandoc.utils and pandoc.utils.stringify then
    local str = pandoc.utils.stringify(val)
    return str ~= '' and str or nil
  end
  local str = tostring(val)
  return str ~= '' and str or nil
end

-- ============================================================================
-- ESCAPE UTILITIES
-- ============================================================================

--- Escape special LaTeX characters in text.
--- @param text string The text to escape
--- @return string The escaped text safe for LaTeX
function M.escape_latex(text)
  text = string.gsub(text, '\\', '\\textbackslash{}')
  text = string.gsub(text, '%{', '\\{')
  text = string.gsub(text, '%}', '\\}')
  text = string.gsub(text, '%$', '\\$')
  text = string.gsub(text, '%&', '\\&')
  text = string.gsub(text, '%%', '\\%%')
  text = string.gsub(text, '%#', '\\#')
  text = string.gsub(text, '%^', '\\textasciicircum{}')
  text = string.gsub(text, '%_', '\\_')
  text = string.gsub(text, '~', '\\textasciitilde{}')
  return text
end

--- Escape special Typst characters in text.
--- @param text string The text to escape
--- @return string The escaped text safe for Typst
function M.escape_typst(text)
  text = string.gsub(text, '%#', '\\#')
  return text
end

--- Escape characters for Typst string literals (inside `"..."`).
--- @param text string The text to escape
--- @return string The escaped text safe for Typst string literals
function M.escape_typst_string(text)
  return text:gsub('\\', '\\\\'):gsub('"', '\\"')
end

--- Escape special Lua pattern characters for use in string.gsub.
--- @param text string The text containing characters to escape
--- @return string The escaped text safe for Lua patterns
function M.escape_lua_pattern(text)
  text = string.gsub(text, '%%', '%%%%')
  text = string.gsub(text, '%^', '%%^')
  text = string.gsub(text, '%$', '%%$')
  text = string.gsub(text, '%(', '%%(')
  text = string.gsub(text, '%)', '%%)')
  text = string.gsub(text, '%.', '%%.')
  text = string.gsub(text, '%[', '%%[')
  text = string.gsub(text, '%]', '%%]')
  text = string.gsub(text, '%*', '%%*')
  text = string.gsub(text, '%+', '%%+')
  text = string.gsub(text, '%-', '%%-')
  text = string.gsub(text, '%?', '%%?')
  return text
end

--- Escape special HTML characters in text.
--- Escapes &, <, >, ", and ' to prevent XSS and ensure valid HTML.
--- @param text string The text to escape
--- @return string Escaped text safe for use in HTML
--- @usage local escaped = M.escape_html('Hello <World>')
function M.escape_html(text)
  if text == nil then return '' end
  if type(text) ~= 'string' then text = tostring(text) end
  local result = text
      :gsub('&', '&amp;')
      :gsub('<', '&lt;')
      :gsub('>', '&gt;')
      :gsub('"', '&quot;')
      :gsub("'", '&#39;')
  return result
end

--- Escape special HTML attribute characters.
--- Escapes characters that could break attribute values.
--- @param value string The attribute value to escape
--- @return string Escaped value safe for use in HTML attributes
--- @usage local escaped = M.escape_attribute('Hello "World"')
function M.escape_attribute(value)
  if value == nil then return '' end
  if type(value) ~= 'string' then value = tostring(value) end
  local result = value
      :gsub('&', '&amp;')
      :gsub('"', '&quot;')
      :gsub('<', '&lt;')
      :gsub('>', '&gt;')
  return result
end

--- Escape text for different formats.
--- @param text string The text to escape
--- @param format string The format to escape for (e.g., "latex", "typst", "lua")
--- @return string The escaped text
function M.escape_text(text, format)
  local escape_functions = {
    latex = M.escape_latex,
    typst = M.escape_typst,
    lua = M.escape_lua_pattern
  }

  local escape = escape_functions[format]
  if escape then
    return escape(text)
  else
    error('Unsupported escape format: ' .. format)
  end
end

--- Converts a string to a valid HTML id by lowercasing and replacing spaces.
--- @param text string The text to convert
--- @return string The HTML id
function M.ascii_id(text)
  local id = text:lower():gsub('[^a-z0-9 ]', ''):gsub(' +', '-')
  return id
end

-- ============================================================================
-- MODULE EXPORT
-- ============================================================================

return M
