--- MC Metadata - Extension configuration and metadata access for Quarto Lua filters and shortcodes
--- @module metadata
--- @license MIT
--- @copyright 2026 Mickaël Canouil
--- @author Mickaël Canouil
--- @version 1.0.0

local M = {}

--- Load a sibling module from the same directory as this file.
--- @param filename string The sibling module filename (e.g., 'string.lua')
--- @return table The loaded module
local function load_sibling(filename)
  local source = debug.getinfo(1, 'S').source:sub(2)
  local dir = source:match('(.*[/\\])') or ''
  return require((dir .. filename):gsub('%.lua$', ''))
end

--- Load required modules
local str = load_sibling('string.lua')
local log = load_sibling('logging.lua')

-- ============================================================================
-- METADATA UTILITIES
-- ============================================================================

--- Get configuration from extensions.name namespace.
--- @param meta table Document metadata
--- @param extension_name string The extension name (e.g., "github", "iconify")
--- @return any The value/table or nil
--- @usage local value = M.get_extension_config(meta, 'section-outline')
function M.get_extension_config(meta, extension_name)
  local config_ext = meta.extensions and meta.extensions[extension_name]
  if not config_ext then return nil end
  return config_ext
end

--- Extract metadata value from document meta using nested structure.
--- Supports the extensions.{extension-name}.{key} pattern.
--- @param meta table The document metadata table
--- @param extension_name string The extension name (e.g., "github", "iconify")
--- @param key string The metadata key to retrieve
--- @return string|nil The metadata value as a string, or nil if not found
--- @usage local repo = M.get_metadata_value(meta, "github", "repository-name")
function M.get_metadata_value(meta, extension_name, key)
  if meta['extensions'] and meta['extensions'][extension_name] and meta['extensions'][extension_name][key] then
    return str.stringify(meta['extensions'][extension_name][key])
  end
  return nil
end

--- Check for deprecated top-level configuration and emit warning
--- @param meta table The document metadata table
--- @param extension_name string The extension name
--- @param key string|nil The configuration key being accessed (nil to check entire extension config)
--- @param deprecation_warning_shown boolean Flag to track if warning has been shown
--- @return any|nil The value from deprecated config, or nil if not found
--- @return boolean Updated deprecation warning flag
function M.check_deprecated_config(meta, extension_name, key, deprecation_warning_shown)
  -- Handle array-based configuration (when key is nil)
  if key == nil then
    if not str.is_empty(meta[extension_name]) then
      if not deprecation_warning_shown then
        log.log_warning(
          extension_name,
          'Top-level "' .. extension_name .. '" configuration is deprecated. ' ..
          'Please use:\n' ..
          'extensions:\n' ..
          '  ' .. extension_name .. ':\n' ..
          '    - (configuration array)'
        )
        deprecation_warning_shown = true
      end
      return meta[extension_name], deprecation_warning_shown
    end
    return nil, deprecation_warning_shown
  end

  -- Handle key-value configuration (original behaviour)
  if not str.is_empty(meta[extension_name]) and not str.is_empty(meta[extension_name][key]) then
    if not deprecation_warning_shown then
      log.log_warning(
        extension_name,
        'Top-level "' .. extension_name .. '" configuration is deprecated. ' ..
        'Please use:\n' ..
        'extensions:\n' ..
        '  ' .. extension_name .. ':\n' ..
        '    ' .. key .. ': value'
      )
      deprecation_warning_shown = true
    end
    return str.stringify(meta[extension_name][key]), deprecation_warning_shown
  end
  return nil, deprecation_warning_shown
end

-- ============================================================================
-- ENHANCED METADATA/CONFIGURATION UTILITIES
-- ============================================================================

--- Get option value with fallback hierarchy: args -> extensions.{extension}.{key} -> defaults.
--- Provides a standardised way to read configuration values with multiple fallback levels.
--- Priority: 1. Named arguments (kwargs), 2. Document metadata, 3. Default values.
---
--- @param spec table Configuration spec with fields: extension (string), key (string), args (table|nil), meta (table|nil), default (any|nil)
--- @return any The resolved option value (type depends on what's stored in config)
--- @usage local duration = M.get_option_with_fallbacks({extension = 'animate', key = 'duration', args = kwargs, meta = meta, default = '3s'})
function M.get_option_with_fallbacks(spec)
  -- Validate required fields
  if not spec.extension or not spec.key then
    error("Configuration spec must include 'extension' and 'key' fields")
  end

  --- @type string The extension name
  local extension = spec.extension
  --- @type string The configuration key
  local key = spec.key
  --- @type table|nil Named arguments table
  local args = spec.args
  --- @type table|nil Document metadata
  local meta = spec.meta
  --- @type any Default value if not found elsewhere
  local default = spec.default

  -- Priority 1: Check named arguments (kwargs)
  if args and args[key] then
    local arg_value = str.stringify(args[key])
    if not str.is_empty(arg_value) then
      return arg_value
    end
  end

  -- Priority 2: Check metadata extensions.{extension}.{key}
  if meta then
    local meta_value = M.get_metadata_value(meta, extension, key)
    if not str.is_empty(meta_value) then
      return meta_value
    end
  end

  -- Priority 3: Return default value
  return default
end

--- Get multiple option values at once with fallback hierarchy.
--- Batch version of get_option_with_fallbacks for retrieving multiple configuration values.
--- Returns a table mapping each key to its resolved value.
---
--- @param spec table Configuration spec with fields: extension (string), keys (table<integer, string>), args (table|nil), meta (table|nil), defaults (table<string, any>|nil)
--- @return table<string, any> Table mapping each key to its resolved value
--- @usage local opts = M.get_options({extension = 'animate', keys = {'duration', 'delay'}, args = kwargs, meta = meta, defaults = {duration = '3s', delay = '2s'}})
function M.get_options(spec)
  -- Validate required fields
  if not spec.extension or not spec.keys then
    error("Configuration spec must include 'extension' and 'keys' fields")
  end

  --- @type table<string, any> Result table
  local result = {}

  --- @type table Default values table
  local defaults = spec.defaults or {}

  -- Get each key using the single-option fallback logic
  for _, key in ipairs(spec.keys) do
    result[key] = M.get_option_with_fallbacks({
      extension = spec.extension,
      key = key,
      args = spec.args,
      meta = spec.meta,
      default = defaults[key]
    })
  end

  return result
end

-- ============================================================================
-- MODULE EXPORT
-- ============================================================================

return M
