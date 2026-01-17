/**
 * Node.js bindings for tree-sitter-ggsql
 *
 * This module provides the tree-sitter language definition for ggsql,
 * a SQL extension for declarative data visualization.
 */

try {
  module.exports = require("../../build/Release/tree_sitter_ggsql_binding");
} catch (error1) {
  if (error1.code !== 'MODULE_NOT_FOUND') {
    throw error1;
  }
  try {
    module.exports = require("../../build/Debug/tree_sitter_ggsql_binding");
  } catch (error2) {
    if (error2.code !== 'MODULE_NOT_FOUND') {
      throw error2;
    }
    throw new Error(
      'Could not load tree-sitter-ggsql binding. ' +
      'Make sure you have run `npm install` or `node-gyp rebuild`.'
    );
  }
}
