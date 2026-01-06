/**
 * Minimal ggSQL grammar without external scanner
 *
 * Uses a simple regex to capture SQL portion as opaque text
 */

// Helper to create case-insensitive keyword patterns
function caseInsensitive(keyword) {
  return new RegExp(
    keyword
      .split('')
      .map(letter => `[${letter.toLowerCase()}${letter.toUpperCase()}]`)
      .join('')
  );
}

module.exports = grammar({
  name: 'ggsql',

  conflicts: $ => [
    [$.sql_portion],
  ],

  rules: {
    // Main entry point - SQL followed by VISUALISE statements
    query: $ => seq(
      optional($.sql_portion),
      repeat1($.visualise_statement)
    ),

    // SQL portion - multiple statements separated by semicolons
    sql_portion: $ => choice(
      // Multiple statements with semicolons
      prec.right(seq(
        $.sql_statement,
        repeat1(seq(';', optional($.sql_statement))),
        optional(';')
      )),
      // Single statement (no semicolon before VISUALISE)
      $.sql_statement
    ),

    // A single SQL statement - order matters! More specific first
    sql_statement: $ => choice(
      $.with_statement,  // Check WITH first (can contain SELECT)
      $.select_statement,
      $.create_statement,
      $.insert_statement,
      $.update_statement,
      $.delete_statement,
      $.other_sql_statement  // Fallback for other SQL
    ),

    // SELECT statement
    select_statement: $ => prec(2, seq(
      caseInsensitive('SELECT'),
      $.select_body
    )),

    select_body: $ => prec.left(repeat1(choice(
      $.sql_keyword,
      $.string,
      $.number,
      ',', '*', '.', '=', '<', '>', '!',
      $.subquery,
      token(/[^\s;(),'"VvWwSsCcIiUuDd]+/),  // Other SQL tokens, excluding keyword start letters
      $.identifier
    ))),

    // WITH statement (CTEs) - WITH must be followed by SELECT
    with_statement: $ => prec(2, seq(
      caseInsensitive('WITH'),
      optional(caseInsensitive('RECURSIVE')),
      $.cte_definition,
      repeat(seq(',', $.cte_definition)),
      optional($.select_statement)  // WITH can optionally be followed by SELECT
    )),

    cte_definition: $ => seq(
      $.identifier,
      caseInsensitive('AS'),
      '(',
      $.select_statement,
      ')'
    ),

    // CREATE statement
    create_statement: $ => prec.right(seq(
      caseInsensitive('CREATE'),
      repeat1(choice(
        $.sql_keyword,
        $.identifier,
        $.string,
        $.number,
        $.subquery,
        ',', '(', ')', '*', '.', '=',
        /[^\s;(),'"]+/
      ))
    )),

    // INSERT statement
    insert_statement: $ => prec.right(seq(
      caseInsensitive('INSERT'),
      repeat1(choice(
        $.sql_keyword,
        $.identifier,
        $.string,
        $.number,
        $.subquery,
        ',', '(', ')', '*', '.', '=',
        /[^\s;(),'"]+/
      ))
    )),

    // UPDATE statement
    update_statement: $ => prec.right(seq(
      caseInsensitive('UPDATE'),
      repeat1(choice(
        $.sql_keyword,
        $.identifier,
        $.string,
        $.number,
        $.subquery,
        ',', '(', ')', '*', '.', '=',
        /[^\s;(),'"]+/
      ))
    )),

    // DELETE statement
    delete_statement: $ => prec.right(seq(
      caseInsensitive('DELETE'),
      repeat1(choice(
        $.sql_keyword,
        $.identifier,
        $.string,
        $.number,
        $.subquery,
        ',', '(', ')', '*', '.', '=',
        /[^\s;(),'"]+/
      ))
    )),

    // Other SQL statements - DO NOT match if starts with keywords we handle
    // explicitly (WITH, SELECT, CREATE, INSERT, UPDATE, DELETE, VISUALISE)
    other_sql_statement: $ => {
      const exclude_pattern = /[^\s;(),'"WwSsCcIiUuDdVv]+/;
      return prec(-1, repeat1(choice(
        $.sql_keyword,
        token(exclude_pattern),  // Tokens not starting with excluded letters
        $.string,
        $.number,
        $.subquery,
        ',', '(', ')', '*', '.', '='
      )));
    },

    // Subquery in parentheses - fully recursive, can contain any SQL
    // Higher precedence to prefer subquery interpretation over other_sql_statement
    subquery: $ => prec(1, seq(
      '(',
      repeat1(choice(
        $.select_statement,
        $.sql_keyword,
        $.string,
        $.number,
        $.identifier,
        $.subquery,  // Nested subqueries
        ',', '*', '.', '=', '<', '>', '!',
        token(/[^\s;(),'\"]+/)  // Any other SQL tokens
      )),
      ')'
    )),

    // Common SQL keywords (to help parser recognize structure)
    sql_keyword: $ => choice(
      caseInsensitive('FROM'),
      caseInsensitive('WHERE'),
      caseInsensitive('JOIN'),
      caseInsensitive('LEFT'),
      caseInsensitive('RIGHT'),
      caseInsensitive('INNER'),
      caseInsensitive('OUTER'),
      caseInsensitive('ON'),
      caseInsensitive('AND'),
      caseInsensitive('OR'),
      caseInsensitive('NOT'),
      caseInsensitive('IN'),
      caseInsensitive('EXISTS'),
      caseInsensitive('BETWEEN'),
      caseInsensitive('LIKE'),
      caseInsensitive('ORDER'),
      caseInsensitive('GROUP'),
      caseInsensitive('BY'),
      caseInsensitive('HAVING'),
      caseInsensitive('LIMIT'),
      caseInsensitive('OFFSET'),
      caseInsensitive('DISTINCT'),
      caseInsensitive('ALL'),
      caseInsensitive('ASC'),
      caseInsensitive('DESC'),
      caseInsensitive('INTO'),
      caseInsensitive('VALUES'),
      caseInsensitive('SET'),
      caseInsensitive('TABLE'),
      caseInsensitive('TEMP'),
      caseInsensitive('TEMPORARY'),
      caseInsensitive('VIEW'),
      caseInsensitive('INDEX'),
      caseInsensitive('DATABASE'),
      caseInsensitive('SCHEMA')
    ),

    // VISUALISE/VISUALIZE [global_mapping] [FROM source] with clauses
    // Global mapping sets default aesthetics for all layers
    // FROM source can be an identifier (table/CTE) or string (file path)
    visualise_statement: $ => seq(
      choice(caseInsensitive('VISUALISE'), caseInsensitive('VISUALIZE')),
      optional($.global_mapping),
      optional(seq(caseInsensitive('FROM'), field('source', choice($.identifier, $.string)))),
      repeat($.viz_clause)
    ),

    // Global mapping: wildcard (*), or list of explicit/implicit mappings
    global_mapping: $ => choice(
      $.wildcard_mapping,
      seq(
        $.global_mapping_item,
        repeat(seq(',', $.global_mapping_item))
      )
    ),

    // Wildcard mapping: VISUALISE * (maps all columns implicitly)
    wildcard_mapping: $ => '*',

    // Global mapping item: explicit (col AS aes) or implicit (col)
    global_mapping_item: $ => choice(
      $.explicit_mapping,   // date AS x
      $.implicit_mapping    // x (becomes x AS x)
    ),

    // Explicit mapping: value AS aesthetic
    explicit_mapping: $ => seq(
      field('value', $.mapping_value),
      caseInsensitive('AS'),
      field('aesthetic', $.aesthetic_name)
    ),

    // Implicit mapping: just an identifier (column name = aesthetic name)
    implicit_mapping: $ => $.identifier,

    // All the visualization clauses (same as current grammar)
    viz_clause: $ => choice(
      $.draw_clause,
      $.scale_clause,
      $.facet_clause,
      $.coord_clause,
      $.label_clause,
      $.guide_clause,
      $.theme_clause,
    ),

    // DRAW clause - syntax: DRAW geom [MAPPING ...] [SETTING ...] [PARTITION BY ...] [FILTER ...]
    draw_clause: $ => seq(
      caseInsensitive('DRAW'),
      $.geom_type,
      optional($.mapping_clause),
      optional($.setting_clause),
      optional($.partition_clause),
      optional($.filter_clause)
    ),

    geom_type: $ => choice(
      'point', 'line', 'path', 'bar', 'col', 'area', 'tile', 'polygon', 'ribbon',
      'histogram', 'density', 'smooth', 'boxplot', 'violin',
      'text', 'label', 'segment', 'arrow', 'hline', 'vline', 'abline', 'errorbar'
    ),

    // MAPPING clause for aesthetic mappings: MAPPING col AS x, "blue" AS color
    mapping_clause: $ => seq(
      caseInsensitive('MAPPING'),
      $.mapping_item,
      repeat(seq(',', $.mapping_item))
    ),

    mapping_item: $ => seq(
      field('value', $.mapping_value),
      caseInsensitive('AS'),
      field('aesthetic', $.aesthetic_name)
    ),

    mapping_value: $ => choice(
      $.column_reference,
      $.literal_value
    ),

    // SETTING clause for parameters: SETTING opacity => 0.5, size => 3
    setting_clause: $ => seq(
      caseInsensitive('SETTING'),
      $.parameter_assignment,
      repeat(seq(',', $.parameter_assignment))
    ),

    parameter_assignment: $ => seq(
      field('param', $.parameter_name),
      caseInsensitive('=>'),
      field('value', $.parameter_value)
    ),

    parameter_name: $ => $.identifier,

    parameter_value: $ => choice(
      $.string,
      $.number,
      $.boolean
    ),

    // PARTITION BY clause for grouping: PARTITION BY category, region
    partition_clause: $ => seq(
      caseInsensitive('PARTITION'),
      caseInsensitive('BY'),
      $.partition_columns
    ),

    partition_columns: $ => seq(
      $.identifier,
      repeat(seq(',', $.identifier))
    ),

    // FILTER clause for layer filtering: FILTER x > 10 AND y < 20
    filter_clause: $ => seq(
      caseInsensitive('FILTER'),
      $.filter_expression
    ),

    filter_expression: $ => choice(
      $.filter_and_expression,
      $.filter_or_expression,
      $.filter_primary
    ),

    filter_primary: $ => choice(
      $.filter_comparison,
      seq('(', $.filter_expression, ')')
    ),

    filter_and_expression: $ => prec.left(2, seq(
      $.filter_primary,
      caseInsensitive('AND'),
      $.filter_expression
    )),

    filter_or_expression: $ => prec.left(1, seq(
      $.filter_primary,
      caseInsensitive('OR'),
      $.filter_expression
    )),

    filter_comparison: $ => seq(
      $.identifier,
      $.comparison_operator,
      choice($.string, $.number, $.boolean, $.identifier)
    ),

    // Basic comparison operators only
    comparison_operator: $ => choice('=', '!=', '<>', '<', '>', '<=', '>='),

    aesthetic_name: $ => choice(
      // Position aesthetics
      'x', 'y', 'xmin', 'xmax', 'ymin', 'ymax', 'xend', 'yend',
      // Color aesthetics
      'color', 'colour', 'fill', 'alpha',
      // Size and shape
      'size', 'shape', 'linetype', 'linewidth', 'width', 'height',
      // Text aesthetics
      'label', 'family', 'fontface', 'hjust', 'vjust'
    ),

    column_reference: $ => $.identifier,

    literal_value: $ => choice(
      $.string,
      $.number,
      $.boolean
    ),

    // SCALE clause - SCALE aesthetic SETTING prop => value, ...
    scale_clause: $ => seq(
      caseInsensitive('SCALE'),
      $.aesthetic_name,
      caseInsensitive('SETTING'),
      optional(seq(
        $.scale_property,
        repeat(seq(',', $.scale_property))
      ))
    ),

    scale_property: $ => seq(
      $.scale_property_name,
      caseInsensitive('=>'),
      $.scale_property_value
    ),

    scale_property_name: $ => choice(
      'type', 'limits', 'breaks', 'labels', 'expand',
      'direction', 'na_value', 'palette', 'domain', 'range'
    ),

    scale_property_value: $ => choice(
      $.string,
      $.number,
      $.boolean,
      $.array
    ),

    // FACET clause - FACET ... SETTING scales => ...
    facet_clause: $ => choice(
      // FACET row_vars BY col_vars
      seq(
        caseInsensitive('FACET'),
        $.facet_vars,
        alias(caseInsensitive('BY'), $.facet_by),
        $.facet_vars,
        optional(seq(caseInsensitive('SETTING'), caseInsensitive('scales'), caseInsensitive('=>'), $.facet_scales))
      ),
      // FACET WRAP vars
      seq(
        caseInsensitive('FACET'),
        alias(caseInsensitive('WRAP'), $.facet_wrap),
        $.facet_vars,
        optional(seq(caseInsensitive('SETTING'), caseInsensitive('scales'), caseInsensitive('=>'), $.facet_scales))
      )
    ),

    facet_wrap: $ => 'WRAP',
    facet_by: $ => 'BY',

    facet_vars: $ => seq(
      $.identifier,
      repeat(seq(',', $.identifier))
    ),

    facet_scales: $ => choice(
      'fixed', 'free', 'free_x', 'free_y'
    ),

    // COORD clause - COORD [type] [SETTING prop => value, ...]
    coord_clause: $ => seq(
      caseInsensitive('COORD'),
      choice(
        // Type with optional SETTING: COORD polar SETTING theta => y
        seq($.coord_type, optional(seq(caseInsensitive('SETTING'), $.coord_properties))),
        // Just SETTING: COORD SETTING xlim => [0, 100] (defaults to cartesian)
        seq(caseInsensitive('SETTING'), $.coord_properties)
      )
    ),

    coord_type: $ => choice(
      'cartesian', 'polar', 'flip', 'fixed', 'trans', 'map', 'quickmap'
    ),

    coord_properties: $ => seq(
      $.coord_property,
      repeat(seq(',', $.coord_property))
    ),

    coord_property: $ => seq(
      $.coord_property_name,
      caseInsensitive('=>'),
      choice($.string, $.number, $.boolean, $.array, $.identifier)
    ),

    coord_property_name: $ => choice(
      'xlim', 'ylim', 'ratio', 'theta', 'clip',
      // Also allow aesthetic names as properties (for domain specification)
      $.aesthetic_name
    ),

    // LABEL clause (repeatable)
    label_clause: $ => seq(
      caseInsensitive('LABEL'),
      optional(seq(
        $.label_assignment,
        repeat(seq(',', $.label_assignment))
      ))
    ),

    label_assignment: $ => seq(
      $.label_type,
      '=',
      $.string
    ),

    label_type: $ => choice(
      'title', 'subtitle', 'x', 'y', 'caption', 'tag',
      // Aesthetic names for legend titles
      'color', 'colour', 'fill', 'size', 'shape', 'linetype'
    ),

    // GUIDE clause - GUIDE aesthetic SETTING prop => value, ...
    guide_clause: $ => seq(
      caseInsensitive('GUIDE'),
      $.aesthetic_name,
      caseInsensitive('SETTING'),
      optional(seq(
        $.guide_property,
        repeat(seq(',', $.guide_property))
      ))
    ),

    guide_property: $ => choice(
      seq('type', caseInsensitive('=>'), $.guide_type),
      seq($.guide_property_name, caseInsensitive('=>'), choice($.string, $.number, $.boolean))
    ),

    guide_type: $ => choice(
      'legend', 'colorbar', 'axis', 'none'
    ),

    guide_property_name: $ => choice(
      'position', 'direction', 'nrow', 'ncol', 'title',
      'title_position', 'label_position', 'text_angle', 'text_size',
      'reverse', 'order'
    ),

    // THEME clause - THEME [name] [SETTING prop => value, ...]
    theme_clause: $ => choice(
      // Just theme name
      seq(caseInsensitive('THEME'), $.theme_name),
      // Theme name with properties
      seq(
        caseInsensitive('THEME'), $.theme_name, caseInsensitive('SETTING'),
        $.theme_property,
        repeat(seq(',', $.theme_property))
      ),
      // Just properties (custom theme)
      seq(
        caseInsensitive('THEME'), caseInsensitive('SETTING'),
        $.theme_property,
        repeat(seq(',', $.theme_property))
      )
    ),

    theme_name: $ => choice(
      'minimal', 'classic', 'gray', 'grey', 'bw', 'dark', 'light', 'void'
    ),

    theme_property: $ => seq(
      $.theme_property_name,
      caseInsensitive('=>'),
      choice($.string, $.number, $.boolean)
    ),

    theme_property_name: $ => choice(
      'background', 'panel_background', 'panel_grid', 'panel_grid_major',
      'panel_grid_minor', 'text_size', 'text_family', 'title_size',
      'axis_text_size', 'axis_line', 'axis_line_width', 'panel_border',
      'plot_margin', 'panel_spacing', 'legend_background', 'legend_position',
      'legend_direction'
    ),

    // Basic tokens
    identifier: $ => /[a-zA-Z_][a-zA-Z0-9_]*/,

    number: $ => token(seq(
      optional('-'),
      choice(
        /\d+/,
        /\d+\.\d*/,
        /\.\d+/
      )
    )),

    string: $ => choice(
      seq("'", repeat(choice(/[^'\\]/, seq('\\', /.*/))), "'"),
      seq('"', repeat(choice(/[^"\\]/, seq('\\', /.*/))), '"')
    ),

    boolean: $ => choice('true', 'false'),

    array: $ => seq(
      '[',
      optional(seq(
        $.array_element,
        repeat(seq(',', $.array_element))
      )),
      ']'
    ),

    array_element: $ => choice(
      $.string,
      $.number,
      $.boolean
    ),

    // Comments
    comment: $ => choice(
      seq('//', /.*/),
      seq('/*', /[^*]*\*+([^/*][^*]*\*+)*/, '/'),
      seq('--', /.*/),
    ),
  },

  extras: $ => [
    /\s+/,        // Whitespace
    $.comment,    // Comments
  ],

  word: $ => $.identifier,
});
