/**
 * Minimal ggsql grammar without external scanner
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
      repeat($.visualise_statement)
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
      $.from_clause,
      $.window_function,  // Window functions like ROW_NUMBER() OVER (...)
      $.cast_expression,  // CAST(expr AS type), TRY_CAST(expr AS type)
      $.function_call,    // Regular function calls like COUNT(), SUM()
      $.sql_keyword,
      $.string,
      $.number,
      ',', '*', '.', '=', '<', '>', '!', '+', '-', '/', '%', '|', '&', '^', '~', '::',
      $.subquery,
      $.identifier
    ))),

    // WITH statement (CTEs) - WITH must be followed by SELECT
    with_statement: $ => prec.right(2, seq(
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
      choice(
        $.with_statement,    // Allow nested CTEs
        $.select_statement
      ),
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
      )),
      optional($.select_statement)
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
    // Prioritizes WITH/SELECT statements, falls back to token-by-token parsing
    subquery: $ => prec(1, seq(
      '(',
      choice(
        $.with_statement,
        $.select_statement,
        $.subquery_body
      ),
      ')'
    )),

    // Token-by-token fallback for any other subquery content
    subquery_body: $ => repeat1(choice(
      $.window_function,
      $.cast_expression,
      $.function_call,
      $.sql_keyword,
      $.string,
      $.number,
      $.identifier,
      $.subquery,
      ',', '*', '.', '=', '<', '>', '!', '::',
      token(/[^\s;(),'\"]+/)
    )),

    // CAST/TRY_CAST expression: CAST(expr AS type) or TRY_CAST(expr AS type)
    // Higher precedence than function_call to win over treating CAST as a regular function
    cast_expression: $ => prec(3, seq(
      choice(caseInsensitive('CAST'), caseInsensitive('TRY_CAST')),
      '(',
      $.positional_arg,
      caseInsensitive('AS'),
      $.type_name,
      ')'
    )),

    // Type name for CAST expressions: DATE, VARCHAR, DECIMAL(10,2), etc.
    type_name: $ => seq(
      $.identifier,
      optional(seq('(', $.number, optional(seq(',', $.number)), ')'))
    ),

    // Function call with parentheses (can be empty like ROW_NUMBER())
    // Used in window functions and general SQL
    function_call: $ => prec(2, seq(
      $.identifier,
      '(',
      optional($.function_args),
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
      caseInsensitive('LATERAL'),
      caseInsensitive('CROSS'),
      caseInsensitive('NATURAL'),
      caseInsensitive('FULL'),
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
      caseInsensitive('SCHEMA'),
      caseInsensitive('OVER'),
      caseInsensitive('ROWS'),
      caseInsensitive('RANGE'),
      caseInsensitive('UNBOUNDED'),
      caseInsensitive('PRECEDING'),
      caseInsensitive('FOLLOWING'),
      caseInsensitive('CURRENT'),
      caseInsensitive('ROW'),
      caseInsensitive('NULLS'),
      caseInsensitive('FIRST'),
      caseInsensitive('LAST')
    ),

    // Window function: func() OVER (PARTITION BY ... ORDER BY ... frame)
    // Higher precedence to match before generic function_call
    window_function: $ => prec(4, seq(
      field('function', $.identifier),
      '(',
      optional($.function_args),
      ')',
      caseInsensitive('OVER'),
      $.window_specification
    )),

    function_args: $ => seq(
      $.function_arg,
      repeat(seq(',', $.function_arg))
    ),

    // Function argument: positional or named
    function_arg: $ => choice(
      $.named_arg,
      $.positional_arg
    ),

    named_arg: $ => seq(
      field('name', $.identifier),
      choice(':=', '=>'),
      field('value', $.positional_arg)
    ),

    // Positional argument: supports complex expressions including:
    // - Simple values: identifier, number, string, *
    // - Qualified names: table.column
    // - Nested function calls: ROUND(AVG(x), 2)
    // - Arithmetic expressions: quantity * price
    // - Type casts: value::type
    positional_arg: $ => prec.left(choice(
      // Simple values
      $.qualified_name,  // Handles both simple identifiers and table.column
      $.number,
      $.string,
      '*',
      // CAST/TRY_CAST expression
      $.cast_expression,
      // Nested function call
      $.function_call,
      // Arithmetic/comparison expression (binary operators)
      seq($.positional_arg, choice('+', '-', '*', '/', '%', '||', '::', '<', '>', '<=', '>=', '=', '!=', '<>'), $.positional_arg),
      // Parenthesized expression
      seq('(', $.positional_arg, ')')
    )),

    // Namespaced identifier: matches "namespace:name" pattern
    // Examples: ggsql:penguins, ggsql:airquality
    namespaced_identifier: $ => {
      const pattern = /[a-zA-Z_][a-zA-Z0-9_]*:[a-zA-Z_][a-zA-Z0-9_]*/;
      return token(choice(
        pattern,
        seq('`', pattern, '`'),
        seq('"', pattern, '"')
      ));
    },

    window_specification: $ => seq(
      '(',
      optional($.window_partition_clause),
      optional($.window_order_clause),
      optional($.frame_clause),
      ')'
    ),

    window_partition_clause: $ => seq(
      caseInsensitive('PARTITION'),
      caseInsensitive('BY'),
      $.identifier,
      repeat(seq(',', $.identifier))
    ),

    window_order_clause: $ => seq(
      caseInsensitive('ORDER'),
      caseInsensitive('BY'),
      $.order_item,
      repeat(seq(',', $.order_item))
    ),

    order_item: $ => seq(
      $.identifier,
      optional(choice(caseInsensitive('ASC'), caseInsensitive('DESC'))),
      optional(seq(caseInsensitive('NULLS'), choice(caseInsensitive('FIRST'), caseInsensitive('LAST'))))
    ),

    frame_clause: $ => seq(
      choice(caseInsensitive('ROWS'), caseInsensitive('RANGE')),
      choice(
        seq(caseInsensitive('BETWEEN'), $.frame_bound, caseInsensitive('AND'), $.frame_bound),
        $.frame_bound
      )
    ),

    frame_bound: $ => choice(
      seq(caseInsensitive('UNBOUNDED'), choice(caseInsensitive('PRECEDING'), caseInsensitive('FOLLOWING'))),
      seq(caseInsensitive('CURRENT'), caseInsensitive('ROW')),
      seq($.number, choice(caseInsensitive('PRECEDING'), caseInsensitive('FOLLOWING')))
    ),

    // Dotted identifier (for catalog.schema.table)
    qualified_name: $ => prec.right(seq(
      $.identifier,
      repeat(seq('.', $.identifier))
    )),

    table_ref: $ => prec.right(seq(
      choice(
        field('table', choice($.qualified_name, $.string, $.namespaced_identifier)),
        $.subquery,
      ),
      optional(seq(
        optional(caseInsensitive('AS')),
        field('alias', $.identifier)
      ))
    )),

    from_clause: $ => prec.right(1, seq(
      caseInsensitive('FROM'),
      $.table_ref,
      repeat(seq(',', $.table_ref))
    )),

    // VISUALISE/VISUALIZE [global_mapping] [FROM source] with clauses
    // Global mapping sets default aesthetics for all layers
    // FROM source can be an identifier (table/CTE) or string (file path)
    visualise_statement: $ => prec.dynamic(1, seq(
      $.visualise_keyword,
      optional($.global_mapping),
      optional($.from_clause),
      repeat($.viz_clause)
    )),

    // VISUALISE keyword as explicit high-precedence token
    visualise_keyword: $ => token(prec(10, choice(
      caseInsensitive("VISUALISE"),
      caseInsensitive("VISUALIZE")
    ))),

    // Shared mapping list: comma-separated mapping elements
    // Used by both global (VISUALISE) and layer (MAPPING) mappings
    mapping_list: $ => seq(
      $.mapping_element,
      repeat(seq(',', $.mapping_element))
    ),

    // Mapping element: wildcard, explicit, or implicit
    mapping_element: $ => choice(
      $.wildcard_mapping,   // *
      $.explicit_mapping,   // date AS x
      $.implicit_mapping    // x (becomes x AS x)
    ),

    // Wildcard mapping: maps all columns to aesthetics with matching names
    wildcard_mapping: $ => '*',

    // Explicit mapping: value AS aesthetic (name)
    explicit_mapping: $ => seq(
      field('value', $.mapping_value),
      caseInsensitive('AS'),
      field('name', $.aesthetic_name)
    ),

    // Implicit mapping: just an identifier (column name = aesthetic name)
    implicit_mapping: $ => $.identifier,

    // Global mapping after VISUALISE - uses shared mapping_list
    global_mapping: $ => $.mapping_list,

    // All the visualization clauses (same as current grammar)
    viz_clause: $ => choice(
      $.draw_clause,
      $.scale_clause,
      $.facet_clause,
      $.coord_clause,
      $.label_clause,
      $.theme_clause,
    ),

    // DRAW clause - syntax: DRAW geom [MAPPING ...] [REMAPPING ...] [SETTING ...] [FILTER ...] [PARTITION BY ...] [ORDER BY ...]
    draw_clause: $ => seq(
      caseInsensitive('DRAW'),
      $.geom_type,
      optional($.mapping_clause),
      optional($.remapping_clause),
      optional($.setting_clause),
      optional($.filter_clause),
      optional($.partition_clause),
      optional($.order_clause)
    ),

    // REMAPPING clause: maps stat-computed columns to aesthetics
    // Syntax: REMAPPING count AS y, sum AS size
    // Reuses mapping_list for parsing - stat names are treated as column references
    remapping_clause: $ => seq(
      caseInsensitive('REMAPPING'),
      $.mapping_list
    ),

    geom_type: $ => choice(
      'point', 'line', 'path', 'bar', 'area', 'tile', 'polygon', 'ribbon',
      'histogram', 'density', 'smooth', 'boxplot', 'violin',
      'text', 'label', 'segment', 'arrow', 'hline', 'vline', 'abline', 'errorbar'
    ),

    // MAPPING clause for aesthetic mappings: MAPPING col AS x, "blue" AS color [FROM source]
    // Supports: MAPPING x AS x, y AS y FROM cte
    //           MAPPING FROM cte (inherits global mappings)
    //           MAPPING * (wildcard)
    //           MAPPING *, x AS color (wildcard with explicit)
    //           MAPPING x, y (implicit mappings)
    // Requires at least one of: aesthetic mappings or FROM clause
    mapping_clause: $ => seq(
      caseInsensitive('MAPPING'),
      choice(
        // Option 1: Just FROM (inherit global mappings)
        seq(
          caseInsensitive('FROM'),
          field('layer_source', choice($.qualified_name, $.string, $.namespaced_identifier))
        ),
        // Option 2: Mapping list (uses shared structure), optionally followed by FROM
        seq(
          $.mapping_list,
          optional(seq(
            caseInsensitive('FROM'),
            field('layer_source', choice($.qualified_name, $.string, $.namespaced_identifier))
          ))
        )
      )
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
      field('name', $.parameter_name),
      '=>',
      field('value', $.parameter_value)
    ),

    parameter_name: $ => $.identifier,

    parameter_value: $ => choice(
      $.string,
      $.number,
      $.boolean,
      $.null_literal,
      $.array
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

    // FILTER clause for layer filtering: FILTER <raw SQL WHERE expression>
    // The filter_expression captures any valid SQL WHERE clause verbatim
    // and passes it to the database backend
    filter_clause: $ => seq(
      caseInsensitive('FILTER'),
      $.filter_expression
    ),

    // Raw SQL expression - captures everything that's valid in a WHERE clause
    // Uses prec.right to greedily consume tokens until a clause keyword is hit
    filter_expression: $ => prec.right(repeat1($.filter_token)),

    // Individual tokens that can appear in a filter expression
    // NOTE: This must NOT match PARTITION or ORDER as identifiers, since those
    // keywords start subsequent clauses in draw_clause
    filter_token: $ => choice(
      // SQL keywords commonly used in WHERE clauses
      caseInsensitive('AND'),
      caseInsensitive('OR'),
      caseInsensitive('NOT'),
      caseInsensitive('IN'),
      caseInsensitive('IS'),
      caseInsensitive('NULL'),
      caseInsensitive('LIKE'),
      caseInsensitive('ILIKE'),
      caseInsensitive('BETWEEN'),
      caseInsensitive('EXISTS'),
      caseInsensitive('ANY'),
      caseInsensitive('ALL'),
      caseInsensitive('CASE'),
      caseInsensitive('WHEN'),
      caseInsensitive('THEN'),
      caseInsensitive('ELSE'),
      caseInsensitive('END'),
      caseInsensitive('CAST'),
      caseInsensitive('AS'),
      caseInsensitive('TRUE'),
      caseInsensitive('FALSE'),
      // Values and identifiers (lower precedence to allow keywords to take priority)
      $.string,
      $.number,
      $.filter_identifier,
      // Comparison operators (as explicit tokens)
      token('='),
      token('!='),
      token('<>'),
      token('<='),
      token('>='),
      token('<'),
      token('>'),
      // Regex operators (DuckDB/PostgreSQL)
      token('~*'),   // case-insensitive regex match
      token('!~*'),  // case-insensitive regex not match
      token('!~'),   // regex not match
      token('~'),    // regex match
      // Arithmetic operators
      token('+'),
      token('-'),
      token('*'),
      token('/'),
      token('%'),
      token('||'),
      // Type cast operator (PostgreSQL style)
      token('::'),
      // Parentheses for grouping
      token('('),
      token(')'),
      token(','),
      token('.')
    ),

    // ORDER BY clause for layer sorting: ORDER BY date ASC, value DESC
    order_clause: $ => seq(
      caseInsensitive('ORDER'),
      caseInsensitive('BY'),
      $.order_expression
    ),

    // Raw SQL ORDER BY expression - captures column names and sort directions
    order_expression: $ => prec.right(repeat1($.order_token)),

    // Individual tokens that can appear in an order expression
    order_token: $ => choice(
      $.identifier,
      $.number,
      caseInsensitive('ASC'),
      caseInsensitive('DESC'),
      caseInsensitive('NULLS'),
      caseInsensitive('FIRST'),
      caseInsensitive('LAST'),
      ',',
      '.',
      '(',
      ')'
    ),

    aesthetic_name: $ => choice(
      // Position aesthetics
      'x', 'y', 'xmin', 'xmax', 'ymin', 'ymax', 'xend', 'yend',
      // Aggregation aesthetic (for bar charts)
      'weight',
      // Color aesthetics
      'color', 'colour', 'fill', 'stroke', 'opacity',
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

    // SCALE clause - SCALE [TYPE] aesthetic [FROM ...] [TO ...] [VIA ...] [SETTING ...] [RENAMING ...]
    // Examples:
    //   SCALE DATE x
    //   SCALE CONTINUOUS y FROM [0, 100]
    //   SCALE DISCRETE color FROM ['A', 'B'] TO ['red', 'blue']
    //   SCALE color TO viridis
    //   SCALE x FROM [0, 100] SETTING breaks => '1 month'
    //   SCALE DISCRETE x RENAMING 'A' => 'Alpha', 'B' => 'Beta'
    scale_clause: $ => seq(
      caseInsensitive('SCALE'),
      optional($.scale_type_identifier),  // optional type before aesthetic
      $.aesthetic_name,
      optional($.scale_from_clause),
      optional($.scale_to_clause),
      optional($.scale_via_clause),
      optional($.setting_clause),  // reuse existing setting_clause from DRAW
      optional($.scale_renaming_clause)  // custom label mappings
    ),

    // RENAMING clause for custom axis/legend labels
    // Syntax: RENAMING 'A' => 'Alpha', 'B' => 'Beta', 'C' => NULL
    scale_renaming_clause: $ => seq(
      caseInsensitive('RENAMING'),
      $.renaming_assignment,
      repeat(seq(',', $.renaming_assignment))
    ),

    renaming_assignment: $ => seq(
      field('name', choice(
        '*',                              // Wildcard for template
        $.string,
        $.number
      )),
      '=>',
      field('value', choice($.string, $.null_literal))  // String label or NULL to suppress
    ),

    // Scale types - describe the nature of the data
    scale_type_identifier: $ => choice(
      caseInsensitive('CONTINUOUS'),  // continuous numeric data
      caseInsensitive('DISCRETE'),    // categorical/discrete data
      caseInsensitive('BINNED'),      // binned/bucketed data
      caseInsensitive('ORDINAL'),     // ordered categorical data with interpolated output
      caseInsensitive('IDENTITY')     // pass-through scale (data already in output format)
    ),

    // FROM clause - input range specification
    scale_from_clause: $ => seq(
      caseInsensitive('FROM'),
      $.array
    ),

    // TO clause - output range (explicit array or named palette)
    scale_to_clause: $ => seq(
      caseInsensitive('TO'),
      choice(
        $.array,      // ['red', 'blue'] - explicit values
        $.identifier  // viridis - named palette
      )
    ),

    // VIA clause - transformation method
    scale_via_clause: $ => seq(
      caseInsensitive('VIA'),
      $.identifier
    ),

    // FACET clause - FACET ... SETTING scales => ...
    facet_clause: $ => choice(
      // FACET row_vars BY col_vars
      seq(
        caseInsensitive('FACET'),
        $.facet_vars,
        alias(caseInsensitive('BY'), $.facet_by),
        $.facet_vars,
        optional(seq(caseInsensitive('SETTING'), caseInsensitive('scales'), '=>', $.facet_scales))
      ),
      // FACET WRAP vars
      seq(
        caseInsensitive('FACET'),
        alias(caseInsensitive('WRAP'), $.facet_wrap),
        $.facet_vars,
        optional(seq(caseInsensitive('SETTING'), caseInsensitive('scales'), '=>', $.facet_scales))
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
      field('name', $.coord_property_name),
      '=>',
      field('value', choice($.string, $.number, $.boolean, $.array, $.identifier))
    ),

    coord_property_name: $ => choice(
      'xlim', 'ylim', 'ratio', 'theta', 'clip',
      // Also allow aesthetic names as properties (for range specification)
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
      field('name', $.label_type),
      '=>',
      field('value', $.string)
    ),

    label_type: $ => choice(
      'title', 'subtitle', 'x', 'y', 'caption', 'tag',
      // Aesthetic names for legend titles
      'color', 'colour', 'fill', 'size', 'shape', 'linetype'
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
      field('name', $.theme_property_name),
      '=>',
      field('value', choice($.string, $.number, $.boolean))
    ),

    theme_property_name: $ => choice(
      'background', 'panel_background', 'panel_grid', 'panel_grid_major',
      'panel_grid_minor', 'text_size', 'text_family', 'title_size',
      'axis_text_size', 'axis_line', 'axis_line_width', 'panel_border',
      'plot_margin', 'panel_spacing', 'legend_background', 'legend_position',
      'legend_direction'
    ),

    // Basic tokens
    bare_identifier: $ => token(/[a-zA-Z_][a-zA-Z0-9_]*/),
    quoted_identifier: $ => token(choice(
      seq('`', /[^`]+/, '`'),
      seq('"', /[^"]+/, '"')
    )),

    identifier: $ => choice(
      $.bare_identifier,
      $.quoted_identifier
    ),

    // Identifier for use in filter expressions - uses lower precedence so that
    // keywords like PARTITION and ORDER can take priority and end the filter
    filter_identifier: $ => token(prec(-1, /[a-zA-Z_][a-zA-Z0-9_]*/)),

    number: $ => token(seq(
      optional('-'),
      choice(
        /\d+/,
        /\d+\.\d*/,
        /\.\d+/
      )
    )),

    string: $ => seq("'", repeat(choice(/[^'\\]/, seq('\\', /.*/))), "'"),

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
      $.boolean,
      $.null_literal
    ),

    null_literal: $ => caseInsensitive('NULL'),

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

  word: $ => $.bare_identifier,
});
