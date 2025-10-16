/**
 * Minimal VizQL grammar without external scanner
 *
 * Uses a simple regex to capture SQL portion as opaque text
 */

module.exports = grammar({
  name: 'vizql',

  rules: {
    // Main entry point - supports multiple VISUALISE/VISUALIZE statements
    // The SQL portion will be handled by the calling code (splitter)
    query: $ => repeat1($.visualise_statement),

    // VISUALISE/VISUALIZE AS <type> with clauses
    visualise_statement: $ => seq(
      choice('VISUALISE', 'VISUALIZE'),
      'AS',
      $.viz_type,
      repeat($.viz_clause)
    ),

    // Visualization output types
    viz_type: $ => choice(
      'PLOT',
      'TABLE',
      'MAP'
    ),

    // All the visualization clauses (same as current grammar)
    viz_clause: $ => choice(
      $.with_clause,
      $.scale_clause,
      $.facet_clause,
      $.coord_clause,
      $.label_clause,
      $.guide_clause,
      $.theme_clause,
    ),

    // WITH clause
    with_clause: $ => seq(
      'WITH',
      $.geom_type,
      'USING',
      $.aesthetic_mapping,
      repeat(seq(',', $.aesthetic_mapping)),
      optional(seq('AS', $.identifier))
    ),

    geom_type: $ => choice(
      'point', 'line', 'path', 'bar', 'col', 'area', 'tile', 'polygon', 'ribbon',
      'histogram', 'density', 'smooth', 'boxplot', 'violin',
      'text', 'label', 'segment', 'arrow', 'hline', 'vline', 'abline', 'errorbar'
    ),

    aesthetic_mapping: $ => seq(
      field('aesthetic', $.aesthetic_name),
      '=',
      field('value', $.aesthetic_value)
    ),

    aesthetic_name: $ => choice(
      // Position aesthetics
      'x', 'y', 'xmin', 'xmax', 'ymin', 'ymax', 'xend', 'yend',
      // Color aesthetics
      'color', 'colour', 'fill', 'alpha',
      // Size and shape
      'size', 'shape', 'linetype', 'linewidth', 'width', 'height',
      // Text aesthetics
      'label', 'family', 'fontface', 'hjust', 'vjust',
      // Grouping
      'group'
    ),

    aesthetic_value: $ => choice(
      $.column_reference,
      $.literal_value
    ),

    column_reference: $ => $.identifier,

    literal_value: $ => choice(
      $.string,
      $.number,
      $.boolean
    ),

    // SCALE clause
    scale_clause: $ => seq(
      'SCALE',
      $.aesthetic_name,
      'USING',
      optional(seq(
        $.scale_property,
        repeat(seq(',', $.scale_property))
      ))
    ),

    scale_property: $ => seq(
      $.scale_property_name,
      '=',
      $.scale_property_value
    ),

    scale_property_name: $ => choice(
      'type', 'limits', 'breaks', 'labels', 'expand',
      'direction', 'na_value', 'palette'
    ),

    scale_property_value: $ => choice(
      $.string,
      $.number,
      $.boolean,
      $.array
    ),

    // FACET clause
    facet_clause: $ => choice(
      // FACET row_vars BY col_vars
      seq(
        'FACET',
        $.facet_vars,
        'BY',
        $.facet_vars,
        optional(seq('USING', 'scales', '=', $.facet_scales))
      ),
      // FACET WRAP vars
      seq(
        'FACET', 'WRAP',
        $.facet_vars,
        optional(seq('USING', 'scales', '=', $.facet_scales))
      )
    ),

    facet_vars: $ => seq(
      $.identifier,
      repeat(seq(',', $.identifier))
    ),

    facet_scales: $ => choice(
      'fixed', 'free', 'free_x', 'free_y'
    ),

    // COORD clause
    coord_clause: $ => seq(
      'COORD', 'USING',
      'type', '=', $.coord_type,
      repeat(seq(',', $.coord_property))
    ),

    coord_type: $ => choice(
      'cartesian', 'polar', 'flip', 'fixed', 'trans', 'map', 'quickmap'
    ),

    coord_property: $ => seq(
      $.coord_property_name,
      '=',
      choice($.string, $.number, $.boolean, $.array)
    ),

    coord_property_name: $ => choice(
      'xlim', 'ylim', 'ratio', 'theta', 'clip'
    ),

    // LABEL clause (repeatable)
    label_clause: $ => seq(
      'LABEL',
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

    // GUIDE clause
    guide_clause: $ => seq(
      'GUIDE',
      $.aesthetic_name,
      'USING',
      optional(seq(
        $.guide_property,
        repeat(seq(',', $.guide_property))
      ))
    ),

    guide_property: $ => choice(
      seq('type', '=', $.guide_type),
      seq($.guide_property_name, '=', choice($.string, $.number, $.boolean))
    ),

    guide_type: $ => choice(
      'legend', 'colorbar', 'axis', 'none'
    ),

    guide_property_name: $ => choice(
      'position', 'direction', 'nrow', 'ncol', 'title',
      'title_position', 'label_position', 'text_angle', 'text_size',
      'reverse', 'order'
    ),

    // THEME clause
    theme_clause: $ => choice(
      // Just theme name
      seq('THEME', $.theme_name),
      // Theme name with properties
      seq(
        'THEME', $.theme_name, 'USING',
        $.theme_property,
        repeat(seq(',', $.theme_property))
      ),
      // Just properties (custom theme)
      seq(
        'THEME', 'USING',
        $.theme_property,
        repeat(seq(',', $.theme_property))
      )
    ),

    theme_name: $ => choice(
      'minimal', 'classic', 'gray', 'grey', 'bw', 'dark', 'light', 'void'
    ),

    theme_property: $ => seq(
      $.theme_property_name,
      '=',
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

    number: $ => choice(
      /\d+/,
      /\d+\.\d*/,
      /\.\d+/
    ),

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