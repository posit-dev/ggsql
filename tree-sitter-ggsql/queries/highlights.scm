; Tree-sitter highlighting queries for ggsql

; Note: Keywords are case-insensitive via regex patterns in grammar.js,
; so we can't match them directly in highlight queries. Instead, we rely
; on structural matching (e.g., visualise_statement node type) for semantic
; highlighting.

; Geom types
[
  "point"
  "line"
  "path"
  "bar"
  "area"
  "tile"
  "polygon"
  "ribbon"
  "histogram"
  "density"
  "smooth"
  "boxplot"
  "violin"
  "text"
  "label"
  "segment"
  "arrow"
  "hline"
  "vline"
  "abline"
  "errorbar"
] @type.builtin

; Aesthetic names
[
  "x"
  "y"
  "weight"
  "xmin"
  "xmax"
  "ymin"
  "ymax"
  "xend"
  "yend"
  "color"
  "colour"
  "fill"
  "opacity"
  "size"
  "shape"
  "linetype"
  "linewidth"
  "width"
  "height"
  "label"
  "family"
  "fontface"
  "hjust"
  "vjust"
] @attribute

; String literals
(string) @string

; Numbers
(number) @number

; Booleans
(boolean) @constant.builtin

; Comments
(comment) @comment

; Identifiers (column references)
(column_reference) @variable

; Property names
(scale_property_name) @property
(coord_property_name) @property
(guide_property_name) @property
(theme_property_name) @property
(label_type) @property

; Operators
"=" @operator
"!=" @operator
"<>" @operator
"<" @operator
">" @operator
"<=" @operator
">=" @operator
"~" @operator
"~*" @operator
"!~" @operator
"!~*" @operator
"::" @operator
"||" @operator

; Punctuation
["," "[" "]" "(" ")"] @punctuation.delimiter

; Parameter names (in SETTING clause)
(parameter_name) @variable.parameter
