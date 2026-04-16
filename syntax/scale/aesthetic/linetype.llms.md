# Linetype

The `linetype` aesthetic defines the stroke pattern of lines: a succession of “on” and “off” parts of the line to make different stipple patterns. Linetype is most useful for [line](../../../syntax/layer/type/line.llms.md) and [path](../../../syntax/layer/type/path.llms.md) layers as a colorless way to differentiate between categories.

While linetypes are most useful for differentiating between discrete data, ggsql also comes with a sequential palette that gradually increases the amount of ink used for the pattern.

## Literal values

Linetypes can be specified in two different ways. A couple of patterns are named and can be referred to as such:

| Name | Pattern | Stroke-dasharray | Description |
|----|----|----|----|
| `solid` | [![](examples/linetype_solid.svg)](examples/linetype_solid.svg) | (none) | Continuous solid line |
| `dashed` | [![](examples/linetype_dashed.svg)](examples/linetype_dashed.svg) | 6 4 | Standard dashed line |
| `dotted` | [![](examples/linetype_dotted.svg)](examples/linetype_dotted.svg) | 1 2 | Dotted line |
| `dotdash` | [![](examples/linetype_dotdash.svg)](examples/linetype_dotdash.svg) | 1 2 6 2 | Alternating dot and dash |
| `longdash` | [![](examples/linetype_longdash.svg)](examples/linetype_longdash.svg) | 10 4 | Long dashes |
| `twodash` | [![](examples/linetype_twodash.svg)](examples/linetype_twodash.svg) | 6 2 2 2 | Two different dash lengths |

You can e.g. use these names when defining a manual palette for the scale:

``` ggsql
SCALE linetype TO ['dashed', 'dotted', 'twodash']
```

You can alternatively specify a custom pattern using hex strings:

``` ggsql
SCALE linetype TO ['44', '1343', '3c6c9c']
```

Each pair of hex digits represents on/off lengths:

- `'44'` = 4 on, 4 off (50% ink, similar to dashed)
- `'1343'` = 1 on, 3 off, 4 on, 3 off (dot-dash pattern)
- `'3c6c9c'` = 3 on, 12 off, 6 on, 12 off, 9 on, 12 off

Valid hex patterns:

- Must have 2, 4, 6, or 8 hex digits
- Digits can be 1-9 or a-f (0 is not allowed)
- Digits represent line lengths as multiples of the linewidth

## Palettes

ggsql provides two linetype palettes which are generally enough for every need

The `categorical` palette is the default palette for discrete linetype scales. It consists of the 6 named patterns [shown above](#literal-values) in the same order. Since it is the only palette for discrete linetypes, there is rarely a need to specify it. The `categorical` palette is the default palette for discrete linetype scales. It consists of the 6 named patterns [shown above](#literal-values) in the same order. Since it is the only palette for discrete linetypes there is rarely a need to specify it.

### Sequential palette

The `sequential` palette is the default for binned and ordinal linetype scales. It consists of up to 15 patterns with increasing amount of “on” and decreasing amount of “off”. This creates a visual progression from sparse (low ink) to solid (100% ink).

#### Example: 5-level sequential

| Level | Pattern | Ink Density | Description |
|----|----|----|----|
| 1 | [![](examples/linetype_seq_1.svg)](examples/linetype_seq_1.svg) | ~6% | Sparse dots |
| 2 | [![](examples/linetype_seq_2.svg)](examples/linetype_seq_2.svg) | ~29% | Short dashes |
| 3 | [![](examples/linetype_seq_3.svg)](examples/linetype_seq_3.svg) | ~52% | Medium dashes |
| 4 | [![](examples/linetype_seq_4.svg)](examples/linetype_seq_4.svg) | ~75% | Long dashes |
| 5 | [![](examples/linetype_solid.svg)](examples/linetype_solid.svg) | 100% | Solid |

## Accessibility

- Linetypes work well for colorblind viewers
- Combine with color for redundant encoding
- Limit to 4-6 distinct linetypes for readability
- Solid lines are most visible; use for primary data
