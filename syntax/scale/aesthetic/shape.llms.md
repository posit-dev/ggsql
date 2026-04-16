# Shape

The `shape` aesthetic governs how points look. Shape can be used instead of color to differentiate between categories of points (or even better, in concert with color). Shape is only meaningful in relation to discrete data.

## Literal values

Shapes are given by name from the following list of 15 possible shapes

| Name | Shape | Description |
|----|----|----|
| `circle` | [![](examples/shape_circle.svg)](examples/shape_circle.svg) | Circular point (default) |
| `square` | [![](examples/shape_square.svg)](examples/shape_square.svg) | Square point |
| `diamond` | [![](examples/shape_diamond.svg)](examples/shape_diamond.svg) | Diamond/rhombus |
| `triangle-up` | [![](examples/shape_triangle_up.svg)](examples/shape_triangle_up.svg) | Upward-pointing triangle |
| `triangle-down` | [![](examples/shape_triangle_down.svg)](examples/shape_triangle_down.svg) | Downward-pointing triangle |
| `star` | [![](examples/shape_star.svg)](examples/shape_star.svg) | 5-pointed star |
| `square-cross` | [![](examples/shape_square_cross.svg)](examples/shape_square_cross.svg) | Square with X cutout |
| `circle-plus` | [![](examples/shape_circle_plus.svg)](examples/shape_circle_plus.svg) | Circle with + cutout |
| `square-plus` | [![](examples/shape_square_plus.svg)](examples/shape_square_plus.svg) | Square with + cutout |
| `cross` | [![](examples/shape_cross.svg)](examples/shape_cross.svg) | X shape |
| `plus` | [![](examples/shape_plus.svg)](examples/shape_plus.svg) | \+ shape |
| `asterisk` | [![](examples/shape_asterisk.svg)](examples/shape_asterisk.svg) | 6-pointed asterisk |
| `bowtie` | [![](examples/shape_bowtie.svg)](examples/shape_bowtie.svg) | Bowtie/hourglass |
| `hline` | [![](examples/shape_hline.svg)](examples/shape_hline.svg) | Horizontal line |
| `vline` | [![](examples/shape_vline.svg)](examples/shape_vline.svg) | Vertical line |

You can use these names directly when setting the shape of a point:

``` ggsql
DRAW point
  SETTING shape => 'star'
```

## Palettes

ggsql provides three built-in shape palettes, which is often all you need.

Palettes are used by giving them as names in the `TO` clause:

``` ggsql
VISUALISE FROM ggsql:penguins
DRAW point
  MAPPING bill_dep AS x, body_mass AS y, species AS shape
  SETTING linewidth => 1, size => 5
SCALE shape TO open
```

Instead of using a named palette you can create one on the fly using an array of shape names:

``` ggsql
VISUALISE FROM ggsql:penguins
DRAW point
  MAPPING bill_dep AS x, body_mass AS y, species AS shape
  SETTING linewidth => 1, size => 5
SCALE shape TO ['star', 'bowtie', 'square-plus']
```

### Default palette (`closed`)

The default palette contains the 9 closed shapes (first nine in the table above). This is the recommended palette for most use cases, as closed shapes are more visually prominent and easier to distinguish at small sizes.

While the closed shapes are most often used filled, you can also turn if fill and only draw the stroke for a lighter look.

### Open palette (`open`)

The `open` palette contains the last 6 shapes in the table. None of these have a fill. You may use this palette when you want transparent shapes that don’t obscure data.

### All shapes palette (`shapes`)

This palette combines the two palettes above to provide all the possible shapes. Since not all shapes have a fill you should not map fill to anything if using this palette, and you should also be aware that differentiating 15 different shapes in the same plot will require a lot of mental effort from the viewer.

### Accessibility considerations

- Limit to 6-7 distinct shapes for readability
- Combine with color for redundant encoding
- Use larger point sizes when using complex shapes
- Closed shapes are more visible than open shapes
