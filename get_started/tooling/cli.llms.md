# Command line interface

The `ggsql` command line interface lets you execute queries and validate syntax directly from the terminal. While it may not be the most ergonomic way to interact directly with ggsql, it is useful for scripting, automation, and building tools around ggsql.

## Installation

Install the ggsql CLI on your system using the standard [installation instructions](../installation.llms.md).

## Executing a query

You can execute a ggsql query from a string with `ggsql exec`:

``` bash
ggsql exec "VISUALISE species AS fill FROM ggsql:penguins DRAW bar"
```

Or run a `.gsql` file:

``` bash
ggsql run my_query.gsql
```

In both cases, output is written to `stdout` as a Vega-Lite JSON spec. You can redirect it to a file:

``` bash
ggsql run my_query.gsql > chart.vl.json
```

Such files can be rendered as images using tools that work with Vega-Lite specs, such as the [Online Vega Editor](https://vega.github.io/editor/) or [vl-convert](https://github.com/vega/vl-convert) command line tool. For example,

``` bash
vl-convert vl2png -i chart.vl.json -o chart.png
```

`ggsql` can also render an image itself, without a second tool — see [Output format](#output-format) below.

A standard SQL query can also be provided to ggsql. If the query returns a table, the resulting values will be written to `stdout`.

## Validating a query

If you only want to check that a query is syntactically valid without executing it, use `ggsql validate`:

``` bash
$ ggsql validate "VISUALISE x, y FROM table DRAW point"
✓ Query syntax is valid
```

## Database connections

`ggsql exec`, `ggsql run` and `ggsql view` all accept a `--reader` flag (short `-r`) that can be used to specify a connection string to be used when executing the query. If not provided, ggsql will use an empty in-memory duckdb connection, equivalent to `--reader duckdb://memory`.

``` bash
$ ggsql exec --reader sqlite://sample/ggsql_test.sqlite \
  "SELECT * FROM test_table LIMIT 3"
col_a,  col_b, col_c
215.87, 75.11, delta
418.78, 71.75, delta
495.75, 12.55, delta

$ ggsql exec --reader odbc://DSN=ggsql-pg-test \
  "SELECT * FROM test_table LIMIT 3"
col_a,  col_b, col_c
319.34, 91.45, gamma
299.08, 49.36, epsilon
12.5,   29.48, gamma
```

### Caching reads

A remote or write-constrained database can be wrapped in an in-memory cache with `--cache`, naming the backend to cache into (`duckdb` or `sqlite`). It is off by default. Repeated reads of the same query — several layers over one table, or the same plot re-run while you iterate on it — are then served from memory instead of the remote:

``` bash
$ ggsql view --reader odbc://DSN=ggsql-pg-test --cache duckdb \
  "SELECT * FROM test_table VISUALISE col_a AS x, col_b AS y DRAW point"
```

The same thing can be spelled as a composite connection string, `<cache>+<primary>://…`, which is what `--cache` is rewritten to:

``` bash
$ ggsql exec --reader duckdb+odbc://DSN=ggsql-pg-test \
  "SELECT * FROM test_table LIMIT 3"
```

The two forms cannot be combined — there would be no saying which cache was meant.

## Output format

`ggsql exec` and `ggsql run` render with the writer named by `--writer` (short `-w`), defaulting to `--writer vegalite` — the Vega-Lite JSON above. A standard build also writes three other formats directly, with nothing to enable:

| `--writer` | Output | Needs a GPU |
|----|----|----|
| `vegalite` | A Vega-Lite JSON spec | no |
| `svg` | An SVG image; text stays selectable and editable | no |
| `pdf` | A one-page PDF with embedded subset fonts; text stays selectable | no |
| `hep` | A `.hep` plot document — the resolved plot rather than a picture | no |
| `png` | A PNG image | **yes** |
| `jpeg` (`jpg`) | A JPEG image | **yes** |
| `tiff` | A TIFF image | **yes** |
| `webp` | A lossless WebP image | **yes** |

**`--output` picks the writer for you.** If you name an output file and leave `--writer` off, the extension decides — so `-o chart.svg` writes SVG and `-o chart.pdf` writes a PDF, with no `-w` needed:

``` bash
ggsql exec -o chart.pdf "VISUALISE species AS fill FROM ggsql:penguins DRAW bar"
```

`json` and `vl.json` mean Vega-Lite; `svg`, `pdf`, `hep`, `png`, `jpg`/`jpeg`, `tif`/`tiff` and `webp` indicate file extensions. An extension `ggsql` doesn’t recognise, or no `--output` at all, falls back to Vega-Lite. An explicit `--writer` always wins: if it disagrees with the extension, the file holds the format `--writer` named — a `.png` written by `-w svg` is an SVG — and the mismatch is noted on `stderr`.

**A binary format needs somewhere to go.** `ggsql` will not print the bytes of a `pdf`, `hep` or raster format to a terminal: it says so on `stderr` and exits non-zero. Give the bytes a destination with `--output`, or pipe them somewhere; to just look at the plot, [`ggsql view`](#viewing-a-plot) draws it in a window and writes no file at all. A query with no `VISUALISE` prints its table, and honours `--output` the same way.

The four raster formats render through the GPU, so they need a graphics adapter — hardware or software — available where `ggsql` runs, and they are not compiled into every build. `svg` and `pdf` need neither, which makes them the ones to reach for on a server, in a container or in CI. Run `ggsql exec --help` to see which writers your build has; asking for one it doesn’t says so, and names the feature that would add it.

`hep` is the odd one out: it produces no picture at all. It captures the *resolved* plot — scales, breaks, labels, theme, geometry and data — so a host can draw it itself at any size, and redraw it on resize, without re-running the query.

### Writer settings

A writer is configured with `--writer-option key=value`, repeated once per setting:

``` bash
ggsql exec --writer png \
  --writer-option width=6 \
  --writer-option height=4 \
  --writer-option units=in \
  --writer-option dpi=150 \
  --output chart.png \
  "VISUALISE species AS fill FROM ggsql:penguins DRAW bar"
```

Several settings can also be collapsed into one flag, separated by `;`. With `-D` short for `--writer-option` (and `--writer-options` accepted as well), plus `-w` for `--writer` and `-o` for `--output`, the same call reads:

``` bash
ggsql exec -w png -D 'width=6;height=4;units=in;dpi=150' -o chart.png \
  "VISUALISE species AS fill FROM ggsql:penguins DRAW bar"
```

**Quote the collapsed form.** Most shells — bash, zsh, PowerShell — read `;` as a command separator, so unquoted it silently runs something else rather than failing. Single quotes, double quotes and `\;` all work. The two forms mix freely, and a repeated key takes its last value.

`;` is the only separator; `,` is not, because values contain commas — `background='rgb(255, 0, 0)'` has to survive intact.

#### The canvas

Every writer above except `vegalite` takes the same five canvas settings:

| Option | Value | Default |
|----|----|----|
| `width` | Canvas width, in `units` | `1500` (px) |
| `height` | Canvas height, in `units` | `1000` (px) |
| `units` | `px`, `in`, `cm`, `mm`, or `pt` — how `width` and `height` are read | `px` |
| `dpi` | Pixels per inch. Sets the print resolution of a physical size, and how large text and other chrome are relative to the canvas | `300` |
| `background` | Any CSS color, e.g. `white`, `#faf3e0`, `rgb(0 0 0 / 50%)`, or `transparent` | `white` |

`units` applies to the `width` and `height` you supply — the defaults are pixel counts either way, so `-D 'width=6;units=in'` gives a canvas 6 inches wide and 1000 pixels tall.

`svg` and `pdf` honour a physical size in the file itself, so `-D 'width=6;height=4;units=in;dpi=300'` yields a file that *prints* six inches wide as well as carrying an 1800-pixel coordinate space. For `hep`, the canvas is recorded as a hint — the aspect and resolution a consumer should default to — rather than a fixed size.

#### Per-format settings

Each format offers additional settings:

| Writer | Option | Value | Default |
|----|----|----|----|
| `png` | `compression` | `none`, `fast`, `balanced`, `small` | `balanced` |
| `jpeg` | `quality` | `1`–`100` | `90` |
| `tiff` | `compression` | `none`, `deflate`, `lzw`, `packbits` | `deflate` |
| `svg` | `text` | `text` keeps real text; `outline` converts it to paths | `text` |
| `svg` | `embed-fonts` | Inline font faces so the file stands alone | `false` |
| `svg` | `id-prefix` | Prefix for generated element ids | none |
| `pdf` | `compress` | Compress the content stream; `false` makes it readable | `true` |
| `pdf` | `links` | Turn markdown links into PDF link annotations | `true` |
| `hep` | `lossy` | Allow writing a plot the format cannot fully express | `false` |
| `hep` | `embed-fonts` | Embed the fonts the plot uses | `false` |

`jpeg` has no alpha channel, so it refuses `background=transparent` rather than quietly compositing onto black. `svg`’s `id-prefix` matters when two SVGs are inlined in one page: without it, both define the same gradient ids and each resolves to the other’s.

`webp` is lossless, with no rate control to set, so it takes the canvas settings and nothing else. The Vega-Lite writer takes no options at all: its output is resolution-independent, so size, resolution and background belong to whatever renders the spec. Passing an option a writer doesn’t understand or can’t honour is an error rather than a setting quietly ignored. Such a misunderstanding is reported before the query runs rather than after.

When a plot asks for a graphic a format cannot express, `ggsql` says so on `stderr` and still writes the file. This is a defect in the file you produced, so the defect is reported whether or not `--verbose` is set, and it stays out of a piped `svg`.

## Viewing a plot

`ggsql view` shows a query’s plot in a native window instead of writing a file, and blocks until the window is closed:

``` bash
ggsql view "VISUALISE species AS fill FROM ggsql:penguins DRAW bar"
```

Resizing the window re-lays-out the plot rather than stretching it, so labels and gridlines stay correct at any size. `-D` (`--viewer-option`) takes `width`, `height`, `background` and `title`; `units` and `dpi` are refused, since a window is sized in logical pixels and its resolution belongs to the display it is on.

Like the raster writers, the viewer renders through the GPU and is not in every build. The subcommand exists either way and says what would enable it.

## Documentation

The ggsql CLI has built-in documentation for ggsql syntax and usage. Run `ggsql docs` for an overview of available documentation topics, and `ggsql docs [topic]` to read about a specific topic.

``` bash
$ ggsql docs draw
DRAW is perhaps the most important clause in ggsql as it defines a layer in your 
visualisation. A layer is a single instance of a visual representation of a dataset.
[...]
```

A ggsql [skill](../../syntax/skill.llms.md), a usage guide intended for AI assistants and humans, can also be output using the `ggsql skill` command (also available as `ggsql agent-info`).

``` bash
$ ggsql skill
                          ggsql Query Writer
ggsql is a SQL extension for declarative data visualization based on
Grammar of Graphics principles.
[...]
```
