# Configure column display using `FORMAT`

The `FORMAT` clause is used to tweak the formatting and structure of one or more whole columns. It is the workhorse for column-focussed changes: value formatting, display properties and stub design.

## Clause syntax

```ggsql
FORMAT [ANY | BODY | STUB] 
  (* | <column>, ...)
  SETTING <parameter> => <value>, ...
  RENAMING <value> => <string>, ...
```

## Target

    [ANY | BODY | STUB]

The target directly following the `FORMAT` clause determines what kind of column is being formatted. Setting the target to `BODY` affects the cells in the body part of the table. The target can be set to `STUB`, which will move the column to the left of the table to serve as row labels. Without spelling out a target, the default will be `ANY`. The `ANY` target is relevant when the same column is formatted by multiple `FORMAT` statements. `ANY` will give priority to `BODY` and `STUB` targets in other statements. Unless instructed otherwise, `ANY` will resolve to `BODY`.

```ggsql
-- Same as `FORMAT ANY foo, bar`
FORMAT foo, bar

-- Designate column as stub
FORMAT STUB foo
```

A column named `any`, `body` or `stub` is read as a target when it comes first in the column list, so quote it to format the column itself:

```ggsql
-- Format the column named `body`, not the BODY target
FORMAT `body`, foo
```

## Column selection

```ggsql
(* | <column>, ...)
```

The `FORMAT` clause requires a comma-separated list of columns to format. In addition to column names, it can use [`SPAN`](../../syntax/clause/span.llms.md) identifiers as well to target the spanner’s columns. It can also take \*, meaning to format all columns.

```ggsql
-- Targeting regular columns
FORMAT foo, bar

-- Targeting every column
FORMAT *

-- Targeting spanner id: bar, baz and qux become targeted
SPAN foo OVER bar, baz
FORMAT foo, qux
```

## Settings

```ggsql
SETTING <parameter> => <value>, ...
```

It takes a typical `SETTING` clause with key-value pairs. It currently takes the following parameters:

- `hjust`: numeric value between 0-1 or string, for horizontal justification of cell content. Unset, the default depends on the column: right-aligned for numbers and stub columns, left-aligned otherwise. Strings can be `'left'`, `'right'`, `'centre'`, or `'center'`. By default content is left-aligned for strings, and right-aligned for numbers or stub columns.
- `width`: a unit as a string giving the number of pixels (e.g. `'100px'`) or a percentage (`'20%'`). Sets the display width of the column.

## Renaming

```ggsql
RENAMING <value> => <string>, ...
```

The `RENAMING` clause is *the* place to format the cell contents. It works the same as the namesake in the [`SCALE`](../../syntax/clause/scale.llms.md#renaming) clause. Generally, it works not unlike the [`LABEL`](../../syntax/clause/label-tabulate.llms.md) clause in the sense that you specify a cell’s content on the left and a replacement value on the right. Alternatively, you can use a formatter with the `* => '...'` syntax. Specifically see the [break formatting](../../syntax/clause/scale.llms.md#break-formatting) section for a list of formatters.

```ggsql
-- Replace a specific piece of content
FORMAT species RENAMING 'adelie' => 'Pygoscelis adeliae'

-- Suppress/hide a specific piece of content
FORMAT species RENAMING 'adelie' => null

-- Format all cells in the column as lower-case
FORMAT species RENAMING * => '{:lower}'
```

## Examples

Declare a stub column.

```ggsql
CREATE TEMP TABLE my_data AS SELECT * FROM ggsql:penguins LIMIT 5

TABULATE * FROM my_data
  FORMAT STUB island
```

|           | species | bill_len | bill_dep | flipper_len | body_mass | sex    | year |
|----------:|:--------|---------:|---------:|------------:|----------:|:-------|-----:|
| Torgersen | Adelie  |     39.1 |     18.7 |         181 |      3750 | male   | 2007 |
| Torgersen | Adelie  |     39.5 |     17.4 |         186 |      3800 | female | 2007 |
| Torgersen | Adelie  |     40.3 |       18 |         195 |      3250 | female | 2007 |
| Torgersen | Adelie  |     null |     null |        null |      null | null   | 2007 |
| Torgersen | Adelie  |     36.7 |     19.3 |         193 |      3450 | female | 2007 |

Stubheads, the column labels for stub columns, are hidden by default. To display these, you can use the `LABEL` clause.

```ggsql
TABULATE * FROM my_data
  FORMAT STUB island
  LABEL island => 'Island'
```

|    Island | species | bill_len | bill_dep | flipper_len | body_mass | sex    | year |
|----------:|:--------|---------:|---------:|------------:|----------:|:-------|-----:|
| Torgersen | Adelie  |     39.1 |     18.7 |         181 |      3750 | male   | 2007 |
| Torgersen | Adelie  |     39.5 |     17.4 |         186 |      3800 | female | 2007 |
| Torgersen | Adelie  |     40.3 |       18 |         195 |      3250 | female | 2007 |
| Torgersen | Adelie  |     null |     null |        null |      null | null   | 2007 |
| Torgersen | Adelie  |     36.7 |     19.3 |         193 |      3450 | female | 2007 |

Stub columns cannot be targetted by spans. The boundary between stub and body is un-spannable.

```ggsql
TABULATE * FROM my_data
  FORMAT STUB island
  SPAN "My label" OVER island, species
```

    ExecutionError: Validation error: SPAN 1: cannot include stub column 'island'.
    Validation error: SPAN 1: cannot include stub column 'island'.

When a column is targeted by multiple `FORMAT` clauses, the clauses are combined; in case of conflicts the last `FORMAT` wins. Without target, `ANY` target is assumed which will not override targets. Note that the `island` column is not in the stub, because `BODY` overrides `STUB`.

```ggsql
TABULATE * FROM my_data
  FORMAT STUB island
  FORMAT BODY island
```

| species | island    | bill_len | bill_dep | flipper_len | body_mass | sex    | year |
|:--------|:----------|---------:|---------:|------------:|----------:|:-------|-----:|
| Adelie  | Torgersen |     39.1 |     18.7 |         181 |      3750 | male   | 2007 |
| Adelie  | Torgersen |     39.5 |     17.4 |         186 |      3800 | female | 2007 |
| Adelie  | Torgersen |     40.3 |       18 |         195 |      3250 | female | 2007 |
| Adelie  | Torgersen |     null |     null |        null |      null | null   | 2007 |
| Adelie  | Torgersen |     36.7 |     19.3 |         193 |      3450 | female | 2007 |

Control content alignment.

```ggsql
TABULATE * FROM my_data
  FORMAT sex
    SETTING hjust => 'right'
```

| species | island    | bill_len | bill_dep | flipper_len | body_mass |    sex | year |
|:--------|:----------|---------:|---------:|------------:|----------:|-------:|-----:|
| Adelie  | Torgersen |     39.1 |     18.7 |         181 |      3750 |   male | 2007 |
| Adelie  | Torgersen |     39.5 |     17.4 |         186 |      3800 | female | 2007 |
| Adelie  | Torgersen |     40.3 |       18 |         195 |      3250 | female | 2007 |
| Adelie  | Torgersen |     null |     null |        null |      null |   null | 2007 |
| Adelie  | Torgersen |     36.7 |     19.3 |         193 |      3450 | female | 2007 |

Set the width of columns.

```ggsql
TABULATE * FROM my_data
  FORMAT species, island
    SETTING width => '150px'
```

| species | island    | bill_len | bill_dep | flipper_len | body_mass | sex    | year |
|:--------|:----------|---------:|---------:|------------:|----------:|:-------|-----:|
| Adelie  | Torgersen |     39.1 |     18.7 |         181 |      3750 | male   | 2007 |
| Adelie  | Torgersen |     39.5 |     17.4 |         186 |      3800 | female | 2007 |
| Adelie  | Torgersen |     40.3 |       18 |         195 |      3250 | female | 2007 |
| Adelie  | Torgersen |     null |     null |        null |      null | null   | 2007 |
| Adelie  | Torgersen |     36.7 |     19.3 |         193 |      3450 | female | 2007 |

Replace null values everywhere by selecting all columns.

```ggsql
TABULATE * FROM my_data
  FORMAT * RENAMING null => '---'
```

| species | island    | bill_len | bill_dep | flipper_len | body_mass | sex    | year |
|:--------|:----------|---------:|---------:|------------:|----------:|:-------|-----:|
| Adelie  | Torgersen |     39.1 |     18.7 |         181 |      3750 | male   | 2007 |
| Adelie  | Torgersen |     39.5 |     17.4 |         186 |      3800 | female | 2007 |
| Adelie  | Torgersen |     40.3 |       18 |         195 |      3250 | female | 2007 |
| Adelie  | Torgersen |      --- |      --- |         --- |       --- | ---    | 2007 |
| Adelie  | Torgersen |     36.7 |     19.3 |         193 |      3450 | female | 2007 |

Replacing specific pieces of content.

```ggsql
TABULATE * FROM my_data
  FORMAT body_mass, sex
    RENAMING 
      null => '?',
      'male' => '♂',
      'female' => '♀'
```

| species | island    | bill_len | bill_dep | flipper_len | body_mass | sex | year |
|:--------|:----------|---------:|---------:|------------:|----------:|:----|-----:|
| Adelie  | Torgersen |     39.1 |     18.7 |         181 |      3750 | ♂   | 2007 |
| Adelie  | Torgersen |     39.5 |     17.4 |         186 |      3800 | ♀   | 2007 |
| Adelie  | Torgersen |     40.3 |       18 |         195 |      3250 | ♀   | 2007 |
| Adelie  | Torgersen |     null |     null |        null |         ? | ?   | 2007 |
| Adelie  | Torgersen |     36.7 |     19.3 |         193 |      3450 | ♀   | 2007 |

Using a formatter to target all cells in a column.

```ggsql
TABULATE * FROM my_data
  FORMAT body_mass
    RENAMING * => '{:num %.0f}g'
```

| species | island    | bill_len | bill_dep | flipper_len | body_mass | sex    | year |
|:--------|:----------|---------:|---------:|------------:|----------:|:-------|-----:|
| Adelie  | Torgersen |     39.1 |     18.7 |         181 |     3750g | male   | 2007 |
| Adelie  | Torgersen |     39.5 |     17.4 |         186 |     3800g | female | 2007 |
| Adelie  | Torgersen |     40.3 |       18 |         195 |     3250g | female | 2007 |
| Adelie  | Torgersen |     null |     null |        null |      null | null   | 2007 |
| Adelie  | Torgersen |     36.7 |     19.3 |         193 |     3450g | female | 2007 |
