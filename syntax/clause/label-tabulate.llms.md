# Define different titles with `LABEL`

> **NOTE:**
>
> This page is about the `TABULATE` usage of the `LABEL` clause. For the `VISUALISE` usage, see [this page](../../syntax/clause/label.llms.md).

The `LABEL` clause is a straightforward clause allowing you to override the default naming of column labels, spans, (sub)titles and captions. Unlike `VISUALISE`’s usage, there is no automatic labelling: a column with no label keeps its name as header.

## Clause syntax

The `LABEL` clause requires a comma-separated list of key-value pairs

```ggsql
LABEL <column/title> => <string/null>, ...
```

On the left hand side of the arrow, the following keys can be used:

- Column names as they occur in the data
- Reserved keywords for special labels:
  - `title` as the main title
  - `subtitle` as the subtitle
  - `caption` as the table caption
- Spanner IDs declared by `SPAN` clauses.

On the right hand side of the arrow, one can use a string to give new label, or `null` to hide the label.

## Rich text

The support of rich text in tables is planned but not yet implemented.

## Examples

Unlike using aliases in `TABULATE`, the `LABEL` clause lets you change column labels without changing column order

```ggsql
CREATE TEMP TABLE my_data AS SELECT * FROM ggsql:penguins LIMIT 5

TABULATE * FROM my_data
  LABEL bill_len => 'Bill length', bill_dep => 'Bill depth'
```

| species | island    | Bill length | Bill depth | flipper_len | body_mass | sex    | year |
|:--------|:----------|------------:|-----------:|------------:|----------:|:-------|-----:|
| Adelie  | Torgersen |        39.1 |       18.7 |         181 |      3750 | male   | 2007 |
| Adelie  | Torgersen |        39.5 |       17.4 |         186 |      3800 | female | 2007 |
| Adelie  | Torgersen |        40.3 |         18 |         195 |      3250 | female | 2007 |
| Adelie  | Torgersen |        null |       null |        null |      null | null   | 2007 |
| Adelie  | Torgersen |        36.7 |       19.3 |         193 |      3450 | female | 2007 |

Hiding labels.

```ggsql
TABULATE * FROM my_data
  LABEL island => null, species => null
```

|        |           | bill_len | bill_dep | flipper_len | body_mass | sex    | year |
|:-------|:----------|---------:|---------:|------------:|----------:|:-------|-----:|
| Adelie | Torgersen |     39.1 |     18.7 |         181 |      3750 | male   | 2007 |
| Adelie | Torgersen |     39.5 |     17.4 |         186 |      3800 | female | 2007 |
| Adelie | Torgersen |     40.3 |       18 |         195 |      3250 | female | 2007 |
| Adelie | Torgersen |     null |     null |        null |      null | null   | 2007 |
| Adelie | Torgersen |     36.7 |     19.3 |         193 |      3450 | female | 2007 |

Adding titles to a table.

```ggsql
TABULATE * FROM my_data
  LABEL
    title => 'Penguins!',
    subtitle => 'Structural size measurements',
    caption => 'Source: Palmer station, Antarctica'
```

| Penguins! |  |  |  |  |  |  |  |
|:---|:---|---:|---:|---:|---:|:---|---:|
| Structural size measurements |  |  |  |  |  |  |  |
| species | island | bill_len | bill_dep | flipper_len | body_mass | sex | year |
| Adelie | Torgersen | 39.1 | 18.7 | 181 | 3750 | male | 2007 |
| Adelie | Torgersen | 39.5 | 17.4 | 186 | 3800 | female | 2007 |
| Adelie | Torgersen | 40.3 | 18 | 195 | 3250 | female | 2007 |
| Adelie | Torgersen | null | null | null | null | null | 2007 |
| Adelie | Torgersen | 36.7 | 19.3 | 193 | 3450 | female | 2007 |

Source: Palmer station, Antarctica {quarto-disable-processing="true" quarto-bootstrap="false" style="display: table; border-collapse: collapse; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3"}

When column names and reserved keywords compete for a label, the reserved keyword wins.

```ggsql
-- data has column named 'title'
CREATE TEMP TABLE title_data AS 
SELECT * FROM (VALUES (1, 'a'), (2, 'b'), (3, 'c')) AS t(id, title)

TABULATE * FROM title_data
  LABEL
    title => 'Overrules column'
```

| Overrules column |       |
|-----------------:|:------|
|               id | title |
|                1 | a     |
|                2 | b     |
|                3 | c     |

You can target an aliased column instead.

```ggsql
TABULATE *, title AS title_col FROM title_data
  LABEL
    title => 'Main title',
    title_col => 'Column title'
```

| Main title |              |
|-----------:|:-------------|
|         id | Column title |
|          1 | a            |
|          2 | b            |
|          3 | c            |
