# Row-grouping using `FACET`

The `FACET` clause allows you to split your data and display each group alongside a group label. This technique is great for making structure in the data explicit.

> **NOTE:**
>
> This page is about the `TABULATE` usage of the `FACET` clause. For the `VISUALISE` usage, see [this page](../../syntax/clause/facet.llms.md).

## Clause syntax

The `FACET` syntax contains a number of subclauses.

```ggsql
FACET <column>, ...
  SETTING <parameter> => <value>, ...
```

## Column selection

```ggsql
<column>, ...
```

In `FACET`, one or more comma separated columns can be used to split the data. The first `column` is mandatory, and subsequent columns add nesting levels. If the columns contain missing `null` values, these are handled by the `missing` setting.

## Settings

```ggsql
SETTING <parameter> => <value>, ...
```

Facets take a typical `SETTING` clause with key-value pairs. It currently takes the following parameters:

- `missing`: Determines what happens with `null` values in the grouping columns. Can be one of:
  - `'first'` Keep the rows with missing group values and display these on top of regular groups.
  - `'last'` (default) Keep the rows with missing group values and display these underneath regular groups.
  - `'drop'` Removes the rows with missing group values from the output table.
- `place`: Placement options of the group labels. One of the following:
  - `'left'` (default) Places group labels in the stub on the left. Group labels are vertically merged across member rows.
  - `'top'` Interleaves table body rows with group label rows. Group labels are placed above its member rows spanning all columns.
- `separator`: Optional join string to collapse group labels across nesting levels. The default (`null`) keeps levels in separate label rows or stub columns.

## Examples

Declare a facet column. By default, these are placed on the left as a stub.

```ggsql
CREATE TEMP TABLE my_data AS 
SELECT DISTINCT ON (species, island, sex) * FROM ggsql:penguins

TABULATE * FROM my_data
FACET sex
```

|        | species   | island    | bill_len | bill_dep | flipper_len | body_mass | year |
|:-------|:----------|:----------|---------:|---------:|------------:|----------:|-----:|
| female | Adelie    | Dream     |     39.5 |     16.7 |         178 |      3250 | 2007 |
|        | Chinstrap | Dream     |     46.5 |     17.9 |         192 |      3500 | 2007 |
|        | Gentoo    | Biscoe    |     46.1 |     13.2 |         211 |      4500 | 2007 |
|        | Adelie    | Torgersen |     39.5 |     17.4 |         186 |      3800 | 2007 |
|        | Adelie    | Biscoe    |     37.8 |     18.3 |         174 |      3400 | 2007 |
| male   | Gentoo    | Biscoe    |       50 |     16.3 |         230 |      5700 | 2007 |
|        | Adelie    | Dream     |     37.2 |     18.1 |         178 |      3900 | 2007 |
|        | Chinstrap | Dream     |       50 |     19.5 |         196 |      3900 | 2007 |
|        | Adelie    | Torgersen |     39.1 |     18.7 |         181 |      3750 | 2007 |
|        | Adelie    | Biscoe    |     37.7 |     18.7 |         180 |      3600 | 2007 |
| null   | Adelie    | Torgersen |     null |     null |        null |      null | 2007 |
|        | Adelie    | Dream     |     37.5 |     18.9 |         179 |      2975 | 2007 |
|        | Gentoo    | Biscoe    |     44.5 |     14.3 |         216 |      4100 | 2007 |

Omit missing group variables from the table.

```ggsql
TABULATE * FROM my_data
FACET sex SETTING missing => 'drop'
```

|        | species   | island    | bill_len | bill_dep | flipper_len | body_mass | year |
|:-------|:----------|:----------|---------:|---------:|------------:|----------:|-----:|
| female | Adelie    | Dream     |     39.5 |     16.7 |         178 |      3250 | 2007 |
|        | Chinstrap | Dream     |     46.5 |     17.9 |         192 |      3500 | 2007 |
|        | Gentoo    | Biscoe    |     46.1 |     13.2 |         211 |      4500 | 2007 |
|        | Adelie    | Torgersen |     39.5 |     17.4 |         186 |      3800 | 2007 |
|        | Adelie    | Biscoe    |     37.8 |     18.3 |         174 |      3400 | 2007 |
| male   | Gentoo    | Biscoe    |       50 |     16.3 |         230 |      5700 | 2007 |
|        | Adelie    | Dream     |     37.2 |     18.1 |         178 |      3900 | 2007 |
|        | Chinstrap | Dream     |       50 |     19.5 |         196 |      3900 | 2007 |
|        | Adelie    | Torgersen |     39.1 |     18.7 |         181 |      3750 | 2007 |
|        | Adelie    | Biscoe    |     37.7 |     18.7 |         180 |      3600 | 2007 |

Alternatively, missing group variables can be given a proper label using `FORMAT`. Note that `missing => 'drop'` has priority over any relabeling in `FORMAT`.

```ggsql
TABULATE * FROM my_data
FACET sex
FORMAT sex RENAMING null => 'unknown'
```

|         | species   | island    | bill_len | bill_dep | flipper_len | body_mass | year |
|:--------|:----------|:----------|---------:|---------:|------------:|----------:|-----:|
| female  | Adelie    | Dream     |     39.5 |     16.7 |         178 |      3250 | 2007 |
|         | Chinstrap | Dream     |     46.5 |     17.9 |         192 |      3500 | 2007 |
|         | Gentoo    | Biscoe    |     46.1 |     13.2 |         211 |      4500 | 2007 |
|         | Adelie    | Torgersen |     39.5 |     17.4 |         186 |      3800 | 2007 |
|         | Adelie    | Biscoe    |     37.8 |     18.3 |         174 |      3400 | 2007 |
| male    | Gentoo    | Biscoe    |       50 |     16.3 |         230 |      5700 | 2007 |
|         | Adelie    | Dream     |     37.2 |     18.1 |         178 |      3900 | 2007 |
|         | Chinstrap | Dream     |       50 |     19.5 |         196 |      3900 | 2007 |
|         | Adelie    | Torgersen |     39.1 |     18.7 |         181 |      3750 | 2007 |
|         | Adelie    | Biscoe    |     37.7 |     18.7 |         180 |      3600 | 2007 |
| unknown | Adelie    | Torgersen |     null |     null |        null |      null | 2007 |
|         | Adelie    | Dream     |     37.5 |     18.9 |         179 |      2975 | 2007 |
|         | Gentoo    | Biscoe    |     44.5 |     14.3 |         216 |      4100 | 2007 |

Place grouping labels in rows above their members.

```ggsql
TABULATE * FROM my_data
FACET sex SETTING place => 'top'
```

| species   | island    | bill_len | bill_dep | flipper_len | body_mass | year |
|:----------|:----------|:---------|:---------|:------------|:----------|:-----|
| female    |           |          |          |             |           |      |
| Adelie    | Dream     | 39.5     | 16.7     | 178         | 3250      | 2007 |
| Chinstrap | Dream     | 46.5     | 17.9     | 192         | 3500      | 2007 |
| Gentoo    | Biscoe    | 46.1     | 13.2     | 211         | 4500      | 2007 |
| Adelie    | Torgersen | 39.5     | 17.4     | 186         | 3800      | 2007 |
| Adelie    | Biscoe    | 37.8     | 18.3     | 174         | 3400      | 2007 |
| male      |           |          |          |             |           |      |
| Gentoo    | Biscoe    | 50       | 16.3     | 230         | 5700      | 2007 |
| Adelie    | Dream     | 37.2     | 18.1     | 178         | 3900      | 2007 |
| Chinstrap | Dream     | 50       | 19.5     | 196         | 3900      | 2007 |
| Adelie    | Torgersen | 39.1     | 18.7     | 181         | 3750      | 2007 |
| Adelie    | Biscoe    | 37.7     | 18.7     | 180         | 3600      | 2007 |
| null      |           |          |          |             |           |      |
| Adelie    | Torgersen | null     | null     | null        | null      | 2007 |
| Adelie    | Dream     | 37.5     | 18.9     | 179         | 2975      | 2007 |
| Gentoo    | Biscoe    | 44.5     | 14.3     | 216         | 4100      | 2007 |

Using multiple group columns to group gives nested labels.

```ggsql
TABULATE * FROM my_data
FACET species, island
```

|           |           | bill_len | bill_dep | flipper_len | body_mass | sex    | year |
|:----------|:----------|---------:|---------:|------------:|----------:|:-------|-----:|
| Adelie    | Biscoe    |     37.8 |     18.3 |         174 |      3400 | female | 2007 |
|           |           |     37.7 |     18.7 |         180 |      3600 | male   | 2007 |
|           | Dream     |     39.5 |     16.7 |         178 |      3250 | female | 2007 |
|           |           |     37.2 |     18.1 |         178 |      3900 | male   | 2007 |
|           |           |     37.5 |     18.9 |         179 |      2975 | null   | 2007 |
|           | Torgersen |     null |     null |        null |      null | null   | 2007 |
|           |           |     39.5 |     17.4 |         186 |      3800 | female | 2007 |
|           |           |     39.1 |     18.7 |         181 |      3750 | male   | 2007 |
| Chinstrap | Dream     |     46.5 |     17.9 |         192 |      3500 | female | 2007 |
|           |           |       50 |     19.5 |         196 |      3900 | male   | 2007 |
| Gentoo    | Biscoe    |       50 |     16.3 |         230 |      5700 | male   | 2007 |
|           |           |     46.1 |     13.2 |         211 |      4500 | female | 2007 |
|           |           |     44.5 |     14.3 |         216 |      4100 | null   | 2007 |

The labels from multiple groups can be collapsed to a single stub column (or single rows) by using the `separator` setting to join the labels. Labels are joined after `FORMAT RENAMING` is applied.

```ggsql
TABULATE * FROM my_data
FACET species, island
  SETTING separator => ' - '
```

|                    | bill_len | bill_dep | flipper_len | body_mass | sex    | year |
|:-------------------|---------:|---------:|------------:|----------:|:-------|-----:|
| Adelie - Biscoe    |     37.8 |     18.3 |         174 |      3400 | female | 2007 |
|                    |     37.7 |     18.7 |         180 |      3600 | male   | 2007 |
| Adelie - Dream     |     39.5 |     16.7 |         178 |      3250 | female | 2007 |
|                    |     37.2 |     18.1 |         178 |      3900 | male   | 2007 |
|                    |     37.5 |     18.9 |         179 |      2975 | null   | 2007 |
| Adelie - Torgersen |     null |     null |        null |      null | null   | 2007 |
|                    |     39.5 |     17.4 |         186 |      3800 | female | 2007 |
|                    |     39.1 |     18.7 |         181 |      3750 | male   | 2007 |
| Chinstrap - Dream  |     46.5 |     17.9 |         192 |      3500 | female | 2007 |
|                    |       50 |     19.5 |         196 |      3900 | male   | 2007 |
| Gentoo - Biscoe    |       50 |     16.3 |         230 |      5700 | male   | 2007 |
|                    |     46.1 |     13.2 |         211 |      4500 | female | 2007 |
|                    |     44.5 |     14.3 |         216 |      4100 | null   | 2007 |

When group labels are placed in the stub, the labels can be controlled by `LABEL`. In case of `separator`-joined labels, the first variable (`species` in the example) displays the label.

```ggsql
TABULATE * FROM my_data
FACET species, island
LABEL island => 'Place', species => 'Type'
```

| Type      | Place     | bill_len | bill_dep | flipper_len | body_mass | sex    | year |
|:----------|:----------|---------:|---------:|------------:|----------:|:-------|-----:|
| Adelie    | Biscoe    |     37.8 |     18.3 |         174 |      3400 | female | 2007 |
|           |           |     37.7 |     18.7 |         180 |      3600 | male   | 2007 |
|           | Dream     |     39.5 |     16.7 |         178 |      3250 | female | 2007 |
|           |           |     37.2 |     18.1 |         178 |      3900 | male   | 2007 |
|           |           |     37.5 |     18.9 |         179 |      2975 | null   | 2007 |
|           | Torgersen |     null |     null |        null |      null | null   | 2007 |
|           |           |     39.5 |     17.4 |         186 |      3800 | female | 2007 |
|           |           |     39.1 |     18.7 |         181 |      3750 | male   | 2007 |
| Chinstrap | Dream     |     46.5 |     17.9 |         192 |      3500 | female | 2007 |
|           |           |       50 |     19.5 |         196 |      3900 | male   | 2007 |
| Gentoo    | Biscoe    |       50 |     16.3 |         230 |      5700 | male   | 2007 |
|           |           |     46.1 |     13.2 |         211 |      4500 | female | 2007 |
|           |           |     44.5 |     14.3 |         216 |      4100 | null   | 2007 |
