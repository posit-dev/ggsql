# Create annotation layers with `PLACE`

The `PLACE` clause is the little sibling of the [`DRAW` clause](../../syntax/clause/draw.llms.md) that also creates layers. A layer created with `PLACE` has no mappings to data, all aesthetics are set using literal values instead.

## Clause syntax

The `PLACE` clause takes just a type and a setting clause, all of them required.

``` ggsql
PLACE <layer-type>
  SETTING <parameter/aesthetic> => <value>, ...
```

Like `DRAW`, the layer type is required and specifies the type of layer to draw, like `point` or `text`. It defines how the remaining settings are interpreted. The [main syntax page](../../syntax/index.llms.md#layers) has a list of all available layer types

Unlike the `DRAW` clause, the `PLACE` clause does not support `FILTER`, `PARTITION BY`, and `ORDER BY` clauses since everything is declared inline.

### `SETTING`

``` ggsql
SETTING <parameter/aesthetic> => <value>, ...
```

The `SETTING` clause can be used for two different things:

- *Setting aesthetics*: All aesthetics in `PLACE` layers are specified using literal value, e.g. ‘red’ (as in the color red). Aesthetics that are set will not go through a scale but will use the provided value as-is. You cannot set an aesthetic to a column, only to a literal values. Contrary to `DRAW` layers, `PLACE` layers can take multiple literal values in an array.
- *Setting parameters*: Some layers take additional arguments that control how they behave. Often, but not always, these modify the statistical transformation in some way. An example would be the binwidth parameter in histogram which controls the width of each bin during histogram calculation. This is not a statistical property since it is not related to each record, but to the calculation as a whole.
