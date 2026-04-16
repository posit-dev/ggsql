# Position

Position aesthetics relate data to placement in the coordinate system of the plot. All layers need at least one of each position aesthetic mapped in order to show its data. However, the layer may compute position aesthetics from the mapping. For example, a bar plot calculates the `y` aesthetic by counting the number of records in each group defined by the mapped `x` aesthetic.

In the above we use `x`and `y` as examples of position aesthetics, but in reality the position aesthetic names are defined by the coordinate system in use. A Cartesian coordinate system recognizes `x` and `y` whereas a polar coordinate system recognizes `radius` and `angle`. You can implicitly choose the coordinate system by mapping to the aesthetics that it uses, i.e. if you map to `radius` and `angle` then a polar coordinate system will be chosen for you.

## Literal values

Scales for position aesthetics never use an output range and always relate to the input range. This is a practical decision by ggsql because different writers may treat the position aesthetic in different ways. [^1]

> **NOTE:**
>
> The lack of true literal values in position means that it is currently hard to place data and annotation “in-between” breaks in a discrete position scale.

## Aesthetic families

Position aesthetics have variants defined by a suffix attached to the primary name (e.g. `x`). The recognized suffixes are:

- `end`
- `min`
- `max`

Which version of an aesthetic to use depends on the layer. All aesthetics within a family use the same scale, which is named after its primary aesthetic. This means that even when rendering a layer that only uses `xmin` and `xmax`, you will still scale it by writing `SCALE x ...` and label it by writing `LABEL x => ...`

## Footnotes

[^1]: In reality one could easily think of position literal values as either normalized position along the x or y axis, or absolute units of distance from the bottom left corner of the coordinate system.
