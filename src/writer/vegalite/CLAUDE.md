# Writer Module Internal Architecture

This document describes internal implementation details of the Vega-Lite writer that are critical for extending or modifying layer rendering behavior.

## Unified Dataset Architecture

**Key Concept**: All layer data is merged into a single top-level dataset with source tagging.

### Why Unified?

Instead of each layer having its own named dataset, all data is combined into one array. This enables:
- Shared axes and scales across layers
- Simplified faceting (one dataset split across panels)
- Consistent row indexing for ordering (line/path/polygon geoms)

### How It Works

```
┌─────────────────────────────────────────────────────────┐
│  Layer 0 DataFrame    │  Layer 1 DataFrame              │
│  [row0, row1, row2]   │  [row0, row1]                   │
└───────────────┬───────┴──────────────┬──────────────────┘
                │                      │
                ▼                      ▼
         datasets[data_key_0]    datasets[data_key_1]
                │                      │
                └──────────┬───────────┘
                           ▼
                   unify_datasets()
                           │
                           ▼
        ┌──────────────────────────────────────────┐
        │  Unified Dataset (top-level data.values) │
        │  [{...row, __ggsql_source__: key_0},     │
        │   {...row, __ggsql_source__: key_0},     │
        │   {...row, __ggsql_source__: key_0},     │
        │   {...row, __ggsql_source__: key_1},     │
        │   {...row, __ggsql_source__: key_1}]     │
        └──────────────────────────────────────────┘
                           │
                           ▼
        Each layer has source filter transform:
        {"filter": {"field": "__ggsql_source__", "equal": "data_key"}}
```

### Source Tagging

When `unify_datasets()` merges datasets:
1. Adds `__ggsql_source__` column to each row with its dataset key
2. Adds `__ggsql_row_index__` column for order preservation (0-indexed, global)
3. Fills missing columns with `null` (union schema)

Example:
```json
{
  "__ggsql_aes_pos1__": 1,
  "__ggsql_aes_pos2__": 2,
  "__ggsql_source__": "__ggsql_layer_0__",
  "__ggsql_row_index__": 0
}
```

## Layer Building Pipeline

### High-Level Flow

```
prepare_layer_data()
    └─> renderer.prepare_data(df) → PreparedData
            │
            ▼
    Register datasets:
    - Single: datasets[data_key] = values
    - Composite: datasets[data_key + component] = values for each component
            │
            ▼
    unify_datasets(datasets) → unified array
            │
            ▼
    vl_spec["data"] = {"values": unified}
            │
            ▼
build_layers()
    for each layer:
        1. Create base layer_spec with mark
        2. Add source filter transform (if renderer.needs_source_filter())
        3. Build encoding channels
        4. renderer.modify_spec(layer_spec) → modify mark properties
        5. renderer.finalize(layer_spec) → add transforms, expand to multiple layers
            │
            ▼
    Return array of layer specs
```

### PreparedData Semantics

**PreparedData::Single**:
```rust
PreparedData::Single { values }
```
- Dataset registered as: `datasets[data_key] = values`
- `__ggsql_source__` in unified data: `data_key`
- Source filter: `{"field": "__ggsql_source__", "equal": "data_key"}`
- **Use case**: Standard single-dataset layers (point, line, bar, etc.)

**PreparedData::Composite**:
```rust
PreparedData::Composite {
    components: HashMap<String, Vec<Value>>,  // component_name → values
    metadata: Box<dyn Any>,
}
```
- Datasets registered as: `datasets[data_key + component_name] = values` for each component
- `__ggsql_source__` in unified data: `data_key + component_name`
- **Use case**: Geoms that decompose into multiple sub-layers (boxplot, violin, errorbar)
- **Special case**: Empty component name (`""`) creates `data_key + "" = data_key` (useful for passing metadata without changing dataset key)

### Important Rules for Finalize

**DO**:
- ✅ Preserve existing transforms:
  ```rust
  let mut transforms = layer_spec
      .get("transform")
      .and_then(|t| t.as_array())
      .cloned()
      .unwrap_or_default();
  // Add your transforms
  transforms.extend(your_transforms);
  layer_spec["transform"] = json!(transforms);
  ```
- ✅ Return multiple layers if needed (composite geoms)
- ✅ Modify encodings that reference transformed fields

**DON'T**:
- ❌ Set `layer_spec["data"]` - layers use the unified top-level dataset
- ❌ Replace the transforms array without preserving existing ones
- ❌ Assume the layer is isolated - it shares data with other layers

## GeomRenderer Trait Lifecycle

### When Each Method Is Called

```rust
// 1. Data Preparation Phase (before unification)
renderer.prepare_data(df, layer, data_key, binned_columns)
    → PreparedData
    // Context: You have the full DataFrame
    // Can: Transform data, compute statistics, create metadata
    // Returns: Data values + optional metadata

// 2. Layer Spec Building Phase (after unification)
let layer_spec = create_base_spec();
add_source_filter(layer_spec);  // if needs_source_filter() == true
build_encoding(layer_spec);

renderer.modify_spec(layer_spec, layer, context)
    // Context: Base spec created, encoding built
    // Can: Modify mark properties, add mark-level calculations
    // Example: Bar width, violin fill settings

// 3. Finalization Phase
renderer.finalize(layer_spec, layer, data_key, prepared)
    → Vec<Value>  // Can return multiple layers
    // Context: Complete layer spec with transforms and encoding
    // Can: Add transforms, expand to multiple layers, modify encodings
    // Must: Preserve existing transforms
```

### Method Purposes

**`prepare_data`**:
- Access full DataFrame before unification
- Perform data transformations (e.g., create segment endpoints)
- Compute statistics (e.g., boxplot quartiles)
- Return metadata for later use

**`modify_encoding`**:
- Called during encoding construction (before spec is built)
- Modify encoding channels (add detail, change field names)
- Add geom-specific encoding logic (e.g., order for path/line)

**`modify_spec`**:
- Called after encoding is built but before finalize
- Modify mark properties (width, baseline, etc.)
- Add mark-level calculations

**`needs_source_filter`**:
- Return `false` if your geom handles filtering internally
- Default: `true` (use standard source filter)
- Example: Boxplot returns `false` and adds component-specific filters

**`finalize`**:
- Add Vega-Lite transforms (window, flatten, calculate)
- Expand into multiple layers (composite geoms)
- Modify encodings that reference transformed fields
- Return final layer array

## Common Patterns

### Pattern 1: Single Layer with Transforms (Line Segmentation)

```rust
fn prepare_data(...) -> Result<PreparedData> {
    // Detect if segmentation is needed
    let needs_segmentation = check_within_group_variation(df);
    
    let values = dataframe_to_values(df)?;
    
    if needs_segmentation {
        // Use Composite with empty component name to pass metadata
        Ok(PreparedData::Composite {
            components: [("".to_string(), values)].into_iter().collect(),
            metadata: Box::new(SegmentMetadata { ... }),
        })
    } else {
        Ok(PreparedData::Single { values })
    }
}

fn finalize(...) -> Result<Vec<Value>> {
    match prepared {
        PreparedData::Composite { metadata, .. } => {
            // Preserve existing transforms (source filter!)
            let mut transforms = layer_spec
                .get("transform")
                .and_then(|t| t.as_array())
                .cloned()
                .unwrap_or_default();
            
            // Add segmentation transforms
            transforms.extend(vec![
                json!({"window": [...]}),
                json!({"flatten": [...]}),
                // ...
            ]);
            
            layer_spec["transform"] = json!(transforms);
            
            // Don't set layer_spec["data"] - use unified dataset
            
            Ok(vec![layer_spec])
        }
        PreparedData::Single { .. } => Ok(vec![layer_spec])
    }
}
```

### Pattern 2: Multi-Layer Decomposition (Boxplot)

```rust
fn prepare_data(...) -> Result<PreparedData> {
    // Compute boxplot statistics
    let components = compute_boxplot_components(df);
    
    // Return multiple component datasets
    Ok(PreparedData::Composite {
        components: [
            ("box".to_string(), box_values),
            ("median".to_string(), median_values),
            ("whisker_lower".to_string(), whisker_values),
            // ...
        ].into_iter().collect(),
        metadata: Box::new(BoxplotMetadata { has_outliers }),
    })
}

fn needs_source_filter() -> bool {
    // We'll add component-specific filters ourselves
    false
}

fn finalize(...) -> Result<Vec<Value>> {
    // Create separate layer for each component
    let mut layers = Vec::new();
    
    // Box layer
    let mut box_layer = layer_spec.clone();
    box_layer["transform"] = json!([{
        "filter": {
            "field": "__ggsql_source__",
            "equal": format!("{}box", data_key)  // Component-specific filter
        }
    }]);
    layers.push(box_layer);
    
    // Median layer
    let mut median_layer = create_median_layer();
    median_layer["transform"] = json!([{
        "filter": {
            "field": "__ggsql_source__",
            "equal": format!("{}median", data_key)
        }
    }]);
    layers.push(median_layer);
    
    // ... more layers
    
    Ok(layers)
}
```

### Pattern 3: Mark Property Modification (Bar Width)

```rust
fn modify_spec(...) -> Result<()> {
    let width = layer.parameters.get("width")
        .and_then(|v| v.as_number())
        .unwrap_or(0.9);
    
    layer_spec["mark"] = json!({
        "type": "bar",
        "width": {"band": width},
        "clip": true
    });
    
    Ok(())
}
```

## Debugging Tips

### Issue: Layer has no data / nothing renders

**Check**:
1. Is `__ggsql_source__` value correct in unified data?
   - For Single: should match `data_key`
   - For Composite: should match `data_key + component_name`
2. Does layer have source filter transform?
   - Check `layer_spec["transform"][0]` for filter
   - If using Composite, ensure component name matches
3. Did you accidentally set `layer_spec["data"]`?
   - Remove it - layers use unified top-level dataset

**Debug**:
```rust
eprintln!("Dataset key: {}", data_key);
eprintln!("Source values in data: {:?}", 
    unified_data.iter()
        .filter_map(|row| row.get("__ggsql_source__"))
        .collect::<Vec<_>>());
eprintln!("Layer transforms: {:?}", layer_spec["transform"]);
```

### Issue: Transforms not working correctly

**Check**:
1. Are you preserving existing transforms in finalize?
2. Are transform fields available in the data?
3. Is filter ordering correct? (source filter should come first)

### Issue: Composite geom breaks with wrong source keys

**Solution**: Use empty component name (`""`) if you want `data_key` as the source value:
```rust
PreparedData::Composite {
    components: [("".to_string(), values)].into_iter().collect(),
    metadata: Box::new(YourMetadata { ... }),
}
```

## Testing Rendered Output

To inspect generated Vega-Lite specs:

```bash
# Generate spec to file
cargo run --bin ggsql -- exec "YOUR QUERY" \
    --reader "duckdb://memory" \
    --output /tmp/test.vl.json

# Check structure
grep -A10 '"data"' /tmp/test.vl.json  # Top-level data
grep -A5 '"transform"' /tmp/test.vl.json  # Layer transforms
grep '__ggsql_source__' /tmp/test.vl.json  # Source values
```

Paste the spec into [Vega-Lite Editor](https://vega.github.io/editor/#/url/vega-lite/) to visualize.

## References

- **Main implementation**: `src/writer/vegalite/mod.rs`
- **Layer rendering**: `src/writer/vegalite/layer.rs`
- **Data unification**: `src/writer/vegalite/data.rs` (`unify_datasets()`)
- **Renderer trait**: `src/writer/vegalite/layer.rs` (`GeomRenderer` trait)
- **Example renderers**: `LineRenderer`, `BoxplotRenderer`, `ViolinRenderer` in `layer.rs`
