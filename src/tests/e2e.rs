//! End-to-end execution tests: full queries through the DuckDB reader,
//! asserting on the resolved `Spec` and the rendered Vega-Lite output.
//!
//! Extracted from `reader/mod.rs` in the Phase 5 cleanup.
#![cfg(all(feature = "duckdb", feature = "vegalite"))]

use ggsql::df;
use ggsql::reader::{DuckDBReader, Reader};
use ggsql::writer::{VegaLiteWriter, Writer};

fn data_layer(json: &serde_json::Value, index: usize) -> &serde_json::Value {
    json["layer"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|l| {
            !matches!(
                l.get("description").and_then(|d| d.as_str()),
                Some("background" | "foreground")
            )
        })
        .nth(index)
        .expect("data layer not found at index")
}

#[test]
fn test_execute_and_render() {
    let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
    let spec = reader
        .execute("SELECT 1 as x, 2 as y VISUALISE x, y DRAW point")
        .unwrap();

    assert_eq!(spec.plot().layers.len(), 1);
    assert_eq!(spec.metadata().layer_count, 1);
    assert!(spec.layer_data(0).is_some());

    let writer = VegaLiteWriter::new();
    let result = writer.render(&spec).unwrap();
    assert!(result.contains("point"));
}

#[test]
fn test_execute_metadata() {
    let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
    let spec = reader
        .execute(
            "SELECT * FROM (VALUES (1, 10), (2, 20), (3, 30)) AS t(x, y) VISUALISE x, y DRAW point",
        )
        .unwrap();

    let metadata = spec.metadata();
    assert_eq!(metadata.rows, 3);
    // Columns now includes both user mappings (pos1, pos2) and resolved defaults (size, stroke, fill, opacity, shape, linewidth)
    // Aesthetics are transformed to internal names (x -> pos1, y -> pos2)
    assert!(metadata.columns.contains(&"pos1".to_string()));
    assert!(metadata.columns.contains(&"pos2".to_string()));
    assert_eq!(metadata.layer_count, 1);
}

#[test]
fn test_execute_with_cte() {
    let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
    let query = r#"
        WITH data AS (
            SELECT * FROM (VALUES (1, 10), (2, 20)) AS t(x, y)
        )
        SELECT * FROM data
        VISUALISE x, y DRAW point
    "#;

    let spec = reader.execute(query).unwrap();

    assert_eq!(spec.plot().layers.len(), 1);
    assert!(spec.layer_data(0).is_some());
    let df = spec.layer_data(0).unwrap();
    assert_eq!(df.height(), 2);
}

#[test]
fn test_render_multi_layer() {
    let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
    let query = r#"
        SELECT * FROM (VALUES (1, 10), (2, 20), (3, 30)) AS t(x, y)
        VISUALISE
        DRAW point MAPPING x AS x, y AS y
        DRAW line MAPPING x AS x, y AS y
    "#;

    let spec = reader.execute(query).unwrap();
    let writer = VegaLiteWriter::new();
    let result = writer.render(&spec).unwrap();

    assert!(result.contains("layer"));
}

#[test]
fn test_polar_project_with_start() {
    let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
    let query = r#"
        SELECT * FROM (VALUES ('A', 10), ('B', 20), ('C', 30)) AS t(category, value)
        VISUALISE value AS y, category AS fill
        DRAW bar
        PROJECT y, x TO polar SETTING start => 90
    "#;

    let spec = reader.execute(query).unwrap();
    let writer = VegaLiteWriter::new();
    let result = writer.render(&spec).unwrap();

    // Parse the JSON to verify the theta scale range is set correctly
    let json: serde_json::Value = serde_json::from_str(&result).unwrap();

    // The encoding should have a theta channel with a scale range offset by 90 degrees
    // 90 degrees = π/2 radians
    let layer = data_layer(&json, 0);
    let theta = &layer["encoding"]["theta"];
    assert!(theta.is_object(), "theta encoding should exist");

    // Check that the scale has a range with the start offset
    let scale = &theta["scale"];
    let range = scale["range"].as_array().unwrap();
    assert_eq!(range.len(), 2);

    // π/2 ≈ 1.5707963
    let start = range[0].as_f64().unwrap();
    assert!(
        (start - std::f64::consts::FRAC_PI_2).abs() < 0.001,
        "start should be π/2 (90 degrees), got {}",
        start
    );

    // π/2 + 2π ≈ 7.8539816
    let end = range[1].as_f64().unwrap();
    let expected_end = std::f64::consts::FRAC_PI_2 + 2.0 * std::f64::consts::PI;
    assert!(
        (end - expected_end).abs() < 0.001,
        "end should be π/2 + 2π, got {}",
        end
    );
}

#[test]
fn test_polar_project_default_start() {
    let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
    let query = r#"
        SELECT * FROM (VALUES ('A', 10), ('B', 20), ('C', 30)) AS t(category, value)
        VISUALISE value AS y, category AS fill
        DRAW bar
        PROJECT y, x TO polar
    "#;

    let spec = reader.execute(query).unwrap();
    let writer = VegaLiteWriter::new();
    let result = writer.render(&spec).unwrap();

    // Parse the JSON
    let json: serde_json::Value = serde_json::from_str(&result).unwrap();

    // The theta encoding should NOT have a scale with range when start is 0 (default)
    let layer = data_layer(&json, 0);
    let theta = &layer["encoding"]["theta"];
    assert!(theta.is_object(), "theta encoding should exist");

    // Either no scale, or no range in scale (since default is 0)
    if let Some(scale) = theta.get("scale") {
        assert!(
            scale.get("range").is_none(),
            "theta scale should not have range when start is 0"
        );
    }
}

#[test]
fn test_polar_project_with_end() {
    let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
    let query = r#"
        SELECT * FROM (VALUES ('A', 10), ('B', 20)) AS t(category, value)
        VISUALISE value AS y, category AS fill
        DRAW bar
        PROJECT y, x TO polar SETTING start => -90, end => 90
    "#;

    let spec = reader.execute(query).unwrap();
    let writer = VegaLiteWriter::new();
    let result = writer.render(&spec).unwrap();

    let json: serde_json::Value = serde_json::from_str(&result).unwrap();
    let layer = data_layer(&json, 0);
    let theta = &layer["encoding"]["theta"];
    let range = theta["scale"]["range"].as_array().unwrap();

    // -90° = -π/2 ≈ -1.5708, 90° = π/2 ≈ 1.5708
    let start = range[0].as_f64().unwrap();
    let end = range[1].as_f64().unwrap();
    assert!(
        (start - (-std::f64::consts::FRAC_PI_2)).abs() < 0.001,
        "start should be -π/2 (-90 degrees), got {}",
        start
    );
    assert!(
        (end - std::f64::consts::FRAC_PI_2).abs() < 0.001,
        "end should be π/2 (90 degrees), got {}",
        end
    );
}

#[test]
fn test_polar_project_with_end_only() {
    // Test using end without explicit start (start defaults to 0)
    let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
    let query = r#"
        SELECT * FROM (VALUES ('A', 10), ('B', 20)) AS t(category, value)
        VISUALISE value AS y, category AS fill
        DRAW bar
        PROJECT y, x TO polar SETTING end => 180
    "#;

    let spec = reader.execute(query).unwrap();
    let writer = VegaLiteWriter::new();
    let result = writer.render(&spec).unwrap();

    let json: serde_json::Value = serde_json::from_str(&result).unwrap();
    let layer = data_layer(&json, 0);
    let theta = &layer["encoding"]["theta"];
    let range = theta["scale"]["range"].as_array().unwrap();

    // start=0 (default), end=180° = π
    let start = range[0].as_f64().unwrap();
    let end = range[1].as_f64().unwrap();
    assert!(
        start.abs() < 0.001,
        "start should be 0 (default), got {}",
        start
    );
    assert!(
        (end - std::f64::consts::PI).abs() < 0.001,
        "end should be π (180 degrees), got {}",
        end
    );
}

#[test]
fn test_polar_encoding_keys_independent_of_user_names() {
    // This test verifies that polar projections always produce theta/radius encoding keys
    // in Vega-Lite output, regardless of what position names the user specified in PROJECT.
    // This is critical because Vega-Lite expects specific channel names for polar marks.
    let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

    // Helper to check encoding keys
    fn check_encoding_keys(json: &serde_json::Value, test_name: &str) {
        let layer = data_layer(json, 0);
        assert!(
            layer["encoding"].get("theta").is_some(),
            "{} should produce theta encoding, got keys: {:?}",
            test_name,
            layer["encoding"]
                .as_object()
                .map(|o| o.keys().collect::<Vec<_>>())
        );
        // Also verify no x or y keys exist (they should be mapped to theta/radius)
        assert!(
            layer["encoding"].get("x").is_none(),
            "{} should NOT have x encoding in polar mode",
            test_name
        );
        assert!(
            layer["encoding"].get("y").is_none(),
            "{} should NOT have y encoding in polar mode",
            test_name
        );
    }

    // Test case 1: PROJECT y, x TO polar (y as pos1→radius, x as pos2→theta)
    let query1 = r#"
        SELECT * FROM (VALUES ('A', 10), ('B', 20)) AS t(category, value)
        VISUALISE value AS y, category AS fill
        DRAW bar
        PROJECT y, x TO polar
    "#;
    let spec1 = reader.execute(query1).unwrap();
    let writer = VegaLiteWriter::new();
    let result1 = writer.render(&spec1).unwrap();
    let json1: serde_json::Value = serde_json::from_str(&result1).unwrap();
    check_encoding_keys(&json1, "PROJECT y, x TO polar");

    // Test case 2: PROJECT x, y TO polar (x as pos1→radius, y as pos2→theta)
    let query2 = r#"
        SELECT * FROM (VALUES ('A', 10), ('B', 20)) AS t(category, value)
        VISUALISE value AS x, category AS fill
        DRAW bar
        PROJECT x, y TO polar
    "#;
    let spec2 = reader.execute(query2).unwrap();
    let result2 = writer.render(&spec2).unwrap();
    let json2: serde_json::Value = serde_json::from_str(&result2).unwrap();
    check_encoding_keys(&json2, "PROJECT x, y TO polar");

    // Test case 3: PROJECT TO polar (default radius/angle names)
    let query3 = r#"
        SELECT * FROM (VALUES ('A', 10), ('B', 20)) AS t(category, value)
        VISUALISE value AS angle, category AS fill
        DRAW bar
        PROJECT TO polar
    "#;
    let spec3 = reader.execute(query3).unwrap();
    let result3 = writer.render(&spec3).unwrap();
    let json3: serde_json::Value = serde_json::from_str(&result3).unwrap();
    check_encoding_keys(&json3, "PROJECT TO polar");

    // Test case 4: PROJECT a, b TO polar (custom aesthetic names)
    let query4 = r#"
        SELECT * FROM (VALUES ('A', 10), ('B', 20)) AS t(category, value)
        VISUALISE value AS a, category AS fill
        DRAW bar
        PROJECT a, b TO polar
    "#;
    let spec4 = reader.execute(query4).unwrap();
    let result4 = writer.render(&spec4).unwrap();
    let json4: serde_json::Value = serde_json::from_str(&result4).unwrap();
    check_encoding_keys(&json4, "PROJECT a, b TO polar (custom names)");
}

#[test]
fn test_cartesian_encoding_keys_with_custom_names() {
    // This test verifies that cartesian projections produce x/y encoding keys
    // even when custom position names are used in PROJECT.
    let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

    fn check_cartesian_keys(json: &serde_json::Value, test_name: &str) {
        let layer = data_layer(json, 0);
        assert!(
            layer["encoding"].get("x").is_some(),
            "{} should produce x encoding, got keys: {:?}",
            test_name,
            layer["encoding"]
                .as_object()
                .map(|o| o.keys().collect::<Vec<_>>())
        );
        // Verify no theta/radius keys exist
        assert!(
            layer["encoding"].get("theta").is_none(),
            "{} should NOT have theta encoding in cartesian mode",
            test_name
        );
    }

    // Test case: PROJECT a, b TO cartesian (custom aesthetic names)
    let query = r#"
        SELECT * FROM (VALUES ('A', 10), ('B', 20)) AS t(category, value)
        VISUALISE category AS a, value AS b
        DRAW bar
        PROJECT a, b TO cartesian
    "#;
    let spec = reader.execute(query).unwrap();
    let writer = VegaLiteWriter::new();
    let result = writer.render(&spec).unwrap();
    let json: serde_json::Value = serde_json::from_str(&result).unwrap();
    check_cartesian_keys(&json, "PROJECT a, b TO cartesian (custom names)");
}

#[test]
fn test_register_and_query() {
    let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

    let df = df! {
        "x" => vec![1i32, 2, 3],
        "y" => vec![10i32, 20, 30],
    }
    .unwrap();

    reader.register("my_data", df, false).unwrap();

    let query = "SELECT * FROM my_data VISUALISE x, y DRAW point";
    let spec = reader.execute(query).unwrap();

    assert_eq!(spec.metadata().rows, 3);
    // Aesthetics are transformed to internal names (x -> pos1)
    assert!(spec.metadata().columns.contains(&"pos1".to_string()));

    let writer = VegaLiteWriter::new();
    let result = writer.render(&spec).unwrap();
    assert!(result.contains("point"));
}

#[test]
fn test_register_and_join() {
    let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

    let sales = df! {
        "id" => vec![1i32, 2, 3],
        "amount" => vec![100i32, 200, 300],
        "product_id" => vec![1i32, 1, 2],
    }
    .unwrap();

    let products = df! {
        "id" => vec![1i32, 2],
        "name" => vec!["Widget", "Gadget"],
    }
    .unwrap();

    reader.register("sales", sales, false).unwrap();
    reader.register("products", products, false).unwrap();

    let query = r#"
        SELECT s.id, s.amount, p.name
        FROM sales s
        JOIN products p ON s.product_id = p.id
        VISUALISE id AS x, amount AS y
        DRAW bar
    "#;

    let spec = reader.execute(query).unwrap();
    assert_eq!(spec.metadata().rows, 3);
}

#[test]
fn test_execute_no_viz_fails() {
    let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
    let query = "SELECT 1 as x, 2 as y";

    let result = reader.execute(query);
    assert!(result.is_err());
}

#[test]
fn test_binned_fill_legend_renders_threshold_scale() {
    // End-to-end test for binned fill scale rendering to Vega-Lite
    // Verifies that binned material aesthetics use threshold scale type
    let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

    // Create data with values that span the binned range
    // Binned scales use FROM [min, max] for range and SETTING breaks => [...] for explicit breaks
    let query = r#"
        SELECT * FROM (VALUES
            (1, 10, 15.0),
            (2, 20, 35.0),
            (3, 30, 55.0),
            (4, 40, 85.0)
        ) AS t(x, y, value)
        VISUALISE
        DRAW point MAPPING x AS x, y AS y, value AS fill
        SCALE BINNED fill FROM [0, 100] TO viridis SETTING breaks => [0, 25, 50, 75, 100]
    "#;

    let spec = reader.execute(query).unwrap();

    // Verify spec structure
    assert_eq!(spec.plot().layers.len(), 1);
    // Note: scales may include auto-generated x/y scales plus the explicit fill scale
    assert!(
        spec.plot().find_scale("fill").is_some(),
        "Should have a fill scale"
    );

    // Render to Vega-Lite
    let writer = VegaLiteWriter::new();
    let result = writer.render(&spec).unwrap();
    let vl: serde_json::Value = serde_json::from_str(&result).unwrap();

    // Verify threshold scale type for fill
    let fill_scale = &vl["layer"][0]["encoding"]["fill"]["scale"];
    assert_eq!(
        fill_scale["type"],
        "threshold",
        "Binned fill should use threshold scale type. Got: {}",
        serde_json::to_string_pretty(&vl["layer"][0]["encoding"]["fill"]).unwrap()
    );

    // Verify internal breaks as domain (excludes first and last terminals)
    // breaks = [0, 25, 50, 75, 100] → domain = [25, 50, 75]
    let domain = fill_scale["domain"].as_array().unwrap();
    assert_eq!(
        domain.len(),
        3,
        "Threshold domain should have internal breaks only. Got: {:?}",
        domain
    );
    assert_eq!(domain[0], 25.0);
    assert_eq!(domain[1], 50.0);
    assert_eq!(domain[2], 75.0);

    // Verify color output - viridis palette gets expanded to an explicit range array
    // for threshold scales (Vega-Lite needs explicit colors for threshold domain)
    assert!(
        fill_scale["range"].is_array() || fill_scale["scheme"] == "viridis",
        "Should have color range or scheme. Got scale: {}",
        serde_json::to_string_pretty(fill_scale).unwrap()
    );

    // Verify legend values
    // For `fill` alone (single binned legend scale), uses gradient legend with all 5 break values
    // For symbol legends (multiple binned scales or non-gradient aesthetics), would have N-1 values
    let legend_values = &vl["layer"][0]["encoding"]["fill"]["legend"]["values"];
    assert!(
        legend_values.is_array(),
        "Legend should have values array. Got: {}",
        serde_json::to_string_pretty(&vl["layer"][0]["encoding"]["fill"]["legend"]).unwrap()
    );
    let values = legend_values.as_array().unwrap();
    assert_eq!(
        values.len(),
        5,
        "Gradient legend should have all 5 break values. Got: {:?}",
        values
    );
}

#[test]
fn test_binned_color_legend_with_label_mapping() {
    // Test binned color scale with custom labels renders correctly
    let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

    let query = r#"
        SELECT * FROM (VALUES
            (1, 10, 20.0),
            (2, 20, 60.0),
            (3, 30, 90.0)
        ) AS t(x, y, score)
        VISUALISE
        DRAW point MAPPING x AS x, y AS y, score AS color
        SCALE BINNED color FROM [0, 100] TO ['blue', 'yellow', 'red'] SETTING breaks => [0, 50, 100]
            RENAMING 0 => 'Low', 50 => 'High'
    "#;

    let spec = reader.execute(query).unwrap();

    let writer = VegaLiteWriter::new();
    let result = writer.render(&spec).unwrap();
    let vl: serde_json::Value = serde_json::from_str(&result).unwrap();

    // Verify threshold scale
    // Note: "color" aesthetic is mapped to "stroke" for point geom (not fill)
    let encoding = if vl["layer"].is_array() {
        &vl["layer"][0]["encoding"]
    } else {
        &vl["encoding"]
    };
    // Find the stroke or fill encoding (color maps to one of these)
    let color_encoding = if encoding["stroke"].is_object() {
        &encoding["stroke"]
    } else {
        &encoding["fill"]
    };
    assert_eq!(
        color_encoding["scale"]["type"],
        "threshold",
        "Binned color should use threshold scale. Got encoding: {}",
        serde_json::to_string_pretty(color_encoding).unwrap()
    );

    // Verify labelExpr exists for custom labels
    let legend = &color_encoding["legend"];
    assert!(
        legend["labelExpr"].is_string(),
        "Legend should have labelExpr for custom labels. Got legend: {}",
        serde_json::to_string_pretty(legend).unwrap()
    );

    let label_expr = legend["labelExpr"].as_str().unwrap_or("");
    // For symbol legends, VL generates range-style labels like "0 – 50"
    // Our labelExpr should map these to custom range formats
    assert!(
        label_expr.contains("Low") || label_expr.contains("High"),
        "labelExpr should contain custom labels, got: {}",
        label_expr
    );
}

#[test]
fn test_polar_project_with_inner() {
    let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
    let query = r#"
        SELECT * FROM (VALUES ('A', 10), ('B', 20)) AS t(category, value)
        VISUALISE value AS y, category AS fill
        DRAW bar
        PROJECT y, x TO polar SETTING inner => 0.5
    "#;

    let spec = reader.execute(query).unwrap();
    let writer = VegaLiteWriter::new();
    let result = writer.render(&spec).unwrap();

    let json: serde_json::Value = serde_json::from_str(&result).unwrap();
    let layer = data_layer(&json, 0);

    // Check radius scale has range with expressions
    let radius = &layer["encoding"]["radius"];
    assert!(radius["scale"]["range"].is_array());
    let range = radius["scale"]["range"].as_array().unwrap();

    // First element should be inner proportion expression
    assert!(
        range[0]["expr"].as_str().unwrap().contains("0.5"),
        "Inner radius expression should contain 0.5, got: {:?}",
        range[0]
    );

    // Second element should be the outer radius expression
    assert!(
        range[1]["expr"]
            .as_str()
            .unwrap()
            .contains("min(width, height) / 2"),
        "Outer radius expression should contain min(width, height) / 2, got: {:?}",
        range[1]
    );
}

#[test]
fn test_stacked_bar_chart() {
    // Test stacked bar chart via position => 'stack'
    let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
    let query = r#"
        SELECT * FROM (VALUES
            ('A', 'X', 10),
            ('A', 'Y', 20),
            ('B', 'X', 15),
            ('B', 'Y', 25)
        ) AS t(cat, grp, val)
        VISUALISE
        DRAW bar MAPPING cat AS x, val AS y, grp AS fill
        SETTING position => 'stack'
    "#;

    let spec = reader.execute(query).unwrap();
    let writer = VegaLiteWriter::new();
    let result = writer.render(&spec).unwrap();

    let json: serde_json::Value = serde_json::from_str(&result).unwrap();
    let layer = data_layer(&json, 0);

    // Verify y and y2 encodings exist (stacked bars use y/y2 for range)
    let encoding = &layer["encoding"];
    assert!(encoding["y"].is_object(), "Should have y encoding");
    assert!(
        encoding["y2"].is_object(),
        "Should have y2 encoding for stacked bars"
    );

    // Verify Vega-Lite stacking is disabled (we handle it ourselves)
    assert!(
        encoding["y"]["stack"].is_null(),
        "y encoding should have stack: null to disable VL stacking. Got: {}",
        serde_json::to_string_pretty(&encoding["y"]).unwrap()
    );
}

#[cfg(feature = "builtin-data")]
#[test]
fn test_stacked_bar_chart_dummy_x() {
    // Test stacked bar chart with no x mapping (dummy x column)
    // This is the case where only fill is mapped: all bars at same x position should stack
    let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
    let query = r#"
        VISUALISE FROM ggsql:penguins
        DRAW bar MAPPING species AS fill
    "#;

    let spec = reader.execute(query).unwrap();
    let writer = VegaLiteWriter::new();
    let result = writer.render(&spec).unwrap();

    let json: serde_json::Value = serde_json::from_str(&result).unwrap();
    let layer = data_layer(&json, 0);

    // Verify y and y2 encodings exist (stacked bars use y/y2 for range)
    let encoding = &layer["encoding"];
    assert!(encoding["y"].is_object(), "Should have y encoding");
    assert!(
        encoding["y2"].is_object(),
        "Should have y2 encoding for stacked bars with dummy x. Encoding: {}",
        serde_json::to_string_pretty(encoding).unwrap()
    );

    // Verify Vega-Lite stacking is disabled (we handle it ourselves)
    assert!(
        encoding["y"]["stack"].is_null(),
        "y encoding should have stack: null to disable VL stacking. Got: {}",
        serde_json::to_string_pretty(&encoding["y"]).unwrap()
    );
}

#[cfg(feature = "builtin-data")]
#[test]
fn test_boxplot_dummy_x() {
    // Boxplot with only y mapped: should render a single boxplot of the
    // whole distribution and suppress the categorical x axis.
    let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
    let query = r#"
        VISUALISE FROM ggsql:penguins
        DRAW boxplot MAPPING bill_len AS y
    "#;

    let spec = reader.execute(query).unwrap();
    let writer = VegaLiteWriter::new();
    let result = writer.render(&spec).unwrap();

    let json: serde_json::Value = serde_json::from_str(&result).unwrap();
    // Boxplot is a composite renderer (multiple sub-layers). Check that
    // the first layer's x encoding suppresses its axis.
    let layer = data_layer(&json, 0);
    let encoding = &layer["encoding"];
    assert!(
        encoding["x"]["axis"].is_null(),
        "Boxplot dummy x should have axis: null. Encoding: {}",
        serde_json::to_string_pretty(encoding).unwrap()
    );
}

#[cfg(feature = "builtin-data")]
#[test]
fn test_violin_dummy_x() {
    // Violin with only y mapped: single violin spanning the whole dataset.
    let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
    let query = r#"
        VISUALISE FROM ggsql:penguins
        DRAW violin MAPPING bill_len AS y
    "#;

    let spec = reader.execute(query).unwrap();
    let writer = VegaLiteWriter::new();
    let result = writer.render(&spec).unwrap();

    let json: serde_json::Value = serde_json::from_str(&result).unwrap();
    let layer = data_layer(&json, 0);
    let encoding = &layer["encoding"];
    assert!(
        encoding["x"]["axis"].is_null(),
        "Violin dummy x should have axis: null. Encoding: {}",
        serde_json::to_string_pretty(encoding).unwrap()
    );
}

#[cfg(feature = "builtin-data")]
#[test]
fn test_point_dummy_x() {
    // Point with only y mapped: strip plot at a single dummy x position.
    let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
    let query = r#"
        VISUALISE FROM ggsql:penguins
        DRAW point MAPPING bill_len AS y
    "#;

    let spec = reader.execute(query).unwrap();
    let writer = VegaLiteWriter::new();
    let result = writer.render(&spec).unwrap();

    let json: serde_json::Value = serde_json::from_str(&result).unwrap();
    let layer = data_layer(&json, 0);
    let encoding = &layer["encoding"];
    assert!(
        encoding["x"]["axis"].is_null(),
        "Point dummy x should have axis: null. Encoding: {}",
        serde_json::to_string_pretty(encoding).unwrap()
    );
}

#[test]
fn test_range_dummy_x() {
    // Range with only ymin/ymax mapped: a single vertical interval.
    let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
    let query = r#"
        SELECT 10.0 AS lo, 20.0 AS hi
        VISUALISE
        DRAW range MAPPING lo AS ymin, hi AS ymax
    "#;

    let spec = reader.execute(query).unwrap();
    let writer = VegaLiteWriter::new();
    let result = writer.render(&spec).unwrap();

    let json: serde_json::Value = serde_json::from_str(&result).unwrap();
    let layer = data_layer(&json, 0);
    let encoding = &layer["encoding"];
    assert!(
        encoding["x"]["axis"].is_null(),
        "Range dummy x should have axis: null. Encoding: {}",
        serde_json::to_string_pretty(encoding).unwrap()
    );
}

#[cfg(feature = "builtin-data")]
#[test]
fn test_point_dummy_y() {
    // Symmetric to test_point_dummy_x: only x mapped means dummy y.
    let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
    let query = r#"
        VISUALISE FROM ggsql:penguins
        DRAW point MAPPING bill_len AS x
    "#;

    let spec = reader.execute(query).unwrap();
    let writer = VegaLiteWriter::new();
    let result = writer.render(&spec).unwrap();

    let json: serde_json::Value = serde_json::from_str(&result).unwrap();
    let layer = data_layer(&json, 0);
    let encoding = &layer["encoding"];
    assert!(
        encoding["y"]["axis"].is_null(),
        "Point dummy y should have axis: null. Encoding: {}",
        serde_json::to_string_pretty(encoding).unwrap()
    );
}

#[cfg(feature = "builtin-data")]
#[test]
fn test_point_dummy_both_with_aggregate() {
    // Both axes omitted, but aggregate gives the single point meaning:
    // a count of all rows at the dummy x/y intersection.
    let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
    let query = r#"
        VISUALISE FROM ggsql:penguins
        DRAW point MAPPING bill_len AS size
        SETTING aggregate => 'size:count'
    "#;

    let spec = reader.execute(query).unwrap();
    let writer = VegaLiteWriter::new();
    let result = writer.render(&spec).unwrap();

    let json: serde_json::Value = serde_json::from_str(&result).unwrap();
    let layer = data_layer(&json, 0);
    let encoding = &layer["encoding"];
    assert!(
        encoding["x"]["axis"].is_null(),
        "Both-dummy point should hide x axis. Encoding: {}",
        serde_json::to_string_pretty(encoding).unwrap()
    );
    assert!(
        encoding["y"]["axis"].is_null(),
        "Both-dummy point should hide y axis. Encoding: {}",
        serde_json::to_string_pretty(encoding).unwrap()
    );
}

#[cfg(feature = "builtin-data")]
#[test]
fn test_point_dummy_x_with_aggregate() {
    // Point with aggregate SETTING and no x mapping: should aggregate the
    // whole dataset to a single point and suppress the dummy x axis.
    let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
    let query = r#"
        VISUALISE FROM ggsql:penguins
        DRAW point MAPPING bill_len AS y
        SETTING aggregate => 'mean'
    "#;

    let spec = reader.execute(query).unwrap();
    let writer = VegaLiteWriter::new();
    let result = writer.render(&spec).unwrap();

    let json: serde_json::Value = serde_json::from_str(&result).unwrap();
    let layer = data_layer(&json, 0);
    let encoding = &layer["encoding"];
    assert!(
        encoding["x"]["axis"].is_null(),
        "Aggregated point with dummy x should have axis: null. Encoding: {}",
        serde_json::to_string_pretty(encoding).unwrap()
    );
}

#[cfg(feature = "builtin-data")]
#[test]
fn test_bar_chart_with_expand_setting() {
    // Test bar chart with SCALE y SETTING expand - should work even when y is stat-derived
    // This tests that:
    // 1. Scale type inference works for stat-generated count columns
    // 2. Stacking still works (y2 encoding exists) when SCALE y is specified
    let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
    let query = r#"
        VISUALISE FROM ggsql:penguins
        DRAW bar MAPPING species AS fill
        SCALE y SETTING expand => [0.05, 0.05]
    "#;

    let spec = reader.execute(query).unwrap();
    let writer = VegaLiteWriter::new();
    let result = writer.render(&spec).unwrap();

    // Should succeed without "discrete scale does not support SETTING 'expand'" error
    let json: serde_json::Value = serde_json::from_str(&result).unwrap();
    let layer = data_layer(&json, 0);

    // Verify stacking works (y2 encoding exists for stacked bars)
    let encoding = &layer["encoding"];
    assert!(
        encoding["y2"].is_object(),
        "Should have y2 encoding for stacked bars. Encoding: {}",
        serde_json::to_string_pretty(encoding).unwrap()
    );
}

#[test]
fn test_dodged_bar_chart() {
    // Test dodged bar chart via position => 'dodge'
    let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
    let query = r#"
        SELECT * FROM (VALUES
            ('A', 'X', 10),
            ('A', 'Y', 20),
            ('B', 'X', 15),
            ('B', 'Y', 25)
        ) AS t(cat, grp, val)
        VISUALISE
        DRAW bar MAPPING cat AS x, val AS y, grp AS fill
        SETTING position => 'dodge'
    "#;

    let spec = reader.execute(query).unwrap();
    let writer = VegaLiteWriter::new();
    let result = writer.render(&spec).unwrap();

    let json: serde_json::Value = serde_json::from_str(&result).unwrap();
    let layer = data_layer(&json, 0);

    // Verify xOffset encoding exists (dodged bars use xOffset for displacement)
    let encoding = &layer["encoding"];
    assert!(
        encoding["xOffset"].is_object(),
        "Should have xOffset encoding for dodged bars. Encoding: {}",
        serde_json::to_string_pretty(encoding).unwrap()
    );

    // Verify bar width uses bandwidth expression with adjusted_width for dodged bars
    // For 2 groups with default width 0.9: adjusted_width = 0.9 / 2 = 0.45
    let mark = &layer["mark"];
    let width_expr = mark["width"]["expr"].as_str();
    assert!(
        width_expr.is_some(),
        "Dodged bars should have expression-based width. Mark: {}",
        serde_json::to_string_pretty(mark).unwrap()
    );
    let expr = width_expr.unwrap();
    assert!(
        expr.contains("bandwidth('x')") && expr.contains("0.45"),
        "Width expression should use bandwidth('x') * adjusted_width, got: {}",
        expr
    );
}

#[test]
fn test_position_identity_default() {
    // Test that identity position (default) doesn't modify data
    let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
    let query = r#"
        SELECT * FROM (VALUES
            ('A', 10),
            ('B', 20)
        ) AS t(cat, val)
        VISUALISE
        DRAW bar MAPPING cat AS x, val AS y
    "#;

    let spec = reader.execute(query).unwrap();
    let writer = VegaLiteWriter::new();
    let result = writer.render(&spec).unwrap();

    let json: serde_json::Value = serde_json::from_str(&result).unwrap();
    let layer = data_layer(&json, 0);

    // Verify no xOffset encoding (identity position)
    let encoding = &layer["encoding"];
    assert!(
        encoding.get("xOffset").is_none(),
        "Identity position should not have xOffset encoding"
    );
}

#[test]
fn test_label_with_flipped_project() {
    // End-to-end test: LABEL x/y with PROJECT y, x TO cartesian
    // Labels should be correctly applied to the flipped axes
    let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
    let query = r#"
        SELECT * FROM (VALUES (1, 10), (2, 20)) AS t(x, y)
        VISUALISE
        DRAW bar MAPPING x AS y, y AS x
        PROJECT y, x TO cartesian
        LABEL x => 'Value', y => 'Category'
    "#;

    let spec = reader.execute(query).unwrap();
    let writer = VegaLiteWriter::new();
    let result = writer.render(&spec).unwrap();

    let json: serde_json::Value = serde_json::from_str(&result).unwrap();
    let layer = data_layer(&json, 0);
    let encoding = &layer["encoding"];

    // With PROJECT y, x TO cartesian:
    // - y is pos1 (first position), renders to VL x-axis in cartesian
    // - x is pos2 (second position), renders to VL y-axis in cartesian
    // So LABEL y => 'Category' should appear on VL x-axis, LABEL x => 'Value' on VL y-axis
    let x_title = encoding["x"]["title"].as_str();
    let y_title = encoding["y"]["title"].as_str();

    assert_eq!(
        x_title,
        Some("Category"),
        "x-axis should have 'Category' title (from LABEL y). Got encoding: {}",
        serde_json::to_string_pretty(encoding).unwrap()
    );
    assert_eq!(
        y_title,
        Some("Value"),
        "y-axis should have 'Value' title (from LABEL x). Got encoding: {}",
        serde_json::to_string_pretty(encoding).unwrap()
    );
}

#[test]
fn test_label_with_polar_project() {
    // End-to-end test: LABEL angle/radius with PROJECT TO polar
    let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
    let query = r#"
        SELECT * FROM (VALUES ('A', 10), ('B', 20)) AS t(category, value)
        VISUALISE value AS angle, category AS fill
        DRAW bar
        PROJECT TO polar
        LABEL angle => 'Angle', radius => 'Distance'
    "#;

    let spec = reader.execute(query).unwrap();
    let writer = VegaLiteWriter::new();
    let result = writer.render(&spec).unwrap();

    let json: serde_json::Value = serde_json::from_str(&result).unwrap();
    let layer = data_layer(&json, 0);
    let encoding = &layer["encoding"];

    // Verify theta encoding has the label
    let theta_title = encoding["theta"]["title"].as_str();
    assert_eq!(
        theta_title,
        Some("Angle"),
        "theta encoding should have 'Angle' title. Got encoding: {}",
        serde_json::to_string_pretty(encoding).unwrap()
    );
}
