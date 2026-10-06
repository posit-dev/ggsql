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

    let json: serde_json::Value = serde_json::from_str(&result).unwrap();

    let layer = data_layer(&json, 0);
    let theta = &layer["encoding"]["theta"];
    assert!(theta.is_object(), "theta encoding should exist");

    let scale = &theta["scale"];
    let range = scale["range"].as_array().unwrap();
    assert_eq!(range.len(), 2);

    let start = range[0].as_f64().unwrap();
    assert!(
        (start - std::f64::consts::FRAC_PI_2).abs() < 0.001,
        "start should be π/2 (90 degrees), got {}",
        start
    );

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

    let json: serde_json::Value = serde_json::from_str(&result).unwrap();

    let layer = data_layer(&json, 0);
    let theta = &layer["encoding"]["theta"];
    assert!(theta.is_object(), "theta encoding should exist");

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
        assert!(
            layer["encoding"].get("theta").is_none(),
            "{} should NOT have theta encoding in cartesian mode",
            test_name
        );
    }

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
    let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();

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

    assert_eq!(spec.plot().layers.len(), 1);
    assert!(
        spec.plot().find_scale("fill").is_some(),
        "Should have a fill scale"
    );

    let writer = VegaLiteWriter::new();
    let result = writer.render(&spec).unwrap();
    let vl: serde_json::Value = serde_json::from_str(&result).unwrap();

    let fill_scale = &vl["layer"][0]["encoding"]["fill"]["scale"];
    assert_eq!(
        fill_scale["type"],
        "threshold",
        "Binned fill should use threshold scale type. Got: {}",
        serde_json::to_string_pretty(&vl["layer"][0]["encoding"]["fill"]).unwrap()
    );

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

    assert!(
        fill_scale["range"].is_array() || fill_scale["scheme"] == "viridis",
        "Should have color range or scheme. Got scale: {}",
        serde_json::to_string_pretty(fill_scale).unwrap()
    );

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

    let encoding = if vl["layer"].is_array() {
        &vl["layer"][0]["encoding"]
    } else {
        &vl["encoding"]
    };
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

    let legend = &color_encoding["legend"];
    assert!(
        legend["labelExpr"].is_string(),
        "Legend should have labelExpr for custom labels. Got legend: {}",
        serde_json::to_string_pretty(legend).unwrap()
    );

    let label_expr = legend["labelExpr"].as_str().unwrap_or("");
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

    let radius = &layer["encoding"]["radius"];
    assert!(radius["scale"]["range"].is_array());
    let range = radius["scale"]["range"].as_array().unwrap();

    assert!(
        range[0]["expr"].as_str().unwrap().contains("0.5"),
        "Inner radius expression should contain 0.5, got: {:?}",
        range[0]
    );

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

    let encoding = &layer["encoding"];
    assert!(encoding["y"].is_object(), "Should have y encoding");
    assert!(
        encoding["y2"].is_object(),
        "Should have y2 encoding for stacked bars"
    );

    assert!(
        encoding["y"]["stack"].is_null(),
        "y encoding should have stack: null to disable VL stacking. Got: {}",
        serde_json::to_string_pretty(&encoding["y"]).unwrap()
    );
}

#[cfg(feature = "builtin-data")]
#[test]
fn test_stacked_bar_chart_dummy_x() {
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

    let encoding = &layer["encoding"];
    assert!(encoding["y"].is_object(), "Should have y encoding");
    assert!(
        encoding["y2"].is_object(),
        "Should have y2 encoding for stacked bars with dummy x. Encoding: {}",
        serde_json::to_string_pretty(encoding).unwrap()
    );

    assert!(
        encoding["y"]["stack"].is_null(),
        "y encoding should have stack: null to disable VL stacking. Got: {}",
        serde_json::to_string_pretty(&encoding["y"]).unwrap()
    );
}

#[cfg(feature = "builtin-data")]
#[test]
fn test_boxplot_dummy_x() {
    let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
    let query = r#"
        VISUALISE FROM ggsql:penguins
        DRAW boxplot MAPPING bill_len AS y
    "#;

    let spec = reader.execute(query).unwrap();
    let writer = VegaLiteWriter::new();
    let result = writer.render(&spec).unwrap();

    let json: serde_json::Value = serde_json::from_str(&result).unwrap();
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
    let reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
    let query = r#"
        VISUALISE FROM ggsql:penguins
        DRAW bar MAPPING species AS fill
        SCALE y SETTING expand => [0.05, 0.05]
    "#;

    let spec = reader.execute(query).unwrap();
    let writer = VegaLiteWriter::new();
    let result = writer.render(&spec).unwrap();

    let json: serde_json::Value = serde_json::from_str(&result).unwrap();
    let layer = data_layer(&json, 0);

    let encoding = &layer["encoding"];
    assert!(
        encoding["y2"].is_object(),
        "Should have y2 encoding for stacked bars. Encoding: {}",
        serde_json::to_string_pretty(encoding).unwrap()
    );
}

#[test]
fn test_dodged_bar_chart() {
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

    let encoding = &layer["encoding"];
    assert!(
        encoding["xOffset"].is_object(),
        "Should have xOffset encoding for dodged bars. Encoding: {}",
        serde_json::to_string_pretty(encoding).unwrap()
    );

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

    let encoding = &layer["encoding"];
    assert!(
        encoding.get("xOffset").is_none(),
        "Identity position should not have xOffset encoding"
    );
}

#[test]
fn test_label_with_flipped_project() {
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

    let theta_title = encoding["theta"]["title"].as_str();
    assert_eq!(
        theta_title,
        Some("Angle"),
        "theta encoding should have 'Angle' title. Got encoding: {}",
        serde_json::to_string_pretty(encoding).unwrap()
    );
}
