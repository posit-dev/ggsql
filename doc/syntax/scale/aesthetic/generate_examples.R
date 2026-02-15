#!/usr/bin/env Rscript
# Generate SVG files for all color palettes referenced in color_cont.qmd and color_disc.qmd.
#
# This script parses the palette definitions from src/plot/scale/palettes.rs
# and creates horizontal gradient SVG files (continuous) and swatch SVG files (discrete).

# Continuous palettes referenced in color_cont.qmd
CONTINUOUS_PALETTES <- c(
  # Crameri Sequential
  "navia", "batlow", "batlowk", "batloww", "hawaii", "lajolla", "tokyo",
  "turku", "acton", "bamako", "bilbao", "buda", "davos", "devon", "glasgow",
  "grayc", "imola", "lapaz", "lipari", "nuuk", "oslo",
  # Crameri Multi-Sequential
  "bukavu", "fes", "oleron",
  # Crameri Diverging
  "vik", "berlin", "roma", "bam", "broc", "cork", "lisbon", "managua",
  "tofino", "vanimo",
  # Crameri Cyclic
  "romao", "bamo", "broco", "corko", "viko",
  # ColorBrewer Single-hue
  "blues", "greens", "oranges", "reds", "purples", "greys",
  # ColorBrewer Multi-hue
  "ylorbr", "ylorrd", "ylgn", "ylgnbu", "gnbu", "bugn", "bupu", "pubu",
  "pubugn", "purd", "rdpu", "orrd",
  # ColorBrewer Diverging
  "rdbu", "rdylbu", "rdylgn", "spectral", "brbg", "prgn", "piyg", "rdgy", "puor",
  # Matplotlib
  "viridis", "plasma", "magma", "inferno", "cividis"
)

# Discrete palettes referenced in color_disc.qmd
DISCRETE_PALETTES <- c(
  # ggsql default
  "ggsql10",
  # Tableau
  "tableau10",
  # D3
  "category10",
  # ColorBrewer qualitative
  "set1", "set2", "set3",
  "pastel1", "pastel2",
  "dark2",
  "paired",
  "accent",
  # Kelly
  "kelly22"
)

# Named linetypes with their stroke-dasharray values
# Format: name = c(dash, gap, dash, gap, ...)
# Empty vector means solid line
NAMED_LINETYPES <- list(
  solid    = c(),
  dashed   = c(6, 4),
  dotted   = c(1, 2),
  dotdash  = c(1, 2, 6, 2),
  longdash = c(10, 4),
  twodash  = c(6, 2, 2, 2)
)

# Shape definitions
# Closed shapes (filled)
SHAPES_CLOSED <- c("circle", "square", "diamond", "triangle-up", "triangle-down",
                   "star", "square-cross", "circle-plus", "square-plus")
# Open shapes (stroke only)
SHAPES_OPEN <- c("cross", "plus", "stroke", "vline", "asterisk", "bowtie")
# All shapes
SHAPES_ALL <- c(SHAPES_CLOSED, SHAPES_OPEN)

#' Parse palette color arrays from the Rust source file
#' @param rust_file Path to palettes.rs
#' @return Named list of color vectors
parse_palettes <- function(rust_file) {
  content <- readLines(rust_file, warn = FALSE)
  content <- paste(content, collapse = "\n")

  palettes <- list()


  # Pattern to match: pub const NAME: &[&str] = &[ ... ];
  # We need to find each palette definition
  pattern <- 'pub const ([A-Z_0-9]+): &\\[&str\\] = &\\[([^;]+)\\];'
  matches <- gregexpr(pattern, content, perl = TRUE)

  # Extract all matches
  all_matches <- regmatches(content, matches)[[1]]

  for (match in all_matches) {
    # Extract name
    name_match <- regmatches(match, regexec('pub const ([A-Z_0-9]+):', match, perl = TRUE))[[1]]
    if (length(name_match) < 2) next
    name <- tolower(name_match[2])

    # Extract colors
    colors_section <- sub('.*&\\[', '', match)
    colors_section <- sub('\\];.*', '', colors_section)
    color_matches <- gregexpr('"(#[0-9A-Fa-f]{6})"', colors_section, perl = TRUE)
    colors <- regmatches(colors_section, color_matches)[[1]]
    colors <- gsub('"', '', colors)

    if (length(colors) > 0) {
      palettes[[name]] <- colors
    }
  }

  palettes
}

#' Generate an SVG with a horizontal gradient bar (for continuous palettes)
#' @param colors Vector of hex colors
#' @param width SVG width in pixels
#' @param height SVG height in pixels
#' @return SVG content as string
generate_gradient_svg <- function(colors, width = 600, height = 60) {
  # Sample colors evenly for the gradient
  if (length(colors) <= 20) {
    sampled <- colors
  } else {
    # Sample evenly across the palette
    indices <- round(seq(1, length(colors), length.out = 20))
    sampled <- colors[indices]
  }

  # Build gradient stops
  stops <- vapply(seq_along(sampled), function(i) {
    offset <- (i - 1) / (length(sampled) - 1) * 100
    sprintf('      <stop offset="%.1f%%" stop-color="%s"/>', offset, sampled[i])
  }, character(1))

  stops_str <- paste(stops, collapse = "\n")

  svg <- sprintf(
'<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d">
  <defs>
    <linearGradient id="grad" x1="0%%" y1="0%%" x2="100%%" y2="0%%">
%s
    </linearGradient>
  </defs>
  <rect width="%d" height="%d" fill="url(#grad)" rx="3" ry="3"/>
</svg>',
    width, height, stops_str, width, height
  )

  svg
}

#' Generate an SVG with color swatches (for discrete palettes)
#' @param colors Vector of hex colors
#' @param swatch_width Width of each swatch in pixels
#' @param swatch_height Height of each swatch in pixels
#' @param gap Gap between swatches in pixels
#' @return SVG content as string
generate_swatch_svg <- function(colors, swatch_width = 40, swatch_height = 60, gap = 2) {
  n <- length(colors)
  total_width <- n * swatch_width + (n - 1) * gap

  # Build rectangles for each color
  rects <- vapply(seq_along(colors), function(i) {
    x <- (i - 1) * (swatch_width + gap)
    sprintf('  <rect x="%d" y="0" width="%d" height="%d" fill="%s" rx="3" ry="3"/>',
            x, swatch_width, swatch_height, colors[i])
  }, character(1))

  rects_str <- paste(rects, collapse = "\n")

  svg <- sprintf(
'<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d">
%s
</svg>',
    total_width, swatch_height, rects_str
  )

  svg
}

#' Generate an SVG showing a single linetype
#' @param name Name of the linetype
#' @param dasharray Stroke-dasharray values (empty for solid)
#' @param width SVG width in pixels
#' @param height SVG height in pixels
#' @param stroke_width Width of the line
#' @return SVG content as string
generate_linetype_svg <- function(name, dasharray, width = 200, height = 40, stroke_width = 3) {
  y <- height / 2

  if (length(dasharray) == 0) {
    dash_attr <- ""
  } else {
    # Scale dasharray values by stroke width (ggplot2 convention)
    scaled_dash <- dasharray * stroke_width
    dash_attr <- sprintf(' stroke-dasharray="%s"', paste(scaled_dash, collapse=" "))
  }

  svg <- sprintf(
'<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d">
  <line x1="10" y1="%.1f" x2="%d" y2="%.1f" stroke="#333333" stroke-width="%d"%s stroke-linecap="butt"/>
</svg>',
    width, height, y, width - 10, y, stroke_width, dash_attr
  )

  svg
}

#' Get SVG path data for a shape
#' @param name Shape name
#' @param size Size of the shape (radius)
#' @param cx Center x coordinate
#' @param cy Center y coordinate
#' @return SVG path string
get_shape_path <- function(name, size = 12, cx = 20, cy = 20) {
  # Area-equalized scale factors (matching src/plot/scale/shape.rs)
  # Reference: circle with radius 1.0 has area π
  # All shapes scaled to have approximately equal visual area

  path <- switch(name,
    "circle" = {
      # Circle as SVG circle element (handled separately)
      NULL
    },
    "square" = {
      # Half-side 0.71/0.8 = 0.89 of base size for equal area
      s <- size * 0.89
      sprintf("M%.1f,%.1f L%.1f,%.1f L%.1f,%.1f L%.1f,%.1f Z",
              cx - s, cy - s, cx + s, cy - s, cx + s, cy + s, cx - s, cy + s)
    },
    "diamond" = {
      # Half-diagonal 0.89/0.8 = 1.11 of base size for equal area
      d <- size * 1.11
      sprintf("M%.1f,%.1f L%.1f,%.1f L%.1f,%.1f L%.1f,%.1f Z",
              cx, cy - d, cx + d, cy, cx, cy + d, cx - d, cy)
    },
    "triangle-up" = {
      # Scaled up by 0.92/0.8 = 1.15 for equal area
      r <- size * 1.15
      h <- r * 0.75
      sprintf("M%.1f,%.1f L%.1f,%.1f L%.1f,%.1f Z",
              cx, cy - r, cx + r, cy + h, cx - r, cy + h)
    },
    "triangle-down" = {
      r <- size * 1.15
      h <- r * 0.75
      sprintf("M%.1f,%.1f L%.1f,%.1f L%.1f,%.1f Z",
              cx - r, cy - h, cx + r, cy - h, cx, cy + r)
    },
    "star" = {
      # Outer radius 0.95/0.8 = 1.19 of base size for equal area
      outer <- size * 1.19
      inner <- outer * 0.4
      points <- character(10)
      for (i in 0:9) {
        angle <- (i * 36 - 90) * pi / 180
        r <- if (i %% 2 == 0) outer else inner
        x <- cx + r * cos(angle)
        y <- cy + r * sin(angle)
        points[i + 1] <- sprintf("%.1f,%.1f", x, y)
      }
      paste0("M", points[1], " L", paste(points[-1], collapse = " L"), " Z")
    },
    "cross" = {
      # X shape - open shapes use base size
      s <- size
      sprintf("M%.1f,%.1f L%.1f,%.1f M%.1f,%.1f L%.1f,%.1f",
              cx - s, cy - s, cx + s, cy + s, cx + s, cy - s, cx - s, cy + s)
    },
    "plus" = {
      # + shape
      s <- size
      sprintf("M%.1f,%.1f L%.1f,%.1f M%.1f,%.1f L%.1f,%.1f",
              cx - s, cy, cx + s, cy, cx, cy - s, cx, cy + s)
    },
    "stroke" = {
      # Horizontal line
      s <- size
      sprintf("M%.1f,%.1f L%.1f,%.1f", cx - s, cy, cx + s, cy)
    },
    "vline" = {
      # Vertical line
      s <- size
      sprintf("M%.1f,%.1f L%.1f,%.1f", cx, cy - s, cx, cy + s)
    },
    "asterisk" = {
      # 6-pointed asterisk
      s <- size
      lines <- character(3)
      for (i in 0:2) {
        angle <- (i * 60) * pi / 180
        x1 <- cx + s * cos(angle)
        y1 <- cy + s * sin(angle)
        x2 <- cx - s * cos(angle)
        y2 <- cy - s * sin(angle)
        lines[i + 1] <- sprintf("M%.1f,%.1f L%.1f,%.1f", x1, y1, x2, y2)
      }
      paste(lines, collapse = " ")
    },
    "bowtie" = {
      # Two triangles pointing inward
      s <- size
      sprintf("M%.1f,%.1f L%.1f,%.1f L%.1f,%.1f Z M%.1f,%.1f L%.1f,%.1f L%.1f,%.1f Z",
              cx - s, cy - s * 0.7, cx, cy, cx - s, cy + s * 0.7,
              cx + s, cy - s * 0.7, cx, cy, cx + s, cy + s * 0.7)
    },
    "square-cross" = {
      # Square with X-shaped cutout (handled specially in generate_shape_svg)
      NULL
    },
    "circle-plus" = {
      # Circle with +-shaped cutout (handled specially in generate_shape_svg)
      NULL
    },
    "square-plus" = {
      # Square with +-shaped cutout (handled specially in generate_shape_svg)
      NULL
    },
    NULL
  )
  path
}


#' Generate an SVG showing a single shape
#' @param name Shape name
#' @param size Size of the SVG
#' @param is_closed Whether this is a closed (filled) shape
#' @return SVG content as string
generate_shape_svg <- function(name, size = 40, is_closed = TRUE) {
  cx <- size / 2
  cy <- size / 2
  shape_size <- size * 0.35

  # Special handling for circle
  if (name == "circle") {
    svg <- sprintf(
'<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d">
  <circle cx="%.1f" cy="%.1f" r="%.1f" fill="%s"/>
</svg>',
      size, size, cx, cy, shape_size,
      if (is_closed) "#333333" else "none"
    )
    return(svg)
  }

  # Special handling for circle-plus (circle divided into 4 quarters with constant-width gap)
  if (name == "circle-plus") {
    r <- shape_size
    g <- shape_size * 0.15 / sqrt(2)  # gap half-width, scaled to match X visually
    n <- 8  # points per quarter arc

    # Where circle intersects gap edge
    edge <- sqrt(r^2 - g^2)

    # Start angle for arc (where y = g on circle)
    start_angle <- asin(g / r)
    end_angle <- pi/2 - start_angle

    paths <- character(4)
    for (q in 0:3) {
      base_angle <- q * pi / 2

      # Inner corner point
      corner <- switch(q + 1,
        c(cx + g, cy + g),   # q=0: top-right
        c(cx - g, cy + g),   # q=1: top-left
        c(cx - g, cy - g),   # q=2: bottom-left
        c(cx + g, cy - g)    # q=3: bottom-right
      )

      # Point where gap meets circle (start of arc)
      gap_start <- switch(q + 1,
        c(cx + edge, cy + g),   # right edge of horizontal gap
        c(cx - g, cy + edge),   # top edge of vertical gap
        c(cx - edge, cy - g),   # left edge of horizontal gap
        c(cx + g, cy - edge)    # bottom edge of vertical gap
      )

      # Arc points
      arc_start <- base_angle + start_angle
      arc_span <- end_angle - start_angle
      arc_pts <- sapply(0:n, function(i) {
        t <- i / n
        angle <- arc_start + t * arc_span
        sprintf("%.1f,%.1f", cx + r * cos(angle), cy + r * sin(angle))
      })

      # Point where arc meets gap (end of arc)
      gap_end <- switch(q + 1,
        c(cx + g, cy + edge),   # top edge of vertical gap
        c(cx - edge, cy + g),   # left edge of horizontal gap
        c(cx - g, cy - edge),   # bottom edge of vertical gap
        c(cx + edge, cy - g)    # right edge of horizontal gap
      )

      # Build path: corner -> gap_start -> arc -> gap_end -> close
      paths[q + 1] <- sprintf("M%.1f,%.1f L%.1f,%.1f L%s L%.1f,%.1f Z",
        corner[1], corner[2],
        gap_start[1], gap_start[2],
        paste(arc_pts, collapse = " L"),
        gap_end[1], gap_end[2]
      )
    }
    combined_path <- paste(paths, collapse = " ")
    svg <- sprintf(
'<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d">
  <path d="%s" fill="#333333"/>
</svg>',
      size, size, combined_path
    )
    return(svg)
  }

  # Special handling for square-cross (square divided into 4 triangles by X)
  if (name == "square-cross") {
    s <- shape_size * 0.89
    g <- shape_size * 0.15  # gap half-width

    # 4 triangles (top, right, bottom, left)
    paths <- c(
      sprintf("M%.1f,%.1f L%.1f,%.1f L%.1f,%.1f Z",
              cx - s + g, cy - s, cx + s - g, cy - s, cx, cy - g),  # top
      sprintf("M%.1f,%.1f L%.1f,%.1f L%.1f,%.1f Z",
              cx + s, cy - s + g, cx + s, cy + s - g, cx + g, cy),  # right
      sprintf("M%.1f,%.1f L%.1f,%.1f L%.1f,%.1f Z",
              cx + s - g, cy + s, cx - s + g, cy + s, cx, cy + g),  # bottom
      sprintf("M%.1f,%.1f L%.1f,%.1f L%.1f,%.1f Z",
              cx - s, cy + s - g, cx - s, cy - s + g, cx - g, cy)   # left
    )
    combined_path <- paste(paths, collapse = " ")
    svg <- sprintf(
'<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d">
  <path d="%s" fill="#333333"/>
</svg>',
      size, size, combined_path
    )
    return(svg)
  }

  # Special handling for square-plus (square divided into 4 smaller squares by +)
  if (name == "square-plus") {
    s <- shape_size * 0.89
    g <- shape_size * 0.15 / sqrt(2)  # gap half-width, scaled to match X visually

    # 4 smaller squares in corners
    paths <- c(
      sprintf("M%.1f,%.1f L%.1f,%.1f L%.1f,%.1f L%.1f,%.1f Z",
              cx - s, cy - s, cx - g, cy - s, cx - g, cy - g, cx - s, cy - g),  # top-left
      sprintf("M%.1f,%.1f L%.1f,%.1f L%.1f,%.1f L%.1f,%.1f Z",
              cx + g, cy - s, cx + s, cy - s, cx + s, cy - g, cx + g, cy - g),  # top-right
      sprintf("M%.1f,%.1f L%.1f,%.1f L%.1f,%.1f L%.1f,%.1f Z",
              cx + g, cy + g, cx + s, cy + g, cx + s, cy + s, cx + g, cy + s),  # bottom-right
      sprintf("M%.1f,%.1f L%.1f,%.1f L%.1f,%.1f L%.1f,%.1f Z",
              cx - s, cy + g, cx - g, cy + g, cx - g, cy + s, cx - s, cy + s)   # bottom-left
    )
    combined_path <- paste(paths, collapse = " ")
    svg <- sprintf(
'<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d">
  <path d="%s" fill="#333333"/>
</svg>',
      size, size, combined_path
    )
    return(svg)
  }

  path <- get_shape_path(name, shape_size, cx, cy)
  if (is.null(path)) return(NULL)

  if (is_closed) {
    # Closed shapes: fill only, no stroke
    svg <- sprintf(
'<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d">
  <path d="%s" fill="#333333"/>
</svg>',
      size, size, path
    )
  } else {
    # Open shapes: stroke only, no fill
    svg <- sprintf(
'<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d">
  <path d="%s" fill="none" stroke="#333333" stroke-width="2.5" stroke-linecap="butt" stroke-linejoin="round"/>
</svg>',
      size, size, path
    )
  }

  svg
}

#' Generate sequential linetypes with evenly-spaced ink densities
#' @param count Number of linetypes to generate
#' @return List with names and dasharray values
generate_sequential_linetypes <- function(count) {
  if (count <= 0) return(list())
  if (count == 1) return(list(solid = c()))

  # Generate linetypes from sparse to solid

  # Ink density ranges from ~6.25% to 100%
  result <- list()

  for (i in seq_len(count)) {
    if (i == count) {
      # Last one is always solid
      result[[paste0("seq_", i)]] <- c()
    } else {
      # Calculate ink ratio (evenly spaced from min to max)
      min_ink <- 1/16
      max_ink <- 15/16
      ink_ratio <- min_ink + (i - 1) * (max_ink - min_ink) / (count - 1)

      # Convert to on/off pattern (cycle of 16 units)
      on_length <- round(ink_ratio * 16)
      on_length <- max(1, min(15, on_length))
      off_length <- 16 - on_length

      result[[paste0("seq_", i)]] <- c(on_length, off_length)
    }
  }

  result
}

# Main execution
main <- function() {
  # Determine paths
  script_dir <- if (interactive()) {
    getwd()
  } else {
    dirname(commandArgs(trailingOnly = FALSE)[grep("--file=", commandArgs(trailingOnly = FALSE))])
    script_dir <- sub("--file=", "", commandArgs(trailingOnly = FALSE)[grep("--file=", commandArgs(trailingOnly = FALSE))])
    dirname(script_dir)
  }

  # Try to find the repo root
  # Navigate up from doc/syntax/scale/palette to repo root
  repo_root <- normalizePath(file.path(script_dir, "..", "..", "..", ".."), mustWork = FALSE)
  palettes_file <- file.path(repo_root, "src", "plot", "scale", "palettes.rs")

 # If that doesn't work, try relative to current working directory
  if (!file.exists(palettes_file)) {
    # Try from repo root directly
    palettes_file <- "src/plot/scale/palettes.rs"
  }

  if (!file.exists(palettes_file)) {
    stop("Error: Could not find palettes.rs. Run from repo root or palette directory.")
  }

  # Output directory (subfolder called "examples")
  base_dir <- if (interactive()) {
    "doc/syntax/scale/palette"
  } else {
    script_dir
  }

  if (!dir.exists(base_dir)) {
    base_dir <- "."
  }

  output_dir <- file.path(base_dir, "examples")
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
    message(sprintf("Created output directory: %s", output_dir))
  }

  # Parse palettes from Rust source
  all_palettes <- parse_palettes(palettes_file)
  message(sprintf("Found %d palettes in source file", length(all_palettes)))

  # Generate gradient SVGs for continuous palettes
  generated <- 0
  missing <- character(0)

  message("\n=== Continuous Palettes (gradients) ===")
  for (name in CONTINUOUS_PALETTES) {
    if (!(name %in% names(all_palettes))) {
      missing <- c(missing, name)
      next
    }

    colors <- all_palettes[[name]]
    svg_content <- generate_gradient_svg(colors)
    output_file <- file.path(output_dir, sprintf("gradient_%s.svg", name))
    writeLines(svg_content, output_file)
    generated <- generated + 1
    message(sprintf("Generated: gradient_%s.svg (%d colors)", name, length(colors)))
  }

  # Generate swatch SVGs for discrete palettes
  message("\n=== Discrete Palettes (swatches) ===")
  for (name in DISCRETE_PALETTES) {
    if (!(name %in% names(all_palettes))) {
      missing <- c(missing, name)
      next
    }

    colors <- all_palettes[[name]]
    svg_content <- generate_swatch_svg(colors)
    output_file <- file.path(output_dir, sprintf("swatch_%s.svg", name))
    writeLines(svg_content, output_file)
    generated <- generated + 1
    message(sprintf("Generated: swatch_%s.svg (%d colors)", name, length(colors)))
  }

  # Generate linetype SVGs for named linetypes
  message("\n=== Named Linetypes ===")
  for (name in names(NAMED_LINETYPES)) {
    dasharray <- NAMED_LINETYPES[[name]]
    svg_content <- generate_linetype_svg(name, dasharray)
    output_file <- file.path(output_dir, sprintf("linetype_%s.svg", name))
    writeLines(svg_content, output_file)
    generated <- generated + 1
    message(sprintf("Generated: linetype_%s.svg", name))
  }

  # Generate sequential linetype examples (5 levels)
  message("\n=== Sequential Linetypes (5 levels) ===")
  seq_linetypes <- generate_sequential_linetypes(5)
  for (i in seq_along(seq_linetypes)) {
    name <- names(seq_linetypes)[i]
    dasharray <- seq_linetypes[[name]]
    svg_content <- generate_linetype_svg(name, dasharray)
    output_file <- file.path(output_dir, sprintf("linetype_%s.svg", name))
    writeLines(svg_content, output_file)
    generated <- generated + 1
    message(sprintf("Generated: linetype_%s.svg", name))
  }

  # Generate shape SVGs - closed shapes
  message("\n=== Closed Shapes ===")
  for (name in SHAPES_CLOSED) {
    svg_content <- generate_shape_svg(name, size = 40, is_closed = TRUE)
    if (!is.null(svg_content)) {
      output_file <- file.path(output_dir, sprintf("shape_%s.svg", gsub("-", "_", name)))
      writeLines(svg_content, output_file)
      generated <- generated + 1
      message(sprintf("Generated: shape_%s.svg", gsub("-", "_", name)))
    }
  }

  # Generate shape SVGs - open shapes
  message("\n=== Open Shapes ===")
  for (name in SHAPES_OPEN) {
    svg_content <- generate_shape_svg(name, size = 40, is_closed = FALSE)
    if (!is.null(svg_content)) {
      output_file <- file.path(output_dir, sprintf("shape_%s.svg", gsub("-", "_", name)))
      writeLines(svg_content, output_file)
      generated <- generated + 1
      message(sprintf("Generated: shape_%s.svg", gsub("-", "_", name)))
    }
  }

  message(sprintf("\nGenerated %d SVG files total", generated))

  if (length(missing) > 0) {
    message(sprintf("\nWarning: %d palettes not found in source:", length(missing)))
    for (name in missing) {
      message(sprintf("  - %s", name))
    }
  }

  invisible(generated)
}

main()
