export interface Example {
  name: string;
  query: string;
  section: string;
  loadExtension?: string;
}

export const examples: Example[] = [
  // === Layers ===
  {
    section: "Layers",
    name: "Area",
    query: `VISUALISE FROM ggsql:airquality
DRAW area 
  MAPPING Date AS x, Wind AS y`,
  },
  {
    section: "Layers",
    name: "Bar",
    query: `VISUALISE FROM ggsql:penguins
DRAW bar
    MAPPING species AS x`,
  },
  {
    section: "Layers",
    name: "Boxplot",
    query: `VISUALISE FROM ggsql:penguins
DRAW boxplot
  MAPPING species AS x, bill_len AS y, island AS fill`,
  },
  {
    section: "Layers",
    name: "Density",
    query: `VISUALISE bill_dep AS x, species AS colour FROM ggsql:penguins
  DRAW density MAPPING body_mass AS weight`,
  },
  {
    section: "Layers",
    name: "Histogram",
    query: `VISUALISE FROM ggsql:penguins
DRAW histogram
    MAPPING body_mass AS x`,
  },
  {
    section: "Layers",
    name: "Line",
    query: `VISUALISE FROM ggsql:airquality
DRAW line
    MAPPING Day AS x, Temp AS y, Month AS color`,
  },
  {
    section: "Layers",
    name: "Path",
    query: `WITH df(x, y, id) AS (VALUES
    (1.0, 1.0, 'A'),
    (2.0, 1.0, 'A'),
    (1.0, 3.0, 'A'),
    (3.0, 1.0, 'B'),
    (2.0, 3.0, 'B'),
    (3.0, 3.0, 'B')
)
VISUALIZE x, y FROM df
DRAW line
    MAPPING id AS colour`,
  },
  {
    section: "Layers",
    name: "Point",
    query: `SELECT * FROM ggsql:penguins
VISUALISE
DRAW point MAPPING bill_len AS x, bill_dep AS y, body_mass AS size, species AS color
LABEL title => 'Penguin Measurements', x => 'Bill Length (mm)', y => 'Bill Depth (mm)'`,
  },
  {
    section: "Layers",
    name: "Polygon",
    query: `WITH df(x, y, id) AS (VALUES
    (1.0, 1.0, 'A'),
    (2.0, 1.0, 'A'),
    (1.0, 3.0, 'A'),
    (3.0, 1.0, 'B'),
    (2.0, 3.0, 'B'),
    (3.0, 3.0, 'B')
)
VISUALIZE x, y FROM df
DRAW polygon
    MAPPING id AS colour`,
  },
  {
    section: "Layers",
    name: "Ribbon",
    query: `  VISUALISE FROM ggsql:airquality
  DRAW ribbon
    MAPPING Date AS x, Wind AS ymin, Temp AS ymax`,
  },
  {
    section: "Layers",
    name: "Violin",
    query: `VISUALISE species AS x, bill_dep AS y FROM ggsql:penguins
  DRAW violin`,
  },
  // === Scales ===
  {
    section: "Scales",
    name: "Binned",
    query: `VISUALISE bill_len AS x, bill_dep AS y, body_mass AS color FROM ggsql:penguins
DRAW point
SCALE BINNED color TO viridis`,
  },
  {
    section: "Scales",
    name: "Continuous",
    query: `VISUALISE bill_len AS x, bill_dep AS y FROM ggsql:penguins
DRAW point
SCALE x FROM [0, null]`,
  },
  {
    section: "Scales",
    name: "Discrete",
    query: `VISUALISE bill_len AS x, bill_dep AS y, island AS shape, island AS color FROM ggsql:penguins
DRAW point
  SETTING size => 6
SCALE shape TO ['star', 'circle', 'diamond']
SCALE color`,
  },
  {
    section: "Scales",
    name: "Identity",
    query: `WITH t(category, value, style) AS (VALUES
      ('A', 45, 'forestgreen'),
      ('B', 72, '#3401e3'),
      ('C', 38, 'hsl(150deg 30% 60%)')
)
VISUALISE category AS x, value AS y, style AS fill FROM t
DRAW bar
SCALE IDENTITY fill`,
  },
  {
    section: "Scales",
    name: "Ordinal",
    query: `VISUALISE Ozone AS x, Temp AS y FROM ggsql:airquality
DRAW point
    MAPPING Month AS color
SCALE ORDINAL color
    RENAMING * => '{}th month'`,
  },
  {
    section: "Scales",
    name: "Faceting",
    query: `VISUALISE sex AS x FROM ggsql:penguins
DRAW bar
FACET species
SCALE panel FROM ['Adelie', null]
    RENAMING null => 'The rest'`,
  },

  // === Aesthetics ===
  {
    section: "Aesthetics",
    name: "Position",
    query: `SELECT * FROM ggsql:penguins
VISUALISE
DRAW point MAPPING bill_len AS x, bill_dep AS y`,
  },
  {
    section: "Aesthetics",
    name: "Fill",
    query: `VISUALISE FROM ggsql:penguins
DRAW point
    MAPPING bill_dep AS x, body_mass AS y, species AS fill
    SETTING stroke => null
SCALE color TO category10`,
  },
  {
    section: "Aesthetics",
    name: "Opacity",
    query: `VISUALISE FROM ggsql:airquality
DRAW area 
  MAPPING Date AS x, Wind AS y
  SETTING opacity => 0.2`,
  },
  {
    section: "Aesthetics",
    name: "Linetype",
    query: `VISUALISE FROM ggsql:airquality
DRAW line
  MAPPING Day AS x, Temp AS y, Month AS linetype
SCALE ORDINAL linetype`,
  },
  {
    section: "Aesthetics",
    name: "Linewidth",
    query: `VISUALISE FROM ggsql:airquality
DRAW line
  MAPPING Day AS x, Temp AS y, Month AS colour
  SETTING linewidth => 5`,
  },
  {
    section: "Aesthetics",
    name: "Shape",
    query: `VISUALISE FROM ggsql:penguins
DRAW point
    MAPPING bill_dep AS x, body_mass AS y, species AS shape
    SETTING linewidth => 1, size => 5
SCALE shape TO ['star', 'bowtie', 'square-plus']`,
  },
  {
    section: "Aesthetics",
    name: "Size",
    query: `SELECT * FROM ggsql:penguins
VISUALISE
DRAW point MAPPING bill_len AS x, bill_dep AS y, body_mass AS size
LABEL title => 'Penguin Measurements', x => 'Bill Length (mm)', y => 'Bill Depth (mm)'`,
  },

  // === Extensions ===
  {
    section: "Extensions",
    name: "Wasm Extension",
    query: `-- Loaded from test_ext.wasm via the SQLite extension API
SELECT test_ext_hello() AS greeting`,
    loadExtension: "test_ext",
  },
  {
    section: "Extensions",
    name: "SpatiaLite",
    query: `-- SpatiaLite reprojects world cities from WGS84 lon/lat (EPSG:4326)
-- to Web Mercator metres (EPSG:3857) via PROJ, then plots them as a map.
WITH cities(name, lon, lat) AS (
  VALUES
    ('London',          -0.1276,  51.5074),
    ('New York',       -74.0060,  40.7128),
    ('Tokyo',          139.6917,  35.6895),
    ('Sydney',         151.2093, -33.8688),
    ('Cape Town',       18.4241, -33.9249),
    ('Rio de Janeiro', -43.1729, -22.9068),
    ('Moscow',          37.6173,  55.7558)
)
SELECT
  name,
  ST_X(ST_Transform(MakePoint(lon, lat, 4326), 3857)) AS x,
  ST_Y(ST_Transform(MakePoint(lon, lat, 4326), 3857)) AS y
FROM cities
VISUALISE x AS x, y AS y
DRAW point SETTING size => 6
DRAW text MAPPING name AS label SETTING vjust => 'bottom', offset => [0, -8]
LABEL
  title => 'World cities in Web Mercator (EPSG:3857)',
  subtitle => 'Reprojected from WGS84 with SpatiaLite ST_Transform (PROJ)',
  x => 'Easting (m)',
  y => 'Northing (m)'`,
    loadExtension: "mod_spatialite",
  },
  {
    section: "Extensions",
    name: "World map",
    query: `-- Country outlines from the built-in ggsql:world dataset. SpatiaLite
-- reprojects each country (PROJ, to equal-area EPSG:6933), simplifies it
-- (GEOS, 25 km) to thin the geometry, then a numbers table explodes every
-- polygon ring into ordered vertices for the path layer (ggsql can't pass
-- recursive CTEs to SQLite, so the vertex indices come from a join).
WITH
nums(i) AS (
  SELECT 1 + d0.d + 10 * d1.d + 100 * d2.d AS i
  FROM (SELECT 0 AS d UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9) d0,
       (SELECT 0 AS d UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9) d1,
       (SELECT 0 AS d UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9) d2
),
geo AS (
  SELECT name, continent,
         ST_Simplify(ST_Transform(CastToMulti(GeomFromWKB(geom, 4326)), 6933), 25000) AS g
  FROM ggsql:world
),
ring AS (
  SELECT name, continent, ST_ExteriorRing(ST_GeometryN(g, nums.i)) AS r, nums.i AS pidx
  FROM geo JOIN nums ON nums.i <= ST_NumGeometries(g)
)
SELECT
  ring.name || '-' || ring.pidx AS ring_id,
  ring.continent,
  nums.i AS vidx,
  ST_X(ST_PointN(ring.r, nums.i)) AS x,
  ST_Y(ST_PointN(ring.r, nums.i)) AS y
FROM ring JOIN nums ON nums.i <= ST_NumPoints(ring.r)
VISUALISE x AS x, y AS y
DRAW path MAPPING continent AS color SETTING linewidth => 0.5 PARTITION BY ring_id ORDER BY vidx
LABEL
  title => 'World country outlines (EPSG:6933 equal-area)',
  color => 'Continent'`,
    loadExtension: "mod_spatialite",
  },
];
