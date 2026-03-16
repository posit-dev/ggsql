export interface Example {
  name: string;
  query: string;
  section: string;
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
];
