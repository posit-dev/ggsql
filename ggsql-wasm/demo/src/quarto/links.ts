import * as monaco from "monaco-editor";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface DocLink {
  url: string;
  label: string;
}

interface LinkMatch {
  startCol: number; // 1-based column
  endCol: number; // 1-based column (exclusive)
  link: DocLink;
}

// ---------------------------------------------------------------------------
// Keyword → doc URL mappings
// ---------------------------------------------------------------------------

const CLAUSE_LINKS: Record<string, DocLink> = {
  visualise: { url: "syntax/clause/visualise", label: "VISUALISE clause" },
  visualize: { url: "syntax/clause/visualise", label: "VISUALISE clause" },
  draw: { url: "syntax/clause/draw", label: "DRAW clause" },
  place: { url: "syntax/clause/place", label: "PLACE clause" },
  scale: { url: "syntax/clause/scale", label: "SCALE clause" },
  facet: { url: "syntax/clause/facet", label: "FACET clause" },
  project: { url: "syntax/clause/project", label: "PROJECT clause" },
  label: { url: "syntax/clause/label", label: "LABEL clause" },
};

const GEOM_LINKS: Record<string, DocLink> = {
  point: { url: "syntax/layer/type/point", label: "point layer" },
  line: { url: "syntax/layer/type/line", label: "line layer" },
  path: { url: "syntax/layer/type/path", label: "path layer" },
  bar: { url: "syntax/layer/type/bar", label: "bar layer" },
  area: { url: "syntax/layer/type/area", label: "area layer" },
  tile: { url: "syntax/layer/type/tile", label: "tile layer" },
  polygon: { url: "syntax/layer/type/polygon", label: "polygon layer" },
  ribbon: { url: "syntax/layer/type/ribbon", label: "ribbon layer" },
  histogram: { url: "syntax/layer/type/histogram", label: "histogram layer" },
  density: { url: "syntax/layer/type/density", label: "density layer" },
  smooth: { url: "syntax/layer/type/smooth", label: "smooth layer" },
  boxplot: { url: "syntax/layer/type/boxplot", label: "boxplot layer" },
  violin: { url: "syntax/layer/type/violin", label: "violin layer" },
  text: { url: "syntax/layer/type/text", label: "text layer" },
  segment: { url: "syntax/layer/type/segment", label: "segment layer" },
  rule: { url: "syntax/layer/type/rule", label: "rule layer" },
  linear: { url: "syntax/layer/type/linear", label: "linear layer" },
  range: { url: "syntax/layer/type/range", label: "range layer" },
};

const COORD_LINKS: Record<string, DocLink> = {
  cartesian: { url: "syntax/coord/cartesian", label: "cartesian coordinates" },
  polar: { url: "syntax/coord/polar", label: "polar coordinates" },
};

const SCALE_TYPE_LINKS: Record<string, DocLink> = {
  continuous: {
    url: "syntax/scale/type/continuous",
    label: "continuous scale",
  },
  discrete: { url: "syntax/scale/type/discrete", label: "discrete scale" },
  binned: { url: "syntax/scale/type/binned", label: "binned scale" },
  ordinal: { url: "syntax/scale/type/ordinal", label: "ordinal scale" },
  identity: { url: "syntax/scale/type/identity", label: "identity scale" },
};

const AESTHETIC_LINKS: Record<string, DocLink> = {
  x: { url: "syntax/scale/aesthetic/0_position", label: "position aesthetic" },
  y: { url: "syntax/scale/aesthetic/0_position", label: "position aesthetic" },
  xmin: {
    url: "syntax/scale/aesthetic/0_position",
    label: "position aesthetic",
  },
  xmax: {
    url: "syntax/scale/aesthetic/0_position",
    label: "position aesthetic",
  },
  ymin: {
    url: "syntax/scale/aesthetic/0_position",
    label: "position aesthetic",
  },
  ymax: {
    url: "syntax/scale/aesthetic/0_position",
    label: "position aesthetic",
  },
  xend: {
    url: "syntax/scale/aesthetic/0_position",
    label: "position aesthetic",
  },
  yend: {
    url: "syntax/scale/aesthetic/0_position",
    label: "position aesthetic",
  },
  color: { url: "syntax/scale/aesthetic/1_color", label: "color aesthetic" },
  colour: { url: "syntax/scale/aesthetic/1_color", label: "color aesthetic" },
  fill: { url: "syntax/scale/aesthetic/1_color", label: "color aesthetic" },
  stroke: { url: "syntax/scale/aesthetic/1_color", label: "color aesthetic" },
  opacity: {
    url: "syntax/scale/aesthetic/2_opacity",
    label: "opacity aesthetic",
  },
  linetype: {
    url: "syntax/scale/aesthetic/linetype",
    label: "linetype aesthetic",
  },
  linewidth: {
    url: "syntax/scale/aesthetic/linewidth",
    label: "linewidth aesthetic",
  },
  shape: { url: "syntax/scale/aesthetic/shape", label: "shape aesthetic" },
  size: { url: "syntax/scale/aesthetic/size", label: "size aesthetic" },
  panel: {
    url: "syntax/scale/aesthetic/Z_faceting",
    label: "faceting aesthetic",
  },
  row: {
    url: "syntax/scale/aesthetic/Z_faceting",
    label: "faceting aesthetic",
  },
  column: {
    url: "syntax/scale/aesthetic/Z_faceting",
    label: "faceting aesthetic",
  },
};

const POSITION_LINKS: Record<string, DocLink> = {
  stack: { url: "syntax/layer/position/stack", label: "stack position" },
  dodge: { url: "syntax/layer/position/dodge", label: "dodge position" },
  jitter: { url: "syntax/layer/position/jitter", label: "jitter position" },
  identity: {
    url: "syntax/layer/position/identity",
    label: "identity position",
  },
};

// ---------------------------------------------------------------------------
// Regex patterns
// ---------------------------------------------------------------------------

const CLAUSE_RE =
  /\b(VISUALISE|VISUALIZE|DRAW|PLACE|SCALE|FACET|PROJECT|LABEL)\b/gi;

const GEOM_RE =
  /\b(?:DRAW|PLACE)\s+(point|line|path|bar|area|tile|polygon|ribbon|histogram|density|smooth|boxplot|violin|text|segment|rule|linear|range)\b/gi;

const COORD_RE = /\bTO\s+(cartesian|polar)\b/gi;

const SCALE_TYPE_RE =
  /\bSCALE\s+(CONTINUOUS|DISCRETE|BINNED|ORDINAL|IDENTITY)\b/gi;

const AESTHETIC_AFTER_AS_RE =
  /\bAS\s+(x|y|xmin|xmax|ymin|ymax|xend|yend|color|colour|fill|stroke|opacity|size|shape|linetype|linewidth|panel|row|column)\b/gi;

const AESTHETIC_AFTER_SCALE_RE =
  /\bSCALE\s+(?:(?:CONTINUOUS|DISCRETE|BINNED|ORDINAL|IDENTITY)\s+)?(x|y|xmin|xmax|ymin|ymax|xend|yend|color|colour|fill|stroke|opacity|size|shape|linetype|linewidth|panel|row|column)\b/gi;

const POSITION_RE =
  /\bposition\s*=>\s*'(stack|dodge|jitter|identity)'/gi;

// ---------------------------------------------------------------------------
// Line scanning
// ---------------------------------------------------------------------------

/**
 * Find all linkable keyword matches in a single line of text.
 * Returns matches with 1-based column positions.
 */
function findLinksInLine(lineText: string): LinkMatch[] {
  const matches: LinkMatch[] = [];

  // Helper: run a regex, look up the capture group in a map, push match
  function scan(
    re: RegExp,
    map: Record<string, DocLink>,
    captureGroup: number = 0,
  ) {
    re.lastIndex = 0;
    let m: RegExpExecArray | null;
    while ((m = re.exec(lineText)) !== null) {
      const word = m[captureGroup];
      const link = map[word.toLowerCase()];
      if (!link) continue;

      // The capture group is always the last token in our patterns
      const startOffset =
        captureGroup === 0
          ? m.index
          : m.index + m[0].length - word.length;

      matches.push({
        startCol: startOffset + 1, // 1-based
        endCol: startOffset + word.length + 1,
        link,
      });
    }
  }

  // 1. Clause keywords (full match)
  scan(CLAUSE_RE, CLAUSE_LINKS, 1);

  // 2. Geom types (capture group 1, after DRAW/PLACE)
  scan(GEOM_RE, GEOM_LINKS, 1);

  // 3. Coord types (capture group 1, after TO)
  scan(COORD_RE, COORD_LINKS, 1);

  // 4. Scale modifiers (capture group 1, after SCALE)
  scan(SCALE_TYPE_RE, SCALE_TYPE_LINKS, 1);

  // 5. Aesthetics after AS (capture group 1)
  scan(AESTHETIC_AFTER_AS_RE, AESTHETIC_LINKS, 1);

  // 6. Aesthetics after SCALE [modifier?] (capture group 1)
  scan(AESTHETIC_AFTER_SCALE_RE, AESTHETIC_LINKS, 1);

  // 7. Position adjustments (capture group 1, inside quotes)
  scan(POSITION_RE, POSITION_LINKS, 1);

  return matches;
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

let registered = false;

export function registerGgsqlLinks(siteRoot: string): void {
  if (registered) return;
  registered = true;

  // Normalize siteRoot to end with /
  if (!siteRoot.endsWith("/")) siteRoot += "/";

  // Resolve site root to an absolute URL for Monaco's opener
  const baseUrl = new URL(siteRoot, window.location.href).href;

  // --- Link Provider ---
  monaco.languages.registerLinkProvider("ggsql", {
    provideLinks(model) {
      const links: monaco.languages.ILink[] = [];
      const lineCount = model.getLineCount();

      for (let lineNum = 1; lineNum <= lineCount; lineNum++) {
        const lineText = model.getLineContent(lineNum);
        const lineMatches = findLinksInLine(lineText);

        for (const lm of lineMatches) {
          links.push({
            range: new monaco.Range(
              lineNum,
              lm.startCol,
              lineNum,
              lm.endCol,
            ),
            url: new URL(lm.link.url + ".html", baseUrl).href,
            tooltip: lm.link.label,
          });
        }
      }

      return { links };
    },
  });

}
