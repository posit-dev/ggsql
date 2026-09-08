// The ggsql wasm client.
//
// The Rust side draws a plot to SVG markup and stops there. Everything
// browser-shaped — measuring a container, observing resizes, fetching fonts,
// putting markup in the page — lives here, because wasm bytes are expensive
// and JavaScript is not.

import init, {
  convert_csv,
  convert_parquet,
  GgsqlContext,
  GgsqlPlot,
  hasFonts,
  initExtensionLoader,
  installExtension,
  registerFont,
  setGenericFamily,
  SvgRender,
} from './ggsql_wasm.js';

// One entry point for the package: everything the glue exposes, plus the
// browser-shaped helpers below. `ggsql.d.ts` declares exactly this list, so a
// name missing here type-checks and is `undefined` at run time.
export {
  init as default,
  convert_csv,
  convert_parquet,
  GgsqlContext,
  GgsqlPlot,
  hasFonts,
  initExtensionLoader,
  installExtension,
  registerFont,
  setGenericFamily,
  SvgRender,
};

// The faces shipped with the package, one file per (weight, style).
//
// One file per weight and style is a rule: the shaper selects within a family
// by weight, width and style and knows nothing of `unicode-range`, so two
// subset files sharing a family name let the wrong one win the attribute match.
const FACES = [
  { file: 'roboto-regular.ttf', weight: 400, style: 'normal' },
  { file: 'roboto-bold.ttf', weight: 700, style: 'normal' },
  { file: 'roboto-italic.ttf', weight: 400, style: 'italic' },
  { file: 'roboto-bolditalic.ttf', weight: 700, style: 'italic' },
];

const DEFAULT_FAMILY = 'Roboto';

let fontsPromise = null;

// What each generic has been pointed at, so a drawn plot can be told which
// concrete family its theme's generic resolves to. Mirrors the font context's
// own state, which is not readable back out of it.
const genericFamilies = new Map();

// The generics a plot's theme can name. Only these get redirected at the faces
// this module registered; a theme naming a family outright asked for it.
const GENERICS = new Set([
  'sans-serif', 'serif', 'monospace', 'cursive', 'fantasy', 'system-ui',
]);

/**
 * Register the bundled faces, and tell the browser about them too.
 *
 * Both halves are needed. The shaper measures every string to lay the plot out,
 * so without the faces a plot has no text and the wrong margins. The browser
 * needs them because the SVG places each run with one anchor plus `textLength`,
 * and a face other than the measured one gets squeezed into the measured box.
 *
 * Process-global and permanent, so this is once per page. Safe to call
 * repeatedly; the work happens once.
 */
export function registerDefaultFonts(baseUrl) {
  if (fontsPromise) return fontsPromise;
  // Resolved against the document first, since a URL base has to be absolute
  // and a caller naturally passes a relative path. The trailing slash matters:
  // to `new URL`, `/assets` names a file, not a directory.
  const base = baseUrl
    ? new URL(
        baseUrl.replace(/\/?$/, '/'),
        typeof document !== 'undefined' ? document.baseURI : import.meta.url,
      ).href
    : new URL('./fonts/', import.meta.url).href;

  const attempt = (async () => {
    const families = new Set();
    for (const face of FACES) {
      const url = new URL(face.file, base).href;
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`could not fetch ${url}: ${response.status}`);
      }
      const bytes = new Uint8Array(await response.arrayBuffer());
      for (const family of registerFont(bytes)) families.add(family);
      injectFontFace(face, url);
    }
    // A generic is an indirection through the font context rather than a name,
    // so registering Roboto does not on its own make `sans-serif` mean Roboto.
    const names = [...families];
    if (names.length) pointGenericAt('sans-serif', names);
    return names;
  })();

  // Only success is memoised: a cached failure would make one offline moment
  // permanent and leave every later plot textless. Registration is idempotent,
  // so retrying after a partial attempt costs nothing.
  fontsPromise = attempt.catch((e) => {
    fontsPromise = null;
    throw e;
  });
  return fontsPromise;
}

/** Point a generic at concrete families, remembering it for the SVG fixup. */
function pointGenericAt(kind, families) {
  setGenericFamily(kind, families);
  genericFamilies.set(kind, families);
}

/**
 * Register a font from a URL, and optionally make a generic mean it.
 *
 * The two steps belong together: a generic is an indirection through the font
 * context, so a theme asking for `sans-serif` resolves to nothing until
 * something says what it means — and only the file knows its own family name.
 *
 * WOFF and WOFF2 are accepted, so a font CDN's URL works directly.
 *
 * Process-global, permanent, and must precede the first draw — a plot shaped
 * without a font has no text and the wrong layout.
 */
export async function registerFontFromUrl(url, opts = {}) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`could not fetch the font at ${url}: ${response.status}`);
  }
  const families = registerFont(new Uint8Array(await response.arrayBuffer()));
  if (opts.genericFor) pointGenericAt(opts.genericFor, families);
  return families;
}

/** Give the browser the same face the shaper just measured from. */
function injectFontFace(face, url) {
  if (typeof document === 'undefined') return;
  const id = `ggsql-font-${face.file}`;
  if (document.getElementById(id)) return;
  const style = document.createElement('style');
  style.id = id;
  style.textContent =
    `@font-face{font-family:'${DEFAULT_FAMILY}';` +
    `src:url('${url}') format('truetype');` +
    `font-weight:${face.weight};font-style:${face.style};font-display:block}`;
  document.head.appendChild(style);
}

/**
 * Point the drawn SVG at the face its advances were measured from.
 *
 * Every run is placed with one anchor plus `textLength`, so the browser fits
 * whatever it resolves into the width the shaper measured. Left to the generic
 * alone it resolves its own default and `textLength` scales that face to fit —
 * plausible, wrong, and wrong differently on every platform.
 *
 * Naming the family on the root is enough: `font-family` inherits, and a span
 * that named its own keeps it. The generic stays on as the fallback.
 */
function nameRegisteredFamily(root) {
  if (!root || root.tagName?.toLowerCase() !== 'svg') return;
  const current = root.getAttribute('font-family');
  // Only a generic is ambiguous. A theme that named a family outright asked
  // for it, and the browser can resolve that name as well as we can.
  if (!current || !GENERICS.has(current)) return;
  const families = genericFamilies.get(current);
  if (!families?.length) return;
  const named = families.map((f) => `'${f}'`).join(', ');
  root.setAttribute('font-family', `${named}, ${current}`);
}

/**
 * The container's content box, for the first draw — before the observer has
 * reported one. `clientWidth` / `clientHeight` include padding, so it is
 * subtracted back off rather than drawn over.
 */
function contentBox(el) {
  const style = getComputedStyle(el);
  const x = parseFloat(style.paddingLeft) + parseFloat(style.paddingRight);
  const y = parseFloat(style.paddingTop) + parseFloat(style.paddingBottom);
  return [el.clientWidth - (x || 0), el.clientHeight - (y || 0)];
}

let nextViewId = 0;

/**
 * One plot bound to one container element.
 *
 * Redraws on resize rather than scaling: the layout is re-solved at the new
 * size, so a wider box gets more tick labels instead of stretched ones. That is
 * the whole reason a resize costs a render at all.
 */
export class PlotView {
  /**
   * @param {HTMLElement} container element to draw into; its box sets the size
   * @param {object} [opts]
   * @param {string} [opts.idPrefix] namespace for generated element ids
   * @param {number} [opts.aspect] width/height; height follows width when set
   */
  constructor(container, opts = {}) {
    this.container = container;
    // Without this the height comes from the container, which is fine when CSS
    // gives it one. A container sized *by* its content instead feeds back on
    // itself and collapses; deriving height from width breaks that loop.
    this.aspect = opts.aspect && opts.aspect > 0 ? opts.aspect : null;
    // Inline SVGs share the page's id space, so two plots on one page collide
    // on gradient and clip-path ids without this. A docs page carries several.
    this.idPrefix = opts.idPrefix || `ggsql-${nextViewId++}-`;
    this.plot = null;
    this.warnings = [];
    this._frame = null;
    this._lastSize = null;
    this._freed = false;
    this._box = null;

    // `contentRect` is the content box; `clientHeight` includes padding, so
    // each draw would be taller than its space and, in a container free to
    // grow, climb by the padding every frame.
    this._observer = new ResizeObserver((entries) => {
      const rect = entries[entries.length - 1]?.contentRect;
      if (rect) this._box = [rect.width, rect.height];
      this._schedule();
    });
    this._observer.observe(this.container);
  }

  /**
   * Show a plot, or clear the view when given `null`.
   *
   * Takes ownership of `plot`: the previous one is freed, since a wasm object
   * is not reclaimed by the garbage collector.
   */
  setPlot(plot) {
    // Ownership transfers even to a freed view, so the plot is released rather
    // than leaked — nothing else holds a reference to reclaim it.
    if (this._freed) {
      plot?.free();
      return;
    }
    if (this.plot && this.plot !== plot) this.plot.free();
    this.plot = plot;
    this._lastSize = null;
    // Cleared before drawing: a draw that cannot happen yet — a container in a
    // hidden tab — would leave the last plot's warnings standing.
    this.warnings = [];
    if (!plot) {
      this.container.replaceChildren();
      return;
    }
    this._renderNow();
  }

  /** Redraw at the container's current size. */
  redraw() {
    this._lastSize = null;
    this._renderNow();
  }

  _schedule() {
    if (this._freed || !this.plot) return;
    // Coalesce to one draw per frame. Unlike a canvas, the markup already in
    // the page stays visible until it is replaced, so nothing flickers.
    if (this._frame !== null) return;
    this._frame = requestAnimationFrame(() => {
      this._frame = null;
      this._renderNow();
    });
  }

  _renderNow() {
    if (this._freed || !this.plot) return;
    const [boxWidth, boxHeight] = this._box || contentBox(this.container);
    const width = Math.round(boxWidth);
    const height = this.aspect
      ? Math.round(width / this.aspect)
      : Math.round(boxHeight);
    if (width < 1 || height < 1) return;
    // A hidden element reports zero and a scrollbar can settle a pixel either
    // way, so skip a size we already drew.
    if (this._lastSize && this._lastSize[0] === width && this._lastSize[1] === height) return;

    const render = this.plot.toSvg(width, height, this.idPrefix);
    try {
      this.warnings = render.warnings;
      this.container.innerHTML = render.svg;
      const root = this.container.firstElementChild;
      // An SVG is inline by default, reserving descender space under it — the
      // same feedback loop the padding caused, in miniature.
      if (root) root.style.display = 'block';
      nameRegisteredFamily(root);
      this._lastSize = [width, height];
    } finally {
      render.free();
    }
  }

  /** Stop observing and release the plot. */
  free() {
    if (this._freed) return;
    this._freed = true;
    if (this._frame !== null) cancelAnimationFrame(this._frame);
    this._observer.disconnect();
    if (this.plot) this.plot.free();
    this.plot = null;
  }
}
