// The ggsql wasm client.
//
// The Rust side draws a plot to SVG markup and stops there. Everything
// browser-shaped — measuring a container, observing resizes, fetching fonts,
// putting markup in the page — lives here, because wasm bytes are expensive
// and JavaScript is not.

import glueInit, {
  initSync as glueInitSync,
  registerFont,
  setConverters,
  setGenericFamily,
} from "./ggsql_wasm.js";
import type {
  GgsqlPlot,
  InitInput,
  InitOutput,
  SyncInitInput,
} from "./ggsql_wasm.js";
import { convert_csv } from "./csv";
import { convert_parquet } from "./parquet";
import { initExtensionLoader } from "./extensions";

// One entry point for the package: the glue's API, minus the plumbing `init`
// uses to reach back into it, plus the browser-shaped helpers below. The list
// is explicit rather than `export *` so that `setConverters` stays internal —
// a caller who replaced the converters would break every registered format.
export {
  GgsqlContext,
  GgsqlPlot,
  SvgRender,
  hasFonts,
  registerFont,
  setGenericFamily,
} from "./ggsql_wasm.js";
export type { InitInput, InitOutput, SyncInitInput } from "./ggsql_wasm.js";
export { convert_csv } from "./csv";
export { convert_parquet } from "./parquet";
export { installExtension } from "./extensions";
export type { ColumnDescriptor, ColumnType } from "./columns";

type AsyncInitInput =
  | { module_or_path: InitInput | Promise<InitInput> }
  | InitInput
  | Promise<InitInput>;

// Distinguishing the options bag from a bare module source, which is itself an
// object — a `Response`, `URL`, `WebAssembly.Module` or `Promise`. A plain
// object prototype is what separates the two; passing the source directly is
// what a caller reaches for, and what the glue now warns about.
function isAsyncInitOptions(
  input: AsyncInitInput,
): input is { module_or_path: InitInput | Promise<InitInput> } {
  return (
    typeof input === "object" &&
    input !== null &&
    Object.getPrototypeOf(input) === Object.prototype &&
    "module_or_path" in input
  );
}

/**
 * Instantiate ggsql and connect the package's converters and extension loader.
 *
 * The wiring belongs here rather than in the caller's hands: the Rust side
 * calls back into JavaScript for CSV and Parquet, and a context built before
 * `setConverters` fails on the first registered file. Entering through the
 * package is therefore the contract, and the glue's own `init` is not exported.
 */
export default async function init(input?: AsyncInitInput): Promise<InitOutput> {
  const output =
    input === undefined
      ? await glueInit()
      : await glueInit(
          isAsyncInitOptions(input) ? input : { module_or_path: input },
        );
  setConverters(convert_csv, convert_parquet);
  initExtensionLoader(output as unknown as WebAssembly.Exports);
  return output;
}

/**
 * Synchronously instantiate ggsql and connect the package helpers.
 *
 * Needs the module's bytes already in hand — a bundler that inlines the wasm,
 * not a URL. Same wiring as `init`, so the same contract holds.
 */
export function initSync(
  input: { module: SyncInitInput } | SyncInitInput,
): InitOutput {
  const options =
    typeof input === "object" &&
    input !== null &&
    Object.getPrototypeOf(input) === Object.prototype &&
    "module" in input
      ? input
      : { module: input };
  const output = glueInitSync(options);
  setConverters(convert_csv, convert_parquet);
  initExtensionLoader(output as unknown as WebAssembly.Exports);
  return output;
}

interface BundledFace {
  file: string;
  weight: number;
  style: string;
}

// The faces shipped with the package, one file per (weight, style).
//
// One file per weight and style is a rule: the shaper selects within a family
// by weight, width and style and knows nothing of `unicode-range`, so two
// subset files sharing a family name let the wrong one win the attribute match.
const FACES: BundledFace[] = [
  { file: "roboto-regular.ttf", weight: 400, style: "normal" },
  { file: "roboto-bold.ttf", weight: 700, style: "normal" },
  { file: "roboto-italic.ttf", weight: 400, style: "italic" },
  { file: "roboto-bolditalic.ttf", weight: 700, style: "italic" },
];

const DEFAULT_FAMILY = "Roboto";

let fontsPromise: Promise<string[]> | null = null;

// What each generic has been pointed at, so a drawn plot can be told which
// concrete family its theme's generic resolves to. Mirrors the font context's
// own state, which is not readable back out of it.
const genericFamilies = new Map<string, string[]>();

// The generics a plot's theme can name. Only these get redirected at the faces
// this module registered; a theme naming a family outright asked for it.
const GENERICS = new Set([
  "sans-serif",
  "serif",
  "monospace",
  "cursive",
  "fantasy",
  "system-ui",
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
export function registerDefaultFonts(baseUrl?: string): Promise<string[]> {
  if (fontsPromise) return fontsPromise;
  // Resolved against the document first, since a URL base has to be absolute
  // and a caller naturally passes a relative path. The trailing slash matters:
  // to `new URL`, `/assets` names a file, not a directory.
  const base = baseUrl
    ? new URL(
        baseUrl.replace(/\/?$/, "/"),
        typeof document !== "undefined" ? document.baseURI : import.meta.url,
      ).href
    : new URL("./fonts/", import.meta.url).href;

  const attempt = (async () => {
    const families = new Set<string>();
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
    if (names.length) pointGenericAt("sans-serif", names);
    return names;
  })();

  // Only success is memoised: a cached failure would make one offline moment
  // permanent and leave every later plot textless. Registration is idempotent,
  // so retrying after a partial attempt costs nothing.
  fontsPromise = attempt.catch((error: unknown) => {
    fontsPromise = null;
    throw error;
  });
  return fontsPromise;
}

/** Point a generic at concrete families, remembering it for the SVG fixup. */
function pointGenericAt(kind: string, families: string[]): void {
  setGenericFamily(kind, families);
  genericFamilies.set(kind, families);
}

export interface RegisterFontOptions {
  /**
   * Generic family to point at the registered face — `sans-serif`, `serif`,
   * `monospace`, `cursive`, `fantasy` or `system-ui`.
   */
  genericFor?: string;
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
export async function registerFontFromUrl(
  url: string,
  opts: RegisterFontOptions = {},
): Promise<string[]> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`could not fetch the font at ${url}: ${response.status}`);
  }
  const families = registerFont(new Uint8Array(await response.arrayBuffer()));
  if (opts.genericFor) pointGenericAt(opts.genericFor, families);
  return families;
}

/** Give the browser the same face the shaper just measured from. */
function injectFontFace(face: BundledFace, url: string): void {
  if (typeof document === "undefined") return;
  const id = `ggsql-font-${face.file}`;
  if (document.getElementById(id)) return;
  const style = document.createElement("style");
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
function nameRegisteredFamily(root: Element | null): void {
  if (!root || root.tagName.toLowerCase() !== "svg") return;
  const current = root.getAttribute("font-family");
  // Only a generic is ambiguous. A theme that named a family outright asked
  // for it, and the browser can resolve that name as well as we can.
  if (!current || !GENERICS.has(current)) return;
  const families = genericFamilies.get(current);
  if (!families?.length) return;
  const named = families.map((family) => `'${family}'`).join(", ");
  root.setAttribute("font-family", `${named}, ${current}`);
}

/**
 * The container's content box, for the first draw — before the observer has
 * reported one. `clientWidth` / `clientHeight` include padding, so it is
 * subtracted back off rather than drawn over.
 */
function contentBox(element: HTMLElement): [number, number] {
  const style = getComputedStyle(element);
  const x = parseFloat(style.paddingLeft) + parseFloat(style.paddingRight);
  const y = parseFloat(style.paddingTop) + parseFloat(style.paddingBottom);
  return [element.clientWidth - (x || 0), element.clientHeight - (y || 0)];
}

let nextViewId = 0;

export interface PlotViewOptions {
  /** Namespace for generated element ids. Defaults to a per-view counter. */
  idPrefix?: string;
  /** Width divided by height; height follows the container width when set. */
  aspect?: number;
}

/**
 * One plot bound to one container element.
 *
 * Redraws on resize rather than scaling: the layout is re-solved at the new
 * size, so a wider box gets more tick labels instead of stretched ones. That is
 * the whole reason a resize costs a render at all.
 */
export class PlotView {
  private readonly container: HTMLElement;
  private readonly aspect: number | null;
  private readonly idPrefix: string;
  private plot: GgsqlPlot | null = null;
  private _warnings: string[] = [];
  private frame: number | null = null;
  private lastSize: [number, number] | null = null;
  private freed = false;
  private box: [number, number] | null = null;
  private readonly observer: ResizeObserver;

  constructor(container: HTMLElement, opts: PlotViewOptions = {}) {
    this.container = container;
    // Without this the height comes from the container, which is fine when CSS
    // gives it one. A container sized by its content instead feeds back on
    // itself and collapses; deriving height from width breaks that loop.
    this.aspect = opts.aspect && opts.aspect > 0 ? opts.aspect : null;
    // Inline SVGs share the page's id space, so two plots on one page collide
    // on gradient and clip-path ids without this. A docs page carries several.
    this.idPrefix = opts.idPrefix || `ggsql-${nextViewId++}-`;

    // `contentRect` is the content box; `clientHeight` includes padding, so
    // each draw would be taller than its space and, in a container free to
    // grow, climb by the padding every frame.
    this.observer = new ResizeObserver((entries) => {
      const rect = entries[entries.length - 1]?.contentRect;
      if (rect) this.box = [rect.width, rect.height];
      this.schedule();
    });
    this.observer.observe(this.container);
  }

  /** Whatever the renderer had to degrade or drop on the last draw. */
  get warnings(): string[] {
    return this._warnings;
  }

  /**
   * Show a plot, or clear the view when given `null`.
   *
   * Takes ownership of `plot`: the previous one is freed, since a wasm object
   * is not reclaimed by the garbage collector.
   */
  setPlot(plot: GgsqlPlot | null): void {
    // Ownership transfers even to a freed view, so the plot is released rather
    // than leaked — nothing else holds a reference to reclaim it.
    if (this.freed) {
      plot?.free();
      return;
    }
    if (this.plot && this.plot !== plot) this.plot.free();
    this.plot = plot;
    this.lastSize = null;
    // Cleared before drawing: a draw that cannot happen yet — a container in a
    // hidden tab — would leave the last plot's warnings standing.
    this._warnings = [];
    if (!plot) {
      this.container.replaceChildren();
      return;
    }
    this.renderNow();
  }

  /** Redraw at the container's current size. */
  redraw(): void {
    this.lastSize = null;
    this.renderNow();
  }

  private schedule(): void {
    // Coalesce to one draw per frame. Unlike a canvas, the markup already in
    // the page stays visible until it is replaced, so nothing flickers.
    if (this.freed || !this.plot || this.frame !== null) return;
    this.frame = requestAnimationFrame(() => {
      this.frame = null;
      this.renderNow();
    });
  }

  private renderNow(): void {
    if (this.freed || !this.plot) return;
    const [boxWidth, boxHeight] = this.box || contentBox(this.container);
    const width = Math.round(boxWidth);
    const height = this.aspect
      ? Math.round(width / this.aspect)
      : Math.round(boxHeight);
    if (width < 1 || height < 1) return;
    // A hidden element reports zero and a scrollbar can settle a pixel either
    // way, so skip a size we already drew.
    if (
      this.lastSize &&
      this.lastSize[0] === width &&
      this.lastSize[1] === height
    ) {
      return;
    }

    const render = this.plot.toSvg(width, height, this.idPrefix);
    try {
      this._warnings = render.warnings;
      this.container.innerHTML = render.svg;
      const root = this.container.firstElementChild;
      // An SVG is inline by default, reserving descender space under it — the
      // same feedback loop the padding caused, in miniature.
      if (root instanceof HTMLElement || root instanceof SVGElement) {
        root.style.display = "block";
      }
      nameRegisteredFamily(root);
      this.lastSize = [width, height];
    } finally {
      render.free();
    }
  }

  /** Stop observing and release the plot. */
  free(): void {
    if (this.freed) return;
    this.freed = true;
    if (this.frame !== null) cancelAnimationFrame(this.frame);
    this.observer.disconnect();
    this.plot?.free();
    this.plot = null;
  }
}
