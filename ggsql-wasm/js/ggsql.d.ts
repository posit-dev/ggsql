// Types for the hand-written wrapper. The generated `ggsql_wasm.d.ts` covers
// everything re-exported from the glue; what is declared here is the
// browser-shaped layer that lives in JavaScript.

export {
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

export { default } from './ggsql_wasm.js';

import type { GgsqlPlot } from './ggsql_wasm.js';

/**
 * Register the bundled Roboto faces with the shaper, and declare them to the
 * browser as `@font-face` so both resolve the same file.
 *
 * Must run before the first plot is drawn. Safe to call repeatedly. Use
 * `registerFontFromUrl` instead to supply the page's own typography.
 *
 * @param baseUrl where the `fonts/` directory is served from. Defaults to
 *   `./fonts/` relative to the module.
 * @returns the family names that were registered.
 */
export function registerDefaultFonts(baseUrl?: string): Promise<string[]>;

export interface RegisterFontOptions {
  /**
   * Generic family to point at the registered face — `sans-serif`, `serif`,
   * `monospace`, `cursive`, `fantasy` or `system-ui`. Without it a theme asking
   * for a generic still finds nothing.
   */
  genericFor?: string;
}

/**
 * Register a font from a URL, and optionally make a generic mean it.
 *
 * Accepts WOFF and WOFF2 as well as sfnt, so a font CDN's URL works directly.
 *
 * Must run before the first plot is drawn: a plot shaped without a font has no
 * text at all, and the wrong layout with it.
 *
 * @returns the family names that were registered.
 */
export function registerFontFromUrl(
  url: string,
  opts?: RegisterFontOptions,
): Promise<string[]>;

export interface PlotViewOptions {
  /** Namespace for generated element ids. Defaults to a per-view counter. */
  idPrefix?: string;
  /**
   * Width divided by height. When set, the height follows the container's
   * width instead of being measured — what a container sized by its own
   * content needs, since measuring it after filling it feeds back on itself.
   */
  aspect?: number;
}

/**
 * One plot bound to one container element, redrawn when the container resizes.
 *
 * The layout is re-solved at each new size rather than scaled to it, so a wider
 * box gets more tick labels instead of stretched ones.
 */
export class PlotView {
  constructor(container: HTMLElement, opts?: PlotViewOptions);

  /** Whatever the renderer had to degrade or drop on the last draw. */
  readonly warnings: string[];

  /**
   * Show a plot, or clear the view with `null`.
   *
   * Takes ownership: the previous plot is freed, and so is this one if the
   * view has already been freed.
   */
  setPlot(plot: GgsqlPlot | null): void;

  /** Redraw at the container's current size. */
  redraw(): void;

  /** Stop observing and release the plot. */
  free(): void;
}
