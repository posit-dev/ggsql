import init, {
  GgsqlContext,
  GgsqlPlot,
  installExtension,
  registerDefaultFonts,
} from "ggsql-wasm";
import { WASM_BASE } from "./wasmBase";

/**
 * What a page can do once the module has trapped, which is nothing but reload.
 *
 * A panic in wasm aborts the instance, not the call: the module is dead for the
 * whole page and re-initialising does not bring it back. Saying so once beats
 * repeating an opaque trap for every subsequent cell.
 */
const TRAPPED =
  "the ggsql engine stopped and cannot recover — reload the page to continue";

export class WasmContextManager {
  private context: GgsqlContext | null = null;
  private initialized = false;
  private trapped = false;

  async initialize(): Promise<void> {
    if (this.initialized) return;

    await this.guardAsync(async () => {
      await init(WASM_BASE + "ggsql_wasm_bg.wasm");
      // Before any plot is drawn: a browser enumerates no system fonts, so a
      // plot rendered without this has no text and the wrong margins.
      await registerDefaultFonts(WASM_BASE + "fonts/");
      this.context = new GgsqlContext();
    });
    this.initialized = true;
  }

  async installExtension(name: string, url: string): Promise<void> {
    await this.guardAsync(() => installExtension(name, url));
  }

  private getContext(): GgsqlContext {
    if (!this.context) {
      throw new Error("Context not initialized. Call initialize() first.");
    }
    return this.context;
  }

  /**
   * Run one wasm call, latching a trap.
   *
   * A `WebAssembly.RuntimeError` is a dead instance, not a failed call, so it is
   * recorded rather than merely reported.
   */
  private guard<T>(call: () => T): T {
    if (this.trapped) throw new Error(TRAPPED);
    try {
      return call();
    } catch (err) {
      throw this.noteError(err);
    }
  }

  /** {@link WasmContextManager.guard} for a call that settles later. */
  private async guardAsync<T>(call: () => Promise<T>): Promise<T> {
    if (this.trapped) throw new Error(TRAPPED);
    try {
      return await call();
    } catch (err) {
      throw this.noteError(err);
    }
  }

  /**
   * Classify an error that came out of wasm, latching a trap, and return what
   * should be reported for it.
   *
   * Public because drawing goes through `PlotView` rather than this class, so a
   * caller catching a panic there hands it here.
   */
  noteError(err: unknown): unknown {
    if (err instanceof WebAssembly.RuntimeError) {
      this.trapped = true;
      return new Error(`${TRAPPED} (${err.message})`);
    }
    return err;
  }

  execute(query: string): GgsqlPlot {
    return this.guard(() => this.getContext().execute(query));
  }

  hasVisual(query: string): boolean {
    return this.guard(() => this.getContext().has_visual(query));
  }

  executeSql(query: string): string {
    return this.guard(() => this.getContext().execute_sql(query));
  }

  registerCSV(name: string, data: Uint8Array): void {
    this.guard(() => this.getContext().register_csv(name, data));
  }

  async registerParquet(name: string, data: Uint8Array): Promise<void> {
    await this.guardAsync(() => this.getContext().register_parquet(name, data));
  }

  async registerBuiltinDatasets(): Promise<void> {
    await this.guardAsync(() => this.getContext().register_builtin_datasets());
  }

  unregister(name: string): void {
    this.guard(() => this.getContext().unregister(name));
  }

  listTables(): string[] {
    return this.guard(() =>
      Array.from(this.getContext().list_tables() as Iterable<string>),
    );
  }

  isInitialized(): boolean {
    return this.initialized;
  }

  /** Whether the module has trapped, and so can no longer serve anything. */
  isTrapped(): boolean {
    return this.trapped;
  }
}
