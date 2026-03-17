import init, { GgsqlContext } from "ggsql-wasm";
import { WASM_BASE } from "./wasmBase";

export class WasmContextManager {
  private context: GgsqlContext | null = null;
  private initialized = false;

  async initialize(): Promise<void> {
    if (this.initialized) return;

    await init(WASM_BASE + "ggsql_wasm_bg.wasm");
    this.context = new GgsqlContext();
    this.initialized = true;
  }

  private getContext(): GgsqlContext {
    if (!this.context) {
      throw new Error("Context not initialized. Call initialize() first.");
    }
    return this.context;
  }

  execute(query: string): string {
    return this.getContext().execute(query);
  }

  hasVisual(query: string): boolean {
    return this.getContext().has_visual(query);
  }

  executeSql(query: string): string {
    return this.getContext().execute_sql(query);
  }

  registerCSV(name: string, data: Uint8Array): void {
    this.getContext().register_csv(name, data);
  }

  async registerParquet(name: string, data: Uint8Array): Promise<void> {
    await this.getContext().register_parquet(name, data);
  }

  async registerBuiltinDatasets(): Promise<void> {
    await this.getContext().register_builtin_datasets();
  }

  unregister(name: string): void {
    this.getContext().unregister(name);
  }

  listTables(): string[] {
    return Array.from(this.getContext().list_tables() as Iterable<string>);
  }

  isInitialized(): boolean {
    return this.initialized;
  }
}
