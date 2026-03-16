import { WasmContextManager } from "./context";

export class TableManager {
  private listElement: HTMLElement;
  private contextManager: WasmContextManager;
  private onChangeCallback: (() => void) | null = null;
  private onClickCallback: ((tableName: string) => void) | null = null;

  constructor(listElement: HTMLElement, contextManager: WasmContextManager) {
    this.listElement = listElement;
    this.contextManager = contextManager;
  }

  refresh(): void {
    const tables = this.contextManager.listTables();

    this.listElement.innerHTML = "";

    if (tables.length === 0) {
      const emptyItem = document.createElement("li");
      emptyItem.textContent = "No tables registered";
      emptyItem.style.color = "#005F73";
      emptyItem.style.fontSize = "12px";
      emptyItem.style.padding = "6px 8px";
      this.listElement.appendChild(emptyItem);
      return;
    }

    tables.forEach((tableName) => {
      const item = document.createElement("li");
      item.className = "table-item";

      const nameSpan = document.createElement("span");
      nameSpan.className = "table-item-name";
      nameSpan.textContent = tableName;

      item.appendChild(nameSpan);
      item.style.cursor = "pointer";
      item.onclick = () => {
        if (this.onClickCallback) this.onClickCallback(tableName);
      };

      if (!tableName.startsWith("ggsql:")) {
        const removeBtn = document.createElement("span");
        removeBtn.className = "table-item-remove";
        removeBtn.textContent = "\u00d7";
        removeBtn.title = "Remove table";
        removeBtn.onclick = (e) => {
          e.stopPropagation();
          this.removeTable(tableName);
        };
        item.appendChild(removeBtn);
      }

      this.listElement.appendChild(item);
    });
  }

  async uploadFile(file: File): Promise<void> {
    const tableName = this.sanitiseTableName(file.name);
    const buffer = await file.arrayBuffer();
    const data = new Uint8Array(buffer);

    if (file.name.endsWith(".parquet")) {
      await this.contextManager.registerParquet(tableName, data);
    } else {
      this.contextManager.registerCSV(tableName, data);
    }

    this.refresh();
    if (this.onChangeCallback) this.onChangeCallback();
  }

  removeTable(name: string): void {
    this.contextManager.unregister(name);
    this.refresh();
    if (this.onChangeCallback) this.onChangeCallback();
  }

  onChange(callback: () => void): void {
    this.onChangeCallback = callback;
  }

  onClickTable(callback: (tableName: string) => void): void {
    this.onClickCallback = callback;
  }

  private sanitiseTableName(filename: string): string {
    return filename
      .replace(/\.(csv|parquet|pq)$/i, "")
      .replace(/[^a-zA-Z0-9_]/g, "_")
      .toLowerCase();
  }
}
