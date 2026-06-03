// Wasm Exception Handling proposal types
declare global {
  namespace WebAssembly {
    interface Tag {}
    const Tag: { new (descriptor: { parameters: ValueType[] }): Tag };
    interface Exception {}
    const Exception: {
      new (tag: Tag, payload: unknown[], options?: { traceStack?: boolean }): Exception;
    };
  }
}

interface LoadedExtension {
  instance: WebAssembly.Instance;
  exports: Record<string, number>;
}

const registry = new Map<string, LoadedExtension>();
let lastError: string | null = null;
let nextHandle = 1;
const handleMap = new Map<number, string>();

let sharedMemory: WebAssembly.Memory | null = null;
let sharedTable: WebAssembly.Table | null = null;
let hostExports: WebAssembly.Exports | null = null;

export function initExtensionLoader(wasmExports: WebAssembly.Exports): void {
  hostExports = wasmExports;
  sharedMemory = wasmExports.memory as WebAssembly.Memory;
  sharedTable = wasmExports.__indirect_function_table as WebAssembly.Table;

  if (!sharedMemory) throw new Error("Main module does not export 'memory'");
  if (!sharedTable) throw new Error("Main module does not export '__indirect_function_table'");

  (globalThis as any).__sqlite_ext = {
    dlOpen,
    dlSym,
    dlClose,
    dlError,
  };
}

export async function installExtension(
  name: string,
  wasmSource: BufferSource | Response | string,
): Promise<void> {
  if (!sharedMemory || !sharedTable || !hostExports) {
    throw new Error("Call initExtensionLoader() before installExtension()");
  }

  let bytes: ArrayBuffer;
  if (typeof wasmSource === "string") {
    const response = await fetch(wasmSource);
    if (!response.ok) throw new Error(`Failed to fetch extension: ${response.status}`);
    bytes = await response.arrayBuffer();
  } else if (wasmSource instanceof Response) {
    bytes = await wasmSource.arrayBuffer();
  } else {
    bytes = wasmSource instanceof ArrayBuffer ? wasmSource : (wasmSource as Uint8Array<ArrayBuffer>).buffer;
  }

  const extModule = await WebAssembly.compile(bytes);

  const currentBytes = sharedMemory.buffer.byteLength;
  const extraPages = Math.ceil(bytes.byteLength / 65536) + 1;
  sharedMemory.grow(extraPages);
  const memBase = currentBytes;

  const stackTop = memBase + extraPages * 65536;
  sharedMemory.grow(1);
  // Small scratch area for __wasm_lpad_context (3 x i32 = 12 bytes)
  const lpadContextAddr = stackTop + 65536 - 64;

  const moduleExportDescs = WebAssembly.Module.exports(extModule);

  // Determine how many table slots the extension needs by inspecting its
  // element segments. The wasm binary encodes table entries for all functions
  // that may be called indirectly — far more than just the exported ones.
  const tableSlots = countElementSegmentEntries(new Uint8Array(bytes));
  const tableBase = sharedTable.length;
  sharedTable.grow(tableSlots + 64);

  const imports: WebAssembly.Imports = {
    env: {
      memory: sharedMemory,
      __indirect_function_table: sharedTable,
      __memory_base: new WebAssembly.Global({ value: "i32", mutable: false }, memBase),
      __table_base: new WebAssembly.Global({ value: "i32", mutable: false }, tableBase),
      __stack_pointer: new WebAssembly.Global({ value: "i32", mutable: true }, stackTop),
    },
  };

  // Know which functions the extension itself exports (before instantiation).
  // PIC --shared modules both import AND export the same symbols — env imports
  // are for direct calls, GOT is for indirect. We use lazy trampolines so that
  // direct calls to self-defined symbols bounce to the extension's own export.
  const extExportNames = new Set(
    moduleExportDescs.filter((e) => e.kind === "function").map((e) => e.name),
  );
  let extInstance: WebAssembly.Instance | null = null;
  let cppExceptionTag: WebAssembly.Tag | null = null;

  const moduleImportDescs = WebAssembly.Module.imports(extModule);
  for (const imp of moduleImportDescs) {
    if (imp.module === "env" && imp.name in (imports.env as Record<string, unknown>)) {
      continue;
    }

    if (imp.module === "env" && imp.kind === "function") {
      const hostFn = hostExports[imp.name];
      if (typeof hostFn === "function") {
        (imports.env as Record<string, unknown>)[imp.name] = hostFn;
      } else if (imp.name === "abort") {
        (imports.env as Record<string, unknown>)[imp.name] = () => {
          throw new Error("[ext] abort() called from extension");
        };
      } else if (imp.name === "exit") {
        (imports.env as Record<string, unknown>)[imp.name] = (code: number) => {
          throw new Error(`[ext] exit(${code}) called from extension`);
        };
      } else if (extExportNames.has(imp.name)) {
        // Symbol defined in the extension itself — lazy trampoline that calls
        // the extension's own export once the instance exists.
        const sym = imp.name;
        (imports.env as Record<string, unknown>)[sym] = (...args: unknown[]) => {
          const fn = extInstance?.exports[sym];
          if (typeof fn === "function") return (fn as Function)(...args);
          return 0;
        };
      } else if (imp.name === "__ext_trap") {
        const trapNames: Record<number, string> = { 1: "abort()", 2: "__assert_fail()", 3: "abort() [stubs]" };
        (imports.env as Record<string, unknown>)[imp.name] = (code: number) => {
          const name = code >= 100 ? `exit(${code - 100})` : (trapNames[code] ?? `trap(${code})`);
          throw new Error(`[ext] ${name} called from extension`);
        };
      } else if (imp.name === "_Unwind_RaiseException") {
        (imports.env as Record<string, unknown>)[imp.name] = (excPtr: number) => {
          if (cppExceptionTag) {
            throw new WebAssembly.Exception(cppExceptionTag, [excPtr], { traceStack: true });
          }
          throw new Error("_Unwind_RaiseException: no cpp exception tag");
        };
      } else if (imp.name === "_Unwind_CallPersonality") {
        (imports.env as Record<string, unknown>)[imp.name] = (excPtr: number) => {
          const view = new DataView(sharedMemory!.buffer);
          // Set adjustedPtr (excPtr - 8) to point to the thrown object (excPtr + 32)
          view.setUint32(excPtr - 8, excPtr + 32, true);
          view.setInt32(lpadContextAddr + 8, 1, true);
          return 6; // _URC_HANDLER_FOUND
        };
      } else if (imp.name === "_Unwind_DeleteException") {
        (imports.env as Record<string, unknown>)[imp.name] = () => {};
      } else {
        // Unresolved import: stub it to return 0. Warn once per symbol so a
        // genuinely missing dependency is visible without flooding the console.
        const unresName = imp.name;
        let warned = false;
        (imports.env as Record<string, unknown>)[imp.name] = (...args: unknown[]) => {
          if (!warned) {
            warned = true;
            console.warn(`[ext] unresolved import '${unresName}' stubbed to return 0`);
          }
          void args;
          return 0;
        };
      }
    }

    if (imp.module === "env" && (imp.kind as string) === "tag") {
      const tag = new WebAssembly.Tag({ parameters: ["i32"] });
      (imports.env as Record<string, unknown>)[imp.name] = tag;
      if (imp.name === "__cpp_exception") {
        cppExceptionTag = tag;
      }
    }

    if ((imp.module === "GOT.func" || imp.module === "GOT.mem") && imp.kind === "global") {
      if (!imports[imp.module]) imports[imp.module] = {};
      const hostFn = hostExports[imp.name];
      if (typeof hostFn === "function") {
        const idx = sharedTable.length;
        sharedTable.grow(1);
        sharedTable.set(idx, hostFn as any);
        (imports[imp.module] as Record<string, unknown>)[imp.name] =
          new WebAssembly.Global({ value: "i32", mutable: true }, idx);
      } else if (imp.module === "GOT.mem" && imp.name === "__wasm_lpad_context") {
        (imports[imp.module] as Record<string, unknown>)[imp.name] =
          new WebAssembly.Global({ value: "i32", mutable: true }, lpadContextAddr);
      } else {
        (imports[imp.module] as Record<string, unknown>)[imp.name] =
          new WebAssembly.Global({ value: "i32", mutable: true }, 0);
      }
    }
  }

  extInstance = new WebAssembly.Instance(extModule, imports);

  // PIC shared modules export __wasm_apply_data_relocs which patches data
  // segment entries (vtables, function pointers) using GOT.func/GOT.mem values.
  const applyRelocs = extInstance.exports.__wasm_apply_data_relocs as Function | undefined;
  if (applyRelocs) {
    applyRelocs();

    // The module's start function (__wasm_apply_global_relocs) initialises
    // GOT.mem entries but never GOT.func: a shared library can't assign its
    // own table indices, so that's the dynamic linker's job — which, in the
    // browser, is us. Resolve the GOT entries the host didn't provide and the
    // module left at 0 (its own vtable/function-pointer symbols) from the
    // module's exports, then re-run the data relocs so vtable slots get
    // patched with the now-correct indices.
    let fixedAny = false;
    for (const imp of moduleImportDescs) {
      if (imp.module === "GOT.func" && imp.kind === "global") {
        const g = (imports["GOT.func"] as Record<string, WebAssembly.Global>)?.[imp.name];
        if (g && g.value === 0) {
          const fn = extInstance.exports[imp.name];
          if (typeof fn === "function") {
            const idx = sharedTable.length;
            sharedTable.grow(1);
            sharedTable.set(idx, fn as any);
            g.value = idx;
            fixedAny = true;
          }
        }
      }
      if (imp.module === "GOT.mem" && imp.kind === "global") {
        const g = (imports["GOT.mem"] as Record<string, WebAssembly.Global>)?.[imp.name];
        if (g && g.value === 0) {
          const exp = extInstance.exports[imp.name];
          if (exp && typeof exp === "object" && "value" in exp) {
            g.value = (exp as WebAssembly.Global).value + memBase;
            fixedAny = true;
          }
        }
      }
    }
    if (fixedAny) {
      applyRelocs();
    }
  }

  const callCtors = extInstance.exports.__wasm_call_ctors as Function | undefined;
  if (callCtors) {
    callCtors();
  }

  const extExports: Record<string, number> = {};
  for (const exp of moduleExportDescs) {
    if (exp.kind === "function") {
      const fn = extInstance.exports[exp.name];
      const idx = sharedTable.length;
      sharedTable.grow(1);
      sharedTable.set(idx, fn as any);
      extExports[exp.name] = idx;
    }
  }

  registry.set(name, {
    instance: extInstance,
    exports: extExports,
  });
}

function readLEB128(data: Uint8Array, pos: number): [number, number] {
  let val = 0, shift = 0;
  while (true) {
    const b = data[pos++];
    val |= (b & 0x7f) << shift;
    shift += 7;
    if (!(b & 0x80)) break;
  }
  return [val, pos];
}

function countElementSegmentEntries(wasm: Uint8Array): number {
  let pos = 8;
  let total = 0;
  while (pos < wasm.length) {
    const sid = wasm[pos++];
    let [size, p] = readLEB128(wasm, pos);
    pos = p;
    const end = pos + size;
    if (sid === 9) {
      let [count, p2] = readLEB128(wasm, pos);
      pos = p2;
      for (let i = 0; i < count; i++) {
        const flags = wasm[pos++];
        if (flags === 0) {
          while (wasm[pos] !== 0x0b) pos++;
          pos++;
          let [numElem, p3] = readLEB128(wasm, pos);
          pos = p3;
          total += numElem;
          for (let j = 0; j < numElem; j++) {
            [, pos] = readLEB128(wasm, pos);
          }
        } else {
          break;
        }
      }
      break;
    }
    pos = end;
  }
  return total || 256;
}

function dlOpen(filename: string): number {
  lastError = null;
  const name = filename.replace(/^.*[\\/]/, "").replace(/\.wasm$/, "");
  if (!registry.has(name)) {
    lastError = `Extension '${name}' not installed. Call installExtension() first.`;
    return 0;
  }
  const handle = nextHandle++;
  handleMap.set(handle, name);
  return handle;
}

function dlSym(handle: number, symbol: string): number {
  lastError = null;
  const name = handleMap.get(handle);
  if (!name) {
    lastError = `Invalid extension handle: ${handle}`;
    return 0;
  }
  const ext = registry.get(name);
  if (!ext) {
    lastError = `Extension '${name}' not found in registry`;
    return 0;
  }
  const idx = ext.exports[symbol];
  if (idx === undefined) {
    lastError = `Symbol '${symbol}' not found in extension '${name}'`;
    return 0;
  }
  return idx;
}

function dlClose(handle: number): void {
  handleMap.delete(handle);
}

function dlError(): string | null {
  return lastError;
}
