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

const PAGE = 65536;

// Dedicated shadow-stack size for each extension. The stack sits above the
// extension's data segment and grows downward.
const EXT_STACK_SIZE = 16 * 1024 * 1024;

const registry = new Map<string, LoadedExtension>();
let lastError: string | null = null;
let nextHandle = 1;
const handleMap = new Map<number, string>();

let sharedMemory: WebAssembly.Memory | null = null;
let sharedTable: WebAssembly.Table | null = null;
let hostExports: WebAssembly.Exports | null = null;

// Canonical table index per function, so the same function always has the
// same "address".
const tableIndexCache = new Map<Function, number>();

function canonicalTableIndex(fn: Function): number {
  const cached = tableIndexCache.get(fn);
  if (cached !== undefined) return cached;
  const idx = sharedTable!.grow(1);
  sharedTable!.set(idx, fn as any);
  tableIndexCache.set(fn, idx);
  return idx;
}

function cacheTableRange(start: number, end: number): void {
  for (let i = start; i < end; i++) {
    const fn = sharedTable!.get(i);
    if (typeof fn === "function" && !tableIndexCache.has(fn)) {
      tableIndexCache.set(fn, i);
    }
  }
}

export function initExtensionLoader(wasmExports: WebAssembly.Exports): void {
  hostExports = wasmExports;
  sharedMemory = wasmExports.memory as WebAssembly.Memory;
  sharedTable = wasmExports.__indirect_function_table as WebAssembly.Table;

  if (!sharedMemory) throw new Error("Main module does not export 'memory'");
  if (!sharedTable) throw new Error("Main module does not export '__indirect_function_table'");

  cacheTableRange(0, sharedTable.length);

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

  if (registry.has(name)) {
    console.warn(`[ext] extension '${name}' is already installed; skipping`);
    return;
  }

  let bytes: ArrayBuffer;
  if (typeof wasmSource === "string") {
    const response = await fetch(wasmSource);
    if (!response.ok) throw new Error(`Failed to fetch extension: ${response.status}`);
    bytes = await response.arrayBuffer();
  } else if (wasmSource instanceof Response) {
    bytes = await wasmSource.arrayBuffer();
  } else if (ArrayBuffer.isView(wasmSource)) {
    bytes =
      wasmSource.byteOffset === 0 && wasmSource.byteLength === wasmSource.buffer.byteLength
        ? (wasmSource.buffer as ArrayBuffer)
        : (wasmSource.buffer.slice(
            wasmSource.byteOffset,
            wasmSource.byteOffset + wasmSource.byteLength,
          ) as ArrayBuffer);
  } else {
    bytes = wasmSource as ArrayBuffer;
  }

  const wasmBytes = new Uint8Array(bytes);
  const extModule = await WebAssembly.compile(bytes);

  // Memory layout: [data segment (dylink.0 memory_size)][stack][lpad page].
  // The dylink.0 section declares the module's data+bss size; the file size
  // is only a (typically over-, possibly under-) estimate kept as a fallback.
  const dylink = parseDylinkMemInfo(wasmBytes);
  let dataSize: number;
  if (dylink) {
    dataSize = dylink.memorySize;
    if (1 << dylink.memoryAlign > PAGE) {
      console.warn(
        `[ext] '${name}' requests 2^${dylink.memoryAlign} memory alignment; only page alignment is provided`,
      );
    }
  } else {
    console.warn(`[ext] '${name}' has no dylink.0 section; sizing data segment from file size`);
    dataSize = bytes.byteLength;
  }

  const dataBytes = alignUp(dataSize, PAGE);
  const currentBytes = sharedMemory.buffer.byteLength;
  sharedMemory.grow((dataBytes + EXT_STACK_SIZE + PAGE) / PAGE);
  const memBase = currentBytes;
  const stackTop = memBase + dataBytes + EXT_STACK_SIZE;
  // Small scratch area for __wasm_lpad_context (3 x i32 = 12 bytes)
  const lpadContextAddr = stackTop + PAGE - 64;

  const moduleExportDescs = WebAssembly.Module.exports(extModule);

  // Table slots for the module's element segments. dylink.0 states the count;
  // fall back to parsing the element section. Later needs (GOT entries,
  // dlSym exports) grow the table on demand via canonicalTableIndex.
  const tableSlots = dylink?.tableSize ?? countElementSegmentEntries(wasmBytes);
  const tableBase = sharedTable.length;
  sharedTable.grow(tableSlots);

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
    if (imp.module === "env" && Object.hasOwn(imports.env as object, imp.name)) {
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
          if (typeof fn !== "function") {
            throw new Error(`[ext] self-import '${sym}' called before instantiation completed`);
          }
          return (fn as Function)(...args);
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
        // Minimal personality: The enclosing landing pad handles the exception
        // Offsets are the libc++abi wasm32 __cxa_exception layout.
        const ADJUSTED_PTR_OFFSET = -8;
        const THROWN_OBJECT_OFFSET = 32;
        const LPAD_SELECTOR_OFFSET = 8;
        const URC_HANDLER_FOUND = 6;
        (imports.env as Record<string, unknown>)[imp.name] = (excPtr: number) => {
          const view = new DataView(sharedMemory!.buffer);
          view.setUint32(excPtr + ADJUSTED_PTR_OFFSET, excPtr + THROWN_OBJECT_OFFSET, true);
          view.setInt32(lpadContextAddr + LPAD_SELECTOR_OFFSET, 1, true);
          return URC_HANDLER_FOUND;
        };
      } else if (imp.name === "_Unwind_DeleteException") {
        (imports.env as Record<string, unknown>)[imp.name] = (excPtr: number) => {
          // _Unwind_Exception holds a cleanup function pointer at offset 8
          // libc++abi points it at the routine that destroys and frees the
          // exception object.
          const URC_FOREIGN_EXCEPTION_CAUGHT = 1;
          const cleanupIdx = new DataView(sharedMemory!.buffer).getUint32(excPtr + 8, true);
          if (cleanupIdx) {
            const fn = sharedTable!.get(cleanupIdx);
            if (typeof fn === "function") fn(URC_FOREIGN_EXCEPTION_CAUGHT, excPtr);
          }
        };
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
      const params = (imp as { type?: { parameters?: WebAssembly.ValueType[] } }).type
        ?.parameters ?? ["i32"];
      const tag = new WebAssembly.Tag({ parameters: params as WebAssembly.ValueType[] });
      (imports.env as Record<string, unknown>)[imp.name] = tag;
      if (imp.name === "__cpp_exception") {
        cppExceptionTag = tag;
      }
    }

    if ((imp.module === "GOT.func" || imp.module === "GOT.mem") && imp.kind === "global") {
      if (!imports[imp.module]) imports[imp.module] = {};
      const hostFn = hostExports[imp.name];
      if (typeof hostFn === "function") {
        (imports[imp.module] as Record<string, unknown>)[imp.name] =
          new WebAssembly.Global({ value: "i32", mutable: true }, canonicalTableIndex(hostFn));
      } else if (imp.module === "GOT.mem" && imp.name === "__wasm_lpad_context") {
        (imports[imp.module] as Record<string, unknown>)[imp.name] =
          new WebAssembly.Global({ value: "i32", mutable: true }, lpadContextAddr);
      } else if (extExportNames.has(imp.name) || moduleExportDescs.some((e) => e.name === imp.name)) {
        // Defined by the extension itself — resolved after instantiation.
        (imports[imp.module] as Record<string, unknown>)[imp.name] =
          new WebAssembly.Global({ value: "i32", mutable: true }, 0);
      } else {
        console.warn(`[ext] unresolved ${imp.module} import '${imp.name}' bound to address 0`);
        (imports[imp.module] as Record<string, unknown>)[imp.name] =
          new WebAssembly.Global({ value: "i32", mutable: true }, 0);
      }
    }
  }

  // Async instantiation: Chrome disallows synchronous WebAssembly.Instance
  // on the main thread for modules larger than 8MB.
  extInstance = await WebAssembly.instantiate(extModule, imports);

  // The element segments just populated [tableBase, tableBase + tableSlots);
  // record those indices as the canonical addresses of the extension's
  // functions so GOT fixups and dlSym reuse them.
  cacheTableRange(tableBase, sharedTable.length);

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
            g.value = canonicalTableIndex(fn);
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
      extExports[exp.name] = canonicalTableIndex(fn as Function);
    }
  }

  registry.set(name, {
    instance: extInstance,
    exports: extExports,
  });
}

function alignUp(value: number, alignment: number): number {
  return Math.ceil(value / alignment) * alignment;
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

interface DylinkMemInfo {
  memorySize: number;
  memoryAlign: number;
  tableSize: number;
  tableAlign: number;
}

// Parse the WASM_DYLINK_MEM_INFO subsection of the dylink.0 custom section,
// which declares the memory (data + bss) and table sizes a PIC shared module
// needs from the dynamic linker.
function parseDylinkMemInfo(wasm: Uint8Array): DylinkMemInfo | null {
  let pos = 8;
  while (pos < wasm.length) {
    const sid = wasm[pos++];
    let size: number;
    [size, pos] = readLEB128(wasm, pos);
    const end = pos + size;
    if (sid === 0) {
      let nlen: number, p: number;
      [nlen, p] = readLEB128(wasm, pos);
      const sectionName = new TextDecoder().decode(wasm.subarray(p, p + nlen));
      if (sectionName === "dylink.0") {
        let q = p + nlen;
        while (q < end) {
          const sub = wasm[q++];
          let ssize: number;
          [ssize, q] = readLEB128(wasm, q);
          const send = q + ssize;
          if (sub === 1) {
            // WASM_DYLINK_MEM_INFO
            let memorySize: number, memoryAlign: number, tableSize: number, tableAlign: number;
            [memorySize, q] = readLEB128(wasm, q);
            [memoryAlign, q] = readLEB128(wasm, q);
            [tableSize, q] = readLEB128(wasm, q);
            [tableAlign, q] = readLEB128(wasm, q);
            return { memorySize, memoryAlign, tableSize, tableAlign };
          }
          q = send;
        }
        return null;
      }
    }
    pos = end;
  }
  return null;
}

// Fallback for modules without a dylink.0 section: count the entries of the
// active element segments to size the table reservation.
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
        if (flags !== 0) break;
        // Offset expression: (i32.const <leb>) or (global.get <leb>), then end.
        const op = wasm[pos++];
        if (op !== 0x41 && op !== 0x23) break;
        [, pos] = readLEB128(wasm, pos);
        if (wasm[pos++] !== 0x0b) break;
        let [numElem, p3] = readLEB128(wasm, pos);
        pos = p3;
        total += numElem;
        for (let j = 0; j < numElem; j++) {
          [, pos] = readLEB128(wasm, pos);
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
