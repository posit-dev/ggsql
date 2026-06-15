/*!
Rust bindings for tree-sitter-ggsql grammar

This crate provides the tree-sitter language definition for ggsql,
a SQL extension for declarative data visualization.
*/

use tree_sitter::Language;

extern "C" {
    fn tree_sitter_ggsql() -> Language;
}

/// Returns the tree-sitter language for ggsql
pub fn language() -> Language {
    unsafe { tree_sitter_ggsql() }
}

/// The node types and field names used by the ggsql grammar
pub const NODE_TYPES: &str = include_str!("../../src/node-types.json");

/// The C libc allocator for wasm32-unknown-unknown builds.
///
/// The C code linked into the module (the generated parser and the
/// tree-sitter runtime) has no libc, so `malloc` and friends are defined
/// here on the Rust global allocator and the whole module shares one heap.
/// Each allocation carries a header recording its size, so the `Layout`
/// can be reconstructed on free.
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
mod wasm_alloc {
    use core::ptr::{null_mut, write_bytes};
    use std::alloc::{alloc, dealloc, realloc as rust_realloc, Layout};

    const HEADER: usize = core::mem::size_of::<usize>() * 2;

    #[no_mangle]
    unsafe extern "C" fn malloc(size: usize) -> *mut u8 {
        let Some(total) = size.checked_add(HEADER) else {
            return null_mut();
        };
        let Ok(layout) = Layout::from_size_align(total, HEADER) else {
            return null_mut();
        };
        let ptr = alloc(layout);
        if ptr.is_null() {
            return null_mut();
        }
        *ptr.cast::<usize>() = size;
        ptr.add(HEADER)
    }

    #[no_mangle]
    unsafe extern "C" fn free(ptr: *mut u8) {
        if ptr.is_null() {
            return;
        }
        let base = ptr.sub(HEADER);
        let size = *base.cast::<usize>();
        dealloc(
            base,
            Layout::from_size_align_unchecked(size + HEADER, HEADER),
        );
    }

    #[no_mangle]
    unsafe extern "C" fn realloc(ptr: *mut u8, new_size: usize) -> *mut u8 {
        if ptr.is_null() {
            return malloc(new_size);
        }
        let Some(new_total) = new_size.checked_add(HEADER) else {
            return null_mut();
        };
        if Layout::from_size_align(new_total, HEADER).is_err() {
            return null_mut();
        }
        let base = ptr.sub(HEADER);
        let size = *base.cast::<usize>();
        let layout = Layout::from_size_align_unchecked(size + HEADER, HEADER);
        let new = rust_realloc(base, layout, new_total);
        if new.is_null() {
            return null_mut();
        }
        *new.cast::<usize>() = new_size;
        new.add(HEADER)
    }

    #[no_mangle]
    unsafe extern "C" fn calloc(count: usize, size: usize) -> *mut u8 {
        let Some(total) = count.checked_mul(size) else {
            return null_mut();
        };
        let ptr = malloc(total);
        if !ptr.is_null() {
            write_bytes(ptr, 0, total);
        }
        ptr
    }

    #[no_mangle]
    unsafe extern "C" fn abort() -> ! {
        std::process::abort()
    }
}

/// The highlighting queries for ggsql syntax
pub const HIGHLIGHTS_QUERY: &str = include_str!("../../queries/highlights.scm");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language() {
        let language = language();
        assert!(language.abi_version() <= tree_sitter::LANGUAGE_VERSION);
        assert!(language.abi_version() >= tree_sitter::MIN_COMPATIBLE_LANGUAGE_VERSION);
    }
}
