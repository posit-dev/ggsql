#include <Python.h>

typedef struct TSLanguage TSLanguage;

TSLanguage *tree_sitter_ggsql(void);

static PyObject* language(PyObject *self, PyObject *args) {
    return PyLong_FromVoidPtr(tree_sitter_ggsql());
}

static PyMethodDef methods[] = {
    {"language", language, METH_NOARGS,
     "Get the tree-sitter language for ggsql."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "tree_sitter_ggsql.binding",
    "Tree-sitter bindings for ggsql",
    -1,
    methods
};

PyMODINIT_FUNC PyInit_binding(void) {
    return PyModule_Create(&module);
}
