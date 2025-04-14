#define NPY_TARGET_VERSION NPY_1_22_API_VERSION
#include <Python.h>
#include <numa.h>
#include "numpy/arrayobject.h"

typedef struct {
    int node;
    // global dict[adress, tuple[node, size]] storing node and address
    PyObject *ptr_dict;
} NumaCtx;

// curret global context
static NumaCtx ctx = {0, NULL};

static void* numa_malloc_(void *ctx, size_t size) {
    NumaCtx *nctx = (NumaCtx*)ctx;
    void *ptr = numa_alloc_onnode(size, nctx->node);
    if (!nctx->ptr_dict) {
        nctx->ptr_dict = PyDict_New();
    }
    
    // store node and address
    PyDict_SetItem(nctx->ptr_dict, PyLong_FromVoidPtr(ptr), PyTuple_Pack(2, PyLong_FromLong(nctx->node), PyLong_FromLong(size)));
    if (!ptr) return NULL;
    return ptr;
}

static void* numa_calloc_(void *ctx, size_t nelem, size_t elsize) {
    NumaCtx *nctx = (NumaCtx *)ctx;
    void *ptr = numa_alloc_onnode(nelem * elsize, nctx->node);
    return ptr;
}

static void numa_free_(void *ctx, void *ptr, size_t size) {
    numa_free(ptr, size);
}

static void* numa_realloc_(void *ctx, void *ptr, size_t new_size) {
    NumaCtx *nctx = (NumaCtx *)ctx;

    size_t old_size = PyLong_AsSize_t(PyTuple_GetItem(PyDict_GetItem(nctx->ptr_dict, PyLong_FromVoidPtr(ptr)), 1));
    int node = PyLong_AsLong(PyTuple_GetItem(PyDict_GetItem(nctx->ptr_dict, PyLong_FromVoidPtr(ptr)), 0));
    PyDict_DelItem(nctx->ptr_dict, PyLong_FromVoidPtr(ptr));
    void *ptr_new = numa_realloc(ptr, old_size, new_size);
    // update ptr
    PyDict_SetItem(nctx->ptr_dict, PyLong_FromVoidPtr(ptr_new), PyTuple_Pack(2, PyLong_FromLong(node), PyLong_FromLong(new_size)));
    return ptr_new;
}

static PyDataMem_Handler handler = {
    "numa_handler",
    1,
    {
        &ctx, /* ctx */
        numa_malloc_,
        numa_calloc_,
        numa_realloc_,
        numa_free_
    }
};

static PyObject* set_numa_policy(PyObject *self, PyObject *args) {
    int node;
    if (!PyArg_ParseTuple(args, "i", &node)) return NULL;

    if (!ctx.ptr_dict) {
        ctx.ptr_dict = PyDict_New();
    }

    if (numa_available() < 0) {
        PyErr_SetString(PyExc_RuntimeError, "NUMA not available");
        return NULL;
    }
    
    ctx.node = node;

    PyObject *capsule = PyCapsule_New(&handler, "mem_handler", NULL);
    PyObject *old = PyDataMem_SetHandler(capsule);
    Py_DECREF(capsule);
    return old;
}

static PyObject* reset_numa_policy(PyObject *self, PyObject *args) {
    PyDataMem_SetHandler(NULL);
    Py_RETURN_NONE;
}

static PyObject* get_numa_nodes(PyObject *self, PyObject *args) {
    if (numa_available() < 0) {
        PyErr_SetString(PyExc_RuntimeError, "NUMA not available");
        return NULL;
    }
    
    struct bitmask *nodes = numa_all_nodes_ptr;
    PyObject *node_list = PyList_New(0);
    int max_node = numa_max_node();
    
    for (int node = 0; node <= max_node; node++) {
        if (numa_bitmask_isbitset(nodes, node)) {
            PyList_Append(node_list, PyLong_FromLong(node));
        }
    }
    
    return node_list;
}

static PyMethodDef module_methods[] = {
    {"set_numa_policy", set_numa_policy, METH_VARARGS, "Set NUMA allocation policy"},
    {"reset_numa_policy", reset_numa_policy, METH_NOARGS, "Reset NUMA allocation policy"},
    {"get_numa_nodes", get_numa_nodes, METH_NOARGS, "Get list of available NUMA nodes"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef numa_module = {
    PyModuleDef_HEAD_INIT,
    "numa",
    NULL,
    -1,
    module_methods
};

PyMODINIT_FUNC PyInit_numa(void) {
    import_array();
    return PyModule_Create(&numa_module);
}
