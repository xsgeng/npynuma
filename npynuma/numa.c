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
    if (!ptr) return NULL;

    if (!nctx->ptr_dict) {
        nctx->ptr_dict = PyDict_New();
        if (!nctx->ptr_dict) {
            numa_free(ptr, size);
            return NULL;
        }
    }

    PyObject *key = PyLong_FromVoidPtr(ptr);
    if (!key) {
        numa_free(ptr, size);
        return NULL;
    }

    PyObject *node_obj = PyLong_FromLong(nctx->node);
    PyObject *size_obj = PyLong_FromLong(size);
    if (!node_obj || !size_obj) {
        if (node_obj) Py_DECREF(node_obj);
        if (size_obj) Py_DECREF(size_obj);
        Py_DECREF(key);
        numa_free(ptr, size);
        return NULL;
    }

    PyObject *tuple = PyTuple_Pack(2, node_obj, size_obj);
    Py_DECREF(node_obj);
    Py_DECREF(size_obj);
    if (!tuple) {
        Py_DECREF(key);
        numa_free(ptr, size);
        return NULL;
    }

    if (PyDict_SetItem(nctx->ptr_dict, key, tuple) == -1) {
        Py_DECREF(key);
        Py_DECREF(tuple);
        numa_free(ptr, size);
        return NULL;
    }

    Py_DECREF(key);
    Py_DECREF(tuple);
    return ptr;
}

static void* numa_calloc_(void *ctx, size_t nelem, size_t elsize) {
    size_t size = nelem * elsize;
    
    // numa_alloc_onnode initializes memory to zero on my system.
    void *ptr = numa_malloc_(ctx, size);

    return ptr;
}

static void numa_free_(void *ctx, void *ptr, size_t size) {
    NumaCtx *nctx = (NumaCtx *)ctx;
    if (nctx->ptr_dict) {
        PyObject *key = PyLong_FromVoidPtr(ptr);
        if (key) {
            PyDict_DelItem(nctx->ptr_dict, key);
            Py_DECREF(key);
        }
    }
    numa_free(ptr, size);
}

static void* numa_realloc_(void *ctx, void *ptr, size_t new_size) {
    NumaCtx *nctx = (NumaCtx *)ctx;
    
    PyObject *key = PyLong_FromVoidPtr(ptr);
    if (!key) return NULL;
    
    PyObject *entry = PyDict_GetItem(nctx->ptr_dict, key);
    if (!entry) {
        Py_DECREF(key);
        PyErr_SetString(PyExc_RuntimeError, "Realloc on untracked pointer");
        return NULL;
    }
    
    PyObject *old_size_obj = PyTuple_GetItem(entry, 1);
    PyObject *node_obj = PyTuple_GetItem(entry, 0);
    if (!old_size_obj || !node_obj) {
        Py_DECREF(key);
        return NULL;
    }
    
    size_t old_size = PyLong_AsSize_t(old_size_obj);
    int node = PyLong_AsLong(node_obj);
    if ((old_size == (size_t)-1 || node == -1) && PyErr_Occurred()) {
        Py_DECREF(key);
        return NULL;
    }
    
    if (PyDict_DelItem(nctx->ptr_dict, key) == -1) {
        Py_DECREF(key);
        return NULL;
    }
    Py_DECREF(key);
    
    void *ptr_new = numa_realloc(ptr, old_size, new_size);
    if (!ptr_new) return NULL;
    
    // Create new entry
    PyObject *new_key = PyLong_FromVoidPtr(ptr_new);
    if (!new_key) {
        numa_free(ptr_new, new_size);
        return NULL;
    }
    
    PyObject *new_node_obj = PyLong_FromLong(node);
    PyObject *new_size_obj = PyLong_FromLong(new_size);
    if (!new_node_obj || !new_size_obj) {
        if (new_node_obj) Py_DECREF(new_node_obj);
        if (new_size_obj) Py_DECREF(new_size_obj);
        Py_DECREF(new_key);
        numa_free(ptr_new, new_size);
        return NULL;
    }
    
    PyObject *new_tuple = PyTuple_Pack(2, new_node_obj, new_size_obj);
    Py_DECREF(new_node_obj);
    Py_DECREF(new_size_obj);
    if (!new_tuple) {
        Py_DECREF(new_key);
        numa_free(ptr_new, new_size);
        return NULL;
    }
    
    if (PyDict_SetItem(nctx->ptr_dict, new_key, new_tuple) == -1) {
        Py_DECREF(new_key);
        Py_DECREF(new_tuple);
        numa_free(ptr_new, new_size);
        return NULL;
    }
    
    Py_DECREF(new_key);
    Py_DECREF(new_tuple);
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
        if (!ctx.ptr_dict) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create ptr_dict");
            return NULL;
        }
    }

    if (numa_available() < 0) {
        PyErr_SetString(PyExc_RuntimeError, "NUMA not available");
        return NULL;
    }
    
    ctx.node = node;

    PyObject *capsule = PyCapsule_New(&handler, "mem_handler", NULL);
    if (!capsule) {
        return NULL;
    }
    
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
    import_array()
    return PyModule_Create(&numa_module);
}
