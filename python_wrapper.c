#include <Python.h>

// Initialize Python interpreter
void init_python() {
    Py_Initialize();
}

// Finalize Python interpreter
void close_python() {
    Py_Finalize();
}

// Run arbitrary Python code
const char* run_python(const char* code) {
    if (!Py_IsInitialized()) {
        Py_Initialize();
    }
    int result = PyRun_SimpleString(code);
    if (result == 0) {
        return "success";
    }
    return "error";
}
