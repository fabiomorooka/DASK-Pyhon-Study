import ctypes as ct
import numpy as np
import os 
libc = None
array_matrice = np.ctypeslib.ndpointer(dtype=np.uint8, ndim=2)

def get_or_load():
    global libc
    if libc is None:
        libc = ct.CDLL(os.getcwd() + "/sobel.so")
    return libc


def sobel(width, height, input_array):
    _lib = get_or_load()
    _lib.sobel.argtypes = [ct.c_int, ct.c_int, array_matrice, array_matrice]
    output_array = np.zeros((height,width), dtype=np.ubyte)
    _lib.sobel(width, height, input_array, output_array)
    return output_array 
