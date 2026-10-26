"""Explicit optional native query backend for lossless 128-D SIFT uint8 rows."""
import ctypes
import numpy as np


class NativeUint8Scan:
    def __init__(self, path):
        self.library = ctypes.CDLL(str(path))
        self.scan = self.library.nora_scan_u8_128
        self.scan.argtypes = [ctypes.c_void_p,ctypes.c_uint64,ctypes.c_void_p,
                              ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
        self.scan.restype = ctypes.c_int

    def distances(self, store, ids, query):
        ids = np.ascontiguousarray(ids,dtype=np.uint32)
        base = store.base
        if ids.ndim != 1:
            raise ValueError("Native scan IDs must be one-dimensional")
        if base.ndim != 2 or base.dtype != np.uint8 or base.shape[1] != 128 or not base.flags.c_contiguous:
            raise ValueError("Native scan requires contiguous original uint8 SIFT-128 rows")
        q = np.ascontiguousarray(query,dtype=np.uint8)
        if q.shape != (128,) or not np.array_equal(q.astype(np.float32),query):
            raise ValueError("Native SIFT query must contain exactly representable uint8 components")
        out = np.empty(len(ids),dtype=np.float32)
        status = self.scan(base.ctypes.data,len(base),ids.ctypes.data,len(ids),
                           q.ctypes.data,out.ctypes.data)
        if status:
            raise RuntimeError(f"Native scanner failed with status {status}")
        for at in np.flatnonzero(ids >= len(base)):
            diff = store.extra[int(ids[at])]-query
            out[at] = np.dot(diff,diff)
        return out
