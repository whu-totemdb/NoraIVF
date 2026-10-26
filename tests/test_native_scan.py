"""Real-vector correctness gate for optional native scan; not paper timings."""
import os,sys,time
from pathlib import Path
sys.path.insert(0,os.getcwd())
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from experiments.build_sift1b_scale import binary_matrix
from noraivf.native_scan import NativeUint8Scan
from noraivf.scale_storage import ArrayPosting,RawVectors,make_index,search

base = binary_matrix(sys.argv[1],np.uint8)
queries = binary_matrix(sys.argv[2],np.uint8)
native = NativeUint8Scan(sys.argv[3])
store = RawVectors(base,len(base)+32)
store[len(base)] = base[0].astype(np.float32)
rng = np.random.default_rng(42)
ids = np.concatenate((rng.integers(0,len(base),size=100000,dtype=np.uint32),
                      np.array([0,0,len(base)],dtype=np.uint32)))
timings=[]
for query in queries[:5].astype(np.float32):
    t=time.perf_counter()
    x=store.gather(ids)-query
    reference=np.einsum("ij,ij->i",x,x)
    numpy_s=time.perf_counter()-t
    t=time.perf_counter()
    got=native.distances(store,ids,query)
    native_s=time.perf_counter()-t
    assert np.array_equal(reference,got)
    assert np.array_equal(np.lexsort((ids,reference))[:100],np.lexsort((ids,got))[:100])
    timings.append(dict(numpy_s=numpy_s,native_s=native_s))
# A uint8 view with stride two must be copied before passing its C pointer.
strided=np.arange(256,dtype=np.uint8)[::2]
diff=store.gather(ids)-strided.astype(np.float32)
assert np.array_equal(native.distances(store,ids,strided),np.einsum("ij,ij->i",diff,diff))
try:
    native.distances(store,ids[:4].reshape(2,2),queries[0])
except ValueError:
    pass
else:
    raise AssertionError("2-D ID arrays must be rejected")
# End-to-end scanner returns the identical ID order even for duplicate/tied IDs.
small=base[:10000]
assign=np.arange(len(small),dtype=np.int32)%16
order=np.argsort(assign,kind="stable").astype(np.uint32)
counts=np.bincount(assign,minlength=16)
sums=np.array([small[assign==c].astype(np.float32).sum(axis=0) for c in range(16)])
centroids=sums/counts[:,None]
idx=make_index("NoraIVF",small,centroids.astype(np.float32),assign,order,counts,sums)
for query in queries[:5].astype(np.float32):
    idx.native_scan=None
    ref=search(idx,query,k=100,nprobe=8)
    idx.native_scan=native
    got=search(idx,query,k=100,nprobe=8)
    assert np.array_equal(ref[0],got[0]) and np.array_equal(ref[1],got[1]) and ref[2]==got[2]
# Every vector has distance zero; chunk boundaries must retain the globally
# smallest IDs even when posting order is descending.
zeros=np.zeros((200,128),dtype=np.uint8)
zidx=make_index("NoraIVF",zeros,np.zeros((1,128),dtype=np.float32),
               np.zeros(200,dtype=np.int32),np.arange(200,dtype=np.uint32),
               np.array([200]),np.zeros((1,128),dtype=np.float32))
zidx.posting_lists=[ArrayPosting(np.arange(199,-1,-1,dtype=np.int64).astype(np.uint32))]
for backend in [None,native]:
    zidx.native_scan=backend
    for batch in [37,200]:
        zi,zd,zc=search(zidx,np.zeros(128,dtype=np.float32),k=100,nprobe=1,batch=batch)
        assert np.array_equal(zi,np.arange(100,dtype=np.uint32))
        assert np.all(zd==0) and zc==200
print("PASS: 600018 exact candidate distances including strided query, ID shape guard, deterministic boundary ties, overlay, and 5 end-to-end queries",flush=True)
print("Development check only, NOT paper experiment timing:",timings,flush=True)
