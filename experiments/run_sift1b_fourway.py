"""Matched real updates and queries on the complete SIFT1B index.

Each policy chooses its own maintenance actions. Evaluation scans do not
alter policy state; separately recorded workload queries update Ada's
partition temperatures. No forced split is used in this comparison.
"""
from __future__ import annotations
import argparse,gc,hashlib,json,os,platform,signal,sys,time,traceback
from contextlib import contextmanager
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import faiss
import numpy as np
from experiments.build_sift1b_scale import atomic_json,binary_matrix,sample_fingerprint
from experiments.run_sift1b_scale import memory,file_hash,prepare_postings,prefault_mappings,evaluate,partition_audit
from noraivf.scale_storage import make_index,ArrayPosting,RawVectors,posting_array,search
from noraivf.lean import IntMap

METHODS=['SPFresh','AdaIVF','DeDrift','NoraIVF']


class RunDeadline:
    """Cooperative phase/chunk checks; an external supervisor sets the hard cap.

    Signals request a stop at the next check rather than interrupting an atomic
    result write. A native FAISS call can finish before Python checks again.
    """
    def __init__(self, start, max_seconds):
        self.start = start
        self.max_seconds = max_seconds
        self.phase = 'initialization'
        self.stop_signal = None

    def request_stop(self, signum, frame):
        self.stop_signal = signal.Signals(signum).name

    def check(self, phase):
        self.phase = phase
        if self.stop_signal:
            raise TimeoutError(f'{self.stop_signal} requested a stop at {phase}')
        if time.perf_counter() - self.start >= self.max_seconds:
            raise TimeoutError(f'Overall time budget exhausted at {phase}')

    @contextmanager
    def during(self, phase):
        self.check('before ' + phase)
        yield
        self.check('after ' + phase)


def checked_file_hash(path, deadline):
    """Hash large cache files without materializing them in process memory."""
    path = Path(path)
    h = hashlib.sha256()
    with deadline.during('SHA-256 ' + path.name):
        with path.open('rb') as handle:
            while True:
                deadline.check('SHA-256 ' + path.name)
                block = handle.read(8 * 1024**2)
                if not block:
                    break
                h.update(block)
    return h.hexdigest()


def validated_squared_norm_sums(cache, n, nlist, deadline):
    """Bind Ada's policy statistics to this complete, unchanged posting cache."""
    values_path = cache / 'squared_norm_sums_reference.npy'
    metadata_path = cache / 'squared_norm_sums_reference.json'
    with deadline.during('Ada squared-norm metadata validation'):
        metadata = json.loads(metadata_path.read_text())
        if metadata.get('complete') is not True:
            raise RuntimeError('Ada squared-norm cache is not complete')
        if metadata.get('n') != n or metadata.get('nlist') != nlist:
            raise RuntimeError('Ada squared-norm cache has the wrong corpus/partition count')
        paths = {'sha256': values_path,
                 'build_state_sha256': cache / 'build_state.json',
                 'counts_sha256': cache / 'counts.npy',
                 'ids_sha256': cache / 'ids.npy'}
        actual = {}
        for key, path in paths.items():
            if key not in metadata:
                raise RuntimeError(f'Ada squared-norm metadata is missing {key}')
            actual[key] = checked_file_hash(path, deadline)
            if metadata[key] != actual[key]:
                raise RuntimeError(f'Ada squared-norm cache provenance mismatch: {key}')
        values = np.load(values_path)
        if values.shape != (nlist,) or not np.isfinite(values).all():
            raise RuntimeError('Ada squared-norm cache has invalid values')
        provenance = dict(path=str(values_path), metadata_path=str(metadata_path),
                          metadata_sha256=checked_file_hash(metadata_path, deadline),
                          metadata=metadata, verified_sha256=actual)
    return values, provenance


def evaluate_checked(idx, queries, gt, nprobes, canonical, deadline, phase, records):
    """Retain each complete operating point before checking the next deadline."""
    for nprobe in nprobes:
        with deadline.during(f'{phase}, nprobe={nprobe}'):
            records.extend(evaluate(idx, queries, gt, [nprobe], canonical))

def make_comparison_index(method,base,centroids,assignments,order,counts,sums,q_sums,seed,extra):
    if method in ('NoraIVF','SPFresh'):
        return make_index(method,base,centroids,assignments,order,counts,sums,seed,extra)
    if method=='DeDrift':
        from noraivf.scale_dedrift import CompactDeDrift,initialize_dedrift
        cls=CompactDeDrift
    elif method=='AdaIVF':
        from noraivf.scale_ada import CompactAdaIVF,initialize_ada
        cls=CompactAdaIVF
    else:raise ValueError(method)
    idx=cls.__new__(cls)
    idx.d=base.shape[1];idx.n=len(base);idx.nlist=idx.k=len(centroids)
    idx.seed=seed;idx.rng=np.random.RandomState(seed)
    idx.centroids=centroids.copy();idx.assignments=assignments;idx.data=None
    idx.id_to_vec=RawVectors(base,len(base)+extra);idx.next_id=len(base)
    bounds=np.r_[0,np.cumsum(counts)]
    idx.posting_lists=[ArrayPosting(order[bounds[c]:bounds[c+1]]) for c in range(idx.k)]
    for name in ['total_insert_time','total_delete_time','total_split_time','total_merge_time']:
        setattr(idx,name,0.)
    for name in ['n_inserts','n_deletes','n_splits','n_merges','n_reassigns']:
        setattr(idx,name,0)
    idx.split_history=[]
    pointer=IntMap(len(base)+extra);pointer._a[:len(base)]=assignments
    if method=='DeDrift':initialize_dedrift(idx,counts,sums,pointer)
    else:
        initialize_ada(idx,counts,sums,pointer,q_sums)
        idx.partition_map=idx.pam
    return idx


def audit(idx,ids):
    result=partition_audit(idx,ids)
    if hasattr(idx,'a_c'):
        assert np.isfinite(idx.a_c).all() and np.isfinite(idx.q_c).all()
        assert np.isfinite(idx.T).all()
        result.update(ada_temperature_min=float(idx.T.min()),ada_temperature_max=float(idx.T.max()),
                      ada_global_indicator=float(idx._global_indicator()))
    return result


def full_id_integrity_audit(idx,base,update_ids,canonical,deadline,identity,record=None,chunk=262144):
    """Prove the final live ID/vector multiset with an O(N)-byte bitmap.

    Setting seen[ids] alone does not find repeated IDs within one chunk. The
    final *actual* bitmap population equalling the scanned occurrence count
    proves uniqueness across and within chunks, without sorting any posting.
    Expected liveness comes from the fixed update trace, not index counters.
    This audit runs after every timed query and is reported separately.
    """
    started=time.perf_counter()
    if not __debug__:
        raise RuntimeError('Full integrity audit requires Python assertions (no -O)')
    if chunk<1:
        raise ValueError('Audit chunk size must be positive')
    record={} if record is None else record
    record.update(complete=False,identity=json.loads(json.dumps(identity)),
                  algorithm='full posting-ID scan and final byte-bitmap population; no per-chunk sorting',
                  included_in_performance_timings=False,chunk_rows=chunk)
    try:
        deadline.check('full ID audit: trace validation')
        initial_n=len(base)
        updates=[int(i) for i in update_ids]
        expected_deleted=set(updates)
        assert len(expected_deleted)==len(updates),'Update trace repeats an original ID'
        assert all(0<=i<initial_n for i in updates),'Update trace contains nonoriginal IDs'
        expected_canonical={initial_n+j:old for j,old in enumerate(updates)}
        assert canonical==expected_canonical,'Reinsert IDs/content mapping differs from the fixed trace'
        next_id=initial_n+len(updates)
        assert idx.next_id==next_id,'Unexpected next_id'
        store=idx.id_to_vec
        assert store.base is base and store.base_n==initial_n,'Original corpus backing changed'
        assert store.deleted==expected_deleted,'Deleted-ID set differs from update trace'
        assert set(store.extra)==set(expected_canonical),'Overlay-ID set differs from update trace'
        assert idx.n_inserts==idx.n_deletes==len(updates),'Public update counts differ from trace'
        assert idx.n==len(store)==initial_n,'Final live count differs from original corpus'
        assert len(idx.posting_lists)==idx.k,'Partition count mismatch'
        pointer=idx.ppm if hasattr(idx,'ppm') else (idx.pam if hasattr(idx,'pam') else idx.partition_map)
        if hasattr(idx,'pam') and hasattr(idx,'partition_map'):
            assert idx.partition_map._a is idx.pam._a,'Ada PAM alias no longer shares its actual lookup map'
        maps=[('partition',pointer)]
        if hasattr(idx,'positions'):maps.append(('position',idx.positions))
        deleted=np.asarray(updates,dtype=np.int64)
        for name,mapping in maps:
            assert mapping._a.ndim==1 and len(mapping._a)>=next_id,f'{name} map capacity mismatch'
            assert mapping._fill==-1,f'{name} map has an unexpected absent sentinel'
            assert np.all(mapping._a[deleted]==-1),f'Deleted IDs retain {name} entries'
            for start in range(next_id,len(mapping._a),chunk):
                deadline.check('full ID audit: unused '+name+' map tail')
                assert np.all(mapping._a[start:start+chunk]==-1),f'Unused IDs retain {name} entries'
        for new_id,old_id in expected_canonical.items():
            deadline.check('full ID audit: overlay vector identity')
            assert np.array_equal(store.extra[new_id],base[old_id].astype(np.float32)),f'Overlay vector {new_id} differs from original {old_id}'
        seen=np.zeros(next_id,dtype=np.uint8)
        record.update(expected_live_ids=initial_n,expected_deleted_ids=len(updates),
                      expected_overlay_ids=len(expected_canonical),bitmap_bytes=seen.nbytes,
                      scanned_posting_entries=0,scanned_partitions=0)
        entries=0
        for c,posting in enumerate(idx.posting_lists):
            deadline.check(f'full ID audit: partition {c}')
            if isinstance(posting,ArrayPosting):
                assert posting.a.ndim==1 and 0<=posting.n<=len(posting.a),'Invalid compact posting bounds'
                ids=posting.a[:posting.n]
            else:
                # Do not silently coerce negative, fractional, or overflowing
                # Python IDs to uint32 before validating their domain.
                ids=np.asarray(posting) if len(posting) else np.empty(0,dtype=np.uint32)
            assert ids.ndim==1 and np.issubdtype(ids.dtype,np.integer),'Posting IDs must be one-dimensional integers'
            assert len(ids)==len(posting),'Actual posting rows differ from the reported length'
            for start in range(0,len(ids),chunk):
                deadline.check(f'full ID audit: partition {c}, row {start}')
                block=ids[start:start+chunk]
                assert int(block.min())>=0 and int(block.max())<next_id,'Posting contains an out-of-domain ID'
                assert np.all(pointer._a[block]==c),f'Partition lookup mismatch in partition {c}'
                if hasattr(idx,'positions'):
                    assert np.array_equal(idx.positions._a[block],np.arange(start,start+len(block))),f'Position lookup mismatch in partition {c}'
                seen[block]=1
                entries+=len(block)
                record['scanned_posting_entries']=entries
            record['scanned_partitions']=c+1
        unique=0
        original_present=0
        for start in range(0,next_id,chunk):
            deadline.check('full ID audit: final bitmap population')
            end=min(start+chunk,next_id)
            unique+=int(np.count_nonzero(seen[start:end]))
            if start<initial_n:
                original_present+=int(np.count_nonzero(seen[start:min(end,initial_n)]))
        record.update(unique_posting_ids=unique,original_live_ids_present=original_present)
        assert entries==unique==initial_n,'Duplicate or missing posting IDs: scanned occurrences != unique bitmap population != expected N'
        assert not np.any(seen[deleted]),'Deleted original IDs remain in postings'
        assert original_present==initial_n-len(updates),'Missing original live IDs'
        assert np.all(seen[initial_n:next_id]==1),'Missing reinserted IDs'
        deadline.check('full ID audit: completed proof')
        record.update(complete=True,checks=['fixed trace and sequential fresh IDs',
                      'exact deleted/overlay sets and every overlay vector',
                      'every posting ID in domain and every actual PAM lookup',
                      'every Nora position lookup where present',
                      'deleted and unused map entries absent',
                      'global uniqueness including same-chunk duplicates',
                      'all original live IDs and all reinserts present'])
        return record
    finally:
        record['wall_s']=time.perf_counter()-started


def counters(idx):
    result=dict(maintenance_s=float(idx.total_split_time+idx.total_merge_time),
                index_slots=int(idx.k),nonempty_partitions=sum(bool(len(p)) for p in idx.posting_lists),
                reference_n_splits=int(idx.n_splits),reference_n_merges=int(idx.n_merges))
    if hasattr(idx,'rebalance_events'):
        result['dedrift_rebalances']=len(idx.rebalance_events)
    if hasattr(idx,'ada_stats'):result['ada_stats']=dict(idx.ada_stats)
    return result


def policy_parameters(idx):
    names=('L','split_factor','merge_factor','avg_size','T_split','T_merge',
           'use_scdkm','do_reassign','k2','rebalance_every','tau_s','alpha',
           'beta','tau_f','r_c','iota','check_every','gamma','tau_g','eta','nu',
           'use_temperature','scan_batch','similarity_bytes','max_split_training_bytes')
    return {name:(getattr(idx,name).item() if isinstance(getattr(idx,name),np.generic)
                  else getattr(idx,name)) for name in names if hasattr(idx,name)}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('base','queries','gt','cache','out','scan-library','validated-gt-from'):
        ap.add_argument('--'+name,required=True)
    ap.add_argument('--methods',nargs='+',choices=METHODS,default=METHODS)
    ap.add_argument('--update-pairs',type=int,default=1000)
    ap.add_argument('--workload-query-every',type=int,default=10)
    ap.add_argument('--query-count',type=int,default=100)
    ap.add_argument('--query-repeats',type=int,default=3)
    ap.add_argument('--nprobes',nargs='+',type=int,default=[1,8,32])
    ap.add_argument('--threads',type=int,default=32)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--audit-sample',type=int,default=20000)
    ap.add_argument('--max-seconds',type=float,default=39600)
    ap.add_argument('--pilot',action='store_true',help='Cost/correctness pilot, never final paper results')
    ap.add_argument('--subset-smoke',action='store_true')
    a=ap.parse_args()
    if sys.flags.optimize:
        ap.error('Run without -O: scientific integrity assertions must remain enabled')
    if not (0<a.query_count<=100 and a.update_pairs>0 and a.query_repeats>0):
        ap.error('query-count must be 1..100; update-pairs/query-repeats must be positive')
    if not np.isfinite(a.max_seconds) or a.max_seconds<=0:
        ap.error('max-seconds must be finite and positive')
    if a.workload_query_every<0 or a.threads<1 or any(p<1 for p in a.nprobes):
        ap.error('query interval must be nonnegative; threads and nprobes must be positive')
    out=Path(a.out)
    if out.exists():raise RuntimeError('Output exists; preserve earlier measurements and choose a new path')
    out.parent.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();cache=Path(a.cache)
    deadline=RunDeadline(start,a.max_seconds)
    result=dict(complete=False,config=vars(a),methods={},
                scope='pilot only' if a.pilot else ('subset smoke only' if a.subset_smoke else 'full SIFT1B matched update/query workload'),
                started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                deadline_policy='Cooperative checks before/after phases, between query operating points, updates, and adapter progress callbacks; SIGTERM/SIGINT checkpoint at the next check. External supervisor enforces the hard deadline.')
    previous_handlers={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
    for sig in previous_handlers:signal.signal(sig,deadline.request_stop)
    try:
        atomic_json(out,result)
        with deadline.during('runtime configuration'):
            faiss.omp_set_num_threads(a.threads)
            np.seterr(divide='raise',over='raise',invalid='raise')
        with deadline.during('build cache and memory validation'):
            build=json.loads((cache/'build_state.json').read_text())
            n=int(build['config']['n'])
            assert build['complete'] and build['completed_rows']==n
            assert a.subset_smoke or n==1000000000
            assert sample_fingerprint(a.base)==build['config']['base']
            available=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
            assert available>n*(build['config']['dim']+16)+32*1024**3,'Insufficient memory reserve'
        with deadline.during('corpus/query loading and query selection'):
            base=binary_matrix(a.base,np.uint8)[:n]
            allqueries=binary_matrix(a.queries,np.uint8);allgt=binary_matrix(a.gt,np.int32)
            rng=np.random.default_rng(a.seed)
            evaluation_ids=rng.choice(len(allqueries),size=min(100,len(allqueries)),replace=False)[:a.query_count]
            queries=allqueries[evaluation_ids].astype(np.float32);gt=allgt[evaluation_ids,:100]
            excluded=np.ones(len(allqueries),dtype=bool);excluded[evaluation_ids]=False
            operational_count=a.update_pairs//a.workload_query_every if a.workload_query_every else 0
            operational_ids=rng.choice(np.flatnonzero(excluded),size=operational_count,replace=False)
            assert not np.intersect1d(evaluation_ids,operational_ids).size
        if a.subset_smoke:
            with deadline.during('exact subset smoke ground truth'):
                exact=faiss.IndexFlatL2(base.shape[1]);exact.add(base.astype(np.float32));_,gt=exact.search(queries,100);del exact
        with deadline.during('ground-truth validation reuse'):
            prior=json.loads(Path(a.validated_gt_from).read_text())
            if not a.subset_smoke:
                assert prior['n']==n and prior['initial_build']['config']==build['config']
                assert set(map(int,evaluation_ids))<=set(prior['query_ids'])
                assert prior['provenance']['query_sha256']==file_hash(a.queries)
                assert prior['provenance']['gt_sha256']==file_hash(a.gt)
                validation=prior['independent_gt_validation']
                assert validation['rows_scanned']==n
                assert validation['n_queries']>0 and len(validation['rows'])==validation['n_queries']
                assert all(x['distance_ranks_match'] for x in validation['rows'])
        with deadline.during('posting cache preparation'):
            assignments=np.load(cache/'assign.npy',mmap_mode='r');centroids=np.load(cache/'centroids.npy');counts=np.load(cache/'counts.npy')
            assert int(counts.sum())==n==len(assignments)
            order,sums,posting_meta=prepare_postings(cache,base,assignments,counts)
        q_sums=q_provenance=None
        if 'AdaIVF' in a.methods:
            q_sums,q_provenance=validated_squared_norm_sums(cache,n,len(counts),deadline)
        with deadline.during('update trace selection and source provenance'):
            bounds=np.r_[0,np.cumsum(counts)];largest=int(np.argmax(counts));pool=order[bounds[largest]:bounds[largest+1]]
            assert a.update_pairs<=len(pool)
            update_ids=pool[np.linspace(0,len(pool)-1,a.update_pairs,dtype=np.int64)]
            sample=rng.integers(0,n,size=min(a.audit_sample,n),dtype=np.int64)
            sources=['experiments/run_sift1b_fourway.py','experiments/run_sift1b_scale.py','noraivf/scale_storage.py',
                     'noraivf/scale_dedrift.py','noraivf/scale_ada.py','noraivf/core.py','experiments/run_claim_evidence_annbench.py',
                     'noraivf/native_scan.py','noraivf/uint8_scan.cpp','noraivf/lean.py','experiments/build_sift1b_scale.py',
                     'experiments/prepare_sift1b_ada_stats.py']
            repository=Path(__file__).resolve().parents[1]
            source_paths={s:str(repository/s) for s in sources}
            source_hashes={s:checked_file_hash(path,deadline) for s,path in source_paths.items()}
            cache_paths={name:str((cache/name).resolve()) for name in
                         ('build_state.json','counts.npy','ids.npy','assign.npy','centroids.npy','sums.npy','postings_state.json')}
            # Ada's q-sum validation has already read these potentially large
            # files. Reuse those verified digests, never unchecked metadata.
            verified=q_provenance['verified_sha256'] if q_provenance else {}
            cache_hashes={name:verified[key] if key in verified else checked_file_hash(cache_paths[name],deadline)
                          for name,key in [('build_state.json','build_state_sha256'),
                                           ('counts.npy','counts_sha256'),('ids.npy','ids_sha256')]}
            for name in ('assign.npy','centroids.npy','sums.npy','postings_state.json'):
                cache_hashes[name]=checked_file_hash(cache_paths[name],deadline)
            provenance=dict(source_sha256=source_hashes,source_paths=source_paths,
                harness_path=str(Path(__file__).resolve()),cache_paths=cache_paths,cache_sha256=cache_hashes,
                native_library_path=str(Path(a.scan_library).resolve()),native_library_sha256=checked_file_hash(a.scan_library,deadline),
                query_path=str(Path(a.queries).resolve()),query_sha256=checked_file_hash(a.queries,deadline),
                gt_path=str(Path(a.gt).resolve()),gt_sha256=checked_file_hash(a.gt,deadline),
                base_sampled_fingerprint=build['config']['base'],
                ground_truth_validation_source_path=str(Path(a.validated_gt_from).resolve()),
                ground_truth_validation_source_sha256=checked_file_hash(a.validated_gt_from,deadline),
                ground_truth_validation=prior.get('independent_gt_validation'),ada_squared_norm_cache=q_provenance)
            identity_descriptor=dict(config=json.loads(json.dumps(vars(a))),source_sha256=source_hashes,
                source_paths=source_paths,cache_sha256=cache_hashes,cache_paths=cache_paths,
                base_sampled_fingerprint=build['config']['base'],native_library_sha256=provenance['native_library_sha256'],
                query_sha256=provenance['query_sha256'],gt_sha256=provenance['gt_sha256'],
                update_original_ids_sha256=hashlib.sha256(update_ids.astype('<i8').tobytes()).hexdigest(),
                evaluation_query_ids=evaluation_ids.tolist(),operational_query_ids=operational_ids.tolist())
            # Canonical JSON is frozen independently from mutable result fields.
            identity_json=json.dumps(identity_descriptor,sort_keys=True,separators=(',',':'))
            identity_sha256=hashlib.sha256(identity_json.encode()).hexdigest()
            result.update(n=n,nlist=len(counts),initial_build=build,postings_build=posting_meta,
                evaluation_query_ids=evaluation_ids.tolist(),operational_query_ids=operational_ids.tolist(),
                update_original_ids=update_ids.tolist(),update_source_partition=largest,update_source_partition_size=len(pool),
                implementation=dict(storage='same lossless uint8 corpus and compact ID postings for all methods',
                    query='same single-thread exact C++ squared-L2 scan; no quantization',
                    maintenance='repository L2 ports with compact containers; original trigger policies and default parameters',
                    ada_numerics='Ada maintenance center comparisons use stable float64 norm/dot with direct near-tie fallback; candidate KNN and balancing use direct float64 squared differences. This deliberately differs from the original float32 norm/dot kernel.',
                    evaluation='initial and final scans plus warmups never change policy state',
                    workload='delete/reinsert pairs of real rows; one disjoint official query at nprobe=8 after each enabled query interval; Ada temperatures updated by workload queries',
                    forced_splits=0,parallelism='same configured FAISS threads; native scan and Nora coordinate loop serial'),
                timing_definitions=dict(
                    primary_metric='total_update_wall_s',
                    total_update_wall_s='Sum of complete public delete and insert call wall times, including policy checks, maintained statistics, and triggered structural maintenance. Excludes input-row gathering and harness bookkeeping.',
                    maintenance_wall_s='Structural maintenance event timers only (reference total_split_time + total_merge_time); excludes policy detection and ordinary per-update statistics. This is not total update cost.',
                    maintenance_s='Counter alias for the same structural maintenance event timers only.',
                    workload_query_wall_s='Shared native query scan plus the method-specific query observation hook; separately reported from update calls.',
                    final_full_id_integrity_wall_s='Untimed O(N) ID/PAM/position and trace-derived vector-liveness audit after all final query repeats; excluded from update and query measurements.'),
                provenance=provenance,identity=dict(sha256=identity_sha256,canonical_json=identity_json),
                environment=dict(host=platform.node(),python=sys.version,numpy=np.__version__,faiss=faiss.__version__,
                    faiss_threads=a.threads,thread_env={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']}),
                limitations=['Finite localized update burst, not sustained billion-update throughput.',
                    'Repository in-memory ports, not upstream production implementations.',
                    'Same sampled centroid training and full-corpus assignment reused; construction excluded from update timings.',
                    'Deletion/reinsertion leaves the vector multiset unchanged; does not model distribution drift.',
                    'Ada uses the disclosed stable float64 maintenance kernel; equivalence to a stable-distance reference is separate from fidelity to the original float32 reference.',
                    'An Ada global-rebuild trigger stops the run as incomplete; a full billion-row global rebuild is not implemented by this adapter.',
                    'Methods run sequentially in declared order; within-method query repeats are not independent update trials.',
                    'Peak memory counters are process cumulative and not per-method peaks.'])
            atomic_json(out,result)
        with deadline.during('shared mapping prefault'):
            result['prefault']=prefault_mappings(base,assignments,order);atomic_json(out,result)
        for method in a.methods:
            with deadline.during(method+' bootstrap'):
                print('METHOD',method,'bootstrap',flush=True);t=time.perf_counter()
                idx=make_comparison_index(method,base,centroids,assignments,order,counts,sums,q_sums,a.seed,a.update_pairs+4096)
                from noraivf.native_scan import NativeUint8Scan
                idx.native_scan=NativeUint8Scan(a.scan_library)
                rec=dict(complete=False,bootstrap_s=time.perf_counter()-t,initial_memory=memory(),
                         update_events=[],workload_query_events=[],frontier={},audits={},
                         policy_parameters=policy_parameters(idx),
                         numeric_kernel=getattr(idx,'scale_numeric_kernel','repository reference numerical policy'))
                if hasattr(idx,'scale_numeric_kernel'):
                    result['implementation'].setdefault('maintenance_numeric_kernels',{})[method]=idx.scale_numeric_kernel
                for attr in ('rebalance_events','scale_events','repair_events'):
                    if hasattr(idx,attr):rec[attr]=getattr(idx,attr)
                result['methods'][method]=rec;atomic_json(out,result)
                if hasattr(idx,'_progress'):
                    def maintenance_progress(**fields):
                        rec['maintenance_progress']=dict(elapsed_s=time.perf_counter()-start,**fields)
                        deadline.check(method+' adapter progress '+str(fields.get('event','maintenance')))
                        print('MAINTENANCE',method,json.dumps(rec['maintenance_progress']),flush=True)
                    idx.scale_progress=maintenance_progress
            rec['frontier']['initial']=[]
            evaluate_checked(idx,queries,gt,a.nprobes,{},deadline,method+' initial evaluation',rec['frontier']['initial'])
            with deadline.during(method+' initial audit'):
                rec['audits']['initial']=audit(idx,sample);rec['initial_counters']=counters(idx);atomic_json(out,result)
            canonical={}
            for number,old_id in enumerate(update_ids,1):
                deadline.check(method+f' before update pair {number}')
                old_id=int(old_id);v=base[old_id].astype(np.float32);before=counters(idx)
                pending=dict(pair=number,original_id=old_id,phase='delete')
                rec['pending_update']=pending
                with deadline.during(method+f' delete in pair {number}'):
                    t=time.perf_counter();idx.delete(old_id);delete_s=time.perf_counter()-t
                    pending.update(delete_s=delete_s,phase='insert')
                with deadline.during(method+f' insert in pair {number}'):
                    t=time.perf_counter();new_id=idx.insert(v);insert_s=time.perf_counter()-t;canonical[new_id]=old_id
                    after=counters(idx)
                    rec['update_events'].append(dict(pair=number,original_id=old_id,new_id=new_id,delete_s=delete_s,insert_s=insert_s,
                                                    counters_after=after))
                    del rec['pending_update']
                if a.workload_query_every and number%a.workload_query_every==0:
                    with deadline.during(method+f' workload query after pair {number}'):
                        qid=int(operational_ids[number//a.workload_query_every-1]);q=allqueries[qid].astype(np.float32)
                        t=time.perf_counter();_,_,scanned=search(idx,q,k=100,nprobe=8)
                        if method=='AdaIVF':idx.observe_query(q,nprobe=8)
                        rec['workload_query_events'].append(dict(after_pair=number,query_id=qid,wall_s=time.perf_counter()-t,candidates=scanned))
                if number%10==0 or after['maintenance_s']-before['maintenance_s']>1:
                    rec['progress']=dict(pairs=number,elapsed_s=time.perf_counter()-start,memory=memory())
                    atomic_json(out,result)
                    print('UPDATE',method,number,'elapsed',round(time.perf_counter()-start,2),'counters',json.dumps(after),flush=True)
                deadline.check(method+f' after update pair {number}')
            with deadline.during(method+' update summary and final audit'):
                rec['update_summary']=dict(pairs=len(update_ids),delete_wall_s=sum(e['delete_s'] for e in rec['update_events']),
                    insert_wall_s=sum(e['insert_s'] for e in rec['update_events']),
                    maintenance_wall_s=float(idx.total_split_time+idx.total_merge_time),
                    workload_query_wall_s=sum(e['wall_s'] for e in rec['workload_query_events']),
                    workload_queries=len(rec['workload_query_events']),final_counters=counters(idx))
                rec['update_summary']['total_update_wall_s']=rec['update_summary']['delete_wall_s']+rec['update_summary']['insert_wall_s']
                rec['audits']['final']=audit(idx,np.r_[sample,list(canonical)])
                assert idx.n==n and len(idx.id_to_vec)==n
                rec['frontier']['final_repeats']=[];atomic_json(out,result)
            for repeat in range(a.query_repeats):
                with deadline.during(method+f' final evaluation repeat {repeat}'):
                    permutation=np.random.default_rng(a.seed+100+repeat).permutation(len(queries))
                    repeated=dict(repeat=repeat,complete=False,query_order=evaluation_ids[permutation].tolist(),measurements=[])
                    rec['frontier']['final_repeats'].append(repeated)
                    evaluate_checked(idx,queries[permutation],gt[permutation],a.nprobes,canonical,deadline,
                                     method+f' final evaluation repeat {repeat}',repeated['measurements'])
                    repeated['complete']=True;atomic_json(out,result)
            rec['final_measured_memory']=memory()
            with deadline.during(method+' final full ID integrity audit'):
                full_audit={}
                rec['audits']['final_full_id_integrity']=full_audit
                atomic_json(out,result)
                full_id_integrity_audit(idx,base,update_ids,canonical,deadline,
                    dict(method=method,run_identity_sha256=identity_sha256,
                         harness_path=str(Path(__file__).resolve()),
                         harness_sha256=source_hashes['experiments/run_sift1b_fourway.py']),full_audit)
                atomic_json(out,result)
            with deadline.during(method+' completion'):
                rec['n_live']=idx.n;rec['n_clusters']=idx.k;rec['final_memory']=memory()
                rec['final_partition_sizes']=[len(p) for p in idx.posting_lists]
                if hasattr(idx,'_detect_violators'):
                    rec['final_ada_violators']=idx._detect_violators().tolist()
                if hasattr(idx,'ada_stats'):rec['ada_stats']=idx.ada_stats
                if method=='NoraIVF':rec['scdkm_wall_s']=idx.total_scdkm_time
                rec['complete']=True;atomic_json(out,result);print('METHOD COMPLETE',method,flush=True)
            with deadline.during(method+' storage release'):
                del idx;gc.collect()
        with deadline.during('final completeness checks'):
            assert all(result['methods'][method]['complete'] for method in a.methods)
            result['complete']=True
    except Exception as exc:
        result['complete']=False
        result['failure']=dict(type=type(exc).__name__,message=str(exc),phase=deadline.phase,
                               signal=deadline.stop_signal,traceback=traceback.format_exc())
        atomic_json(out,result)
        raise
    finally:
        try:
            result['elapsed_s']=time.perf_counter()-start
            result['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
            result['final_memory']=memory();atomic_json(out,result)
        finally:
            for sig,handler in previous_handlers.items():signal.signal(sig,handler)
    print('COMPLETE',out,flush=True)

if __name__=='__main__':main()
