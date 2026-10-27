"""Label-free local PCA, principal-angle restrictions and stable partial edges."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse

CORE3 = ('K562_essential', 'HepG2', 'Orion_HCT116')


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def array_sha(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode()); digest.update(str(value.shape).encode()); digest.update(value.tobytes())
    return digest.hexdigest()


def partial_restrictions(bases, stability, src, dst, scores, min_overlap=0.25):
    overlap = np.einsum('eai,eaj->eij', bases[src], bases[dst])
    u, singular, vt = np.linalg.svd(overlap, full_matrices=False)
    left, right = u.transpose(0, 2, 1), vt
    a = np.einsum('ekd,ed->ek', left ** 2, stability[src])
    b = np.einsum('ekd,ed->ek', right ** 2, stability[dst])
    weight = scores[:, None] * singular ** 2 * np.sqrt(a * b)
    weight *= singular >= min_overlap
    degree = np.zeros(len(bases), np.float64)
    np.add.at(degree, src, scores)
    np.add.at(degree, dst, scores)
    weight /= np.sqrt(np.maximum(degree[src], 1) * np.maximum(degree[dst], 1))[:, None]
    return left.astype('f4'), right.astype('f4'), weight.astype('f4'), singular.astype('f4')


def make_local_frames(features, graph, rank=4, max_neighbors=32, repeats=4, seed=17):
    rng = np.random.default_rng(seed)
    n, ambient = features.shape
    frames = np.zeros((n, ambient, rank), np.float64)
    stability = np.zeros((n, rank), np.float64)
    local_rank = np.zeros(n, np.int64)
    for g in range(n):
        lo, hi = graph.indptr[g:g + 2]
        ids, scores = graph.indices[lo:hi], graph.data[lo:hi]
        take = np.argsort(-scores, kind='stable')[:max_neighbors]
        ids, scores = ids[take], scores[take]
        if not len(ids):
            frames[g] = np.eye(ambient, rank)
            continue
        cloud = (features[ids] - features[g]).T * np.sqrt(scores / max(scores.sum(), 1e-12))[None]
        u, s, _ = np.linalg.svd(cloud, full_matrices=False)
        effective = min(rank, int(np.sum(s > max(s[0] * 1e-4, 1e-8))))
        local_rank[g] = effective
        # Complete deficient bases deterministically; artificial directions are
        # NEVER assigned graph precision but remain available to data fitting.
        seed_frame = np.column_stack((u[:, :effective], np.eye(ambient)))
        q, _ = np.linalg.qr(seed_frame)
        frames[g] = q[:, :rank]
        if effective == 0:
            continue
        for _ in range(repeats):
            ix = rng.choice(len(ids), len(ids), replace=True)
            ub, sb, _ = np.linalg.svd(cloud[:, ix], full_matrices=False)
            eb = min(rank, int(np.sum(sb > max(sb[0] * 1e-4, 1e-8))))
            coverage = np.sum((frames[g].T @ ub[:, :eb]) ** 2, axis=1)
            stability[g, :effective] += coverage[:effective] / repeats
    # Use basis-invariant subspace stability. Per-axis bootstrap diagonals
    # would make the constraints depend on arbitrary rotations of PCA bases.
    stable_nodes = local_rank == rank
    stability = np.repeat(stability.mean(1)[:, None], rank, axis=1)
    stability[~stable_nodes] = 0
    return frames.astype('f4'), np.clip(stability, 0, 1).astype('f4'), local_rank


def build_geometry(assets, directory, rank=4, ambient=32, *, graph_directory=None):
    # Explicit public asset root replaces the original machine-local constant.
    graph_directory = graph_directory or getattr(assets, 'graph_directory', None)
    if graph_directory is None:
        raise ValueError('graph_directory must contain graphs/KS/A.npz and source/node_axis.tsv')
    GRAPH = Path(graph_directory)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'connection_geometry.npz'
    axis = list(assets.gene_names)
    binding_path = directory / 'binding.json'
    training_environments = tuple(getattr(assets, "training_environments", CORE3))
    prior = np.asarray(assets.gene_prior, np.float64)
    controls = np.stack([assets.features.controls[e] for e in training_environments]).astype(np.float64)
    response_scale = np.asarray(assets.features.geometry['scale'], np.float32)
    if len(set(axis)) != len(axis) or prior.ndim != 2 or prior.shape[0] != len(axis) or controls.shape != (len(training_environments),len(axis),7):
        raise ValueError('Invalid connection input gene axes')
    if response_scale.shape != (len(axis),) or not np.isfinite(response_scale).all() or (response_scale <= 0).any():
        raise ValueError('Invalid connection statistic scales')
    if not np.isfinite(prior).all() or not np.isfinite(controls).all():
        raise ValueError('Nonfinite connection input features')
    if not 1 <= rank <= ambient <= min(len(axis),prior.shape[1]+7*len(training_environments)):
        raise ValueError('Infeasible connection geometry rank or ambient dimension')
    definition = dict(schema='connection_geometry_v4_complete_binding_v2', genes=axis,
        rank=rank, ambient=ambient, max_neighbors=32, resampling_repeats=4, seed=17,
        minimum_principal_cosine=.25, control_environments=list(training_environments),
        prior_sha256=array_sha(prior), controls_sha256=array_sha(controls),
        statistic_scale_input_sha256=array_sha(response_scale),
        graph_sha256=file_sha(GRAPH/'graphs/KS/A.npz'),
        graph_node_axis_sha256=file_sha(GRAPH/'source/node_axis.tsv'))
    if path.exists():
        record = json.loads(binding_path.read_text())
        if record.get('fitted_in_new_project') is not True:
            raise ValueError('Historical fitted connection geometry cannot be reused')
        if record.get('definition') != definition:
            raise ValueError('Connection geometry input definition changed')
        if record['genes'] != axis or record['sha256'] != file_sha(path):
            raise ValueError('Connection geometry binding mismatch')
        with np.load(path) as f:
            result = {k: np.array(f[k], copy=True) for k in f.files}
        if result['bases'].shape[2] != rank or result['bases'].shape[1] != ambient:
            raise ValueError('Geometry configuration mismatch')
        return result

    # The inherited prior is a documented mixture, NOT pure GenePT.
    ctrl_center = controls.mean(axis=(0, 1))
    ctrl_scale = np.maximum(controls.std(axis=(0, 1)), 0.05)
    normalized_controls = (controls - ctrl_center) / ctrl_scale
    raw = np.concatenate((prior, normalized_controls.transpose(1, 0, 2).reshape(len(axis), -1)), 1)
    center = raw.mean(0)
    scale = np.maximum(raw.std(0), 0.05)
    normalized = (raw - center) / scale
    _, _, vt = np.linalg.svd(normalized, full_matrices=False)
    projection = vt[:ambient].T
    # This is a feature-space compression, not a compression of all gene outputs.
    features = normalized @ projection
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)

    with (GRAPH / 'source/node_axis.tsv').open() as f:
        nodes = {r['gene_id']: int(r['node_index']) for r in csv.DictReader(f, delimiter='\t')}
    full = sparse.load_npz(GRAPH / 'graphs/KS/A.npz').tocsr()
    locations = np.asarray([nodes.get(g, -1) for g in axis])
    valid = np.flatnonzero(locations >= 0)
    sub = full[locations[valid]][:, locations[valid]].tocoo()
    graph = sparse.csr_matrix((sub.data, (valid[sub.row], valid[sub.col])), shape=(len(axis), len(axis)))
    graph = graph.maximum(graph.T)
    graph.setdiag(0)
    graph.eliminate_zeros()
    if graph.nnz:
        graph.data /= max(float(graph.data.max()), 1e-8)
    bases, stability, local_rank = make_local_frames(features, graph, rank=rank)
    upper = sparse.triu(graph, 1).tocoo()
    left, right, weight, singular = partial_restrictions(
        bases, stability, upper.row, upper.col, upper.data)
    keep = weight.sum(1) > 1e-8
    stat_scale = np.tile(response_scale[:, None], (1, 7))
    stat_scale[:, 2] = 0.25  # detection-rate unit; fixed before any label read
    result = dict(bases=bases, edge_src=upper.row[keep], edge_dst=upper.col[keep],
                  left=left[keep], right=right[keep], weight=weight[keep],
                  principal_cosines=singular[keep], stability=stability, local_rank=local_rank,
                  prior=prior.astype('f4'), ctrl_center=ctrl_center.astype('f4'),
                  ctrl_scale=ctrl_scale.astype('f4'), stat_scale=stat_scale,
                  feature_center=center.astype('f4'), feature_scale=scale.astype('f4'),
                  feature_projection=projection.astype('f4'))
    np.savez(path, **result)
    record = dict(genes=axis, sha256=file_sha(path), rank=rank, ambient=ambient, definition=definition,
                  graph_sha256=file_sha(GRAPH / 'graphs/KS/A.npz'),
                  graph_node_axis_sha256=file_sha(GRAPH / 'source/node_axis.tsv'),
                  original_edges=int(len(keep)), retained_edges=int(keep.sum()),
                  deficient_nodes=int(np.sum(local_rank < rank)),
                  isolated_nodes=int(np.sum(local_rank == 0)),
                  prior='external Tahoe/GenePT/STRING static random projection 256D plus five availability masks; no task-trained weights',
                  fitted_in_new_project=True,
                  pca_control_fit_environments=list(training_environments), target_controls_fit_geometry=False,
                  response_values_fit_geometry=False, invented_graph_edges=0,
                  missing_directions='canonical completion; zero relation weight',
                  stability='four neighborhood-resampling subspace overlaps; not calibrated probability')
    binding_path.write_text(json.dumps(record, indent=2) + '\n')
    return result


def geometry_from_arrays(gene_prior, controls, response_scale, graph, *, rank=4,
                         ambient=32, min_overlap=.25):
    """Original J0 local geometry from explicit, aligned arrays.

    ``controls`` is [training environments, genes, 7]. It must contain only
    training-control statistics. ``graph`` is aligned to the output gene axis.
    No perturbation labels or held-environment controls fit this geometry.
    This is the array equivalent of :func:`build_geometry`, without file I/O.
    """
    prior = np.asarray(gene_prior, np.float64)
    controls = np.asarray(controls, np.float64)
    response_scale = np.asarray(response_scale, np.float32)
    if prior.ndim != 2 or controls.ndim != 3 or controls.shape[1:] != (len(prior), 7):
        raise ValueError('Prior [G,P] and training controls [C,G,7] required')
    if response_scale.shape != (len(prior),) or not np.isfinite(response_scale).all() or (response_scale <= 0).any():
        raise ValueError('Finite positive expression scales required')
    if not np.isfinite(prior).all() or not np.isfinite(controls).all():
        raise ValueError('Finite geometry inputs required')
    if not 1 <= rank <= ambient <= min(len(prior), prior.shape[1] + 7 * len(controls)):
        raise ValueError('Infeasible connection geometry rank or ambient dimension')
    graph = sparse.csr_matrix(graph, dtype=np.float64)
    if graph.shape != (len(prior), len(prior)) or not np.isfinite(graph.data).all() or (graph.data < 0).any():
        raise ValueError('Nonnegative finite graph aligned to the gene axis required')
    ctrl_center = controls.mean(axis=(0, 1))
    ctrl_scale = np.maximum(controls.std(axis=(0, 1)), .05)
    normalized_controls = (controls - ctrl_center) / ctrl_scale
    raw = np.concatenate((prior, normalized_controls.transpose(1, 0, 2).reshape(len(prior), -1)), 1)
    center = raw.mean(0)
    scale = np.maximum(raw.std(0), .05)
    normalized = (raw - center) / scale
    _, _, vt = np.linalg.svd(normalized, full_matrices=False)
    projection = vt[:ambient].T
    features = normalized @ projection
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
    graph = graph.maximum(graph.T)
    graph.setdiag(0)
    graph.eliminate_zeros()
    if graph.nnz:
        graph.data /= max(float(graph.data.max()), 1e-8)
    bases, stability, local_rank = make_local_frames(features, graph, rank=rank)
    upper = sparse.triu(graph, 1).tocoo()
    left, right, weight, singular = partial_restrictions(
        bases, stability, upper.row, upper.col, upper.data, min_overlap=min_overlap)
    keep = weight.sum(1) > 1e-8
    stat_scale = np.tile(response_scale[:, None], (1, 7))
    stat_scale[:, 2] = .25
    return dict(bases=bases, edge_src=upper.row[keep], edge_dst=upper.col[keep],
        left=left[keep], right=right[keep], weight=weight[keep],
        principal_cosines=singular[keep], stability=stability, local_rank=local_rank,
        prior=prior.astype('f4'), ctrl_center=ctrl_center.astype('f4'),
        ctrl_scale=ctrl_scale.astype('f4'), stat_scale=stat_scale,
        feature_center=center.astype('f4'), feature_scale=scale.astype('f4'),
        feature_projection=projection.astype('f4'))
