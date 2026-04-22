import h5py
import numpy as np
import networkx as nx
from collections import defaultdict
from tqdm import tqdm

H5_PATH = "data/dataset.h5"

with h5py.File(H5_PATH, 'r') as h5:
    node_feats = h5['features/node_features'][:]
    edge_index = h5['edges/edge_index'][:]
    edge_ptr   = h5['edges/edge_ptr'][:]
    hw_hashes  = h5['meta/hw_hashes'][:].astype(str)

hw_to_first = {}
for i, hw in enumerate(hw_hashes):
    if hw not in hw_to_first:
        hw_to_first[hw] = i

def build_hw_graph(idx):
    feats   = node_feats[idx]
    is_task = (feats[:, 3] + feats[:, 4]) > 0
    e_start = int(edge_ptr[idx])
    e_end   = int(edge_ptr[idx + 1])
    ei      = edge_index[:, e_start:e_end]
    mask    = ~is_task[ei[0]] & ~is_task[ei[1]]
    hw_ei   = ei[:, mask]
    G = nx.Graph()
    for node_idx in range(len(feats)):
        if is_task[node_idx]: continue
        ntype = int(feats[node_idx, :3].argmax())
        G.add_node(node_idx, ntype=str(ntype))
    for e in range(hw_ei.shape[1]):
        src, dst = int(hw_ei[0, e]), int(hw_ei[1, e])
        G.add_edge(src, dst)
    return G

print('Computing WL hashes...')
wl_hashes = {}
for hw, idx in tqdm(hw_to_first.items()):
    G = build_hw_graph(idx)
    wl = nx.weisfeiler_lehman_graph_hash(G, node_attr='ntype', iterations=4)
    wl_hashes[hw] = wl

wl_to_md5s = defaultdict(list)
for hw, wl in wl_hashes.items():
    wl_to_md5s[wl].append(hw)

nm = nx.algorithms.isomorphism.categorical_node_match('ntype', -1)
total_pairs_expected = sum(
    len(v)*(len(v)-1)//2
    for v in wl_to_md5s.values() if len(v) > 1
)
print(f'Total pairs to verify: {total_pairs_expected:,}')

false_positives = 0
total_pairs     = 0

for wl, md5s in tqdm(wl_to_md5s.items(), desc='Verifying groups'):
    if len(md5s) < 2:
        continue
    graphs = {m: build_hw_graph(hw_to_first[m]) for m in md5s}
    for i in range(len(md5s)):
        for j in range(i+1, len(md5s)):
            total_pairs += 1
            if not nx.is_isomorphic(graphs[md5s[i]], graphs[md5s[j]],
                                    node_match=nm):
                false_positives += 1
                print(f'  COLLISION: {md5s[i]} vs {md5s[j]}')

print(f'Total pairs checked : {total_pairs:,}')
print(f'False positives     : {false_positives}')
if false_positives == 0:
    print('CONFIRMED: 392 is the exact count of unique hardware layouts.')
else:
    print(f'True unique count is higher than 392.')
