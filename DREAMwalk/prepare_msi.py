"""
convert the MSI data into DREAMwalk input files:
  input_network.txt, which has heterogeneous graph: source target edge_type weight edge_id
  hierarchy_file.csv, ATC hierarchy for drugs (child,parent), rooted at 'drug'
  nodetypes.tsv, node -> drug / disease / gene / etc
  dataset.csv, drug-disease pairs with labels (positives + sampled negatives)
  
tab-separated despite the extension, but that's what predict_associations expects

drug-indication edges from (6_drug_indication_df.tsv) are not added to the input_network.txt!!!
"""
import argparse
import os
import random
import pandas as pd

#edge_type ids must be contiguous from 1 (generate_embeddings indexes type-1)
EDGE_FILES = [
    (1, '1_drug_to_protein.tsv'),
    (2, '2_indication_to_protein.tsv'),
    (3, '3_protein_to_protein.tsv'),
    (4, '4_protein_to_biological_function.tsv'),
    (5, '5_biological_function_to_biological_function.tsv'),
    # (6, '6_drug_indication_df.tsv')
]

# MSI node type -> DREAMwalk node type (HeterogeneousSG knows drug/disease/gene/etc)
NODE_TYPES = {
    'drug': 'drug',
    'indication': 'disease',
    'protein': 'gene',
    'biological_function': 'etc',
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--msi_dir', type=str, default='data',
                        help='directory holding the extracted MSI tsv files')
    parser.add_argument('--output_dir', type=str, default='msi_inputs')
    parser.add_argument('--neg_ratio', type=float, default=1.0,
                        help='negatives sampled per positive drug-disease pair')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()

def build_network(msi_dir: str):
    edges = []
    node2type = {}
    for edge_type, fname in EDGE_FILES:
        df = pd.read_csv(os.path.join(msi_dir, fname), sep='\t', dtype=str)
        for n1, n2, t1, t2 in df[['node_1', 'node_2', 'node_1_type', 'node_2_type']].itertuples(index=False):
            node2type[n1] = NODE_TYPES[t1]
            node2type[n2] = NODE_TYPES[t2]
        # the graph is read as undirected, so drop self-loops and reversed duplicates
        pairs = {tuple(sorted(p)) for p in zip(df['node_1'], df['node_2']) if p[0] != p[1]}
        edges += [(n1, n2, edge_type) for n1, n2 in sorted(pairs)]
        print(f'  type {edge_type}: {len(pairs):>7} edges  ({fname})')
    return edges, node2type

def build_atc_hierarchy(msi_dir: str, drugs: set):
    df = pd.read_csv(os.path.join(msi_dir, '7_drug_classification_df.tsv'), sep='\t', dtype=str)
    df = df[df['db_id'].isin(drugs)]
    chain = ['db_id', 'atc_code', 'level_4', 'level_3', 'level_2', 'level_1']
    links = set()
    for row in df[chain].itertuples(index=False):
        path = [c for c in row if isinstance(c, str)] + ['drug']
        links.update(zip(path[:-1], path[1:]))
    # a drug with several ATC codes gets several parents, i.e. several paths to the root
    return pd.DataFrame(sorted(links), columns=['child', 'parent'])

def build_pairs(msi_dir: str, drugs: set, diseases: set, neg_ratio: float, seed: int):
    df = pd.read_csv(os.path.join(msi_dir, '6_drug_indication_df.tsv'), sep='\t', dtype=str)
    # pairs need embeddings for both ends, so keep only nodes present in the network
    df = df[df['drug'].isin(drugs) & df['indication'].isin(diseases)]
    positives = set(zip(df['drug'], df['indication']))

    rng = random.Random(seed)
    pair_drugs, pair_diseases = sorted(df['drug'].unique()), sorted(df['indication'].unique())
    n_neg = int(len(positives) * neg_ratio)
    if n_neg > len(pair_drugs) * len(pair_diseases) - len(positives):
        raise ValueError(f'neg_ratio {neg_ratio} asks for more negatives than there are unlabeled pairs')
    negatives = set()
    while len(negatives) < n_neg:
        pair = (rng.choice(pair_drugs), rng.choice(pair_diseases))
        if pair not in positives:
            negatives.add(pair)

    rows = [(d, i, 1) for d, i in sorted(positives)] + [(d, i, 0) for d, i in sorted(negatives)]
    return pd.DataFrame(rows, columns=['drug', 'disease', 'label'])

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print('Building network...')
    edges, node2type = build_network(args.msi_dir)
    with open(os.path.join(args.output_dir, 'input_network.txt'), 'w') as fw:
        for edge_id, (n1, n2, edge_type) in enumerate(edges):
            # column order must match utils.read_graph: type before weight
            fw.write(f'{n1}\t{n2}\t{edge_type}\t1.0\t{edge_id}\n')

    with open(os.path.join(args.output_dir, 'nodetypes.tsv'), 'w') as fw:
        fw.write('node\ttype\n')
        for node, ntype in sorted(node2type.items()):
            fw.write(f'{node}\t{ntype}\n')

    drugs = {n for n, t in node2type.items() if t == 'drug'}
    diseases = {n for n, t in node2type.items() if t == 'disease'}

    hier = build_atc_hierarchy(args.msi_dir, drugs)
    hier.to_csv(os.path.join(args.output_dir, 'hierarchy_file.csv'), index=False)

    pairs = build_pairs(args.msi_dir, drugs, diseases, args.neg_ratio, args.seed)
    pairs.to_csv(os.path.join(args.output_dir, 'dataset.csv'), sep='\t', index=False)

    n_atc = hier.loc[hier['child'].isin(drugs), 'child'].nunique()
    print(f'Nodes: {len(node2type)} ({len(drugs)} drugs, {len(diseases)} diseases)')
    print(f'Drugs with an ATC code: {n_atc}/{len(drugs)}')
    print(f'Pairs: {int(pairs["label"].sum())} positive, {int((pairs["label"] == 0).sum())} negative')
    print(f'Files written to {args.output_dir}/')

if __name__ == '__main__':
    main()
