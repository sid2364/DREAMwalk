"""
CV for DREAMwalk drug-disease prediction

Per seed, we use one embedding (the graph holds no drug-disease edges, so it carries no fold-dependent label information and can be shared across folds), freshly sampled negatives, and stratified K-fold over pairs with the published XGBoost settings
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from DREAMwalk.generate_embeddings import save_embedding_files
from DREAMwalk.predict_associations import make_classifier, return_scores
from DREAMwalk.prepare_msi import build_pairs
from DREAMwalk.utils import set_seed

METRICS = ['acc', 'auroc', 'aupr', 'f1']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network_file', type=str, required=True)
    parser.add_argument('--sim_network_file', type=str, default='',
                        help='leave out to disable the teleport (tp_factor=0)')
    parser.add_argument('--node_type_file', type=str, required=True)
    parser.add_argument('--msi_dir', type=str, default='data')
    parser.add_argument('--output_dir', type=str, default='msi_cv')

    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--neg_ratio', type=float, default=1.0)

    # embedding settings, same defaults as generate_embeddings.py
    parser.add_argument('--tp_factor', type=float, default=0.5)
    parser.add_argument('--num_walks', type=int, default=100)
    parser.add_argument('--walk_length', type=int, default=10)
    parser.add_argument('--dimension', type=int, default=128)
    parser.add_argument('--window_size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=os.cpu_count())
    return parser.parse_args()


def read_edges(netf: str):
    edges = set()
    with open(netf) as fr:
        for line in fr:
            n1, n2 = line.split('\t')[:2]
            edges.add(frozenset((n1, n2)))
    return edges

# check that the labelled pairs are not edges in the input graph, which would be a leakage (type 1, explicit)
def check_pairs_not_in_graph(pairs: pd.DataFrame, netfs: list):
    pair_set = {frozenset(p) for p in zip(pairs['drug'], pairs['disease'])}
    for netf in netfs:
        leaked = pair_set & read_edges(netf)
        if leaked:
            raise RuntimeError(f'{len(leaked)} labelled pairs are edges in {netf}, e.g. {sorted(next(iter(leaked)))}')

# create a feature vector for each drug-disease pair by subtracting the embeddings of the two nodes
# concatenation is done coz that's how my Master's thesis did it
# TODO also check substraction
def featurize(pairs: pd.DataFrame, embeddings: dict):
    return np.array([np.concatenate((embeddings[d], embeddings[i])) for d, i in zip(pairs['drug'], pairs['disease'])])

def run_seed(args, seed: int, drugs: set, diseases: set):
    # check before embedding, which takes hours
    pairs = build_pairs(args.msi_dir, drugs, diseases, args.neg_ratio, seed)
    check_pairs_not_in_graph(pairs, [f for f in (args.network_file, args.sim_network_file) if f])

    embf = os.path.join(args.output_dir, f'embeddings_seed{seed}.pkl')
    if os.path.exists(embf):
        print(f'Reusing {embf}')
    else:
        save_embedding_files(args.network_file, 
                             args.sim_network_file, embf,
                             nodetypef=args.node_type_file, 
                             tp_factor=args.tp_factor, 
                             seed=seed,
                             num_walks=args.num_walks,
                             walk_length=args.walk_length,
                             workers=args.workers, 
                             dimension=args.dimension,
                             window_size=args.window_size)
    with open(embf, 'rb') as fr:
        embeddings = pickle.load(fr)

    x, y = featurize(pairs, embeddings), pairs['label'].to_numpy()

    pairs['fold'] = -1
    rows = []

    # method 1 to assign folds: stratified K-fold
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seed)
    # TODO method 2 to assign folds: use disease area split
    for fold, (train_idx, test_idx) in enumerate(skf.split(x, y)):
        train_pairs = set(zip(pairs['drug'].iloc[train_idx], pairs['disease'].iloc[train_idx]))
        test_pairs = set(zip(pairs['drug'].iloc[test_idx], pairs['disease'].iloc[test_idx]))
        assert not train_pairs & test_pairs, 'train and test folds share pairs'
        pairs.iloc[test_idx, pairs.columns.get_loc('fold')] = fold

        set_seed(seed)
        clf = make_classifier(seed)
        clf.fit(x[train_idx], y[train_idx])
        scores = return_scores(y[test_idx], clf.predict_proba(x[test_idx])[:, 1])
        rows.append({'seed': seed, 'fold': fold, 'n_train': len(train_idx), 'n_test': len(test_idx),
                     **dict(zip(METRICS, scores))})
        print(f'seed {seed} fold {fold} | ' + ' | '.join(f'{m}: {s:.4f}' for m, s in zip(METRICS, scores)))

    pairs.to_csv(os.path.join(args.output_dir, f'fold_assignments_seed{seed}.tsv'), sep='\t', index=False)
    return rows


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    nodetypes = pd.read_csv(args.node_type_file, sep='\t', dtype=str)
    drugs = set(nodetypes.loc[nodetypes['type'] == 'drug', 'node'])
    diseases = set(nodetypes.loc[nodetypes['type'] == 'disease', 'node'])

    rows = []
    for seed in args.seeds:
        rows += run_seed(args, seed, drugs, diseases)
        # rewrite after every seed so a crash keeps the finished results
        metrics = pd.DataFrame(rows)
        metrics.to_csv(os.path.join(args.output_dir, 'cv_metrics.csv'), index=False)

    print('=' * 50)
    print(metrics.groupby('seed')[METRICS].mean().round(4).to_string())
    print('-' * 50)
    print(f'All {len(metrics)} folds: ' + ' | '.join(
        f'{m}: {metrics[m].mean():.4f} ± {metrics[m].std():.4f}' for m in METRICS))


if __name__ == '__main__':
    main()
