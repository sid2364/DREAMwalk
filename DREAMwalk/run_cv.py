"""
CV for DREAMwalk drug-disease prediction

Per seed, we use one embedding (the graph holds no drug-disease edges, so it carries no fold-dependent label information and can be shared across folds and split modes), freshly sampled negatives, and the published XGBoost settings

--split picks how pairs are divided:
  random:        stratified K-fold over pairs (test diseases also appear in training)
  disease:       stratified group K-fold with the disease as group (test diseases are unseen)
  disease_area:  the paper's disease split: diseases grouped by primary MeSH category (see disease_categories.py),
                 whole categories drawn into test / valid / train at ~1:1:8 of the pairs, repeated --repeats times
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from DREAMwalk.generate_embeddings import save_embedding_files
from DREAMwalk.predict_associations import make_classifier, return_scores
from DREAMwalk.prepare_msi import build_pairs
from DREAMwalk.utils import set_seed

METRICS = ['acc', 'auroc', 'aupr', 'f1']
SPLITS = ['random', 'disease', 'disease_area']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network_file', type=str, required=True)
    parser.add_argument('--sim_network_file', type=str, default='',
                        help='leave out to disable the teleport (tp_factor=0)')
    parser.add_argument('--node_type_file', type=str, required=True)
    parser.add_argument('--msi_dir', type=str, default='data')
    parser.add_argument('--output_dir', type=str, default='msi_cv')

    parser.add_argument('--split', type=str, default='random', choices=SPLITS)
    parser.add_argument('--category_file', type=str, default='msi_inputs/disease_categories.tsv',
                        help='output of disease_categories.py, only used by --split disease_area')
    parser.add_argument('--folds', type=int, default=5, help='folds for --split random / disease')
    parser.add_argument('--repeats', type=int, default=10, help='category splits per seed for --split disease_area')
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

# attach each pair's main MeSH category and drop pairs whose disease could not be mapped
# TODO check positive/negative balance before and after to see if it biases the dataset
def add_categories(pairs: pd.DataFrame, categoryf: str):
    cats = pd.read_csv(categoryf, sep='\t', dtype=str, keep_default_na=False)
    pairs = pairs.merge(cats[['disease', 'primary_category']]
                        .rename(columns={'primary_category': 'category'}), on='disease', how='left')
    # now drop pairs that could not be mapped 
    unmapped = pairs['category'].fillna('') == ''
    print(f'droppping {unmapped.sum()}/{len(pairs)} pairs with no MeSH category ( {int(pairs.loc[unmapped, "label"].sum())} positives)')
    print(f'Before: {len(pairs)} pairs')
    pairs = pairs[~unmapped].reset_index(drop=True)
    print(f'After: {len(pairs)} pairs')
    return pairs[~unmapped].reset_index(drop=True)

# create a feature vector for each drug-disease pair by subtracting the embeddings of the two nodes
# concatenation is done coz that's how my Master's thesis did it
# TODO also check substraction
def featurize(pairs: pd.DataFrame, embeddings: dict):
    return np.array([np.concatenate((embeddings[d], embeddings[i])) for d, i in zip(pairs['drug'], pairs['disease'])])

# DREAMwalk's disease split, according to my understanding: shuffle the categories, fill test then valid with whole categories until each holds >= its share of the pairs, the rest is train (roughly!!) 80:10:10
def category_splits(categories: np.ndarray, repeats: int, seed: int, test_frac=0.1, valid_frac=0.1):
    sizes = pd.Series(categories).value_counts()
    for r in range(repeats):
        # get random order, then fill test and validation
        order = np.random.default_rng([seed, r]).permutation(sorted(sizes.index))
        test, valid, n = [], [], 0
        for c in order:
            if n < test_frac * len(categories):
                test.append(c)
            elif n < (test_frac + valid_frac) * len(categories):
                valid.append(c)
            else:
                break
            n += sizes[c]
        test_idx = np.flatnonzero(np.isin(categories, test))
        valid_idx = np.flatnonzero(np.isin(categories, valid))
        train_idx = np.flatnonzero(~np.isin(categories, test + valid))
        # return r, train_idx, valid_idx, test_idx
        yield r, train_idx, valid_idx, test_idx

# make splits for cv
def make_splits(pairs: pd.DataFrame, y: np.ndarray, args, seed: int):
    no_valid = np.array([], dtype=int)
    if args.split == 'random':
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seed)
        for fold, (train_idx, test_idx) in enumerate(skf.split(pairs, y)):
            yield fold, train_idx, no_valid, test_idx
    elif args.split == 'disease':
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html 
        # same, but with non-overlapping groups (diseases) in train and test
        # this is the harder case, but with least leakage and tests generalisation best
        sgkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=seed)
        for fold, (train_idx, test_idx) in enumerate(sgkf.split(pairs, y, groups=pairs['disease'])):
            yield fold, train_idx, no_valid, test_idx
    else:
        yield from category_splits(pairs['category'].to_numpy(), args.repeats, seed)

def check_split(pairs: pd.DataFrame, split: str, idxs: list):
    pair_sets = [set(zip(pairs['drug'].iloc[i], pairs['disease'].iloc[i])) for i in idxs]
    cols = {'random': [], 'disease': ['disease'], 'disease_area': ['disease', 'category']}[split]
    for col in ['pair'] + cols:
        sets = pair_sets if col == 'pair' else [set(pairs[col].iloc[i]) for i in idxs]
        for a in range(len(sets)):
            for b in range(a + 1, len(sets)):
                assert not sets[a] & sets[b], f'train / valid / test share {col}s'

def run_seed(args, seed: int, drugs: set, diseases: set):
    # check before embedding, which takes hours
    pairs = build_pairs(args.msi_dir, drugs, diseases, args.neg_ratio, seed)
    check_pairs_not_in_graph(pairs, [f for f in (args.network_file, args.sim_network_file) if f])
    if args.split == 'disease_area':
        pairs = add_categories(pairs, args.category_file)

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

    if args.split != 'disease_area':
        pairs['fold'] = -1
    rows = []

    for fold, train_idx, valid_idx, test_idx in make_splits(pairs, y, args, seed):
        check_split(pairs, args.split, [i for i in (train_idx, valid_idx, test_idx) if len(i)])
        if args.split == 'disease_area':
            roles = np.full(len(pairs), 'train', dtype=object)
            roles[valid_idx], roles[test_idx] = 'valid', 'test'
            pairs[f'repeat{fold}'] = roles
        else:
            pairs.iloc[test_idx, pairs.columns.get_loc('fold')] = fold

        set_seed(seed)
        clf = make_classifier(seed)
        clf.fit(x[train_idx], y[train_idx])
        scores = return_scores(y[test_idx], clf.predict_proba(x[test_idx])[:, 1])
        row = {'seed': seed, 'split': args.split, 'fold': fold,
               'n_train': len(train_idx), 'n_valid': len(valid_idx), 'n_test': len(test_idx),
               'test_pos_rate': y[test_idx].mean(), **dict(zip(METRICS, scores))}
        if len(valid_idx):
            val_scores = return_scores(y[valid_idx], clf.predict_proba(x[valid_idx])[:, 1])
            row.update({f'val_{m}': s for m, s in zip(METRICS, val_scores)})
        if args.split == 'disease_area':
            row['test_categories'] = '|'.join(sorted(set(pairs['category'].iloc[test_idx])))
        rows.append(row)
        print(f'seed {seed} {args.split} {fold} | n train/valid/test: {len(train_idx)}/{len(valid_idx)}/{len(test_idx)} | '
              + ' | '.join(f'{m}: {s:.4f}' for m, s in zip(METRICS, scores))
              + (f' | test categories: {row["test_categories"]}' if 'test_categories' in row else ''))

    pairs.to_csv(os.path.join(args.output_dir, f'fold_assignments_{args.split}_seed{seed}.tsv'), sep='\t', index=False)
    return rows


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    nodetypes = pd.read_csv(args.node_type_file, sep='\t', dtype=str)
    drugs = set(nodetypes.loc[nodetypes['type'] == 'drug', 'node'])
    diseases = set(nodetypes.loc[nodetypes['type'] == 'disease', 'node'])

    rows = []
    for seed in args.seeds:
        seed_rows = run_seed(args, seed, drugs, diseases)
        pd.DataFrame(seed_rows).to_csv(os.path.join(args.output_dir, f'cv_metrics_{args.split}_seed{seed}.csv'), index=False)
        rows += seed_rows
        # rewrite after every seed so a crash keeps the finished results
        metrics = pd.DataFrame(rows)
        metrics.to_csv(os.path.join(args.output_dir, f'cv_metrics_{args.split}.csv'), index=False)

    print('=' * 50)
    print(metrics.groupby('seed')[METRICS].mean().round(4).to_string())
    print('-' * 50)
    print(f'All {len(metrics)} {args.split} splits: ' + ' | '.join(
        f'{m}: {metrics[m].mean():.4f} ± {metrics[m].std():.4f}' for m in METRICS))


if __name__ == '__main__':
    main()
