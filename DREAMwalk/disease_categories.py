"""
map msi diseases to top-level MeSH categories (C04, C14, F03, blah, blah) for the disease area split
1. CUI -> MeSH descriptor via MONDO's equivalence xrefs (UMLS:C... and MESH:D... in the same term)
2. fallback: exact match of the MSI indication name against MeSH entry terms
3. otherwise unmapped (problem?)

notes:
the category of a tree number is its first 3 chars (C04.557.470 -> C04),
the primary category is the most frequent one among a disease's tree numbers (ties -> alphabetical)
inputs in --msi_dir: desc2026.xml (NLM MeSH descriptors) and mondo.obo (see README for download!!)
output: disease_categories.tsv in --output_dir
"""
import argparse
import os
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--msi_dir', type=str, default='data')
    parser.add_argument('--output_dir', type=str, default='msi_inputs')
    parser.add_argument('--mesh_file', type=str, default='desc2026.xml')
    parser.add_argument('--mondo_file', type=str, default='mondo.obo')
    return parser.parse_args()

def normalize(name: str):
    return ' '.join(name.lower().replace(',', ' ').split())

def read_mesh(meshf: str):
    trees, term2desc = {}, defaultdict(set)
    for _, elem in ET.iterparse(meshf, events=('end',)):
        if elem.tag != 'DescriptorRecord':
            continue
        ui = elem.findtext('DescriptorUI')
        trees[ui] = [t.text for t in elem.iter('TreeNumber')]
        for term in elem.iter('Term'):
            term2desc[normalize(term.findtext('String'))].add(ui)
        elem.clear()
    return trees, term2desc

def read_mondo(mondof: str):
    cui2mesh = defaultdict(set)
    umls, mesh, obsolete = set(), set(), False

    def flush():
        if not obsolete:
            for cui in umls:
                cui2mesh[cui] |= mesh

    with open(mondof) as fr:
        for line in fr:
            line = line.strip()
            if line.startswith('['):
                flush()
                umls, mesh, obsolete = set(), set(), False
            elif line.startswith('xref: UMLS:'):
                umls.add(line.split()[1][len('UMLS:'):])
            # MESH:C... are supplementary concepts, they have no tree numbers
            elif line.startswith('xref: MESH:D'):
                mesh.add(line.split()[1][len('MESH:'):])
            elif line == 'is_obsolete: true':
                obsolete = True
    flush()
    return cui2mesh

def disease_names(msi_dir: str):
    names = {}
    ind = pd.read_csv(os.path.join(msi_dir, '6_drug_indication_df.tsv'), sep='\t', dtype=str)
    names.update(zip(ind['indication'], ind['indication_name']))
    prot = pd.read_csv(os.path.join(msi_dir, '2_indication_to_protein.tsv'), sep='\t', dtype=str)
    for cui, name in zip(prot['node_1'], prot['node_1_name']):
        names.setdefault(cui, name)
    return names

def primary_category(categories: list):
    counts = Counter(categories)
    return min(counts, key=lambda c: (-counts[c], c))

def map_diseases(diseases, names, trees, term2desc, cui2mesh):
    rows = []
    for cui in sorted(diseases):
        name = names.get(cui, '')
        mesh_ids = sorted(m for m in cui2mesh.get(cui, ()) if trees.get(m))
        source = 'mondo'
        if not mesh_ids:
            mesh_ids = sorted(m for m in term2desc.get(normalize(name), ()) if trees.get(m))
            source = 'name'
        if not mesh_ids:
            rows.append((cui, name, '', '', '', 'none'))
            continue
        categories = [t[:3] for m in mesh_ids for t in trees[m]]
        rows.append((cui, name, '|'.join(mesh_ids), '|'.join(sorted(set(categories))),
                     primary_category(categories), source))
    return pd.DataFrame(rows, columns=['disease', 'name', 'mesh_ids', 'categories', 'primary_category', 'source'])

def main():
    args = parse_args()
    nodetypes = pd.read_csv(os.path.join(args.output_dir, 'nodetypes.tsv'), sep='\t', dtype=str)
    diseases = set(nodetypes.loc[nodetypes['type'] == 'disease', 'node'])

    print("reading MeSH...")
    trees, term2desc = read_mesh(os.path.join(args.msi_dir, args.mesh_file))
    print("reading MONDO...")
    cui2mesh = read_mondo(os.path.join(args.msi_dir, args.mondo_file))

    df = map_diseases(diseases, disease_names(args.msi_dir), trees, term2desc, cui2mesh)
    outf = os.path.join(args.output_dir, 'disease_categories.tsv')
    df.to_csv(outf, sep='\t', index=False)

    ind = pd.read_csv(os.path.join(args.msi_dir, '6_drug_indication_df.tsv'), sep='\t', dtype=str)
    ind = ind[ind['indication'].isin(diseases)].drop_duplicates(['drug', 'indication'])
    ind = ind.merge(df[['disease', 'primary_category']], left_on='indication', right_on='disease')
    mapped = ind['primary_category'] != ''

    print(f'Diseases: {len(df)}, by source: {df["source"].value_counts().to_dict()}')
    print(f'Positive pairs with a mapped disease: {mapped.sum()}/{len(ind)} ({mapped.mean():.1%})')
    n_multi = df['categories'].str.contains('|', regex=False).sum()
    print(f'multi category diseases: {n_multi}')
    print('Positive pairs per primary category:')
    # get indication locs and count the primary categories of the mapped ones
    print(ind.loc[mapped, 'primary_category'].value_counts().to_string())
    print(f'Saved {outf}')

if __name__ == '__main__':
    main()
