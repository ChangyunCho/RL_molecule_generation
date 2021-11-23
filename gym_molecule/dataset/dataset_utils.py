import pandas as pd
import networkx as nx
from rdkit import Chem
from rdkit.Chem import GraphDescriptors
import numpy as np

def mol_to_nx(mol):
    G = nx.Graph()

    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(),
                   symbol=atom.GetSymbol(),
                   formal_charge=atom.GetFormalCharge(),
                   implicit_valence=atom.GetImplicitValence(),
                   ring_atom=atom.IsInRing(),
                   degree=atom.GetDegree(),
                   hybridization=atom.GetHybridization())
        
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   bond_type=bond.GetBondType())
    return G

def nx_to_mol(G):
    mol = Chem.RWMol()
    atomic_nums = nx.get_node_attributes(G, 'atomic_num')
    chiral_tags = nx.get_node_attributes(G, 'chiral_tag')
    formal_charges = nx.get_node_attributes(G, 'formal_charge')
    node_is_aromatics = nx.get_node_attributes(G, 'is_aromatic')
    node_hybridizations = nx.get_node_attributes(G, 'hybridization')
    num_explicit_hss = nx.get_node_attributes(G, 'num_explicit_hs')
    node_to_idx = {}
    for node in G.nodes():
        a=Chem.Atom(atomic_nums[node])
        a.SetChiralTag(chiral_tags[node])
        a.SetFormalCharge(formal_charges[node])
        a.SetIsAromatic(node_is_aromatics[node])
        a.SetHybridization(node_hybridizations[node])
        a.SetNumExplicitHs(num_explicit_hss[node])
        idx = mol.AddAtom(a)
        node_to_idx[node] = idx

    bond_types = nx.get_edge_attributes(G, 'bond_type')
    for edge in G.edges():
        first, second = edge
        ifirst = node_to_idx[first]
        isecond = node_to_idx[second]
        bond_type = bond_types[first, second]
        mol.AddBond(ifirst, isecond, bond_type)

    Chem.SanitizeMol(mol)
    return mol

def load_dataset(path):
    """
    Loads gdb13 dataset from path to pandas dataframe
    :param path:
    :return:
    """
    df = pd.read_csv(path, header=None, names=['smiles'])
    return df

def sort_dataset(in_path, out_path):
    """
    Sorts the dataset of smiles from input path by molecular complexity as
    proxied by the BertzCT index, and outputs the new sorted dataset
    :param in_path:
    :param out_path:
    :return:
    """
    def _calc_bertz_ct(smiles):
        return GraphDescriptors.BertzCT(Chem.MolFromSmiles(smiles))

    in_df = load_dataset(in_path)
    in_df['BertzCT'] = in_df.smiles.apply(_calc_bertz_ct)
    sorted_df = in_df.sort_values(by=['BertzCT'])
    sorted_df['smiles'].to_csv(out_path, index=False)



