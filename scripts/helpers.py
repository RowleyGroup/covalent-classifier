import requests
from rdkit import Chem
from molgraph import chemistry
from molgraph.chemistry import features, Featurizer, MolecularGraphEncoder

aimnet_properties = [
    'f_el',
    'f_nuc',
    'f_rad',
    # 'omega_el',
    # 'omega_nuc',
    # 'omega_rad',
    ]

def set_aimnet_property(atom,
                        prop_name: str,
                        source_df):

    owner_mol = atom.GetOwningMol()
    idx = atom.GetIdx()
    key = Chem.MolToInchiKey(owner_mol)
    val = source_df[source_df.InChI_Key == key].aimnet_data.values[0][prop_name][idx]
    atom.SetProp(prop_name, f"{val}")

def assign_aimnet_properties(smiles_string, source_df):
    mol = Chem.MolFromSmiles(smiles_string)
    for atom in mol.GetAtoms():
        for prop in aimnet_properties:
            try:
                set_aimnet_property(atom=atom, prop_name=prop, source_df=source_df)
            except:
                print(f"property {prop} could not be assing to {smiles_string}. {smiles_string} needs to be removed")
                return None
    return mol

def get_smiles_from_pubchem(pubchem_id):
    return requests.get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{pubchem_id}/property/CanonicalSmiles/TXT").text.strip('\n')

class FukuiEl(chemistry.Feature):
    def __call__(self, atom):
        if atom.HasProp("f_el"):
            return atom.GetProp("myprop")
        return None


class FukuiNuc(chemistry.Feature):
    def __call__(self, atom):
        if atom.HasProp("f_nuc"):
            return atom.GetProp("f_nuc")
        return None


class FukuiRad(chemistry.Feature):
    def __call__(self, atom):
        if atom.HasProp("f_rad"):
            return atom.GetProp("f_rad")
        return None

atom_encoder = Featurizer([

    features.Symbol(),
    features.Hybridization(),
    features.FormalCharge(),
    features.TotalNumHs(),
    features.TotalValence(),
    features.NumRadicalElectrons(),
    features.Degree(),
    features.ChiralCenter(),
    features.Aromatic(),
    features.Ring(),
    features.Hetero(),
    features.HydrogenDonor(),
    features.HydrogenAcceptor(),
    features.CIPCode(),
    features.ChiralCenter(),
    features.RingSize(),
    features.Ring(),
    features.CrippenLogPContribution(),
    features.CrippenMolarRefractivityContribution(),
    features.TPSAContribution(),
    features.LabuteASAContribution(),
    features.GasteigerCharge(),
    FukuiEl(),
    FukuiNuc(),
    FukuiRad()
])

bond_encoder = Featurizer([

    features.BondType(),
    features.Conjugated(),
    features.Rotatable(),
    features.Ring(),
    features.Stereo(),
])

encoder = MolecularGraphEncoder(atom_encoder, bond_encoder)