from molgraph.chemistry import features, Featurizer, MolecularGraphEncoder

atom_encoder = Featurizer([
    features.Symbol(),
    features.TotalNumHs(),
    features.ChiralCenter(),
    features.Aromatic(),
    features.Ring(),
    features.Hetero(),
    features.HydrogenDonor(),
    features.HydrogenAcceptor(),
    features.CIPCode(),
    features.RingSize(),
    features.GasteigerCharge()
])

bond_encoder = Featurizer([
    features.BondType(),
    features.Conjugated(),
    features.Rotatable(),
    features.Ring(),
    features.Stereo(),
])

def encode(InChI: str):
    encoder = MolecularGraphEncoder(atom_encoder, bond_encoder)
    return encoder([InChI])
