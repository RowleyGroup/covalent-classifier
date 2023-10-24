import pandas as pd
from sklearn.utils import shuffle
from molgraph.chemistry import features, Featurizer, MolecularGraphEncoder

RANDOM_STATE = 66

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

encoder = MolecularGraphEncoder(atom_encoder, bond_encoder)

def encode(InChI: str):
    return encoder([InChI])

def upsample_minority(df: pd.DataFrame):
    n_neg = len(df.query("covalent == 0"))
    n_pos = len(df.query("covalent == 1"))

    if n_neg > n_pos:
        n_upsample = n_neg - n_neg
        to_concat = (
            df
            .query("covalent == 1")
            .sample(n_upsample, random_state=RANDOM_STATE, replace=True)
        )

    elif n_neg < n_pos:
        n_upsample = n_pos - n_neg
        to_concat = (
            df
            .query("covalent == 0")
            .sample(n_upsample, random_state=RANDOM_STATE, replace=True)
        )
    else:
        return df

    return shuffle(pd.concat([df, to_concat]), random_state=RANDOM_STATE)
