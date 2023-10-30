import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score, precision_score, recall_score
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
        print("Upsampling the positive class...")
        n_upsample = n_neg - n_pos
        to_concat = (
            df
            .query("covalent == 1")
            .sample(n_upsample, random_state=RANDOM_STATE, replace=True)
        )

    elif n_neg < n_pos:
        print("Upsampling the negative class...")
        n_upsample = n_pos - n_neg
        to_concat = (
            df
            .query("covalent == 0")
            .sample(n_upsample, random_state=RANDOM_STATE, replace=True)
        )
    else:
        return df

    return shuffle(pd.concat([df, to_concat]), random_state=RANDOM_STATE)


def make_graph_data(file, upsample=True, debug=True):
    df_train = pd.read_csv(file)
    df_train = shuffle(
        df_train.reset_index(drop=True),
        random_state=RANDOM_STATE)

    if debug:
        print("Encoding the graphs, this might take a while...", flush=True)
    df_train["graph"] = df_train.InChI.apply(encoder)

    if upsample:
        df_train = upsample_minority(df_train)

    X, y = tf.concat(list(df_train.graph.values), axis=0).separate(), df_train.covalent.values

    if debug:
        print("Encoding complete!", flush=True)

    return X, y

def get_class_weights(y):
    neg = len(y[y==0])
    pos = len(y[y==1])
    total = neg + pos
    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)
    class_weight = {0: weight_for_0, 1: weight_for_1}
    return class_weight


def get_test_metrics(test_file, model):
    X_test, y_test = make_graph_data(test_file, upsample=False, debug=False)
    y_pred = model.predict(X_test)
    y_pred_rounded = np.round(y_pred)
    print(f"""
    Test AUC {roc_auc_score(y_test, y_pred)},
    Test Precision {precision_score(y_test, y_pred_rounded)},
    Test Recall {recall_score(y_test, y_pred_rounded)},
          """, flush=True)
