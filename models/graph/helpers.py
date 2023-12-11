import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from molgraph.chemistry import features, Featurizer, MolecularGraphEncoder

RANDOM_STATE = 66

atom_encoder = Featurizer([
    features.Symbol(),
    features.TotalNumHs(),
    features.Aromatic(),
    features.Ring(),
    features.Hetero(),
    features.HydrogenDonor(),
    features.HydrogenAcceptor(),
    features.RingSize(),
    features.GasteigerCharge()
])
bond_encoder = Featurizer([
    features.BondType(),
    features.Conjugated(),
    features.Rotatable(),
    features.Ring(),
])
encoder = MolecularGraphEncoder(atom_encoder, bond_encoder)


def encode(smiles_string: str):
    return encoder([smiles_string])


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


def make_graph_data(csv_file_cov, csv_file_noncov, upsample=True, debug=True, test_set=False):

    df_cov = pd.read_csv(csv_file_cov)
    df_cov["covalent"] = 1
    df_noncov = pd.read_csv(csv_file_noncov)
    df_noncov["covalent"] = 0

    df_train = pd.concat([df_cov, df_noncov])
    df_train = df_train.drop_duplicates(subset=["SMILES"])
    df_train = shuffle(
        df_train.reset_index(drop=True),
        random_state=RANDOM_STATE)

    if not test_set:
        df_val = df_train.sample(frac=0.1, random_state=RANDOM_STATE)
        df_train = df_train.drop(df_val.index)

        if debug:
            print("Encoding the graphs, this might take a while...", flush=True)

        df_train["graph"] = df_train.SMILES.apply(encoder)
        df_val["graph"] = df_val.SMILES.apply(encoder)
        if upsample:
            df_train = upsample_minority(df_train)

        X_train, y_train = tf.concat(list(df_train.graph.values), axis=0).separate(), df_train.covalent.values
        X_val, y_val= tf.concat(list(df_val.graph.values), axis=0).separate(), df_val.covalent.values


        if debug:
            print("Encoding complete!", flush=True)

        return X_train, X_val, y_train, y_val
    else:
        if debug:
            print("Encoding the graphs, this might take a while...", flush=True)

        df_train["graph"] = df_train.SMILES.apply(encoder)
        X_train, y_train = tf.concat(list(df_train.graph.values), axis=0).separate(), df_train.covalent.values

        if debug:
            print("Encoding complete!", flush=True)

        return X_train, y_train


def get_class_weights(y):
    neg = len(y[y==0])
    pos = len(y[y==1])
    total = neg + pos
    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)
    class_weight = {0: weight_for_0, 1: weight_for_1}
    return class_weight


def get_val_metrics(X_val, y_val, model):
    y_pred = model.predict(X_val)
    y_pred_rounded = np.round(y_pred)
    print(f"""
    Internal AUC {roc_auc_score(y_val, y_pred)},
    Internal Precision {precision_score(y_val, y_pred_rounded)},
    Internal Recall {recall_score(y_val, y_pred_rounded)},
          """, flush=True)


def get_test_metrics(test_file, model):
    X_test, y_test = make_graph_data(test_file, upsample=False, debug=False, test_set=True)
    y_pred = model.predict(X_test)
    y_pred_rounded = np.round(y_pred)
    print(f"""
    External AUC {roc_auc_score(y_test, y_pred)},
    External Precision {precision_score(y_test, y_pred_rounded)},
    External Recall {recall_score(y_test, y_pred_rounded)},
          """, flush=True)
