import tensorflow as tf
import pandas as pd
import numpy as np
import swifter
import logging
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, confusion_matrix
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
    features.Stereo()
])
encoder = MolecularGraphEncoder(atom_encoder, bond_encoder)


def encode(smiles_string: str):
    return encoder([smiles_string])


def upsample_minority(df: pd.DataFrame):
    n_neg = len(df.query("covalent == 0"))
    n_pos = len(df.query("covalent == 1"))

    if n_neg > n_pos:
        logging.info("Upsampling the positive class...")
        n_upsample = n_neg - n_pos
        to_concat = (
            df
            .query("covalent == 1")
            .sample(n_upsample, random_state=RANDOM_STATE, replace=True)
        )

    elif n_neg < n_pos:
        logging.info("Upsampling the negative class...")
        n_upsample = n_pos - n_neg
        to_concat = (
            df
            .query("covalent == 0")
            .sample(n_upsample, random_state=RANDOM_STATE, replace=True)
        )
    else:
        return df

    return shuffle(pd.concat([df, to_concat]), random_state=RANDOM_STATE)


def make_train_val_data(csv_file_cov,
                        csv_file_noncov,
                        upsample=True,
                        debug=True):

        df_cov = pd.read_csv(csv_file_cov)
        df_cov["covalent"] = 1
        df_noncov = pd.read_csv(csv_file_noncov)
        df_noncov["covalent"] = 0

        df = pd.concat([df_cov, df_noncov])
        df = df.drop_duplicates(subset=["SMILES"])

        df_train, df_val = train_test_split(df,
                                            test_size=0.05,
                                            shuffle=True,
                                            stratify=df.covalent.values,
                                            random_state=RANDOM_STATE)

        if debug:
            logging.info("Encoding the graphs, this might take a while...")

        df_val["graph"] = df_val.SMILES.swifter.apply(encoder)
        df_train["graph"] = df_train.SMILES.swifter.apply(encoder)

        if upsample:
            df_train = upsample_minority(df_train)

        X_train, y_train = tf.concat(list(df_train.graph.values), axis=0).separate(), df_train.covalent.values
        X_val, y_val= tf.concat(list(df_val.graph.values), axis=0).separate(), df_val.covalent.values

        if debug:
            logging.info("Encoding complete!")

        return X_train, X_val, y_train, y_val


def make_test_data(csv_test_file):
    df_test = pd.read_csv(csv_test_file)
    df_test["graph"] = df_test.SMILES.swifter.apply(encoder)
    X_test, y_test = tf.concat(list(df_test.graph.values), axis=0).separate(), df_test.covalent.values
    return X_test, y_test


def make_decoy_data(csv_decoy_file):
    df_decoy = pd.read_csv(csv_decoy_file)
    df_decoy["covalent"] = 0
    df_decoy["graph"] = df_decoy.SMILES.swifter.apply(encoder)
    X_decoy, y_decoy = tf.concat(list(df_decoy.graph.values), axis=0).separate(), df_decoy.covalent.values
    return X_decoy, y_decoy


def get_class_weights(y):
    neg = len(y[y==0])
    pos = len(y[y==1])
    total = neg + pos
    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)
    class_weight = {0: weight_for_0, 1: weight_for_1}
    return class_weight


def get_test_metrics(X, y_true, model, decoy_set=False):
    y_pred = model.predict(X)
    y_pred_rounded = np.round(y_pred)
    if not decoy_set:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_rounded).ravel()
        print(f"""
        AUC {roc_auc_score(y_true, y_pred)},
        Precision {precision_score(y_true, y_pred_rounded)},
        External Recall {recall_score(y_true, y_pred_rounded)},
        TN, FP, FN, TP: {tn, fp, fn, tp}
            """, flush=True)
    else:
        acc = accuracy_score(y_true, y_pred_rounded)
        print(f"""
        FALSE POSITIVE RATE: {1-acc}
            """, flush=True)
