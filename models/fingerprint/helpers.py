import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from rdkit import Chem


def get_fingerprint(InChI, fpgen):
    mol = Chem.MolFromInchi(InChI)

    if not mol:
        return None

    return fpgen.GetFingerprintAsNumPy(mol)


def make_train_val_data(csv_file, fpgen, random_state):
    df = pd.read_csv(csv_file)
    df["fp"] = df.InChI.apply(lambda x: get_fingerprint(x, fpgen=fpgen))
    df = df.dropna()
    X = np.stack(df.fp.values)
    y = df.covalent.values
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.1,
                                                        stratify=y,
                                                        random_state=random_state)
    return X_train, X_test, y_train, y_test


def make_test_data(csv_file, fpgen):
    df = pd.read_csv(csv_file)
    df["fp"] = df.InChI.apply(lambda x: get_fingerprint(x, fpgen=fpgen))
    df = df.dropna()
    X = np.stack(df.fp.values)
    y = df.covalent.values
    return X, y


def get_test_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    auroc = roc_auc_score(y_test, y_pred)
    prec = precision_score(y_test, np.round(y_pred))
    recall = recall_score(y_test, np.round(y_pred))
    return auroc, prec, recall