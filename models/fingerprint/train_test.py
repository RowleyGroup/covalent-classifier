import pandas as pd
import rdkit.RDLogger as RDLogger
from rdkit.Chem import AllChem
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from helpers import make_train_val_data, make_test_data,get_test_metrics


RDKIT_LOGGER = RDLogger.logger()
RDKIT_LOGGER.setLevel(RDLogger.CRITICAL)
RANDOM_SEED = 66
VERBOSITY = 1

# Adjust these as needed
RADIUS = 3
N_BITS = 4096
FP_GEN = AllChem.GetMorganGenerator(fpSize=N_BITS, radius=RADIUS)

# Pick one of the models below
MODEL_DICT = {
    "hgb": HistGradientBoostingClassifier(max_leaf_nodes=41, max_depth=None,
                                        class_weight="balanced", verbose=VERBOSITY,random_state=RANDOM_SEED),

    "lr": LogisticRegression(class_weight="balanced", solver='saga', verbose=VERBOSITY),

    "mlp": MLPClassifier(hidden_layer_sizes=(100,100,100), alpha=0.0001,
                  solver='adam', activation='relu', verbose=VERBOSITY),

    "rf": RandomForestClassifier(max_depth=21, verbose=VERBOSITY),

    "svc": SVC(kernel="rbf", C=1, verbose=VERBOSITY)
}
MODEL = MODEL_DICT["mlp"]

def main(debug=True):
    if debug:
          print("Fetching and encoding the data...")

    X_train, X_val, y_train, y_val = make_train_val_data("./data/train_test/training_data_all.csv",
                                                           fpgen=FP_GEN, random_state=RANDOM_SEED)
    X_test, y_test = make_test_data("./data/train_test/test_data_all.csv", fpgen=FP_GEN)
    X_false_positive, y_false_positive = make_test_data("./data/test_noncovalent/false_covalent.csv", fpgen=FP_GEN)

    if debug:
          print("Encoding complete! Begginning training...")

    MODEL.fit(X_train, y_train)

    # validation
    auroc_val, prec_val, recall_val = get_test_metrics(MODEL, X_val, y_val)
    print(f"""
    Internal AUROC: {auroc_val},
    Internal PRECISION : {prec_val},
    Internal RECALL: {recall_val},
          """, flush=True)

    # test
    auroc_test, prec_test, recall_test = get_test_metrics(MODEL, X_test, y_test)
    print(f"""
    External AUROC: {auroc_test},
    External PRECISION : {prec_test},
    External RECALL: {recall_test},
          """, flush=True)

    # false covalent
    accuracy = accuracy_score(y_false_positive, MODEL.predict(X_false_positive))
    print(f"""
    FALSE POSITIVE RATE: {1-accuracy}
          """, flush=True)

if __name__ == "__main__":
    main()