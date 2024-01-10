import tensorflow as tf
import os
import pandas as pd
from molgraph import layers
from molgraph.layers import MinMaxScaling

from helpers import make_train_val_data, make_test_data, make_decoy_data
from helpers import get_test_metrics, get_class_weights

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


TRAIN_DATA_COV = "./data/SMILES_training/trainingset_covalent_smiles.csv"
TRAIN_DATA_NONCOV = "./data/SMILES_training/trainingset_noncovalent_smiles.csv"
TEST_DATA = "./data/SMILES_test/test_data_all.csv"
DECOY_DATA = "./data/SMILES_test/testset_decoy.csv"
UPSAMPLE = True
CHANGE_WEIGTHS = True
MODELNAME = "GCNII"

def train(X_train, y_train,
          class_weight={0:1, 1:1},
          layer = layers.GCNIIConv,
          units=64,
          n_layers=6,
          use_edge_features=True,
          dropout=0.1,
          dense_units=128,
          activation="selu",
          learning_rate=5e-5,
          epochs=20,
          batch_size=16,
          verbosity=2):

    node_preprocessing = MinMaxScaling(
        feature='node_feature', feature_range=(0, 1), threshold=True)
    edge_preprocessing = MinMaxScaling(
        feature='edge_feature', feature_range=(0, 1), threshold=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='roc_auc',
            min_delta=1e-3,
            patience=5,
            mode='max',
            restore_best_weights=True,
        )
    ]

    node_preprocessing.adapt(X_train.merge())
    edge_preprocessing.adapt(X_train.merge())

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(type_spec=X_train.merge().unspecific_spec))
    model.add(node_preprocessing)
    model.add(edge_preprocessing)
    for _ in range(n_layers):
        model.add(layer(units=units, activation=activation, dropout=dropout,
                        use_edge_features=use_edge_features))
    model.add(layers.Readout('mean'))
    model.add(tf.keras.layers.Dense(dense_units, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.Precision(),
                 tf.keras.metrics.Recall(),
                 tf.keras.metrics.AUC(name="roc_auc")])

    model.fit(X_train, y_train,
                epochs=epochs,
                verbose=verbosity,
                batch_size=batch_size,
                callbacks=callbacks,
                class_weight=class_weight)
    model.save(f"./{MODELNAME}")
    return model


def main():
    X_train, X_val, y_train, y_val = make_train_val_data(csv_file_cov=TRAIN_DATA_COV,
                                                         csv_file_noncov=TRAIN_DATA_NONCOV,
                                                         upsample=UPSAMPLE)
    X_test, y_test = make_test_data(csv_test_file=TEST_DATA)
    X_decoy, y_decoy = make_decoy_data(csv_decoy_file=DECOY_DATA)

    class_weight = get_class_weights(y=y_train)

    if CHANGE_WEIGTHS:
        model = train(X_train=X_train,
                    y_train=y_train,
                    class_weight=class_weight)
    else:
        model = train(X_train=X_train,
                    y_train=y_train)

    print("***\n VAL METRICS \n***")
    get_test_metrics(X_val, y_val, model)

    print("***\n TEST METRICS \n***")
    get_test_metrics(X_test, y_test, model)

    print("***\n DECOY METRICS \n***")
    get_test_metrics(X_decoy, y_decoy, model, decoy_set=True)


if __name__ == "__main__":
    main()