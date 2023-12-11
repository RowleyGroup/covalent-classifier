import tensorflow as tf
import os
import pandas as pd
from molgraph import layers
from molgraph.layers import MinMaxScaling
from helpers import make_graph_data, get_test_metrics, get_class_weights, get_val_metrics
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


TRAIN_DATA_COV = "./data/SMILES_training/trainingset_covalent_smiles.csv"
TRAIN_DATA_NONCOV = "./data/SMILES_training/trainingset_noncovalent_smiles.csv"

def train(X_train, y_train, class_weight={0:1, 1:1},
          units=128, n_layers=6, use_edge_features=True,
          dropout=0.15, dense_units=128,
          activation="selu", learning_rate=5e-5, epochs=30, batch_size=64, verbosity=2):

    node_preprocessing = MinMaxScaling(
        feature='node_feature', feature_range=(0, 1), threshold=True)
    edge_preprocessing = MinMaxScaling(
        feature='edge_feature', feature_range=(0, 1), threshold=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='roc_auc',
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
        model.add(layers.GINConv(units=units, activation=activation, dropout=dropout, use_edge_features=use_edge_features))
    model.add(layers.Readout('mean'))
    model.add(tf.keras.layers.Dense(dense_units, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC(name="roc_auc")])

    model.fit(X_train, y_train,
                epochs=epochs,
                verbose=verbosity,
                batch_size=batch_size,
                callbacks=callbacks,
                class_weight=class_weight)
    return model


def main():
    X_train, X_val, y_train, y_val = make_graph_data(csv_file_cov=TRAIN_DATA_COV,
                                                     csv_file_noncov=TRAIN_DATA_NONCOV,
                                                     upsample=True)

    class_weight = get_class_weights(y=y_train)

    model = train(X_train=X_train,
                  y_train=y_train,
                  class_weight=class_weight)

    get_val_metrics(X_val=X_val,
                    y_val=y_val,
                    model=model)

    get_test_metrics(test_file="./data/SMILES_test/test_data_all.csv",
                     model=model)


if __name__ == "__main__":
    main()