import tensorflow as tf
from molgraph import layers
from molgraph.layers import MinMaxScaling
from helpers import make_graph_data, get_test_metrics


def train(X_train, y_train, class_weight,
          units=128, dropout=0.15, dense_units=128,
          activation="selu", learning_rate=5e-5, epochs=300, batch_size=64, verbosity=2):

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
    model.add(layers.GCNIIConv(units=units, activation=activation, dropout=dropout, use_edge_features=True))
    model.add(layers.GCNIIConv(units=units, activation=activation, dropout=dropout, use_edge_features=True))
    model.add(layers.GCNIIConv(units=units, activation=activation, dropout=dropout, use_edge_features=True))
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
    X_train, y_train, class_weight = make_graph_data("./data/train_test/training_data_all.csv", upsample=True, get_class_weights=True)
    model = train(X_train, y_train, class_weight)
    get_test_metrics("../../data/train_test/test_data_all.csv", model)

if __name__ == "__main__":
    main()