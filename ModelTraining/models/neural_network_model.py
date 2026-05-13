import numpy as np

try:
    from tensorflow.keras.layers import Dense, Dropout, Input
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2
except ImportError:  # pragma: no cover - optional deep-learning dependency
    Dense = Dropout = Input = Adam = Sequential = l2 = None


def train_neural_network(
    X_train,
    y_train,
    epochs=10,
    batch_size=256,
    validation_split=0.1,
):
    if Sequential is None:
        raise ImportError("TensorFlow is required to train the neural-network baseline.")

    num_classes = int(np.max(y_train)) + 1
    model = Sequential(
        [
            Input(shape=(X_train.shape[1],)),
            Dense(512, activation="relu", kernel_regularizer=l2(0.01)),
            Dropout(0.4),
            Dense(256, activation="relu", kernel_regularizer=l2(0.01)),
            Dropout(0.3),
            Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1,
    )
    return model
