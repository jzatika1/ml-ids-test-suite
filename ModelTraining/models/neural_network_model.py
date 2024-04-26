from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

def train_neural_network(X_train, y_train):
    
    # Define the model architecture
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(4096, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(2048, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(1024, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(X_train.shape[1], activation='softmax')
    ])
    
    # Compile the model
    model.compile(optimizer=Adam(), 
                  loss='sparse_categorical_crossentropy',  # Suitable for integer labels
                  metrics=['accuracy'])
    
    return model