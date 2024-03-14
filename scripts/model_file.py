from imports import np, tf, sk

def build_model(input_shape):
    """
    function builds a neural network model
    
    Args:
        input_shape: the shape of the input data
    
    Returns:
        model: the built model
    """
    model = tf.keras.models.Sequential([

        tf.keras.layers.Conv1D(filters=64, kernel_size=4, padding = "causal", activation="relu", input_shape=input_shape),
        tf.keras.layers.MaxPool1D(pool_size=2),
        tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding = "same", activation="relu"),
        tf.keras.layers.MaxPool1D(pool_size=2),
        
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),

        tf.keras.layers.Flatten(),
        
        tf.keras.layers.Dense(1024, activation = "relu", kernel_regularizer = tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.GaussianNoise(0.1),
        tf.keras.layers.Dense(512, activation = "relu", kernel_regularizer = tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.GaussianNoise(0.1),
        tf.keras.layers.Dense(256, activation = "relu"),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(128, activation = "relu"),
        tf.keras.layers.Dropout(0.1),


        tf.keras.layers.Dense(5, activation = "softmax")
    ])

    
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0005)
    
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                  optimizer = optimizer,
                  metrics = ["accuracy"])
    
    return model




def find_best_learning_rate(model, train_data, train_labels):
    """"
    function finds best learning rate on the given model and data

    Args:
        model: the model to find the best learning rate for
    
    Returns:
        lr: best learning rate
    """
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch/20))

    optimizer = tf.keras.optimizers.SGD(learning_rate = 1e-8, momentum = 0.9)
    model.compile(loss = tf.keras.losses.CategoricalCrossentropy(),
                  optimizer = optimizer,
                  metrics = ["accuracy"])
    
    history = model.fit(train_data, train_labels, epochs = 100, callbacks = [lr_scheduler])

    return history.history["lr"][np.argmin(history.history["loss"])]
    