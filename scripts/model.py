from imports import np, tf, sk

def build_model(input_shape):
    """
    function builds a neural network model
    
    Args:
        input_shape: the shape of the input data
    
    Returns:
        model: the built model
    """
    model = tf.models.Sequential([
        
    ])
    
    optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001)
    
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                  optimizer = optimizer,
                  metrics = ["accuracy"])
    
    return model