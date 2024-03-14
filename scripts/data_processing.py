from imports import np, pd, sk, os, train_test_split

path = os.path.join("../data","seattle-weather.csv")

def preprocess_data(path = path):
    """"
    function reads data from the file and preprocesses it
    
    Args:
        data: the data to be preprocessed

    Returns:
        data_processed: the preprocessed data as a numpy array
    """
    data = pd.read_csv(path)
    data = data.drop(columns = "date")
    data = data.dropna()
    data_numpy = data.to_numpy()

    return data, data_numpy




def split_data(data, split_ratio = 0.7):
    """
    function splits the data into training and testing sets and assigns labels
    
    Args:
        data: the data to be split
        split_ratio: the ratio of the data to be used for training
    
    Returns:
        X_train: the training data
        y_train: the training labels
        X_test: the testing data
        y_test: the testing labels
    """
    X = data[:,:-1]
    y = data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1 - split_ratio)

    return X_train, y_train, X_test, y_test




def normalize_data(X_train, X_test):
    """
    function normalizes the data to 0-1 range
    
    Args:
        X_train: the training data
        X_test: the testing data
    
    Returns:
        X_train: the normalized training data
        X_test: the normalized testing data
    """
    scaler = sk.preprocessing.MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test




def encoder(y_train, y_test):
    """
    function converts the categorical labels to numerical labels
    
    Args:
        y_train: the training labels
        y_test: the testing labels
    
    Returns:
        y_train: the numerical training labels
        y_test: the numerical testing labels
        encoder: the label encoder used to convert the labels
    """
    encoder = sk.preprocessing.LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)
    
    return y_train, y_test, encoder




def inverse_encoder(y_test, y_pred, encoder):
    """
    function converts the numerical labels back to categorical labels
    
    Args:
        y_test: the numerical testing labels
        y_pred: the numerical predicted labels
    
    Returns:
        y_test: the categorical testing labels
        y_pred: the categorical predicted labels
    """
    y_test = encoder.inverse_transform(y_test)
    y_pred = encoder.inverse_transform(y_pred)
    
    return y_test, y_pred




def create_more_data(X_train, y_train):
    """
    function creates more data by adding noise to the original data
    
    Args:
        data: the original data
    
    Returns:
        data: the original data with added noise
    """
    noise = np.random.normal(0, 0.1, X_train.shape)
    X_train = np.concatenate([X_train, X_train + noise])
    y_train = np.concatenate([y_train, y_train])
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)
    X_train = X_train[indices]
    y_train = y_train[indices]

    return X_train, y_train