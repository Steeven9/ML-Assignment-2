import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

import custom_utils as cu

np.random.seed(69)
tf.random.set_seed(69)

if __name__ == '__main__':
    img_size = 224
    label_to_idx = {
        'rock':0,
        'paper':1,
        'scissors':2
    }
    idx_to_label = {
        0:'rock',
        1:'paper',
        2:'scissors'
    }

    # Load data
    (x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = cu.load_rps()

    # Preprocess data
    # Preprocess and split data
    x, y = cu.make_dataset(x_train_raw, y_train_raw, label_to_idx, (img_size, img_size))
    x_test, y_test = cu.make_dataset(x_test_raw, y_test_raw, label_to_idx, (img_size, img_size))

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)

    # Load the trained model
    model = load_model('./nn_task2_no_augmentation.h5')
    #model = load_model('./nn_task2_no_augmentation.pkl')

    # Predict on the given samples
    y_pred = model.predict(x_test)

    # Evaluate the missclassification error on the test set
    assert y_test.shape == y_pred.shape
    scores = model.evaluate(x_test, y_pred)
    mse = ((y_test - y_pred) ** 2).mean()
    print('-- Stats for task 2 (without augmentation) --')
    print('Test loss: {} - Accuracy: {} - '.format(*scores), end='')
    print('MSE: {}'.format(mse))

    # Load the trained model with augmentation
    model = load_model('./nn_task2_augmentation.h5')
    #model = load_model('./nn_task2_augmentation.pkl')

    # Predict on the given samples
    y_pred = model.predict(x_test)

    # Evaluate the missclassification error on the test set
    assert y_test.shape == y_pred.shape
    scores = model.evaluate(x_test, y_pred)
    mse = ((y_test - y_pred) ** 2).mean()
    print('-- Stats for task 2 (with augmentation) --')
    print('Test loss: {} - Accuracy: {} - '.format(*scores), end='')
    print('MSE: {}'.format(mse))
