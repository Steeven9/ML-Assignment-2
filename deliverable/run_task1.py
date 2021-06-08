import numpy as np
import tensorflow as tf
from tensorflow.keras import utils
from tensorflow.keras.models import load_model

import custom_utils as cu

np.random.seed(69)
tf.random.set_seed(69)

if __name__ == '__main__':
    # Load the data
    (x_train, y_train), (x_test, y_test) = cu.load_cifar10()

    # Normalize to 0-1 range
    x_test = x_test / 255.

    # Split into classes
    n_classes = 3
    y_test = utils.to_categorical(y_test, n_classes)

    # Load the trained model
    model = load_model('./nn_task1.h5')

    # Predict on the given samples
    y_pred = model.predict(x_test)

    # Evaluate the missclassification error on the test set
    assert y_test.shape == y_pred.shape
    scores = model.evaluate(x_test, y_pred)
    mse = ((y_test - y_pred) ** 2).mean()
    print('-- Stats for task 1 --')
    print('Test loss: {} - Accuracy: {} - '.format(*scores), end='')
    print('MSE: {}'.format(mse))
