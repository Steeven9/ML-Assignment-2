import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential, applications, optimizers, utils
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (AveragePooling2D, Conv2D, Dense, Flatten,
                                     MaxPooling2D)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import custom_utils as cu

np.random.seed(69)
tf.random.set_seed(69)

def task_1():
    #### TASK 1 ####

    # Load the data
    (x_train, y_train), (x_test, y_test) = cu.load_cifar10()

    # Normalize to 0-1 range
    x_train = x_train / 255.
    x_test = x_test / 255.

    # Split into classes
    n_classes = 3
    y_train = utils.to_categorical(y_train, n_classes)
    y_test = utils.to_categorical(y_test, n_classes)

    # Build model
    model = Sequential()
    model.add(Conv2D(8, (5, 5), activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (3, 3), strides=(2, 2), activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(8, activation='tanh'))
    model.add(Dense(3, activation='softmax'))

    # Compile model
    model.compile(optimizer=optimizers.RMSprop(learning_rate=0.003), 
                loss='categorical_crossentropy', 
                metrics=['accuracy', 'mse'])
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.summary()

    # Train model    
    history = model.fit(x_train, 
            y_train, 
            shuffle=True,
            batch_size=128, 
            epochs=500,
            validation_split=0.2,
            callbacks=[es])

    # Evaluate model
    scores = model.evaluate(x_test, y_test)
    print('Test loss: {} - Accuracy: {} - MSE: {}'.format(*scores))

    # Save model
    model.save('./nn_task1.h5')

    # Draw plot
    smooth = lambda y: np.polyval(np.polyfit(history.epoch, y, deg=5), history.epoch)
    plt.plot(smooth(history.history['accuracy']), c='C0', alpha=0.7, lw=3)
    plt.plot(smooth(history.history['val_accuracy']), c='C1', alpha=0.7, lw=3)
    plt.plot(history.history['accuracy'], label='Training accuracy', c='C0')
    plt.plot(history.history['val_accuracy'], label='Validation accuracy', c='C1')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.title("Task 1")
    plt.legend()
    plt.savefig('./plot_task1.png')
    plt.close()

    print("Done with task 1.")


def task_2():
    #### TASK 2 ####

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

    # Preprocess and split data
    x, y = cu.make_dataset(x_train_raw, y_train_raw, label_to_idx, (img_size, img_size))
    x_test, y_test = cu.make_dataset(x_test_raw, y_test_raw, label_to_idx, (img_size, img_size))

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)

    # Build the VGG16 network, download pre-trained weights 
    # and remove the last dense layers
    vgg16 = applications.VGG16(weights='imagenet',  
                            include_top=False, 
                            input_shape=(img_size, img_size, 3))

    # Freeze the network weights
    vgg16.trainable = False

    # Build model
    model = Sequential()
    model.add(vgg16)
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    # Compile model
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy', 
                metrics=['accuracy', 'mse'])
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.summary()

    # Train model    
    history = model.fit(x_train, 
            y_train, 
            shuffle=True,
            batch_size=16, 
            epochs=50,
            validation_data=(x_val, y_val),
            callbacks=[es])

    # Evaluate model
    scores = model.evaluate(x_test, y_test)
    print('Test loss: {} - Accuracy: {} - MSE: {}'.format(*scores))

    # Save model
    model.save('./nn_task2_no_augmentation.h5')
    cu.save_vgg16(model, filename='./nn_task2_no_augmentation.pkl')

    # Draw plot
    smooth = lambda y: np.polyval(np.polyfit(history.epoch, y, deg=5), history.epoch)
    plt.plot(smooth(history.history['accuracy']), c='C0', alpha=0.7, lw=3)
    plt.plot(smooth(history.history['val_accuracy']), c='C1', alpha=0.7, lw=3)
    plt.plot(history.history['accuracy'], label='Training accuracy', c='C0')
    plt.plot(history.history['val_accuracy'], label='Validation accuracy', c='C1')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.title("Task 2 - without augmentation")
    plt.legend()
    plt.savefig('./plot_task2_no_augmentation.png')
    plt.close()

    print("Now using data augmentation")

    train_gen = ImageDataGenerator(width_shift_range=0.15,     # horizontal translation
                                    height_shift_range=0.15,   # vertical translation
                                    channel_shift_range=0.3,   # random channel shifts
                                    rotation_range=360,        # rotation
                                    zoom_range=0.3,            # zoom in/out randomly
                                    shear_range=15,            # deformation
                                    )
    val_gen = ImageDataGenerator()

    batch_size = 16
    es = EarlyStopping(monitor='val_loss', patience=200, restore_best_weights=True)

    train_loader = train_gen.flow(x_train, y_train, batch_size=batch_size)
    val_loader = val_gen.flow(x_val, y_val, batch_size=x_val.shape[0])

    history = model.fit(train_loader,
                            steps_per_epoch=x_train.shape[0]//batch_size,
                            epochs=50, 
                            validation_data=val_loader,
                            validation_steps=1,
                            callbacks=[es])

    # Evaluate model
    scores = model.evaluate(x_test, y_test)
    print('Test loss: {} - Accuracy: {} - MSE: {}'.format(*scores))

    # Save model
    model.save('./nn_task2_augmentation.h5')
    cu.save_vgg16(model, filename='./nn_task2_augmentation.pkl')

    # Draw plot
    smooth = lambda y: np.polyval(np.polyfit(history.epoch, y, deg=5), history.epoch)
    plt.plot(smooth(history.history['accuracy']), c='C0', alpha=0.7, lw=3)
    plt.plot(smooth(history.history['val_accuracy']), c='C1', alpha=0.7, lw=3)
    plt.plot(history.history['accuracy'], label='Training accuracy', c='C0')
    plt.plot(history.history['val_accuracy'], label='Validation accuracy', c='C1')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.title("Task 2 - with augmentation")
    plt.legend()
    plt.savefig('./plot_task2_augmentation.png')
    plt.close()

    print("Done with task 2.")

    
if __name__ == '__main__':
    if (len(sys.argv) == 2):
        if sys.argv[1] == "1":
            print("Launching task 1")
            task_1()
        elif sys.argv[1] == "2":
            print("Launching task 2")
            task_2()
        else:
            print("Launching both tasks")
            task_1()
            task_2()
    else:
        print("Launching both tasks")
        task_1()
        task_2()
