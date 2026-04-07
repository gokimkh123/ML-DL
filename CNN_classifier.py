import tensorflow as tf
from tensorflow.keras import layers, models, utils
import matplotlib.pyplot as plt
import numpy as np

def build_CNNModel():
    inputs = layers.Input(shape=(28, 28, 1))

    x = layers.Conv2D(32, (3, 3), strides=(1, 1), padding="valid")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), strides=(1, 1), padding="valid")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    flatten = layers.Flatten()(x)
    hidden = flatten

    for num_nodes in [200, 200]:
        hidden = layers.Dense(units=num_nodes, activation="relu")(hidden)
    outputs = layers.Dense(units=10, activation="softmax")(hidden)

    cnn_model = models.Model(inputs=inputs, outputs=outputs)

    return cnn_model

def classifier_Fashion_MNIST_by_CNN():
    f_mnist = tf.keras.datasets.fashion_mnist
    (x_train_full, y_train_full), (x_test, y_test_orig) = f_mnist.load_data()

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    y_train_full = utils.to_categorical(y_train_full, 10)
    y_test = utils.to_categorical(y_test_orig, 10)

    x_train_full, x_test = x_train_full / 255.0, x_test / 255.0

    x_val, x_train = x_train_full[:5000], x_train_full[5000:]
    y_val, y_train = y_train_full[:5000], y_train_full[5000:]

    epochs = 20
    batch_size = 64

    model = build_CNNModel()

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    print("Start training...")
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val))

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

    print("Final Test Accuracy:{test_acc:.4f")

    predictions = model.predict(x_test)

    plt.figure(figsize=(12,8))

    for i in range(10):
        plt.subplot(2,5,i+1)
        plt.imshow(x_test[i], cmap="gray")

        pred_idx = np.argmax(predictions[i])
        true_idx = y_test_orig[i]

        color = 'blue' if pred_idx == true_idx else 'red'
        plt.axis("off")
    plt.tight_layout()
    plt.show()

    plt.plot(history.history['loss'], label="train_loss")
    plt.plot(history.history['val_loss'], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    classifier_Fashion_MNIST_by_CNN()