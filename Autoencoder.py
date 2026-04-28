import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt
import numpy as np

@staticmethod
def print_56_pari_images(img_data_list1, img_data_list2, label_list):
    num_row = 7
    num_col = 16
    if img_data_list2 is None:
        num_col = 8

    num_pairs = num_row * num_col
    plt.figure(figsize=(10, 8))
    plt.title("Digit pairs")
    num_images = img_data_list1.shape[0]

    if num_images > num_pairs:
        num_images = num_pairs

    if img_data_list2 is None:
        for i in range(num_images):
            plt.subplot(num_row, num_col, i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(img_data_list1[i], cmap=plt.cm.binary)
            plt.xlabel(str(label_list[i]) + "_o")
    else:
        for i in range(num_images):
            plt.subplot(num_row, num_col, 2 * i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(img_data_list1[i], cmap=plt.cm.binary)
            plt.xlabel(str(label_list[i]) + "_o")

        for i in range(num_images):
            plt.subplot(num_row, num_col, 2 * (i + 1))
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(img_data_list1[i], cmap=plt.cm.binary)
            plt.xlabel(str(label_list[i]) + "_r")

    plt.show()

def build_AE(in_out_dim = 784, latent_dim = 32, encoder_h_layers=[300, 150], decoder_h_layers=[150, 300]):
    encoder_input = layers.Input(shape=(in_out_dim, ))
    encoder_h_layer = encoder_input

    for dim in encoder_h_layers:
        encoder_h_layer = layers.Dense(units=dim, activation="relu", use_bias=True)(encoder_h_layer)

    latent = layers.Dense(units=latent_dim, activation="tanh", use_bias=True)(encoder_h_layer)
    encoder = models.Model(inputs=encoder_input, outputs=latent)

    decoder_input = layers.Input(shape=(latent_dim, ))
    decoder_h_layer = decoder_input

    for dim in decoder_h_layers:
        decoder_h_layer = layers.Dense(units=dim, activation="relu", use_bias=True)(decoder_h_layer)

    decoder_output = layers.Dense(units=in_out_dim, activation=None, use_bias=True)(decoder_h_layer)
    decoder = models.Model(inputs=decoder_input, outputs=decoder_output)


    decoder_out = decoder(latent)
    en_decoder = models.Model(inputs=encoder_input, outputs=decoder_out)

    return encoder, decoder, en_decoder

def MNIST_AE():
    pic_w = 28
    pic_h = 28

    in_out_dim = pic_w * pic_h

    paramerter_path="./AE_model/as_model.weights.h5"

    lr = 0.001
    epochs = 20
    batch_size = 128
    latent_dim = 32
    encoder_h_layers = [300, 150]
    decoder_h_layers = [150, 300]

    mnist = tf.keras.datasets.mnist
    (x_train_full, y_train_full), (x_test,y_test) = mnist.load_data()

    x_train_full, x_test = x_train_full / 255.0, x_test / 255.0

    print_56_pari_images(img_data_list1=x_test[0:56], img_data_list2=None, label_list=y_test[0:56])

    x_train_full = x_train_full.reshape((len(x_train_full), np.prod(x_train_full.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    x_val, x_train = x_train_full[:5000], x_train_full[5000:]

    encoder, decoder, en_decoder = build_AE(in_out_dim, latent_dim, encoder_h_layers, decoder_h_layers)

    adam = optimizers.Adam(learning_rate=lr)
    en_decoder.compile(optimizer=adam, loss="mse")

    print("Start training...")
    history = en_decoder.fit(x_train, x_train, epochs=epochs, validation_data=(x_val, x_val), batch_size=batch_size)
    test_loss = en_decoder.evaluate(x_test, x_test, verbose=2)
    print(f"Test loss: {test_loss}")

    en_decoder.save_weights(paramerter_path)

    plt.plot(history.history["loss"], label="train_mse")
    plt.plot(history.history["val_loss"], label="val_mse")
    plt.xlabel("Epoch")
    plt.ylabel("mse")
    plt.legend()
    plt.show()

    encoder, decoder, en_decoder = build_AE(in_out_dim, latent_dim, encoder_h_layers, decoder_h_layers)
    en_decoder.load_weights(paramerter_path)
    print("load model parameters from %s" % paramerter_path)

    latent = encoder.predict(x_test)
    test_reconstruction = decoder.predict(latent)

    test_image_recon = test_reconstruction.reshape(-1, pic_h, pic_w)
    test_img = x_test.reshape(-1, pic_h, pic_w)

    n = 56
    random_indices = np.random.randint(len(x_test), size=n)
    print_truth = test_img[random_indices]
    print_recond = test_image_recon[random_indices]
    label_list = y_test[random_indices]
    print_56_pari_images(img_data_list1=print_truth, img_data_list2=print_recond, label_list=label_list)

if __name__ == '__main__':
    MNIST_AE()

