import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

def print_56_pair_images(img_data_list1, img_data_list2, label_list):
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
            plt.subplot(num_row, num_col, i + 1)
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
            plt.imshow(img_data_list2[i], cmap=plt.cm.binary)
            plt.xlabel(str(label_list[i]) + "_r")

    plt.show()


def add_salt_pepper_noise(images, prob=0.5):
    mask = np.random.binomial(1, 1 - prob, size=images.shape).astype("float32")
    return images * mask


def build_AE(in_out_dim=784, latent_dim=32, encoder_h_layers=[300, 150], decoder_h_layers=[150, 300]):
    encoder_input = layers.Input(shape=(in_out_dim,))
    encoder_h_layer = encoder_input

    for dim in encoder_h_layers:
        encoder_h_layer = layers.Dense(units=dim, activation="relu", use_bias=True)(encoder_h_layer)

    latent = layers.Dense(units=latent_dim, activation="tanh", use_bias=True)(encoder_h_layer)
    encoder = models.Model(inputs=encoder_input, outputs=latent)

    decoder_input = layers.Input(shape=(latent_dim,))
    decoder_h_layer = decoder_input

    for dim in decoder_h_layers:
        decoder_h_layer = layers.Dense(units=dim, activation="relu", use_bias=True)(decoder_h_layer)

    decoder_output = layers.Dense(units=in_out_dim, activation=None, use_bias=True)(decoder_h_layer)
    decoder = models.Model(inputs=decoder_input, outputs=decoder_output)

    decoder_out = decoder(latent)
    en_decoder = models.Model(inputs=encoder_input, outputs=decoder_out)

    return encoder, decoder, en_decoder


def MNIST_DAE():
    pic_w = 28
    pic_h = 28
    in_out_dim = pic_w * pic_h

    lr = 0.001
    epochs = 20
    batch_size = 128
    latent_dim = 32
    encoder_h_layers = [300, 150]
    decoder_h_layers = [150, 300]
    noise_prob = 0.5

    mnist = tf.keras.datasets.mnist
    (x_train_full, y_train_full), (x_test, y_test) = mnist.load_data()

    x_train_full, x_test = x_train_full / 255.0, x_test / 255.0

    x_train_full = x_train_full.reshape((len(x_train_full), np.prod(x_train_full.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    x_val, x_train = x_train_full[:5000], x_train_full[5000:]
    y_val, y_train = y_train_full[:5000], y_train_full[5000:]

    x_train_noised = add_salt_pepper_noise(x_train, noise_prob)
    x_val_noised   = add_salt_pepper_noise(x_val,   noise_prob)
    x_test_noised  = add_salt_pepper_noise(x_test,  noise_prob)

    encoder, decoder, en_decoder = build_AE(in_out_dim, latent_dim, encoder_h_layers, decoder_h_layers)

    adam = optimizers.Adam(learning_rate=lr)
    en_decoder.compile(optimizer=adam, loss="mse")

    print("Start training...")
    history = en_decoder.fit(
        x_train_noised, x_train,
        epochs=epochs,
        validation_data=(x_val_noised, x_val),
        batch_size=batch_size
    )
    test_loss = en_decoder.evaluate(x_test_noised, x_test, verbose=2)
    print(f"Test loss: {test_loss}")

    '''
    # ──────────────────────────────────────────────
    # [1. Denoising autoencoder학습 & Denoising 성능확인]
    # ──────────────────────────────────────────────
    n = 56
    random_idx = np.random.randint(len(x_test), size=n)

    sample_56_noised = x_test_noised[random_idx]
    sample_56_recon  = en_decoder.predict(sample_56_noised)

    noised_img = sample_56_noised.reshape(-1, pic_h, pic_w)
    recon_img  = sample_56_recon.reshape(-1, pic_h, pic_w)
    label_list = y_test[random_idx]

    print_56_pair_images(img_data_list1=noised_img, img_data_list2=recon_img, label_list=label_list)
    '''
    # ──────────────────────────────────────────────
    # [3. 각 숫자(class)별 latent representation 평균값으로 이미지 생성]
    # ──────────────────────────────────────────────
    latent_test = encoder.predict(x_test)

    avg_latents = np.zeros((10, latent_dim), dtype="float32")
    std_latents = np.zeros((10, latent_dim), dtype="float32")

    for digit in range(10):
        idx = (y_test == digit)
        avg_latents[digit] = latent_test[idx].mean(axis=0)
        std_latents[digit] = latent_test[idx].std(axis=0)

    avg_images = decoder.predict(avg_latents).reshape(-1, pic_h, pic_w)

    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    fig.suptitle("Task 3: Images from Average Latent Representations")
    for digit in range(10):
        r, c = divmod(digit, 5)
        axes[r, c].imshow(avg_images[digit], cmap=plt.cm.binary)
        axes[r, c].set_title(f"Digit {digit}")
        axes[r, c].axis("off")
    plt.tight_layout()
    plt.show()
    '''
    # ──────────────────────────────────────────────
    # [과제 4] T-SNE로 평균 latent vector 위치관계 시각화
    # ──────────────────────────────────────────────
    tsne = TSNE(n_components=2, perplexity=3, random_state=42)
    avg_2d = tsne.fit_transform(avg_latents)

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    plt.figure(figsize=(7, 6))
    plt.title("t-SNE Visualization of Avg Codes (per digit)")
    for digit in range(10):
        plt.scatter(avg_2d[digit, 0], avg_2d[digit, 1], color=colors[digit], s=120, label=str(digit))
        plt.annotate(str(digit), xy=(avg_2d[digit, 0], avg_2d[digit, 1]),
                     xytext=(6, 6), textcoords="offset points",
                     fontsize=11, fontweight="bold", color=colors[digit])
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(title="Digit", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()
    '''
    # ──────────────────────────────────────────────
    # [5. 각 숫자별(class 별) latent representation 평균값 및 표준편차를 활용하여 새로운 이미지 생성]
    # ──────────────────────────────────────────────
    N_SAMPLES = 5

    gen_latents = np.zeros((10, N_SAMPLES, latent_dim), dtype="float32")
    for digit in range(10):
        for j in range(N_SAMPLES):
            rand_j = np.random.uniform(-1.0, 1.0, size=(latent_dim,)).astype("float32")
            gen_latents[digit, j] = avg_latents[digit] + std_latents[digit] * rand_j

    gen_images = decoder.predict(gen_latents.reshape(10 * N_SAMPLES, latent_dim))
    gen_images = gen_images.reshape(10, N_SAMPLES, pic_h, pic_w)

    fig, axes = plt.subplots(10, N_SAMPLES, figsize=(N_SAMPLES * 1.6, 10 * 1.6))
    fig.suptitle("Task 5: Generated Images (AvgLatent + StdDev ⊙ rand)")
    for digit in range(10):
        for j in range(N_SAMPLES):
            axes[digit, j].imshow(gen_images[digit, j], cmap=plt.cm.binary)
            axes[digit, j].axis("off")
        axes[digit, 0].set_ylabel(str(digit), fontsize=11, rotation=0, labelpad=15, va="center")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    MNIST_DAE()
