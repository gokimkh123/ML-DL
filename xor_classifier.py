import tensorflow as tf
from tensorflow.keras import Input, layers, Model, optimizers, activations


def xor_classifier_example():
    input_data = tf.constant([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    output_xor = tf.constant([0.0, 1.0, 1.0, 0.0])

    num_input_nodes = 2
    num_output_nodes = 1

    batch_size = 1
    epochs = 300
    nodes_in_hidden_layers = [10, 10]
    lr = 0.1
    loss_type = "mse"

    input_layer = Input(shape=(num_input_nodes,))

    hidden_layers = input_layer
    for num_hidden_nodes in nodes_in_hidden_layers:
        hidden_layers = layers.Dense(units=num_hidden_nodes, activation="relu", use_bias=True)(hidden_layers)

    output_layer = layers.Dense(units=num_output_nodes, activation="sigmoid", use_bias=True)(hidden_layers)


    xor_model = Model(inputs=input_layer, outputs=output_layer)

    sgd = optimizers.SGD(learning_rate=lr)
    xor_model.compile(optimizer=sgd, loss=loss_type)

    xor_model.fit(x=input_data, y=output_xor, batch_size=batch_size, epochs=epochs)


    prediction = xor_model.predict(x=input_data, batch_size=batch_size)

    input_and_result = zip(input_data, prediction)
    print("=========MLP XOR classifier result =============")

    for x, predicted in input_and_result:
        in_val = x.numpy()

        if predicted > 0.5:
            print("%d XOR %d => %.2f => 1" % (in_val[0], in_val[1], predicted[0]))
        else:
            print("%d XOR %d => %.2f => 0" % (in_val[0], in_val[1], predicted[0]))



if __name__ == "__main__":
    xor_classifier_example()