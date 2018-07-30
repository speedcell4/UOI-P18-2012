import keras
import keras.backend as K
from keras_wc_embd import get_embedding_layer


def _loss(layers, input_dim, rnn_units, lmbd=0.01):
    """Generate loss function.

    :param layers: Parallel RNN layers.
    :param input_dim: The input dimension of RNN layers.
    :param rnn_units: Unit size for each RNN.
    :param lmbd: A constant controlling the weights of losses.

    :return loss: The loss function.
    """
    def __loss(y_true, y_pred):
        kernel_cs = []
        for layer in layers:
            kernel_c = layer.cell.get_weights()[0][:, rnn_units * 2: rnn_units * 3]
            kernel_cs.append(kernel_c.reshape((input_dim * rnn_units,)))
        phi = K.stack(kernel_cs)
        loss_sim = K.sum(K.dot(phi, K.transpose(phi)) - K.eye(len(layers)))
        loss_cat = K.sum(y_true * K.log(y_pred))
        return loss_cat + lmbd * loss_sim
    return __loss


def build_model(rnn_num,
                rnn_units,
                word_dict_len,
                char_dict_len,
                max_word_len,
                output_dim,
                word_dim=100,
                char_dim=100,
                char_embd_dim=100):
    """Build model for NER.

    :param rnn_num: Number of parallel RNNs.
    :param rnn_units: Unit size for each RNN.
    :param word_dict_len: The number of words in the dictionary.
    :param char_dict_len: The numbers of characters in the dictionary.
    :param max_word_len: The maximum length of a word in the dictionary.
    :param output_dim: The output dimension / number of NER types.
    :param word_dim: The dimension of word embedding.
    :param char_dim: The final dimension of character embedding.
    :param char_embd_dim: The embedding dimension of characters before bidirectional RNN.

    :return model: The built model.
    """
    inputs, embd_layer = get_embedding_layer(
        word_dict_len=word_dict_len,
        char_dict_len=char_dict_len,
        max_word_len=max_word_len,
        word_embd_dim=word_dim,
        char_hidden_dim=char_dim // 2,
        char_embd_dim=char_embd_dim,
    )
    rnns, layers = [], []
    for i in range(rnn_num):
        lstm_layer = keras.layers.LSTM(
            units=rnn_units,
            return_sequences=True,
            name='LSTM_%d' % (i + 1),
        )
        layers.append(lstm_layer)
        rnns.append(lstm_layer(embd_layer))
    concat_layer = keras.layers.Concatenate(name='Concatenation')(rnns)
    dense_layer = keras.layers.Dense(units=output_dim, activation='softmax', name='Dense')(concat_layer)
    model = keras.models.Model(inputs=inputs, outputs=dense_layer)
    loss = _loss(layers, word_dim + char_dim, rnn_units)
    model.compile(
        optimizer='adam',
        loss=loss,
        acc='categorical_accuracy'
    )
    return model

