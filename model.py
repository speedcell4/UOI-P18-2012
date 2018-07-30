import keras
from keras_wc_embd import get_embedding_layer


def build_model(rnn_num,
                rnn_units,
                word_dict_len,
                char_dict_len,
                max_word_len,
                output_dim,
                word_dim=100,
                char_dim=100,
                char_embd_dim=100):
    inputs, embd_layer = get_embedding_layer(
        word_dict_len=word_dict_len,
        char_dict_len=char_dict_len,
        max_word_len=max_word_len,
        word_embd_dim=word_dim,
        char_hidden_dim=char_dim // 2,
        char_embd_dim=char_embd_dim,
    )
    rnns, kernels = [], []
    for i in range(rnn_num):
        lstm_layer = keras.layers.LSTM(
            units=rnn_units,
            return_sequences=True,
            name='LSTM_%d' % (i + 1),
        )
        rnns.append(lstm_layer(embd_layer))
        kernels.append(lstm_layer.cell.get_weights()[0][:, rnn_units * 2: rnn_units * 3])
    concat_layer = keras.layers.Concatenate(name='Concatenation')(rnns)
    dense_layer = keras.layers.Dense(units=output_dim, activation='softmax', name='Dense')(concat_layer)
    model = keras.models.Model(inputs=inputs, outputs=dense_layer)
    return model

