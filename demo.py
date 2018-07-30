from model import build_model

model = build_model(rnn_num=16,
                    rnn_units=64,
                    char_dict_len=129,
                    word_dict_len=59,
                    max_word_len=20,
                    output_dim=3)
model.summary()
