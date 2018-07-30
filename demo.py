import os
import codecs
import numpy
import keras
from keras_wc_embd import get_dicts_generator, get_batch_input
from model import build_model


MODEL_PATH = 'model.h5'

DATA_ROOT = 'dataset/CoNNL2003eng'
DATA_TRAIN_PATH = os.path.join(DATA_ROOT, 'train.txt')
DATA_VALID_PATH = os.path.join(DATA_ROOT, 'valid.txt')
DATA_TEST_PATH = os.path.join(DATA_ROOT, 'test.txt')

RNN_NUM = 16
RNN_UNITS = 32

BATCH_SIZE = 32
EPOCHS = 1

TAGS = {
    'O': 0,
    'B-PER': 1,
    'I-PER': 2,
    'B-LOC': 3,
    'I-LOC': 4,
    'B-ORG': 5,
    'I-ORG': 6,
    'B-MISC': 7,
    'I-MISC': 8,
}


def load_data(path):
    sentences, taggings = [], []
    with codecs.open(path, 'r', 'utf8') as reader:
        for line in reader:
            line = line.strip()
            if not line:
                if not sentences or len(sentences[-1]) > 0:
                    sentences.append([])
                    taggings.append([])
                continue
            parts = line.split()
            if parts[0] != '-DOCSTART-':
                sentences[-1].append(parts[0])
                taggings[-1].append(TAGS[parts[-1]])
    if not sentences[-1]:
        sentences.pop()
        taggings.pop()
    return sentences, taggings


train_sentences, train_taggings = load_data(DATA_TRAIN_PATH)
valid_sentences, valid_taggings = load_data(DATA_VALID_PATH)

dicts_generator = get_dicts_generator(
    word_min_freq=2,
    char_min_freq=2,
    word_ignore_case=True,
    char_ignore_case=False
)
for sentence in train_sentences:
    dicts_generator(sentence)
word_dict, char_dict, max_word_len = dicts_generator(return_dict=True)

train_steps = (len(train_sentences) + BATCH_SIZE - 1) // BATCH_SIZE
valid_steps = (len(valid_sentences) + BATCH_SIZE - 1) // BATCH_SIZE


def batch_generator(sentences, taggings, steps):
    global word_dict, char_dict, max_word_len
    while True:
        for i in range(steps):
            batch_sentences = sentences[BATCH_SIZE * i:min(BATCH_SIZE * (i + 1), len(sentences))]
            batch_taggings = taggings[BATCH_SIZE * i:min(BATCH_SIZE * (i + 1), len(taggings))]
            word_input, char_input = get_batch_input(
                batch_sentences,
                max_word_len,
                word_dict,
                char_dict,
                word_ignore_case=True,
                char_ignore_case=False
            )
            sentence_len = word_input.shape[1]
            for j in range(len(batch_taggings)):
                batch_taggings[j] = batch_taggings[j] + [0] * (sentence_len - len(batch_taggings[j]))
            batch_taggings = keras.utils.to_categorical(numpy.asarray(batch_taggings), len(TAGS))
            yield [word_input, char_input], batch_taggings


model = build_model(rnn_num=RNN_NUM,
                    rnn_units=RNN_UNITS,
                    word_dict_len=len(word_dict),
                    char_dict_len=len(char_dict),
                    max_word_len=max_word_len,
                    output_dim=len(TAGS))
model.summary()

if os.path.exists(MODEL_PATH):
    model.load_weights(MODEL_PATH, by_name=True)

model.fit_generator(
    generator=batch_generator(train_sentences, train_taggings, train_steps),
    steps_per_epoch=train_steps,
    epochs=EPOCHS,
    validation_data=batch_generator(valid_sentences, valid_taggings, valid_steps),
    validation_steps=valid_steps,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=2),
        keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=3),
    ],
    verbose=True,
)

model.save_weights(MODEL_PATH)

test_sentences, test_taggings = load_data(DATA_TEST_PATH)
test_steps = (len(valid_sentences) + BATCH_SIZE - 1) // BATCH_SIZE

model.evaluate_generator(
    generator=batch_generator(test_sentences, test_taggings, test_steps),
    steps=test_steps,
    verbose=True,
)
