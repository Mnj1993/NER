import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
import spacy
from spacy import displacy

# Load dataset
data = pd.read_csv('ner_dataset.csv', encoding='unicode_escape')

# Get dictionary mappings for tokens and tags
def get_dict_map(data, token_or_tag):
    vocab = list(set(data[token_or_tag].to_list()))
    idx2tok = {idx: tok for idx, tok in enumerate(vocab)}
    tok2idx = {tok: idx for idx, tok in enumerate(vocab)}
    return tok2idx, idx2tok

token2idx, idx2token = get_dict_map(data, 'Word')
tag2idx, idx2tag = get_dict_map(data, 'Tag')

data['Word_idx'] = data['Word'].map(token2idx)
data['Tag_idx'] = data['Tag'].map(tag2idx)

# Group and preprocess data
data_fillna = data.fillna(method='ffill', axis=0)
data_group = data_fillna.groupby(['Sentence #'], as_index=False)['Word', 'POS', 'Tag', 'Word_idx', 'Tag_idx'].agg(lambda x: list(x))

# Split dataset into train, test, and validation sets
def get_pad_train_test_val(data_group, data):
    n_token = len(list(set(data['Word'].to_list())))
    n_tag = len(list(set(data['Tag'].to_list())))

    tokens = pad_sequences(data_group['Word_idx'].tolist(), maxlen=maxlen, dtype='int32', padding='post', value=n_token - 1)
    tags = pad_sequences(data_group['Tag_idx'].tolist(), maxlen=maxlen, dtype='int32', padding='post', value=tag2idx["O"])
    tags = [to_categorical(i, num_classes=n_tag) for i in tags]

    tokens_, test_tokens, tags_, test_tags = train_test_split(tokens, tags, test_size=0.1, train_size=0.9, random_state=2020)
    train_tokens, val_tokens, train_tags, val_tags = train_test_split(tokens_, tags_, test_size=0.25, train_size=0.75, random_state=2020)

    return train_tokens, val_tokens, test_tokens, train_tags, val_tags, test_tags

train_tokens, val_tokens, test_tokens, train_tags, val_tags, test_tags = get_pad_train_test_val(data_group, data)

# Build BiLSTM model
def get_bilstm_lstm_model():
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_length))
    model.add(Bidirectional(LSTM(units=output_dim, return_sequences=True, dropout=0.2, recurrent_dropout=0.2), merge_mode='concat'))
    model.add(LSTM(units=output_dim, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))
    model.add(TimeDistributed(Dense(n_tags, activation="relu")))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

# Train model
def train_model(X, y, model):
    loss = list()
    for i in range(25):
        hist = model.fit(X, y, batch_size=1000, verbose=1, epochs=1, validation_split=0.2)
        loss.append(hist.history['loss'][0])
    return loss

model_bilstm_lstm = get_bilstm_lstm_model()
results['with_add_lstm'] = train_model(train_tokens, np.array(train_tags), model_bilstm_lstm)

# Load spaCy NER model
nlp = spacy.load('en_core_web_sm')

# Perform NER on sample text
text = nlp('Hi, My name is Manoj N \n I am from USA \n I want to be a excellent Data Scientist')
displacy.render(text, style='ent', jupyter=True)
