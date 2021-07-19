# Building the NMT (Neural Machine Translation) model

from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, RepeatVector, TimeDistributed

def Seq2seqModel(input_vocab_size, output_vocab_size, input_maxwords, output_maxwords, hidden_units):
    
    model = Sequential()
    
    # Encoder Part...
    embedding_size = 512
    model.add(Embedding(input_vocab_size, embedding_size, input_length=input_maxwords, mask_zero=True))
    model.add(LSTM(hidden_units))
    
    # Fitting the encoder and decoder parts together by adjusting dimensions...
    model.add(RepeatVector(output_maxwords))
    
    # Decoder Part...
    model.add(LSTM(hidden_units, return_sequences=True))
    model.add(Dense(output_vocab_size, activation='softmax'))
    
    return model