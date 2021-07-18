# Building the NMT (Neural Machine Translation) model from scratch (i.e. connecting layers manually)

from tensorflow.python.keras.layers import Dense, LSTM, Embedding, RepeatVector
from tensorflow.keras import Model, Input

def SimpleNMTModel(input_vocab_size, output_vocab_size, input_maxwords, output_maxwords, hidden_units, teacher_forcing):
     
    embedding_size = 512
    
    # Embedding the inputs...
    encoder_input = Input(shape = (input_maxwords), name = 'encoder_input') 
    encoder_embed_input = Embedding(input_vocab_size, embedding_size, input_length=input_maxwords, mask_zero=True)(encoder_input)
    
    # Encoder Part...
    encoder_lstm = LSTM(hidden_units, return_state=True, name='encoder_lstm')
    encoder_output, state_h, state_c = encoder_lstm(encoder_embed_input)
    encoder_state = [state_h, state_c]
    
    # Fitting the encoder and decoder parts together by usig the hidden state value as decoder's input for each cell...
    if (teacher_forcing):
        input_decoder = Input(shape = (output_maxwords), name = 'decoder_input') 
        decoder_input = Embedding(output_vocab_size, embedding_size, input_length=output_maxwords, mask_zero=True)(input_decoder)
    else:
        decoder_input = RepeatVector(n = output_maxwords)(encoder_output)


    # Decoder Part...
    decoder_lstm = LSTM(hidden_units, return_sequences=True, name='decoder_lstm')
    decoder_output = decoder_lstm(decoder_input, initial_state=encoder_state)
    
    # Dense layer on top of Decoder Part...
    dense = Dense(output_vocab_size, activation='softmax', name='output_dense_layer')
    decoder_pred = dense(decoder_output)
    
    # Initializing Model...
    if (teacher_forcing):
        model = Model(inputs=[encoder_input, input_decoder], outputs=decoder_pred)
    else:
        model = Model(inputs=[encoder_input], outputs=decoder_pred)
    
    return model

    