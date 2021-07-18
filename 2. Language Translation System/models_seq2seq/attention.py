# Instead of importing from 'keras', use 'tensorflow.python.keras'
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Embedding, Attention, AdditiveAttention, Concatenate, TimeDistributed, Dot, Activation
from tensorflow.keras import Model, Input

def AttentionModel(input_vocab_size, output_vocab_size, input_maxwords, output_maxwords, hidden_units):
     
    embedding_size = 300
    
    # Encoder input Embedding...
    encoder_input = Input(shape = (input_maxwords), name = 'encoder_input') 
    encoder_embed_input = Embedding(input_vocab_size, embedding_size, input_length=input_maxwords, mask_zero=True)(encoder_input)

    # Encoder part... 
    encoder_lstm = LSTM(hidden_units, return_sequences=True, return_state=True, name='encoder_lstm')
    encoder_output, state_h, state_c = encoder_lstm(encoder_embed_input)
    encoder_state = [state_h, state_c]
    
    # Decoder input Embedding - Teacher Forcing inputs instead of previous timestep's inputs...
    decoder_input = Input(shape = (output_maxwords), name = 'decoder_input') 
    decoder_embed_input = Embedding(output_vocab_size, embedding_size, input_length=output_maxwords, mask_zero=True)(decoder_input)

    decoder_lstm = LSTM(hidden_units, return_sequences=True, name='decoder_lstm')
    decoder_output = decoder_lstm(decoder_embed_input, initial_state=encoder_state)
    

    # Attention Layer...Luong's Attention Mechanism...
    attention_value_e = Dot(axes=[2, 2], name='attention_score_e')([decoder_output, encoder_output])  # using 'Dot product' -----> (1 of 3 available methods mentioned in Luong's paper)
    attention_weights = Activation('softmax', name='attention_vector')(attention_value_e)
    context_vector = Dot(axes=[2,1], name='context_vector')([attention_weights, encoder_output]) 

    # context_vector = Attention(name='attention_layer')([decoder_output, encoder_output])         # Dot-product attention layer, a.k.a. Luong-style attention (keras)
    # context_vector = AdditiveAttention(name='attention_layer')([decoder_output, encoder_output]) # Additive attention layer, a.k.a. Bahdanau-style attention (keras)
    
    # Inputs for the FC layer present on top of the Decoder part...
    decoder_combined_context = Concatenate(axis=-1)([context_vector, decoder_output])
    # Hidden layer of the FC layer with tanh as activation funtion...
    output = TimeDistributed(Dense(hidden_units, activation="tanh"))(decoder_combined_context)
    
    
    # Final output (Dense) layer of the FC layer...
    dense = Dense(output_vocab_size, activation='softmax', name='output_dense_layer')
    decoder_pred = dense(output)
    
    # Initializing Model...
    model = Model(inputs=[encoder_input, decoder_input], outputs=[decoder_pred])
    
    return model


### 3 Methods to calculate score mentioned in Luong's paper --->this score is then passed into softmax layer to calculate the weights
# Method 1: Dot product of decoder_output and encoder_output
# Method 2: decoder_output * some weigths matrix (Wa) * encoder_output
# Method 3: Wa * [decoder_output, encoder_output] ---> Concatenation method