### Chatbot Training and Inference
This project trains a chatbot using a sequence-to-sequence model with TensorFlow and Keras. The chatbot can generate responses to user inputs based on a provided dataset.

### Features
- Data Preparation: Import, tokenize, and pad input/output sequences.
- Model Architecture: Encoder-decoder LSTM with embeddings.
- Training: Train the model and save it for future use.
- Inference: Generate responses using the trained model.

## Usage
### Data Preparation
- Ensure your dataset is in a tab-separated text file with input and output pairs.
- Prepare the input and output data:

```python
import os
import numpy as np
import pandas as pd
from tensorflow.keras import preprocessing, utils

data_path1 = 'chatbot-dataset.txt'
input_texts, target_texts = [], []
with open(data_path1) as f:
    lines = f.read().split('\n')
for line in lines[: min(600, len(lines) - 1)]:
    input_text, target_text = line.split('\t')
    input_texts.append(input_text)
    target_texts.append(target_text)

lines_df = pd.DataFrame(list(zip(input_texts, target_texts)), columns=['input', 'output'])
```
## Tokenization and Padding
### Prepare the input and output data for the encoder and decoder
 ```python
# Tokenizing input data
tokenizer = preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(input_texts)
tokenized_input_lines = tokenizer.texts_to_sequences(input_texts)
max_input_length = max(len(seq) for seq in tokenized_input_lines)
padded_input_lines = preprocessing.sequence.pad_sequences(tokenized_input_lines, maxlen=max_input_length, padding='post')
encoder_input_data = np.array(padded_input_lines)
num_input_tokens = len(tokenizer.word_index) + 1

# Tokenizing output data
tokenizer.fit_on_texts(['<START> ' + text + ' <END>' for text in target_texts])
tokenized_output_lines = tokenizer.texts_to_sequences(['<START> ' + text + ' <END>' for text in target_texts])
max_output_length = max(len(seq) for seq in tokenized_output_lines)
padded_output_lines = preprocessing.sequence.pad_sequences(tokenized_output_lines, maxlen=max_output_length, padding='post')
decoder_input_data = np.array(padded_output_lines)
num_output_tokens = len(tokenizer.word_index) + 1

# Preparing target data for decoder
decoder_target_data = np.zeros((len(target_texts), max_output_length, num_output_tokens))
for i, seq in enumerate(tokenized_output_lines):
    for t, token in enumerate(seq):
        if t > 0:
            decoder_target_data[i, t - 1, token] = 1
```

## Model Definition and Training
### Define and train the model
 ```python
from tensorflow.keras import layers, models

encoder_inputs = layers.Input(shape=(None,))
encoder_embedding = layers.Embedding(num_input_tokens, 256, mask_zero=True)(encoder_inputs)
encoder_outputs, state_h, state_c = layers.LSTM(256, return_state=True, recurrent_dropout=0.2, dropout=0.2)(encoder_embedding)
encoder_states = [state_h, state_c]

decoder_inputs = layers.Input(shape=(None,))
decoder_embedding = layers.Embedding(num_output_tokens, 256, mask_zero=True)(decoder_inputs)
decoder_lstm = layers.LSTM(256, return_sequences=True, return_state=True, recurrent_dropout=0.2, dropout=0.2)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = layers.Dense(num_output_tokens, activation='softmax')
output = decoder_dense(decoder_outputs)

model = models.Model([encoder_inputs, decoder_inputs], output)
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=124, epochs=500)
model.save('chatbot_model.h5')
 ```
## Inference
### Define inference models and generate responses
 ```python
def make_inference_models():
    encoder_model = models.Model(encoder_inputs, encoder_states)

    decoder_state_input_h = layers.Input(shape=(256,))
    decoder_state_input_c = layers.Input(shape=(256,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_embedding, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    
    decoder_model = models.Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    return encoder_model, decoder_model

enc_model, dec_model = make_inference_models()

def str_to_tokens(sentence: str):
    words = sentence.lower().split()
    tokens_list = [input_word_dict[word] for word in words]
    return preprocessing.sequence.pad_sequences([tokens_list], maxlen=max_input_length, padding='post')

for epoch in range(encoder_input_data.shape[0]):
    states_values = enc_model.predict(str_to_tokens(input('User: ')))
    empty_target_seq = np.zeros((1, 1))
    empty_target_seq[0, 0] = output_word_dict['start']
    stop_condition = False
    decoded_translation = ''
    while not stop_condition:
        dec_outputs, h, c = dec_model.predict([empty_target_seq] + states_values)
        sampled_word_index = np.argmax(dec_outputs[0, -1, :])
        sampled_word = output_word_dict[sampled_word_index]
        decoded_translation += f' {sampled_word}'

        if sampled_word == 'end' or len(decoded_translation.split()) > max_output_length:
            stop_condition = True

        empty_target_seq = np.zeros((1, 1))
        empty_target_seq[0, 0] = sampled_word_index
        states_values = [h, c]

    print(f'Bot: {decoded_translation.replace(" end", "")}\n')

```







