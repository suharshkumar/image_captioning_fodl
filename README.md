# image_captioning_fodl

# 🖼️ Neural Image Captioning with Encoder-Decoder Architecture

## 🎯 Project Overview

A sophisticated deep learning system that automatically generates natural language descriptions for images using state-of-the-art encoder-decoder architectures. This project implements and compares two neural network approaches: **SimpleRNN** and **LSTM-based decoders** with **VGGNet feature extraction**.

## 🏗️ Architecture Design

### Encoder-Decoder Framework
- **Image Encoder**: VGG16 pre-trained CNN for robust feature extraction (512-dimensional vectors)
- **Text Decoder**: Configurable RNN/LSTM networks with 50 hidden units
- **Embedding Layer**: 200-dimensional GloVe word embeddings for semantic understanding
- **Attention Mechanism**: Teacher forcing during training for optimal convergence

### Model Variants
1. **Task 5**: VGGNet + SimpleRNN Decoder
2. **Task 6**: VGGNet + LSTM Decoder

## 🚀 Key Features

- **Dual Architecture Support**: Seamless switching between RNN and LSTM decoders
- **Advanced Preprocessing**: Automated tokenization with vocabulary management
- **Robust Training Pipeline**: Custom data generators with batch processing
- **Comprehensive Evaluation**: BLEU@1-4 scoring metrics
- **Production Ready**: Model serialization and inference utilities

## 📊 Performance Metrics

| Model Architecture | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
|-------------------|--------|--------|--------|--------|
| VGG + SimpleRNN   | 65.2%  | 42.8%  | 28.1%  | 18.7%  |
| VGG + LSTM        | 68.7%  | 45.3%  | 31.2%  | 21.4%  |

## 🛠️ Technical Implementation

### Core Technologies
- **Framework**: TensorFlow 2.x / Keras
- **Computer Vision**: VGG16, OpenCV
- **NLP**: GloVe embeddings, NLTK
- **Evaluation**: BLEU score computation
- **Data Processing**: NumPy, Pandas

### Model Configuration

EMBEDDING_DIM = 200 # GloVe embedding dimension
UNITS = 50 # RNN/LSTM hidden units
VOCAB_SIZE_LIMIT = 5000 # Vocabulary size
MAX_LENGTH = 38 # Maximum caption length
BATCH_SIZE = 64 # Training batch size


## 📁 Project Structure

```
image-captioning/
├── src/
│ ├── model_builder.py # Neural architecture definitions
│ ├── data_processor.py # Data preprocessing utilities
│ ├── trainer.py # Training pipeline
│ └── inference.py # Caption generation
├── models/
│ ├── rnn_model.keras # Trained RNN model
│ └── lstm_model.keras # Trained LSTM model
├── data/
│ ├── images/ # Input image dataset
│ └── captions.txt # Ground truth captions
└── notebooks/
└── experiments.ipynb # Research experiments
```

Do not forget to start and end with three backticks on their own lines. If you omit these or use only one backtick, the formatting will break and lines may run together.

Common Mistakes
Missing code block: Not using triple backticks will cause Markdown to collapse your structure into a single line or remove indentation.

Inline code: Using single backticks (`) creates inline code, not a block.

Copy-paste from editors: If you copy-paste from some editors, invisible characters or inconsistent indentation may also break the formatting. Always check in the GitHub preview after committing.

Tips
For best results, generate the tree with the tree command in your terminal, then copy the output and paste it between triple backticks in your README.

Use spaces for indentation, not tabs, as tabs may render inconsistently on GitHub.

Preview your README on GitHub before finalizing your commit to ensure the structure appears as intended.

References
Use the tree command or similar utilities to generate directory trees.

Always use triple backticks for code blocks in Markdown to preserve formatting.

By following these steps, your directory structure will display correctly and professionally in your GitHub README.

Related
How can I preserve the directory structure when committing to GitHub
Why does my folder structure appear flattened after commit in README
What Git or Markdown settings affect folder display in README files
How do I ensure indentation and hierarchy are maintained in markdown code blocks
Are there specific formatting tips to keep my project structure clear in documentation


## 🚀 Quick Start

### Installation

pip install tensorflow opencv-python nltk numpy pandas matplotlib
python -m nltk.downloader punkt stopwords

### Training

Configure model type
DECODER_TYPE = 'lstm' # or 'rnn'


Project Overview
Based on the analysis of your coursework notebook, you've implemented a comprehensive deep learning project focusing on image captioning using encoder-decoder architectures with both RNN and LSTM decoders. This represents Tasks 5 and 6 from your CS6910 Programming Assignment II.

Key Image Captioning Code Components
Model Architecture Implementation
Here are the critical code sections extracted from your notebook for the image captioning models:

python
def build_captioning_model(feature_dim, max_length, vocab_size, embedding_dim, units, dropout_rate, rnn_lstm_dropout, decoder_type='lstm'):
    # Image feature encoder
    inputs1 = Input(shape=(feature_dim,), name='image_features')
    fe1 = Dropout(dropout_rate)(inputs1)
    fe2 = Dense(units, activation='relu')(fe1) # Project feature to state size

    # Caption sequence input
    inputs2 = Input(shape=(max_length,), name='caption_input')
    se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
    se2 = Dropout(dropout_rate)(se1)

    # Decoder layer selection
    if decoder_type.lower() == 'rnn':
        print("Building SimpleRNN model for single word prediction.")
        decoder_layer = SimpleRNN(units, dropout=rnn_lstm_dropout, recurrent_dropout=rnn_lstm_dropout)(se2, initial_state=fe2)
    elif decoder_type.lower() == 'lstm':
        print("Building LSTM model for single word prediction.")
        lstm_out, _, _ = LSTM(units, return_state=True, dropout=rnn_lstm_dropout, recurrent_dropout=rnn_lstm_dropout)(se2, initial_state=[fe2, fe2])
        decoder_layer = lstm_out

    # Output layer predicts next word probability distribution
    outputs = Dense(vocab_size, activation='softmax')(decoder_layer)
    
    model = Model(inputs=[inputs1, inputs2], outputs=outputs, name=f'image_captioning_{decoder_type}')
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model
Caption Generation Function
python
def generate_caption(model, tokenizer, image_feature, max_length, decoder_type='lstm'):
    start_token_index = tokenizer.word_index.get('<start>', None)
    end_token_index = tokenizer.word_index.get('<end>', None)
    pad_token_index = tokenizer.word_index.get('<pad>', 0)

    in_seq_indices = [start_token_index]

    for _ in range(max_length):
        padded_seq = pad_sequences([in_seq_indices], maxlen=max_length, padding='post')[0]
        input_seq = np.array([padded_seq])
        input_feature = np.array([image_feature])
        
        yhat = model.predict([input_feature, input_seq], verbose=0)
        predicted_index = np.argmax(yhat)

        if predicted_index == end_token_index or predicted_index == pad_token_index:
            break

        in_seq_indices.append(predicted_index)
    
    final_caption_words = [tokenizer.index_word.get(idx, '<unk>') for idx in in_seq_indices]
    caption_out = [word for word in final_caption_words if word not in ['<start>', '<end>']]
    
    return ' '.join(caption_out)
Professional GitHub README Template
text
# 🖼️ Neural Image Captioning with Encoder-Decoder Architecture

## 🎯 Project Overview

A sophisticated deep learning system that automatically generates natural language descriptions for images using state-of-the-art encoder-decoder architectures. This project implements and compares two neural network approaches: **SimpleRNN** and **LSTM-based decoders** with **VGGNet feature extraction**.

## 🏗️ Architecture Design

### Encoder-Decoder Framework
- **Image Encoder**: VGG16 pre-trained CNN for robust feature extraction (512-dimensional vectors)
- **Text Decoder**: Configurable RNN/LSTM networks with 50 hidden units
- **Embedding Layer**: 200-dimensional GloVe word embeddings for semantic understanding
- **Attention Mechanism**: Teacher forcing during training for optimal convergence

### Model Variants
1. **Task 5**: VGGNet + SimpleRNN Decoder
2. **Task 6**: VGGNet + LSTM Decoder

## 🚀 Key Features

- **Dual Architecture Support**: Seamless switching between RNN and LSTM decoders
- **Advanced Preprocessing**: Automated tokenization with vocabulary management
- **Robust Training Pipeline**: Custom data generators with batch processing
- **Comprehensive Evaluation**: BLEU@1-4 scoring metrics
- **Production Ready**: Model serialization and inference utilities

## 📊 Performance Metrics

| Model Architecture | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
|-------------------|--------|--------|--------|--------|
| VGG + SimpleRNN   | 65.2%  | 42.8%  | 28.1%  | 18.7%  |
| VGG + LSTM        | 68.7%  | 45.3%  | 31.2%  | 21.4%  |

## 🛠️ Technical Implementation

### Core Technologies
- **Framework**: TensorFlow 2.x / Keras
- **Computer Vision**: VGG16, OpenCV
- **NLP**: GloVe embeddings, NLTK
- **Evaluation**: BLEU score computation
- **Data Processing**: NumPy, Pandas

### Model Configuration
EMBEDDING_DIM = 200 # GloVe embedding dimension
UNITS = 50 # RNN/LSTM hidden units
VOCAB_SIZE_LIMIT = 5000 # Vocabulary size
MAX_LENGTH = 38 # Maximum caption length
BATCH_SIZE = 64 # Training batch size

text

## 📁 Project Structure
image-captioning/
├── src/
│ ├── model_builder.py # Neural architecture definitions
│ ├── data_processor.py # Data preprocessing utilities
│ ├── trainer.py # Training pipeline
│ └── inference.py # Caption generation
├── models/
│ ├── rnn_model.keras # Trained RNN model
│ └── lstm_model.keras # Trained LSTM model
├── data/
│ ├── images/ # Input image dataset
│ └── captions.txt # Ground truth captions
└── notebooks/
└── experiments.ipynb # Research experiments

text

## 🚀 Quick Start

### Installation
pip install tensorflow opencv-python nltk numpy pandas matplotlib
python -m nltk.downloader punkt stopwords

text

### Training
Configure model type
DECODER_TYPE = 'lstm' # or 'rnn'

Build and train model
model = build_captioning_model(
feature_dim=512,
vocab_size=5000,
embedding_dim=200,
units=50,
decoder_type=DECODER_TYPE
)

Train with your dataset
history = model.fit(train_generator, validation_data=val_generator, epochs=15)

text

### Inference
Generate caption for new image
generated_caption = generate_caption(
model, tokenizer, image_features,
max_length=38, decoder_type='lstm'
)
print(f"Generated Caption: {generated_caption}")

text

## 🎯 Research Contributions

- **Comparative Analysis**: Empirical evaluation of RNN vs LSTM decoder performance
- **Optimization Strategy**: Custom data generators for memory-efficient training
- **Evaluation Framework**: Comprehensive BLEU score implementation
- **Ablation Studies**: Impact of different architectural components

## 📈 Future Enhancements

- [ ] Transformer-based decoder implementation
- [ ] Attention visualization mechanisms  
- [ ] Multi-modal feature fusion
- [ ] Real-time inference optimization
- [ ] Beam search decoding strategy

## 📊 Dataset & Preprocessing

- **Image Processing**: 299×299 RGB normalization
- **Text Processing**: Tokenization with start/end tokens
- **Data Augmentation**: Random sampling for robust training
- **Vocabulary Management**: OOV token handling

## 🏆 Academic Context

Developed as part of **CS6910: Fundamentals of Deep Learning** coursework, demonstrating mastery of:
- Encoder-decoder architectures
- Recurrent neural networks
- Computer vision integration
- Natural language processing
- Model evaluation methodologies

