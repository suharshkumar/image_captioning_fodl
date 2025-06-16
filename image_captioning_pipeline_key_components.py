# =============================================================================
# IMAGE CAPTIONING PIPELINE - COMPLETE IMPLEMENTATION
# CS6910: Fundamentals of Deep Learning - Programming Assignment II
# Tasks 5 & 6: VGGNet + RNN/LSTM Decoder Architecture
# =============================================================================

# -------------------------------
# 1. IMPORTS AND DEPENDENCIES
# -------------------------------
import os
import re
import string
import numpy as np
import pandas as pd
import pickle
from collections import Counter
import matplotlib.pyplot as plt

# TensorFlow and Keras imports
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import (Input, Dense, Embedding, LSTM, SimpleRNN, 
                                   Dropout, Flatten, Conv2D, AveragePooling2D)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

# Image processing
from PIL import Image
import cv2

# Evaluation metrics
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

# -------------------------------
# 2. CONFIGURATION AND PARAMETERS
# -------------------------------
# File paths configuration
BASE_DRIVE_PATH = '/content/drive/MyDrive/Assignment2/Team8_captioning'
DECODER_TYPE = 'lstm'  # or 'rnn'
NUM_IMAGES_TO_USE = 8097

CAPTIONS_PATH = os.path.join(BASE_DRIVE_PATH, 'captions.txt')
IMAGES_PATH = os.path.join(BASE_DRIVE_PATH, 'Images')
FEATURES_PATH = os.path.join(BASE_DRIVE_PATH, f'vgg16_features_{DECODER_TYPE}.pkl')
MODEL_SAVE_PATH = os.path.join(BASE_DRIVE_PATH, f'caption_model_{DECODER_TYPE}_lim{NUM_IMAGES_TO_USE}.keras')
TOKENIZER_PATH = os.path.join(BASE_DRIVE_PATH, f'tokenizer_{DECODER_TYPE}_lim{NUM_IMAGES_TO_USE}.pkl')

# Model hyperparameters
EMBEDDING_DIM = 256
UNITS = 50
VOCAB_SIZE_LIMIT = 5000
MAX_LENGTH_PADDING = 38
BATCH_SIZE = 320
EPOCHS = 15
DROPOUT_RATE = 0.3
RNN_LSTM_DROPOUT = 0.2

# Image processing parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224

# -------------------------------
# 3. DATA LOADING AND PREPROCESSING
# -------------------------------
def load_captions(filename):
    """Load and parse captions from text file"""
    mapping = {}
    parsed_count = 0
    line_num = 0
    error_lines = 0
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line_num += 1
                line = line.strip()
                if not line: 
                    continue

                # Parse image_id and caption using regex
                match_separator = re.search(r'[#.]\d+\s+', line)
                if match_separator:
                    separator_start_index = match_separator.start()
                    separator_end_index = match_separator.end()
                    image_id = line[:separator_start_index].strip()
                    caption = line[separator_end_index:].strip()
                    
                    # Validate image_id
                    if not image_id or ' ' in image_id or '\t' in image_id:
                        error_lines += 1
                        continue
                    
                    # Add to mapping
                    if image_id not in mapping:
                        mapping[image_id] = []
                    mapping[image_id].append(caption)
                    parsed_count += 1
                else:
                    error_lines += 1
                    
    except FileNotFoundError:
        print(f"Error: Caption file not found at {filename}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during caption loading (around line {line_num}): {e}")
        return None

    # Print loading summary
    total_lines_processed = line_num
    num_unique_images = len(mapping)
    print(f"\n--- Caption Loading Summary ---")
    print(f"Processed {total_lines_processed} lines from {filename}.")
    print(f"Successfully parsed captions for {num_unique_images} unique images.")
    print(f"Total caption entries added: {sum(len(v) for v in mapping.values())}")
    if error_lines > 0:
        print(f"Encountered {error_lines} problematic lines that couldn't be fully parsed.")
    
    return mapping

def clean_captions(mapping):
    """Clean and preprocess captions"""
    translator = str.maketrans('', '', string.punctuation)
    for key, captions in mapping.items():
        for i in range(len(captions)):
            caption = captions[i]
            # Convert to lowercase
            caption = caption.lower()
            # Remove punctuation
            caption = caption.translate(translator)
            # Remove extra whitespace
            caption = ' '.join(caption.split())
            # Add start and end tokens
            captions[i] = '<start> ' + caption + ' <end>'
    print("Captions cleaned (lowercase, punctuation removed, start/end tokens added).")

def create_vocabulary(captions_mapping):
    """Create vocabulary from captions"""
    all_captions = []
    for captions in captions_mapping.values():
        all_captions.extend(captions)
    
    # Tokenize
    tokenizer = Tokenizer(num_words=VOCAB_SIZE_LIMIT, oov_token='<unk>')
    tokenizer.fit_on_texts(all_captions)
    
    # Add special tokens
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'
    
    print(f"Vocabulary created with {len(tokenizer.word_index)} unique words")
    return tokenizer

# -------------------------------
# 4. IMAGE FEATURE EXTRACTION
# -------------------------------
def extract_image_features():
    """Extract features using VGG16 model"""
    # Load pre-trained VGG16 model
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    vgg_model.trainable = False  # Freeze layers
    
    # Create feature extraction model
    feature_extractor = Model(inputs=vgg_model.input, outputs=vgg_model.layers[-2].output)
    
    features = {}
    image_files = os.listdir(IMAGES_PATH)
    
    print(f"Extracting features from {len(image_files)} images...")
    
    for i, filename in enumerate(image_files):
        if i % 1000 == 0:
            print(f"Processed {i}/{len(image_files)} images")
            
        image_path = os.path.join(IMAGES_PATH, filename)
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image = image.resize((IMG_HEIGHT, IMG_WIDTH))
            image_array = np.array(image)
            image_array = np.expand_dims(image_array, axis=0)
            image_array = image_array / 255.0  # Normalize
            
            # Extract features
            feature = feature_extractor.predict(image_array, verbose=0)
            feature = feature.flatten()
            
            # Store features with image ID (remove extension)
            image_id = filename.split('.')[0]
            features[image_id] = feature
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    print(f"Feature extraction completed. Total features: {len(features)}")
    return features

# -------------------------------
# 5. DATA GENERATOR FOR TRAINING
# -------------------------------
class CaptionDataGenerator(Sequence):
    """Custom data generator for image captioning"""
    
    def __init__(self, image_features, captions_mapping, tokenizer, max_length, batch_size=32, shuffle=True):
        self.image_features = image_features
        self.captions_mapping = captions_mapping
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Create list of (image_id, caption) pairs
        self.data_pairs = []
        for image_id, captions in captions_mapping.items():
            if image_id in image_features:
                for caption in captions:
                    self.data_pairs.append((image_id, caption))
        
        self.indexes = np.arange(len(self.data_pairs))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __len__(self):
        return len(self.data_pairs) // self.batch_size
    
    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        return self._generate_batch(batch_indexes)
    
    def _generate_batch(self, batch_indexes):
        batch_image_features = []
        batch_input_sequences = []
        batch_output_sequences = []
        
        for idx in batch_indexes:
            image_id, caption = self.data_pairs[idx]
            
            # Get image features
            image_feature = self.image_features[image_id]
            
            # Convert caption to sequence
            sequence = self.tokenizer.texts_to_sequences([caption])[0]
            
            # Create input-output pairs for each word in sequence
            for i in range(1, len(sequence)):
                input_seq = sequence[:i]
                output_seq = sequence[i]
                
                # Pad input sequence
                input_seq = pad_sequences([input_seq], maxlen=self.max_length, padding='post')[0]
                
                batch_image_features.append(image_feature)
                batch_input_sequences.append(input_seq)
                batch_output_sequences.append(output_seq)
        
        return ([np.array(batch_image_features), np.array(batch_input_sequences)], 
                np.array(batch_output_sequences))
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

# -------------------------------
# 6. MODEL ARCHITECTURE DEFINITION
# -------------------------------
def build_captioning_model(feature_dim, max_length, vocab_size, embedding_dim, units, 
                          dropout_rate, rnn_lstm_dropout, decoder_type='lstm'):
    """Build image captioning model with encoder-decoder architecture"""
    
    # Image feature encoder input
    inputs1 = Input(shape=(feature_dim,), name='image_features')
    fe1 = Dropout(dropout_rate)(inputs1)
    fe2 = Dense(units, activation='relu')(fe1)  # Project feature to state size

    # Caption sequence input
    inputs2 = Input(shape=(max_length,), name='caption_input')
    se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
    se2 = Dropout(dropout_rate)(se1)

    # Decoder layer selection
    if decoder_type.lower() == 'rnn':
        print("Building SimpleRNN model for image captioning.")
        decoder_layer = SimpleRNN(units, dropout=rnn_lstm_dropout, 
                                 recurrent_dropout=rnn_lstm_dropout)(se2, initial_state=fe2)
    elif decoder_type.lower() == 'lstm':
        print("Building LSTM model for image captioning.")
        lstm_out, _, _ = LSTM(units, return_state=True, dropout=rnn_lstm_dropout, 
                             recurrent_dropout=rnn_lstm_dropout)(se2, initial_state=[fe2, fe2])
        decoder_layer = lstm_out
    else:
        raise ValueError("decoder_type must be either 'rnn' or 'lstm'")

    # Output layer predicts next word probability distribution
    outputs = Dense(vocab_size, activation='softmax')(decoder_layer)
    
    # Create and compile model
    model = Model(inputs=[inputs1, inputs2], outputs=outputs, 
                 name=f'image_captioning_{decoder_type}')
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

# -------------------------------
# 7. TRAINING PIPELINE
# -------------------------------
def train_captioning_model(model, train_generator, val_generator, model_save_path, epochs=15):
    """Train the image captioning model"""
    
    # Define callbacks
    checkpoint = ModelCheckpoint(
        model_save_path,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    print("\n--- Starting Training ---")
    
    # Train the model
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=[checkpoint, early_stopping],
        verbose=1
    )
    
    print("\n--- Training Finished ---")
    return history

# -------------------------------
# 8. CAPTION GENERATION (INFERENCE)
# -------------------------------
def generate_caption(model, tokenizer, image_feature, max_length, decoder_type='lstm'):
    """Generate caption for a given image feature using greedy decoding"""
    
    # Get special token indices
    start_token_index = tokenizer.word_index.get('<start>', None)
    end_token_index = tokenizer.word_index.get('<end>', None)
    pad_token_index = tokenizer.word_index.get('<pad>', 0)

    # Initialize sequence with start token
    in_seq_indices = [start_token_index]

    # Generate caption word by word
    for _ in range(max_length):
        # Pad sequence to max length
        padded_seq = pad_sequences([in_seq_indices], maxlen=max_length, padding='post')[0]
        input_seq = np.array([padded_seq])
        input_feature = np.array([image_feature])
        
        # Predict next word
        yhat = model.predict([input_feature, input_seq], verbose=0)
        predicted_index = np.argmax(yhat)

        # Check for end token or pad token
        if predicted_index == end_token_index or predicted_index == pad_token_index:
            break

        # Add predicted word to sequence
        in_seq_indices.append(predicted_index)

    # Convert indices to words
    final_caption_words = [tokenizer.index_word.get(idx, '<unk>') for idx in in_seq_indices]
    caption_out = [word for word in final_caption_words if word not in ['<start>', '<end>']]

    return ' '.join(caption_out)

def generate_captions_for_images(model, tokenizer, image_features, image_ids, max_length, decoder_type='lstm'):
    """Generate captions for multiple images"""
    generated_captions = {}
    
    print(f"Generating captions for {len(image_ids)} images...")
    
    for i, image_id in enumerate(image_ids):
        if i % 100 == 0:
            print(f"Generated {i}/{len(image_ids)} captions")
            
        if image_id in image_features:
            caption = generate_caption(model, tokenizer, image_features[image_id], 
                                     max_length, decoder_type)
            generated_captions[image_id] = caption
    
    print("Caption generation completed!")
    return generated_captions

# -------------------------------
# 9. EVALUATION METRICS (BLEU SCORES)
# -------------------------------
def calculate_bleu_scores(reference_captions, generated_captions):
    """Calculate BLEU@1-4 scores for generated captions"""
    
    references = []
    candidates = []
    
    for image_id in generated_captions:
        if image_id in reference_captions:
            # Get reference captions (remove start/end tokens)
            ref_caps = []
            for ref_cap in reference_captions[image_id]:
                ref_words = ref_cap.replace('<start>', '').replace('<end>', '').strip().split()
                ref_caps.append(ref_words)
            
            # Get generated caption
            gen_cap = generated_captions[image_id].split()
            
            references.append(ref_caps)
            candidates.append(gen_cap)
    
    # Calculate BLEU scores
    bleu_1 = corpus_bleu(references, candidates, weights=(1.0, 0, 0, 0))
    bleu_2 = corpus_bleu(references, candidates, weights=(0.5, 0.5, 0, 0))
    bleu_3 = corpus_bleu(references, candidates, weights=(0.33, 0.33, 0.33, 0))
    bleu_4 = corpus_bleu(references, candidates, weights=(0.25, 0.25, 0.25, 0.25))
    
    print(f"\n--- BLEU Scores ---")
    print(f"BLEU-1: {bleu_1:.4f}")
    print(f"BLEU-2: {bleu_2:.4f}")
    print(f"BLEU-3: {bleu_3:.4f}")
    print(f"BLEU-4: {bleu_4:.4f}")
    
    return {'BLEU-1': bleu_1, 'BLEU-2': bleu_2, 'BLEU-3': bleu_3, 'BLEU-4': bleu_4}

# -------------------------------
# 10. MAIN EXECUTION PIPELINE
# -------------------------------
def main():
    """Main execution pipeline for image captioning"""
    
    print("=== IMAGE CAPTIONING PIPELINE STARTED ===")
    
    # Step 1: Load and clean captions
    print("\n1. Loading and cleaning captions...")
    captions_mapping = load_captions(CAPTIONS_PATH)
    if captions_mapping is None:
        return
    clean_captions(captions_mapping)
    
    # Step 2: Create vocabulary
    print("\n2. Creating vocabulary...")
    tokenizer = create_vocabulary(captions_mapping)
    vocab_size = len(tokenizer.word_index)
    
    # Step 3: Extract image features
    print("\n3. Extracting image features...")
    if os.path.exists(FEATURES_PATH):
        print("Loading existing features...")
        with open(FEATURES_PATH, 'rb') as f:
            image_features = pickle.load(f)
    else:
        image_features = extract_image_features()
        with open(FEATURES_PATH, 'wb') as f:
            pickle.dump(image_features, f)
    
    # Step 4: Prepare data generators
    print("\n4. Preparing data generators...")
    # Split data (80-20 train-validation)
    image_ids = list(set(captions_mapping.keys()) & set(image_features.keys()))
    np.random.shuffle(image_ids)
    split_idx = int(0.8 * len(image_ids))
    
    train_ids = image_ids[:split_idx]
    val_ids = image_ids[split_idx:]
    
    train_captions = {img_id: captions_mapping[img_id] for img_id in train_ids}
    val_captions = {img_id: captions_mapping[img_id] for img_id in val_ids}
    
    train_generator = CaptionDataGenerator(
        image_features, train_captions, tokenizer, 
        MAX_LENGTH_PADDING, BATCH_SIZE, shuffle=True
    )
    
    val_generator = CaptionDataGenerator(
        image_features, val_captions, tokenizer, 
        MAX_LENGTH_PADDING, BATCH_SIZE, shuffle=False
    )
    
    print(f"Train generator length: {len(train_generator)} batches")
    print(f"Validation generator length: {len(val_generator)} batches")
    
    # Step 5: Build model
    print("\n5. Building model...")
    model = build_captioning_model(
        feature_dim=512,  # VGG16 feature dimension
        max_length=MAX_LENGTH_PADDING,
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        units=UNITS,
        dropout_rate=DROPOUT_RATE,
        rnn_lstm_dropout=RNN_LSTM_DROPOUT,
        decoder_type=DECODER_TYPE
    )
    
    model.summary()
    
    # Step 6: Train model
    print("\n6. Training model...")
    history = train_captioning_model(
        model, train_generator, val_generator, 
        MODEL_SAVE_PATH, epochs=EPOCHS
    )
    
    # Step 7: Load best model and generate captions
    print("\n7. Loading best model and generating captions...")
    best_model = load_model(MODEL_SAVE_PATH)
    
    # Generate captions for validation set
    val_generated_captions = generate_captions_for_images(
        best_model, tokenizer, image_features, val_ids[:100],  # Sample 100 images
        MAX_LENGTH_PADDING, DECODER_TYPE
    )
    
    # Step 8: Evaluate with BLEU scores
    print("\n8. Evaluating with BLEU scores...")
    val_sample_captions = {img_id: val_captions[img_id] for img_id in val_generated_captions.keys()}
    bleu_scores = calculate_bleu_scores(val_sample_captions, val_generated_captions)
    
    # Step 9: Save results
    print("\n9. Saving results...")
    with open(TOKENIZER_PATH, 'wb') as f:
        pickle.dump(tokenizer, f)
    
    results = {
        'bleu_scores': bleu_scores,
        'generated_captions': val_generated_captions,
        'model_config': {
            'decoder_type': DECODER_TYPE,
            'embedding_dim': EMBEDDING_DIM,
            'units': UNITS,
            'vocab_size': vocab_size,
            'max_length': MAX_LENGTH_PADDING
        }
    }
    
    results_path = os.path.join(BASE_DRIVE_PATH, f'results_{DECODER_TYPE}.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    print("\n=== IMAGE CAPTIONING PIPELINE COMPLETED ===")
    print(f"Best model saved at: {MODEL_SAVE_PATH}")
    print(f"Tokenizer saved at: {TOKENIZER_PATH}")
    print(f"Results saved at: {results_path}")

# -------------------------------
# 11. UTILITY FUNCTIONS
# -------------------------------
def load_and_predict_single_image(image_path, model_path, tokenizer_path, decoder_type='lstm'):
    """Load a single image and generate caption"""
    
    # Load model and tokenizer
    model = load_model(model_path)
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image = image.resize((IMG_HEIGHT, IMG_WIDTH))
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array / 255.0
    
    # Extract features using VGG16
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    feature_extractor = Model(inputs=vgg_model.input, outputs=vgg_model.layers[-2].output)
    feature = feature_extractor.predict(image_array, verbose=0)
    feature = feature.flatten()
    
    # Generate caption
    caption = generate_caption(model, tokenizer, feature, MAX_LENGTH_PADDING, decoder_type)
    
    return caption

# Run the main pipeline
if __name__ == "__main__":
    main()
