# =============================================================================
# IMAGE CAPTIONING PIPELINE - KEY COMPONENTS
# =============================================================================

# -------------------------------
# 1. Data Preparation: Caption Loading & Cleaning
# -------------------------------
import re
import string

def load_captions(filename):
    mapping = {}
    parsed_count = 0
    line_num = 0
    error_lines = 0
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line_num += 1
                line = line.strip()
                if not line: continue

                match_separator = re.search(r'[#.]\d+\s+', line)
                if match_separator:
                    separator_start_index = match_separator.start()
                    separator_end_index = match_separator.end()
                    image_id = line[:separator_start_index].strip()
                    caption = line[separator_end_index:].strip()
                    if not image_id or ' ' in image_id or '\t' in image_id:
                        error_lines += 1
                        continue
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

    total_lines_processed = line_num
    num_unique_images = len(mapping)
    print(f"\n--- Caption Loading Summary ---")
    print(f"Processed {total_lines_processed} lines from {filename}.")
    print(f"Successfully parsed captions for {num_unique_images} unique images.")
    print(f"Total caption entries added: {sum(len(v) for v in mapping.values())}")
    if error_lines > 0:
        print(f"Encountered {error_lines} problematic lines that couldn't be fully parsed.")
    elif total_lines_processed > 0 and num_unique_images == 0:
        print("Warning: No captions were loaded. Check file content and parsing logic.")
    return mapping

def clean_captions(mapping):
    translator = str.maketrans('', '', string.punctuation)
    for key, captions in mapping.items():
        for i in range(len(captions)):
            caption = captions[i]
            caption = caption.lower()
            caption = caption.translate(translator)
            caption = ' '.join(caption.split())
            captions[i] = '<start> ' + caption + ' <end>'
    print("Captions cleaned (lowercase, punctuation removed, start/end tokens added).")

# -------------------------------
# 2. Model Parameters and Paths
# -------------------------------
import os

BASE_DRIVE_PATH = '/content/drive/MyDrive/Assignment2/Team8_captioning'
DECODER_TYPE = 'lstm'
NUM_IMAGES_TO_USE = 8097

CAPTIONS_PATH = os.path.join(BASE_DRIVE_PATH, 'captions.txt')
IMAGES_PATH = os.path.join(BASE_DRIVE_PATH, 'Images')
FEATURES_PATH = os.path.join(BASE_DRIVE_PATH, f'vgg16_features_{DECODER_TYPE}.pkl')
MODEL_SAVE_PATH = os.path.join(BASE_DRIVE_PATH, f'caption_model_{DECODER_TYPE}_lim{NUM_IMAGES_TO_USE}.keras')
TOKENIZER_PATH = os.path.join(BASE_DRIVE_PATH, f'tokenizer_{DECODER_TYPE}_lim{NUM_IMAGES_TO_USE}.pkl')

EMBEDDING_DIM = 256
UNITS = 50
VOCAB_SIZE_LIMIT = 5000
MAX_LENGTH_PADDING = 35
BATCH_SIZE = 320
EPOCHS = 15
DROPOUT_RATE = 0.3
RNN_LSTM_DROPOUT = 0.2

IMG_HEIGHT = 224
IMG_WIDTH = 224

# -------------------------------
# 3. Feature Extraction (VGG16 Example)
# -------------------------------
from tensorflow.keras.applications import VGG16
from tensorflow.keras import models, layers

img_size = 224

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
base_model.trainable = False  # Freeze layers for faster training

model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(5, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# -------------------------------
# 4. Caption Generation Function
# -------------------------------
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

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




