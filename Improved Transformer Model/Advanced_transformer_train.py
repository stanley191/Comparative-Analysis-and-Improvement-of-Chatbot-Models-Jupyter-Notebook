import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import json
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
import re
from collections import Counter
import random

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        
        # Multi-head attention - compatible with inference
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=embed_dim,  # Changed from embed_dim // num_heads to embed_dim for compatibility
            dropout=rate
        )
        
        # Feed-forward network - using ReLU to match inference
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),  # Changed from GELU to ReLU
            layers.Dropout(rate),
            layers.Dense(embed_dim),
        ])
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=None):
        # Self-attention with residual connection - simplified to match inference
        attn_output = self.att(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super(TokenAndPositionEmbedding, self).__init__(**kwargs)
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = layers.Embedding(
            input_dim=vocab_size, 
            output_dim=embed_dim,
            mask_zero=True  # Enable masking for padding tokens
        )
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
        # Remove dropout to match inference exactly

    def call(self, x, training=None):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions  # Removed dropout to match inference
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "maxlen": self.maxlen,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
        })
        return config

def advanced_preprocess_text(text, lemmatizer=None, stop_words=None, remove_stopwords=False):
    """Advanced text preprocessing with lemmatization and stopword removal"""
    if lemmatizer is None:
        lemmatizer = WordNetLemmatizer()
    if stop_words is None:
        stop_words = set(stopwords.words('english'))
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters but keep some punctuation context
    text = re.sub(r'[^\w\s\?\!\.]', ' ', text)
    
    # Tokenize
    words = word_tokenize(text)
    
    # Remove stopwords if specified (be careful with intent classification)
    if remove_stopwords:
        words = [word for word in words if word not in stop_words]
    
    # Lemmatize
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return ' '.join(words)

def augment_training_data(texts, labels, augment_ratio=2.0):
    """Data augmentation techniques for intent classification"""
    augmented_texts = []
    augmented_labels = []
    
    for text, label in zip(texts, labels):
        # Original sample
        augmented_texts.append(text)
        augmented_labels.append(label)
        
        # Generate augmentations
        words = text.split()
        
        # Random word dropping (incomplete sentences)
        if len(words) > 3 and random.random() < 0.3:
            drop_idx = random.randint(1, len(words) - 2)
            aug_text = ' '.join(words[:drop_idx] + words[drop_idx+1:])
            augmented_texts.append(aug_text)
            augmented_labels.append(label)
        
        # Random word shuffling (maintain meaning but change order)
        if len(words) > 2 and random.random() < 0.2:
            # Shuffle middle words, keep first and last
            if len(words) > 4:
                middle = words[1:-1]
                random.shuffle(middle)
                aug_text = ' '.join([words[0]] + middle + [words[-1]])
            else:
                shuffled = words.copy()
                random.shuffle(shuffled)
                aug_text = ' '.join(shuffled)
            augmented_texts.append(aug_text)
            augmented_labels.append(label)
        
        # Add noise (simulate typos)
        if random.random() < 0.1 and len(words) > 2:
            noise_words = words.copy()
            # Randomly modify one character in a random word
            word_idx = random.randint(0, len(noise_words) - 1)
            word = noise_words[word_idx]
            if len(word) > 2:
                char_idx = random.randint(0, len(word) - 1)
                chars = list(word)
                chars[char_idx] = random.choice('abcdefghijklmnopqrstuvwxyz')
                noise_words[word_idx] = ''.join(chars)
            aug_text = ' '.join(noise_words)
            augmented_texts.append(aug_text)
            augmented_labels.append(label)
    
    return augmented_texts, augmented_labels

def create_enhanced_transformer_model(vocab_size, max_length, num_classes, 
                                     embed_dim=128, num_heads=8, ff_dim=256,
                                     num_transformer_blocks=2):
    """Create enhanced transformer model with better architecture"""
    inputs = layers.Input(shape=(max_length,))
    
    # Embedding layer
    embedding_layer = TokenAndPositionEmbedding(max_length, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    
    # Multiple transformer blocks
    for i in range(num_transformer_blocks):
        transformer_block = TransformerBlock(
            embed_dim, num_heads, ff_dim, rate=0.1
        )
        x = transformer_block(x)
    
    # Multiple pooling strategies
    avg_pool = layers.GlobalAveragePooling1D()(x)
    max_pool = layers.GlobalMaxPooling1D()(x)
    
    # Concatenate different pooling outputs
    x = layers.Concatenate()([avg_pool, max_pool])
    
    # Classification head - using ReLU for consistency
    x = layers.Dense(128, activation="relu")(x)  # Changed from GELU to ReLU
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(64, activation="relu")(x)   # Changed from GELU to ReLU
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def train_enhanced_transformer_chatbot():
    """Enhanced training function with better techniques"""
    print("Loading and preprocessing data...")
    
    # Initialize preprocessing tools
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Load data
    texts, labels, intent_to_responses = load_and_preprocess_data()
    
    # Advanced preprocessing
    print("Applying advanced preprocessing...")
    processed_texts = [advanced_preprocess_text(text, lemmatizer, stop_words, remove_stopwords=False) 
                      for text in texts]
    
    print("Augmenting training data...")
    augmented_texts, augmented_labels = augment_training_data(processed_texts, labels)
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(augmented_labels)
    num_classes = len(label_encoder.classes_)
    
    # Create tokenizer
    vocab_size = 5000  # Increased vocabulary
    max_length = 32    # Increased max length
    word_to_idx, idx_to_word, vocab = create_tokenizer(augmented_texts, vocab_size)
    actual_vocab_size = len(vocab)
    
    print(f"Vocabulary size: {actual_vocab_size}")
    print(f"Max sequence length: {max_length}")
    
    # Convert texts to sequences
    X = np.array([text_to_sequence(text, word_to_idx, max_length) for text in augmented_texts])
    y = keras.utils.to_categorical(encoded_labels, num_classes)
    
    # Stratified split for better validation
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(X, encoded_labels))
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    print(f"Training shape: {X_train.shape}")
    print(f"Validation shape: {X_val.shape}")
    
    # Create model
    model = create_enhanced_transformer_model(
        vocab_size=actual_vocab_size,
        max_length=max_length,
        num_classes=num_classes,
        embed_dim=128,
        num_heads=16,
        ff_dim=128,
        num_transformer_blocks=1
    )
    
    # Optimizer with learning rate scheduling
    initial_learning_rate = 0.001
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=100,
        decay_rate=0.96,
        staircase=True
    )
    
    optimizer = keras.optimizers.Adam(learning_rate=initial_learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy", "top_k_categorical_accuracy"]
    )
    
    print("Model architecture:")
    model.summary()
    
    #Callbacks
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=20,
            verbose=1
        )
    ]
    
    # Train model with enhanced settings
    print("\nTraining enhanced transformer model...")
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=200,
        validation_data=(X_val, y_val),
        callbacks=callbacks_list,
        verbose=1
    )
    
    # Final evaluation
    train_loss, train_acc, train_top_k = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc, val_top_k = model.evaluate(X_val, y_val, verbose=0)
    
    print(f"\nFinal Results:")
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")
    
    # Save everything with compatible filename
    print("Saving model and preprocessors...")
    model.save('advanced_transformer_chatbot_model.h5')
    
    # Save with the filename that inference expects
    with open('tokenizer_data.pkl', 'wb') as f:
        pickle.dump({
            'word_to_idx': word_to_idx,
            'idx_to_word': idx_to_word,
            'vocab': vocab,
            'max_length': max_length,
            'label_encoder': label_encoder,
            'intent_to_responses': intent_to_responses,
            'lemmatizer': lemmatizer,
            'stop_words': stop_words
        }, f)
    
    return model, word_to_idx, idx_to_word, label_encoder, intent_to_responses

# Keep the original helper functions
def preprocess_text(text):
    """Clean and preprocess text"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def create_tokenizer(texts, vocab_size=10000):
    """Create a tokenizer from texts"""
    all_text = ' '.join(texts)
    words = word_tokenize(all_text.lower())
    word_freq = Counter(words)
    vocab = ['<PAD>', '<UNK>', '<START>', '<END>'] + [word for word, _ in word_freq.most_common(vocab_size - 4)]
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    return word_to_idx, idx_to_word, vocab

def text_to_sequence(text, word_to_idx, max_length):
    """Convert text to sequence of indices"""
    words = word_tokenize(preprocess_text(text))
    sequence = [word_to_idx.get(word, word_to_idx['<UNK>']) for word in words]
    if len(sequence) < max_length:
        sequence = sequence + [word_to_idx['<PAD>']] * (max_length - len(sequence))
    else:
        sequence = sequence[:max_length]
    return sequence

def load_and_preprocess_data(intents_file='intents.json'):
    """Load and preprocess the intents data"""
    with open(intents_file) as file:
        intents = json.load(file)
    
    texts = []
    labels = []
    intent_to_responses = {}
    
    for intent in intents['intents']:
        tag = intent['tag']
        intent_to_responses[tag] = intent['responses']

        # Oversample certain intents
        if tag == 'joke' or tag == 'compliment' or tag == 'name' or tag == 'weather' or tag == 'time' or tag == 'capabilities':
            repeat = 8
        else:
            repeat = 7

        for j in range(0,repeat):
            for pattern in intent['patterns']:
                texts.append(preprocess_text(pattern))
                labels.append(tag)
    
    return texts, labels, intent_to_responses

if __name__ == "__main__":
    model, word_to_idx, idx_to_word, label_encoder, intent_to_responses = train_enhanced_transformer_chatbot()