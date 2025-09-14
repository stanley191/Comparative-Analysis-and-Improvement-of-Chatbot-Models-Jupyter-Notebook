import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow import layers
import json
import pickle
import nltk
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
import re
from collections import Counter

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
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
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "maxlen": self.maxlen,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
        })
        return config

def preprocess_text(text):
    """Clean and preprocess text"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

def create_tokenizer(texts, vocab_size=10000):
    """Create a tokenizer from texts"""
    # Combine all texts
    all_text = ' '.join(texts)
    words = word_tokenize(all_text.lower())
    
    # Count word frequencies
    word_freq = Counter(words)
    
    # Create vocabulary with most common words
    vocab = ['<PAD>', '<UNK>', '<START>', '<END>'] + [word for word, _ in word_freq.most_common(vocab_size - 4)]
    
    # Create word to index mapping
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    
    return word_to_idx, idx_to_word, vocab

def text_to_sequence(text, word_to_idx, max_length):
    """Convert text to sequence of indices"""
    words = word_tokenize(preprocess_text(text))
    sequence = [word_to_idx.get(word, word_to_idx['<UNK>']) for word in words]
    
    # Pad or truncate to max_length
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
        
        for pattern in intent['patterns']:
            texts.append(preprocess_text(pattern))
            labels.append(tag)
    
    return texts, labels, intent_to_responses

def create_transformer_model(vocab_size, max_length, num_classes, embed_dim=64, num_heads=4, ff_dim=64):
    """Create transformer-based classification model"""
    inputs = layers.Input(shape=(max_length,))
    embedding_layer = TokenAndPositionEmbedding(max_length, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def train_transformer_chatbot():
    """Main training function"""
    print("Loading and preprocessing data...")
    
    # Load data
    texts, labels, intent_to_responses = load_and_preprocess_data()
    
    print(f"Loaded {len(texts)} training examples with {len(set(labels))} unique intents")
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)
    
    # Create tokenizer
    vocab_size = 2000
    max_length = 20
    word_to_idx, idx_to_word, vocab = create_tokenizer(texts, vocab_size)
    actual_vocab_size = len(vocab)
    
    print(f"Vocabulary size: {actual_vocab_size}")
    print(f"Max sequence length: {max_length}")
    
    # Convert texts to sequences
    X = np.array([text_to_sequence(text, word_to_idx, max_length) for text in texts])
    y = keras.utils.to_categorical(encoded_labels, num_classes)
    
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {y.shape}")
    
    # Create and compile model
    model = create_transformer_model(
        vocab_size=actual_vocab_size,
        max_length=max_length,
        num_classes=num_classes,
        embed_dim=64,
        num_heads=4,
        ff_dim=64
    )
    
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    print("Model architecture:")
    model.summary()
    
    # Train model
    print("\nTraining transformer model...")
    history = model.fit(
        X, y,
        batch_size=32,
        epochs=200,
        validation_split=0.2,
        verbose=1
    )
    
    # Save everything
    print("Saving model and preprocessors...")
    model.save('transformer_chatbot_model.h5')
    
    # Save tokenizer and other components
    with open('tokenizer_data.pkl', 'wb') as f:
        pickle.dump({
            'word_to_idx': word_to_idx,
            'idx_to_word': idx_to_word,
            'vocab': vocab,
            'max_length': max_length,
            'label_encoder': label_encoder,
            'intent_to_responses': intent_to_responses
        }, f)
    
    print("Training completed successfully!")
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    
    return model, word_to_idx, idx_to_word, label_encoder, intent_to_responses

if __name__ == "__main__":
    model, word_to_idx, idx_to_word, label_encoder, intent_to_responses = train_transformer_chatbot()