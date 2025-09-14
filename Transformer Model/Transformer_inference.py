import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import pickle
import random
import nltk
from nltk.tokenize import word_tokenize
import re
import time

# Make sure to have the custom layers available
class TransformerBlock(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            keras.layers.Dense(ff_dim, activation="relu"),
            keras.layers.Dense(embed_dim),
        ])
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

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

class TokenAndPositionEmbedding(keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super(TokenAndPositionEmbedding, self).__init__(**kwargs)
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

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

class TransformerChatbot:
    def __init__(self, model_path='transformer_chatbot_model.h5', tokenizer_path='tokenizer_data.pkl'):
        """Initialize the transformer chatbot"""
        print("Loading transformer chatbot...")
        
        # Load the tokenizer and other data
        with open(tokenizer_path, 'rb') as f:
            data = pickle.load(f)
            self.word_to_idx = data['word_to_idx']
            self.idx_to_word = data['idx_to_word']
            self.vocab = data['vocab']
            self.max_length = data['max_length']
            self.label_encoder = data['label_encoder']
            self.intent_to_responses = data['intent_to_responses']
        
        # Load the model with custom objects
        custom_objects = {
            'TransformerBlock': TransformerBlock,
            'TokenAndPositionEmbedding': TokenAndPositionEmbedding
        }
        
        self.model = keras.models.load_model(model_path, custom_objects=custom_objects)
        print("Chatbot loaded successfully!")
        print(f"Vocabulary size: {len(self.vocab)}")
        print(f"Available intents: {list(self.label_encoder.classes_)}")
    
    def preprocess_text(self, text):
        """Clean and preprocess input text"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def text_to_sequence(self, text):
        """Convert text to sequence of indices"""
        words = word_tokenize(self.preprocess_text(text))
        sequence = [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) for word in words]
        
        # Pad or truncate to max_length
        if len(sequence) < self.max_length:
            sequence = sequence + [self.word_to_idx['<PAD>']] * (self.max_length - len(sequence))
        else:
            sequence = sequence[:self.max_length]
        
        return np.array(sequence)
    
    def predict_intent(self, message):
        """Predict the intent of a message"""
        start_time = time.time()
        
        # Convert message to sequence
        sequence = self.text_to_sequence(message)
        sequence = np.expand_dims(sequence, axis=0)  # Add batch dimension
        
        # Get prediction
        prediction = self.model.predict(sequence, verbose=0)
        predicted_class_idx = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class_idx]
        
        # Convert back to intent name
        predicted_intent = self.label_encoder.inverse_transform([predicted_class_idx])[0]
        
        end_time = time.time()
        response_time = end_time - start_time
        
        return predicted_intent, confidence, response_time
    
    def get_response(self, message, confidence_threshold=0.5):
        """Get response for a message"""
        start_time = time.time()
        
        intent, confidence, prediction_time = self.predict_intent(message)
        
        if confidence < confidence_threshold:
            response = "I'm not sure I understand. Could you please rephrase that?"
        elif intent in self.intent_to_responses:
            responses = self.intent_to_responses[intent]
            response = random.choice(responses)
        else:
            response = "I'm not sure how to respond to that."
        
        end_time = time.time()
        total_response_time = end_time - start_time
        
        print(f"Predicted intent: {intent} (confidence: {confidence:.4f}, response time: {total_response_time:.4f}s)")
        
        return response, total_response_time
    
    def chat(self):
        """Start an interactive chat session"""
        print("\n" + "="*50)
        print("Transformer Chatbot is ready!")
        print("Type 'quit' to exit")
        print("="*50)
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Chatbot: Goodbye! Have a great day!")
                break
            
            if not user_input:
                continue
            
            response, response_time = self.get_response(user_input)
            print(f"Chatbot: {response}")
    
    def batch_test(self, test_messages):
        """Test the chatbot with a batch of messages"""
        print("\nBatch Testing:")
        print("-" * 50)

        df = pd.DataFrame(columns=['Transformer', 'Predicted Intent', 'Confidence (0-1)', 'Relevance (0-1)', 'Response Time (s)'])
        counter = 0
        for message in test_messages:
            counter += 1

            start_time = time.time()
            intent, confidence, prediction_time = self.predict_intent(message[0])
            end_time = time.time()

            if intent == message[1]:
                relevance = 1  # Correct intent
            else:
                relevance = 0 # Incorrect intent

            full_response_time = end_time - start_time
            new_line =({'Transformer': f'prompt {counter}',
            'Predicted Intent': intent,
            'Confidence (0-1)': confidence,
            'Relevance (0-1)': relevance,
            'Response Time (s)': full_response_time
            })
            df = pd.concat([df, pd.DataFrame([new_line])], ignore_index=False)
            
        df.to_csv('Transformer_test_data.csv', index=False)
        print("testing complete")

def main():
    """Main function to run the chatbot"""
    try:
        # Initialize the chatbot
        chatbot = TransformerChatbot()
        
        # Start interactive chat
        # chatbot.chat()
        messages = [
        ["Hi", "greeting"],
        ["Hello", "greeting"],
        ["Hey there", "greeting"],
        ["Good morning", "greeting"],
        ["Greetings", "greeting"],
        ["What's up", "greeting"],
        ["Good afternoon", "greeting"],
        ["Nice to meet you", "greeting"],
        ["Hello, how are you?", "greeting"],
        ["Hey, good to see you.", "greeting"],

        ["Bye", "goodbye"],
        ["Goodbye", "goodbye"],
        ["See you later", "goodbye"],
        ["Take care", "goodbye"],
        ["Farewell", "goodbye"],
        ["Catch you later", "goodbye"],
        ["See ya", "goodbye"],
        ["I have to go now, bye!", "goodbye"],
        ["Talk to you later!", "goodbye"],
        ["Peace out", "goodbye"],

        ["Thank you", "thanks"],
        ["Thanks", "thanks"],
        ["Thank you so much", "thanks"],
        ["I appreciate it", "thanks"],
        ["Thanks a lot", "thanks"],
        ["Much appreciated", "thanks"],
        ["I'm grateful for your help", "thanks"],
        ["That was very helpful, thank you.", "thanks"],
        ["Thanks for the assistance.", "thanks"],
        ["Thank you, that's perfect.", "thanks"],

        ["Can you help me?", "help"],
        ["I need help", "help"],
        ["Help me please", "help"],
        ["I need assistance", "help"],
        ["Can you assist me?", "help"],
        ["I'm looking for help", "help"],
        ["Could you help me with something?", "help"],
        ["I need some support", "help"],
        ["Can you guide me through this?", "help"],
        ["I'm stuck, can you help?", "help"],

        ["What's the weather like?", "weather"],
        ["How's the weather today?", "weather"],
        ["Is it raining outside?", "weather"],
        ["Will it rain today?", "weather"],
        ["What's the weather forecast?", "weather"],
        ["What is the temperature today?", "weather"],
        ["Is it sunny right now?", "weather"],
        ["Do I need an umbrella today?", "weather"],
        ["Is it cold outside?", "weather"],
        ["Could it snow later?", "weather"],

        ["What time is it?", "time"],
        ["Can you tell me the current time?", "time"],
        ["What's the time right now?", "time"],
        ["Time please", "time"],
        ["Do you know what time it is?", "time"],
        ["I need to know the time.", "time"],
        ["Could you tell me the time?", "time"],
        ["What is the time where you are?", "time"],
        ["Hey, what time is it?", "time"],
        ["Do you have the time?", "time"],

        ["What's your name?", "name"],
        ["Who are you?", "name"],
        ["Tell me your name", "name"],
        ["What do I call you?", "name"],
        ["Your name please", "name"],
        ["How should I address you?", "name"],
        ["What are you called?", "name"],
        ["Can I know your name?", "name"],
        ["Do you have a name I can use?", "name"],
        ["Please introduce yourself.", "name"],

        ["How are you?", "how_are_you"],
        ["How are you doing?", "how_are_you"],
        ["How's it going?", "how_are_you"],
        ["How do you feel today?", "how_are_you"],
        ["Are you okay?", "how_are_you"],
        ["How's your day going?", "how_are_you"],
        ["What's up with you?", "how_are_you"],
        ["How are things on your end?", "how_are_you"],
        ["You doing alright?", "how_are_you"],
        ["Everything good with you?", "how_are_you"],

        ["What can you do?", "capabilities"],
        ["What are your capabilities?", "capabilities"],
        ["What are your skills?", "capabilities"],
        ["How can you help me?", "capabilities"],
        ["What services do you provide?", "capabilities"],
        ["What's your purpose?", "capabilities"],
        ["What are you designed for?", "capabilities"],
        ["Tell me about your features", "capabilities"],
        ["What functions do you have?", "capabilities"],
        ["What kind of things can you help with?", "capabilities"],

        ["Tell me a joke", "joke"],
        ["Make me laugh", "joke"],
        ["Say something funny", "joke"],
        ["Do you know any jokes?", "joke"],
        ["Can you be funny?", "joke"],
        ["Humor me", "joke"],
        ["Tell me something amusing", "joke"],
        ["Entertain me with a joke", "joke"],
        ["I need to laugh, tell me a joke", "joke"],
        ["Got any good jokes?", "joke"],

        ["You're awesome", "compliment"],
        ["You're great", "compliment"],
        ["Good job on that", "compliment"],
        ["Well done", "compliment"],
        ["You're very helpful", "compliment"],
        ["You're really smart", "compliment"],
        ["I like talking to you", "compliment"],
        ["You're amazing", "compliment"],
        ["You're the best chatbot", "compliment"],
        ["This is excellent work", "compliment"],

        ["How old are you?", "age"],
        ["What's your age?", "age"],
        ["When were you created?", "age"],
        ["When were you born?", "age"],
        ["Your age please", "age"],
        ["Are you young or old?", "age"],
        ["How long have you been around?", "age"],
        ["What year were you made?", "age"],
        ["When is your birthday?", "age"],
        ["Do you have an age?", "age"]
    ]
        chatbot.batch_test(messages)
        
    except FileNotFoundError as e:
        print(f"Error: Could not find required files. Please make sure you have:")
        print("1. transformer_chatbot_model.h5 (trained model)")
        print("2. tokenizer_data.pkl (tokenizer and preprocessing data)")
        print("3. intents.json (original intents file)")
        print(f"\nSpecific error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()