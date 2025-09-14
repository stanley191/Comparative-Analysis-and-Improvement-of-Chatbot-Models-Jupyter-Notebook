import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model
model = load_model('BOW_chatbot_model.h5')
import json
import random
import time
import pandas as pd

# Load data
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

def clean_up_sentence(sentence):
    # tokenize the pattern - splitting words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stemming every word - reducing to base form
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words, show_details=False):
    # tokenizing patterns
    sentence_words = clean_up_sentence(sentence)
    # bag of words - vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,word in enumerate(words):
            if word == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % word)
    return(np.array(bag))

def predict_class(sentence):
    # filter below threshold predictions
    p = bag_of_words(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sorting strength probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    if not ints:
        return "I'm not sure I understand. Could you rephrase that?"
    
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chat():
    print("=" * 60)
    print("CHATBOT TERMINAL")
    print("=" * 60)
    print("Hello! I'm your chatbot assistant.")
    print("Type 'quit', 'exit', or 'bye' to end the conversation.")
    print("-" * 60)
    
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        # Check for exit commands
        if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
            print("\nBot: Goodbye! Have a great day!")
            break
        
        if user_input == '':
            print("Please enter a message.")
            continue
        
        # Start timing
        start_time = time.time()
        
        # Get prediction and response
        ints = predict_class(user_input)
        response = getResponse(ints, intents)
        
        # Calculate response time
        end_time = time.time()
        response_time = round((end_time - start_time), 2) # in seconds
        
        # Extract intent and confidence
        if ints:
            predicted_intent = ints[0]['intent']
            confidence = round(float(ints[0]['probability']), 3)  # decimal
        else:
            predicted_intent = "unknown"
            confidence = 0.0
        
        # Display response with metadata
        print(f"\nBot: {response}")
        print("-" * 40)
        print(f"Intent: {predicted_intent}")
        print(f"Confidence: {confidence}")
        print(f"Response Time: {response_time}s")
        print("-" * 40)

def batch_test(test_messages):
    """Test the chatbot with a batch of messages"""
    print("\nBatch Testing:")
    print("-" * 50)

    df = pd.DataFrame(columns=['Bag of words', 'Predicted Intent', 'Confidence (0-1)', 'Relevance (0-1)', 'Response Time (s)'])
    counter = 0
    for message in test_messages:
        counter += 1

        start_time = time.time()
        ints = predict_class(message[0])
        end_time = time.time()
        
        # Extract intent and confidence
        if ints:
            intent = ints[0]['intent']
            confidence = ints[0]['probability']  # decimal
        else:
            predicted_intent = "unknown"
            confidence = 0.0

        if intent == message[1]:
            relevance = 1  # Correct intent
        else:
            relevance = 0 # Incorrect intent

        full_response_time = end_time - start_time
        new_line =({'Bag of words': f'prompt {counter}',
        'Predicted Intent': intent,
        'Confidence (0-1)': confidence,
        'Relevance (0-1)': relevance,
        'Response Time (s)': full_response_time
        })
        df = pd.concat([df, pd.DataFrame([new_line])], ignore_index=False)
            
    df.to_csv('BOW_test_data.csv', index=False)
    print("testing complete")

if __name__ == "__main__":
    try:
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

        batch_test(messages)
    except KeyboardInterrupt:
        print("\nBot: Goodbye! Chat ended by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please make sure all required files are present:")
        print("- chatbot_model.h5")
        print("- intents.json") 
        print("- words.pkl")
        print("- classes.pkl")