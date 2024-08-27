import random
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

\
lemmatizer = WordNetLemmatizer()

# Load data and model
with open('intents.json', 'r') as file:
    intents = json.load(file)
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model_advanced.h5')

def clean_up_sentence(sentence):
    sentence_words = word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    if not intents_list or float(intents_list[0]['probability']) < 0.5:
        return "I'm not sure how to respond to that. Can you please rephrase or ask something else?"
    
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'].lower() == tag.lower():
            result = random.choice(i['responses'])
            return result
    
    return "I'm having trouble finding the right information. Could you try asking in a different way?"

print("Chatbot is ready to chat! (Type 'quit' to exit)")

# Loop to keep the chatbot running and accepting user input
while True:
    message = input("You: ")
    if message.lower() == 'quit':
        break
    
    ints = predict_class(message)
    res = get_response(ints, intents)
    print("Bot:", res)

print("Chatbot session ended.")