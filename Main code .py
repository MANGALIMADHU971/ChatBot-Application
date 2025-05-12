import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

file_path = os.path.abspath(r"C:\Users\chara\OneDrive\Documents\Python Scripts\intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

vectorizer = TfidfVectorizer(ngram_range=(1, 4))
clf = LogisticRegression(random_state=0, max_iter=10000)

tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response
        
counter = 0

def main():
    global counter
    st.title("Entertainment ChatBot")

    menu = ["Home", "History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.write("Welcome to the Entertainment chatbot. ")
        st.write("Please type a message and press Enter to start the conversation.")

        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        counter += 1
        user_input = st.text_input("You:", key=f"user_input_{counter}")

        if user_input:

            user_input_str = str(user_input)

            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=120, max_chars=None, key=f"chatbot_response_{counter}")

            timestamp = datetime.datetime.now().strftime(f"%Y-%m-%d %H:%M:%S")

            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input_str, response, timestamp])

            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Have a great day!")
                st.stop()

    elif choice == "History":
        st.header("History")
        with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)  
            for row in csv_reader:
                st.text(f"User👨🏻‍💻: {row[0]}")
                st.text(f"Chatbot🤖: {row[1]}")
                st.text(f"Timestamp🕰️: {row[2]}")
                st.markdown("---")

    elif choice == "About":
        st.write("Welcome to the Entertainment Chatbot, your go-to companion for all things fun and exciting! Whether you're looking for movie recommendations, music playlists, TV shows to binge, or simply need a good laugh, our chatbot is here to bring you the latest in entertainment. Powered by advanced AI, this bot tailors its suggestions based on your preferences, keeps you updated on trending content, and even offers games and quizzes to keep you engaged.")
        st.write("Explore the world of entertainment like never before—let’s chat and discover your next favorite show, film, or artist! With the Entertainment Chatbot, the fun never stops!")
        st.subheader("Project Overview:")

        st.write()

        st.subheader("Dataset:")

        st.write()

        st.subheader("Streamlit Chatbot Interface:")

        st.write("The Entertainment Chatbot, built with Streamlit, offers an interactive interface where users can input text and receive personalized responses in a chat window. Powered by a trained model, it provides entertainment recommendations, trivia, and fun conversations to keep users entertained.")

        st.subheader("Conclusion:")

        st.write("In conclusion, the Entertainment Chatbot offers an engaging and personalized experience, making it easier for users to discover new movies, music, shows, and more. With its interactive interface and smart responses, it ensures that entertainment is always just a chat away, bringing fun and excitement to your day.")

if __name__ == '__main__':
    main()


##JavaScript intent


[
    {
        "tag": "greeting",
        "patterns": [
            "Hi",
            "Hello",
            "Hey",
            "Good morning",
            "Good evening",
            "How are you?",
            "What's up?",
            "How's it going?"
        ],
        "responses": [
            "Hello! How can I assist you today?",
            "Hi there! How can I help you?",
            "Good day! What can I do for you?",
            "Hey! How’s everything going?",
            "Hello! What brings you here today?"
        ]
    },
    {
        "tag": "goodbye",
        "patterns": [
            "Bye",
            "Goodbye",
            "See you later",
            "Take care",
            "Catch you later",
            "See you soon"
        ],
        "responses": [
            "Goodbye! Have a great day!",
            "Take care! See you later.",
            "See you soon! Stay safe.",
            "Goodbye, and come back anytime!"
        ]
    },
    {
        "tag": "thanks",
        "patterns": [
            "Thanks",
            "Thank you",
            "Thanks a lot",
            "I appreciate it",
            "You're the best",
            "Thanks for your help"
        ],
        "responses": [
            "You're welcome!",
            "Glad I could help!",
            "Anytime!",
            "Happy to assist!",
            "No problem at all!"
        ]
    },
    {
        "tag": "about",
        "patterns": [
            "What is your purpose?",
            "Who are you?",
            "Tell me about you",
            "What can you do?",
            "What do you help with?"
        ],
        "responses": [
            "I am an entertainment chatbot here to assist you with movie, music, and show recommendations.",
            "I help you find movies, music, TV shows, and provide fun trivia and jokes.",
            "I’m here to keep you entertained and suggest the best in movies, music, and more!",
            "I’m your go-to for entertainment recommendations and fun interactions!"
        ]
    },
    {
        "tag": "movie_recommendation",
        "patterns": [
            "Can you recommend a good movie?",
            "What should I watch?",
            "Give me a movie suggestion",
            "Tell me a good movie to watch"
        ],
        "responses": [
            "How about 'Inception'? It’s a mind-bending thriller. Or maybe 'The Shawshank Redemption' for a classic!",
            "If you're in the mood for something fun, try 'Guardians of the Galaxy' or 'The Grand Budapest Hotel'.",
            "I recommend 'The Matrix' for a sci-fi adventure, or 'The Pursuit of Happyness' for an inspiring drama."
        ]
    },
    {
        "tag": "music_recommendation",
        "patterns": [
            "What music should I listen to?",
            "Can you recommend some songs?",
            "Give me a music suggestion",
            "What’s a good song for a road trip?"
        ],
        "responses": [
            "For a road trip, try some upbeat pop songs like 'Uptown Funk' or 'Shut Up and Dance'.",
            "How about some chill acoustic music? Check out 'Banana Pancakes' by Jack Johnson.",
            "If you're in the mood for something relaxing, listen to 'Weightless' by Marconi Union."
        ]
    },
    {
        "tag": "tv_show_recommendation",
        "patterns": [
            "What TV shows do you recommend?",
            "Tell me a good TV series to watch",
            "Give me a suggestion for a TV show",
            "What’s a good TV show?"
        ],
        "responses": [
            "If you enjoy drama, 'Breaking Bad' is a must-watch! Or try 'Stranger Things' for some thrilling sci-fi.",
            "How about 'Friends' for a classic sitcom, or 'Black Mirror' for some thought-provoking episodes?",
            "For something light-hearted, 'The Office' is a great option. If you prefer mystery, 'Sherlock' is amazing."
        ]
    },
    {
        "tag": "joke",
        "patterns": [
            "Tell me a joke",
            "Make me laugh",
            "Can you tell a funny story?",
            "Do you know any jokes?"
        ],
        "responses": [
            "Why don’t skeletons fight each other? They don’t have the guts.",
            "Why don’t scientists trust atoms? Because they make up everything!",
            "I told my computer I needed a break, now it won’t stop sending me Kit-Kats!"
        ]
    },
    {
        "tag": "help",
        "patterns": [
            "Can you help me?",
            "I need help",
            "Help me please",
            "Can you assist me?"
        ],
        "responses": [
            "Of course! How can I assist you?",
            "I’m here to help. What do you need assistance with?",
            "I’m happy to help! What can I do for you?",
            "Let me know what you need help with, and I’ll do my best!"
        ]
    },
    {
        "tag": "feedback",
        "patterns": [
            "How am I doing?",
            "Can I give feedback?",
            "I have feedback for you",
            "Do you take feedback?"
        ],
        "responses": [
            "I’d love to hear your feedback! Please let me know how I can improve.",
            "I’m always open to feedback! Please share your thoughts.",
            "Feedback is always welcome. What do you think?"
        ]
    },
    {
        "tag": "contact",
        "patterns": [
            "How can I contact you?",
            "Is there a way to reach you?",
            "Can I talk to a human?",
            "Can I send you a message?"
        ],
        "responses": [
            "You can chat with me anytime here! If you need to reach a human, I can direct you to the support team.",
            "I’m always here to chat. For human support, please contact our help desk.",
            "Feel free to reach out to me whenever you need assistance!"
        ]
    }
]
