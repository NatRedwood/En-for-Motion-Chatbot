from utils import postprocessor, preprocess
import tensorflow as tf
from model import model
from tensorflow.keras.models import load_model
import streamlit as st
import webbrowser
from PIL import Image
st.set_page_config(layout = "wide")

# define the saved model path
model = load_model('scripts/results/model3_new_data_balanced_emotion/',compile=False)

# SPECIFY WHICH MODEL TO PREDICT ON
train_data = preprocess('../data/final_datasets/model3/train.txt')
train = train_data.copy()

# predict on train data
predstr = model.predict(train.text)

# urls to webpages and images
url_help = 'https://www.nimh.nih.gov/health/find-help'
image_brain = Image.open('images/brain_hands.png')

# create chatbot interface
st.title("En-for-Motion")
st.image(image_brain, width=200)
st.header("How can we predict one’s emotional state based on language they use?")
st.header("How can internal linguistic features help assess one’s mental health?")

st.subheader("We are here to help you find yourself. If you're feeling a bit worried, lost or are just curious what's the state of your mind, we're here for you! Let us assess your inner balance!")

# Store the initial value of widgets in session state
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

# chatbot starts here
name = st.text_input("Hey, welcome to En-for-Motion! What's your name?",
                    placeholder="Pizza Pussy Santa, Ladies and Gentelmen, etc...",
                    key="name")
answers = []

if len(name) > 0:
    st.success("Submitted")
    answer1 = st.text_input(f"{name}, go back to the moment you woke up in the morning. How did you feel? Were you ready for the day or maybe worried, unstable, anxious...?", 
                                placeholder="I felt...", 
                                key='input1')
    answers.append(answer1)

    if len(answer1) > 0:
        st.success("Submitted")
        answer2 = st.text_input("Now, think about people you have close relationships with. What was the last interaction with one of them? What happened? How did you feel afterwards?", 
                                                    placeholder="I felt...", 
                                                    key='input2')
        answers.append(answer2)
    
        if len(answer2) > 0:
            st.success("Submitted")
            answer3 = st.text_input("Think about the circle of your friends. How do you feel when you meet them?", 
                                                placeholder="I feel...", 
                                                key='answer3')
            answers.append(answer3)

            if len(answer3) > 0:
                st.success("Submitted")
                answer4 = st.text_input("Imagine you are in a room full of strangers. You were invited to the event but you know no one. How does it make you feel?",
                                                        placeholder="I feel...",
                                                        key='answer4')
                answers.append(answer4)
                            
                if len(answer4) > 0:
                    st.success("Submitted")
                    answer5 = st.text_input("Recall the last moment when you achieved something. Did you have any recently? How did it make you feel?",
                                                placeholder="I felt...",
                                                key='answer5')
                    answers.append(answer5)

                    if len(answer5) > 0:
                        st.success("Submitted")
                        answer6 = st.text_input("Life has its ups and downs. Thank about the last time something didn't go as you planned? How did you manage your emotions afterwards?",
                                                placeholder="I feel...",
                                                key='answer6')
                        answers.append(answer6)

                        if len(answer6) > 0:
                            st.success("Submitted")
                            answer7 = st.text_input("Close your eyes and remind yourself about your favorite song. How does it make you feel?",
                                            placeholder="I feel...",
                                            key='answer7')
                            answers.append(answer7)

                            if len(answer7) > 0:
                                st.success("Submitted")
                                answer8 = st.text_input("How are you doing in general recently? What's up? Tell me about where you are in your life now. I'm listening...",
                                                placeholder="I feel...",
                                                key='answer8')
                                answers.append(answer8)

                                results = model.predict(answers)
                                score = postprocessor(results, predstr)

                                if len(answer8) > 0:
                                    if st.button('Get my mental score',key='score'):
                                        st.subheader(f'Your mental health score is: {round(score,2)}')
                                        # model3 - range is 62.900986
                                        if score > 65:
                                            st.subheader("Wow, you're rocking! Don't worry, be happy! You really know what it means. You're in a great shape! Seize the day.")
                                        elif score < 65 and score > 50: # all answers 'good' - 63.713654
                                            st.subheader("You're in a really good shape. Keep it up and you'll live a happy and fulfilled life!")
                                        elif score < 50 and score > 46:
                                            st.subheader("We all know life has ups and downs. It's not always perfect but taking care of yourself will definitelyhelp you improve your well-being. Focus and work on the improvement!")
                                        elif score < 46 and score > 41: # all answers 'bad' - 45.62471
                                            st.subheader("It looks like you're struggling a lot recently. Focus on what you need to feel better. Think how you can change your situation. Don't be afraid to seek help from a trained professional to improve your well-being.")
                                            if st.button('Get help'):
                                                webbrowser.open_new_tab(url_help)
                                        elif score < 41: # all answers 'horrible' - 40.37927
                                            st.subheader("It sucks to see that things have been hard for you lately. I'm wondering if you are struggling with things that a therapist could help with? I hate to see you feeling so down and I want to help you get connected to anything that would be helpful. Maybe you would like talking to someone safe and confidential?")
                                            if st.button('Get help'):
                                                webbrowser.open_new_tab(url_help)
else:
    st.text("Please, answer the question. Trust me, I want to help you.")