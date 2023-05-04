from utils import postprocessor, preprocess
import tensorflow as tf
from model import model
from tensorflow.keras.models import load_model
import streamlit as st

# define the saved model path
model = load_model('D:/En-for-MOTION/scripts/results/model3_new_data_balanced_emotion')

# SPECIFY WHICH MODEL TO PREDICT ON
train_data = preprocess('D:/En-for-MOTION/data/final_datasets/model3/train.txt')
train = train_data.copy()

# predict on train data
predstr = model.predict(train.text)

print("Hey, welcome to En-for-Motion!")
print("\nWe are here to help you find yourself. If you're feeling a bit worried, lost or are just curious what's the state of your mind, we're here for you!\nLet us assess your inner balance!")
answers = []
answers.append(input('\nGo back to the moment you woke up in the morning. How did you feel? Were you ready for the day or maybe worried, unstable, anxious...?'))
answers.append(input('\nThink about people you have close relationships with. What was the last interaction with one of them? What happened? How did you feel afterwards?'))
answers.append(input('\nThink about the circle of your friends. How do you feel when you meet them?'))
answers.append(input('\nImagine you are in a room full of strangers. You were invited to the event but you know no one. How does it make you feel?'))
answers.append(input("\nRecall the last moment when you achieved something. Did you have any recently? How did it make you feel?"))
answers.append(input("\nLife has its ups and downs. Thank about the last time something didn't go as you planned? How did you manage your emotions afterwards?"))
answers.append(input('\nClose your eyes and remind yourself about your favorite song. How does it make you feel?'))
answers.append(input("\nHow are you doing in general recently? What's up? Tell me about where you are in your life now. I'm listening..."))

results = model.predict(answers)
print(results)
print(predstr)
score = postprocessor(results, predstr)
print('Your mental health score is:', score)

# model3 - range is 62.900986
if score > 65:
    print("Wow, you're rocking! Don't worry, be happy! You really know what it means. You're in a great shape! Seize the day.")
elif score < 65 and score > 46: # all answers 'good' - 63.713654
    print("You're in a really good shape. Keep it up and you'll live a happy and fulfilled life!")
elif score < 46 and score > 41: # all answers 'bad' - 45.62471
    print("It looks like you're struggling a lot recently. Focus on what you need to feel better. Think how you can change your situation. Don't be afraid to seek help from a trained professional to improve your well-being.")
elif score < 41: # all answers 'horrible' - 40.37927
    print("It sucks to see that things have been hard for you lately. I'm wondering if you are struggling with things that a therapist could help with? I hate to see you feeling so down and I want to help you get connected to anything that would be helpful. Maybe you would like talking to someone safe and confidential?")


'''if answer:
    answer = st.text_input("Imagine you are in a room full of strangers. You were invited to the event but you know no one. How does it make you feel?", "I feel...", key=input)
    answers.append(answer)
if answer:
    answer = st.text_input("Recall the last moment when you achieved something. Did you have any recently? How did it make you feel?", "I felt...", key=input)
    answers.append(answer)
if answer:
    answer = st.text_input("Life has its ups and downs. Thank about the last time something didn't go as you planned? How did you manage your emotions afterwards?", "I felt...", key=input)
    answers.append(answer)
if answer:
    answer = st.text_input("Close your eyes and remind yourself about your favorite song. How does it make you feel?", "I feel...", key=input)
    answers.append(answer)
if answer:
    answer = st.text_input("How are you doing in general recently? What's up? Tell me about where you are in your life now. I'm listening...", "I feel...", key = input)
    answers.append(answer)



if button:
    if len(name) > 0:
        st.success("Submitted")
        answer1 = st.text_input(f"{name}, go back to the moment you woke up in the morning. How did you feel? Were you ready for the day or maybe worried, unstable, anxious...?", 
                                placeholder="I felt...", 
                                key='input1')
        answers.append(answer1)
        button1 = st.button("Submit",key='1')
    else:
        st.text("Please, type your name. Trust me, I want to help you.")
        
        if button1:
            if len(answer1) > 0:
                st.success("Submitted")
                answer2 = st.text_input("Now, think about people you have close relationships with. What was the last interaction with one of them? What happened? How did you feel afterwards?", 
                                            placeholder="I felt...", 
                                            key='input2')
                answers.append(answer2)
                button2 = st.button("Submit", key='2')
            else:
                st.text("Please, answer the question. Trust me, I want to help you.")
                  
            if button2:
                if len(answer2) > 0:
                    st.success("Submitted")
                    answer3 = st.text_input("Think about the circle of your friends. How do you feel when you meet them?", 
                                                placeholder="I feel...", 
                                                key='answer3')
                    answers.append(answer3)
                    button3 = st.button("Submit", key='3')
                else:
                    st.text("Please, answer the question. Trust me, I want to help you.")

                    if button3:
                            if len(answer3) > 0:
                                st.success("Submitted")
                                answer4 = st.text_input("Imagine you are in a room full of strangers. You were invited to the event but you know no one. How does it make you feel?",
                                                        placeholder="I feel...",
                                                        key='answer4')
                                answers.append(answer4)
                                button4 = st.button("Submit", key='4') 
                            else:
                                st.text("Please, answer the question. Trust me, I want to help you.")

                    if button4:
                        if len(answer4) > 0:
                            st.success("Submitted")
                            answer5 = st.text_input("Recall the last moment when you achieved something. Did you have any recently? How did it make you feel?",
                                                placeholder="I felt...",
                                                key='answer5')
                            answers.append(answer5)
                            button5 = st.button("Submit", key='5')   
                        else:
                            st.text("Please, answer the question. Trust me, I want to help you.")           

                        if button5:
                            if len(answer5) > 0:
                                st.success("Submitted")
                                answer6 = st.text_input("Life has its ups and downs. Thank about the last time something didn't go as you planned? How did you manage your emotions afterwards?",
                                                placeholder="I feel...",
                                                key='answer6')
                                answers.append(answer6)
                                button6 = st.button("Submit", key='6')       
                            else:
                                st.text("Please, answer the question. Trust me, I want to help you.")                 

                            if button6:
                                if len(answer6) > 0:
                                    st.success("Submitted")
                                    answer7 = st.text_input("Close your eyes and remind yourself about your favorite song. How does it make you feel?",
                                                placeholder="I feel...",
                                                key='answer7')
                                    answers.append(answer7)
                                    button7 = st.button("Submit", key='7')       
                                else:
                                    st.text("Please, answer the question. Trust me, I want to help you.")  

                                if button7:
                                    if len(answer7) > 0:
                                        st.success("Submitted")
                                        answer8 = st.text_input("How are you doing in general recently? What's up? Tell me about where you are in your life now. I'm listening...",
                                                placeholder="I feel...",
                                                key='answer8')
                                        answers.append(answer8)
                                        button8 = st.button("Submit", key='8')       
                                    else:
                                        st.text("Please, answer the question. Trust me, I want to help you.")                 

                                    results = model.predict(answers)
                                    print(results)
                                    score = postprocessor(results, predstr)
                                    print(score)

                                    st.subheader('Your mental health score is:', score)


                                    # model3 - range is 62.900986
                                    if score > 65:
                                        st.subheader("Wow, you're rocking! Don't worry, be happy! You really know what it means. You're in a great shape! Seize the day.")
                                    elif score < 65 and score > 46: # all answers 'good' - 63.713654
                                        st.subheader("You're in a really good shape. Keep it up and you'll live a happy and fulfilled life!")
                                    elif score < 46 and score > 41: # all answers 'bad' - 45.62471
                                        st.subheader("It looks like you're struggling a lot recently. Focus on what you need to feel better. Think how you can change your situation. Don't be afraid to seek help from a trained professional to improve your well-being.")
                                    elif score < 41: # all answers 'horrible' - 40.37927
                                        st.subheader("It sucks to see that things have been hard for you lately. I'm wondering if you are struggling with things that a therapist could help with? I hate to see you feeling so down and I want to help you get connected to anything that would be helpful. Maybe you would like talking to someone safe and confidential?")'''