# En-for-Motion-Chatbot

![main_api](https://github.com/NatRedwood/En-for-Motion-Chatbot/blob/main/images/apimain.png)

PDF version of the paper includes all the details about the project: [paper](https://github.com/NatRedwood/En-for-Motion-Chatbot/blob/main/RedwoodN_En_for_Motion_Chatbot.pdf).

## What is En-for-Motion?
Welcome to the En-for-Motion chatbot, a mental health chatbot that uses language to assess an individual’s mental health. RST is a linguistic theory
 that aims to identify how text is structured to convey meaning. I hypothesize that the RST relations, when applied to chatbot conversations, 
 can help assess an individual’s mental health status accurately.

## How does it work?
En-for-Motion Chatbot is an NLP model based program that asks 8 questions about the mental health
condition (shown on the graphics). 

![qs](https://github.com/NatRedwood/En-for-Motion-Chatbot/blob/main/images/qs.png)

Based on the answers, it calculates the
mental health score of the user. 

![final_score](https://github.com/NatRedwood/En-for-Motion-Chatbot/blob/main/images/score_final.png)

To get the predictions, text classification model is trained on the
dataset of sentences labeled for 6 emotions, joy,
love, sadness, fear, anger and surprise. The labels
are then put into 2 buckets: positive (joy and love)
and negative (sadness, fear, anger and surprise).

![em_data_ex](https://github.com/NatRedwood/En-for-Motion-Chatbot/blob/main/images/em_data_ex.png)

## Why bother?
The primary aim is to investigate the potential of the En-for-Motion chatbot in assessing an individual’s mental health. By
leveraging the power of AI, this chatbot can offer
personalized and mental health support to individuals worldwide. With the increasing demand for
mental health services, chatbots like En-for-Motion
can offer a scalable and cost-effective solution to
mental health diagnosis and therapy. While there is
still much work to be done, this study highlights the
potential of chatbots in mental health, paving the
way for further research in this exciting field.
