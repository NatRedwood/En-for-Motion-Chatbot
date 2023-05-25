# En-for-Motion-Chatbot

## What is En-for-Motion?
Welcome to the En-for-Motion chatbot ([paper PDF](https://github.com/NatRedwood/En-for-Motion-Chatbot/blob/main/RedwoodN_En_for_Motion_Chatbot.pdf)), a mental health chatbot that uses language to assess an individual’s mental health. RST is a linguistic theory
 that aims to identify how text is structured to convey meaning. I hypothesize that the RST relations, when applied to chatbot conversations, 
 can help assess an individual’s mental health status accurately .
 
<img src="https://github.com/NatRedwood/En-for-Motion-Chatbot/blob/main/images/apimain.png" width=60% height=60%>

## Why bother?
By leveraging the power of AI, this chatbot can offer
personalized and mental health support to individuals worldwide. With the increasing demand for
mental health services, chatbots like En-for-Motion
can offer a scalable and cost-effective solution to
mental health diagnosis and therapy. While there is
still much work to be done, this study highlights the
potential of chatbots in mental health, paving the
way for further research in this exciting field.

## How does it work?
En-for-Motion Chatbot is an NLP model based program that asks 8 questions about the mental health
condition (shown on the graphics). 

<img src="https://github.com/NatRedwood/En-for-Motion-Chatbot/blob/main/images/qs.png" width=60% height=60%>

Based on the answers, it calculates the
mental health score of the user. 

<img src="https://github.com/NatRedwood/En-for-Motion-Chatbot/blob/main/images/score_final.png" width=60% height=60%>

To get the predictions, text classification model is trained on the
dataset of sentences labeled for 6 emotions, joy,
love, sadness, fear, anger and surprise. The labels
are then put into 2 buckets: positive (joy and love)
and negative (sadness, fear, anger and surprise).

<img src="(https://github.com/NatRedwood/En-for-Motion-Chatbot/blob/main/images/em_data_ex.png" width=60% height=60%>

## More info
For more details, check the paper: [paper PDF](https://github.com/NatRedwood/En-for-Motion-Chatbot/blob/main/RedwoodN_En_for_Motion_Chatbot.pdf).
