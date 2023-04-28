# About the project:

## Problem being studied:

    With social media being as influential is it is on our modern day and age, being able to gather and analyze data from social media can be very powerful in a number of applications, such as predicting market trends or analyzing public perception of a product. However this comes with a challange, as social media communication is different from normal written text, as certain platforms such as twitter feature word limits that encourage cramming information, as well as unusual pieces of text such as hashes, links, emojis, and usernames. However, with a powerful enough model, such as an LSTM RNN, I hypothesisize that with some slight sanitation, it can still acurately predict the sentiment of social media posts.

## What I did:

    For this project, I focused on twitter, as I was able to find a good data set of tweets, labeled with sentiments. To start, I had to prepare my data for the model. I did this by first filtering out undesired labels (nuetral, irrelavant) since I wanted the model to only worry about positive/negative. I then ran a vectorization algorithm (fasttext) on the data, in order to allow it to be fed into the network.  I also decided to catch and filter out commonly used links here, since I felt like they didn't add much to the original message. I then padded responses in order to account for different word length responses. Finally, I converted the response to a tensor, and fed it into the network

    For my network, I used keras's sequential model, with a masking layer, LSTM layer, and two dense layers. The masking layer is used to mask and receive the input into the network. It has the masking value set to 0 (what we used to pad the data), it also takes the shape of incoming data. For the LSTM layer, I used a ________ activation function as I found it had the best results. Then, I used two dense layers to merge the high dimension output into a lower dimension one. I used a ReLu activation function here since I wanted to ignore all negtive outputs. Then I passed it into a softmax dense layer which would output the actual prediction of sentiment. 

    When training my network, I used _________________-

## Data Source:

    - For my dataset, I used a public domain dataset of posts with a sentiment that I found on Kaggle. The dataset was created by user "passionate-nlp", and is linked below.

    - Dataset: https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis

## Contributions:

    - I (Alex Malakov) worked on this project completly on my own :D

# Analysis:

## What I Learned:

    wow data so poggers :D

## What I Would do Next: 

    Make it better