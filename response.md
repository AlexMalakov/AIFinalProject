# About the project:

## NOTE:

    video presentation link: https://youtu.be/dtzrSNb-x1g
    another backup link: https://drive.google.com/file/d/10ggfwJyGDXZya22EBxAA8kCnJXO0zJw8/view?usp=sharing
    In order to run the code, you will need the wiki.en.bin file. You can download it from the follwing website: https://fasttext.cc/docs/en/crawl-vectors.html
    It's very large, so I wasn't able to get it on git, even after zipping it :(

## Problem being studied:

    With social media being as influential is it is on our modern day and age, being able to gather and analyze data from social media can be very powerful in a number of applications, such as predicting market trends or analyzing public perception of a product. However this comes with a challange, as social media communication is different from normal written text, as certain platforms such as twitter feature word limits that encourage cramming information, as well as unusual pieces of text such as hashes, links, emojis, and usernames. However, with a powerful enough model, such as an LSTM RNN, I hypothesisize that with some slight sanitation, it can still acurately predict the sentiment of social media posts.

## What I did:

    For this project, I focused on twitter, as I was able to find a good data set of tweets, labeled with sentiments. To start, I had to prepare my data for the model. I did this by first filtering out undesired labels (nuetral, irrelavant) since I wanted the model to only worry about positive/negative. I then ran a vectorization algorithm (fasttext) on the data, in order to allow it to be fed into the network.  I also decided to catch and filter out commonly used links here, since I felt like they didn't add much to the original message. I then padded responses in order to account for different word length responses. Finally, I converted the response to a tensor, and fed it into the network

    For my network, I used keras's sequential model, with a masking layer, LSTM layer, and two dense layers. The masking layer is used to pad and receive the input into the network. It has the masking value set to 0 (what we used to pad the data), it also takes the shape of incoming data. For the LSTM layer, I used a tanh activation function, as is standard, which I found produced the best results. Then, I used two dense layers to merge the high dimension output into a lower dimension one. I used a ReLu activation function here since I wanted to ignore all negtive outputs. Then I passed it into a sigmoid activation dense layer which would output the actual prediction of sentiment. 

    When training my network, I used a batch size of 150, and ran on 10 epochs, as I found this was the best compromise of results and run time. I used a learning rate of .001 as is standard, and ran my model with the Adam optimizer and cross entropy loss (log likelihood). This resulted in my model gradually improving over multiple epochs, and resulted in pretty strong performance on train and test metrics.

## Source:

    - For my dataset, I used a public domain dataset of posts with a sentiment that I found on Kaggle. The dataset was created by user "passionate-nlp", and is linked below.

    - Dataset: https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis

    - While most of my code was written by me, I used starter code and examples from Tensorflow's website to help me figure out how to get started. 

    - https://www.tensorflow.org/guide/keras

## Contributions:

    - I (Alex Malakov) worked on this project completly on my own :D

# Analysis:

## What I Learned:

    After using the dataset for the duration of the project, I learned a lot about how it is put together, and how a model is able to use it to predict sentiment. 

    First, after studying the dataset, I noticed that it was pretty similar to normal english. In character frequency, it matched up to english with 'e','t','a','o', and 'i' being the most common characters. Additionally, when using tf-idf, common enligh words such as "the", "you', 'it", "then" were the featured as most important. However, it differed due to being a specific medium, and words more common on online social media such as "game", "pic", and "twitter" were also found to be among the most important. 

    Then, running the model, I learned that the data is pretty distinguishable, despite it's quirks, due to the model producing a freakishly high AUCROC score. However, some positive responses were very similar to negative ones, as the model struggled the most on the recall metric, which measures true positive versus false negative predictions. However the model performs well on accuracy and precision, meaing when it guesses that something is positive, it is usually correct. This means that negative data is significantly more obvious to detect than positive data.

## What I Would do Next: 

    If I could continue this projcet, I would try to secure better hardware to train the model on. As it stands, my computer takes about an hour to fully finish training, and the ram requirement is large enough that I can no longer train the model on my laptop if I wanted to. 

    But if I had access to better hardware, I would be interested in playing around with a larger dataset, and maybe adding another LSTM layer to the network. While I suspect it would lead my model to greatly overfit the data, it would still be intersting to see the effects of multilple LSTM layers on performance. 