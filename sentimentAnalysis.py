import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import gensim
import re
import csv


#I don't want to worry about 3/4 classes for classification
#removes data that isn't labeled pos/neg, or data that isn't a string
def pruneData(data, label):
    i = 0
    while i < len(label):
        if label[i]=="Neutral" or label[i] == "Irrelevant" or type(data[i]) != str:
            data = np.delete(data,i)
            label = np.delete(label,i)
        else:
            data[i] = data[i].lower()
            i+=1
    
    return (data,label)


#vectorizes an array of sentences so our model can use the data
def runfasttext(data):
    fasttextout = []
    for element in data:
        element = re.sub('<.*?>', '', element) #remove html text
        words = element.split(' ')
        vectors = []
        for word in words:
            #want to handle links, because they added jubles of unreadable/unusable text
            if "twitter/com/" in word or "t.co/" in word:
                word = "context"
            elif ".org" in word or ".com" in word or ".net" in word or ".gov" in word or "bit.ly" in word:
                word = "link"
            vectors.append(gensimModel.wv[word])
        fasttextout.append(vectors)
    return fasttextout

#padding data (may not be needed)
def padInput(dataTrain, dataTest, maxx):   
    padTrain = []
    padTest = []
    fasttextLen = len(dataTrain[0][0]) #length of fasttext vectorized word
    for i in range(len(dataTrain)):
        #all current columns
        fixedCol = dataTrain[i]
        
        for j in range(maxx - len(dataTrain[i])): #for each missing "word"
            fixedCol.append(np.zeros(fasttextLen)) # attach a pad word
        padTrain.append(fixedCol)

    for i in range(len(dataTest)):

        fixedCol = dataTest[i]
        
        for j in range(maxx - len(dataTest[i])):
            fixedCol.append(np.zeros(fasttextLen))
        padTest.append(fixedCol)
    
    return (padTrain, padTest)

#makes class data usable by model, negative = 0, positive = 1
def assignLabels(label):
    labels = []
    for i in range(len(label)):
        if label[i] == "Negative":
            labels.append(0)
        elif label[i] == "Positive":
            labels.append(1)
        else: 
            #I onlt want positive/negative data
            print("ERROR ASSIGNING LABELS")
            print("label value", label[i])
            return None
    return labels

#open files used for logging
def setUpMetricLog():
    #for batch metrics
    with open('data/batchLog.csv', mode='w') as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow(['Batch Num', 'BinAcc', 'Percision', 'Recall',"AUCROC"])

    #for epoch validation metrics
    with open('data/epochLog.csv', mode='w') as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow(['Epoch Num', 'BinAcc', 'Percision', 'Recall',"AUCROC"])

#custom callback class for writing our csvs :D
class LoggerCallback(keras.callbacks.Callback):
    #initialize batch counter, log files
    def __init__(self, batchLog, epochLog):
        super(LoggerCallback, self).__init__()
        self.batchLog = batchLog
        self.epochLog = epochLog
        self.batchNum = 0

    #at the end of each batch, add the scores to the csv
    def on_batch_end(self, batch, logs = None):
        self.batchNum += 1
        auc = logs["auc"]
        recall = logs["recall"]
        prec = logs["precision"]
        binAcc = logs["binary_accuracy"]

        with open(self.batchLog, 'a', newline ='') as f:
            csvWrite = csv.writer(f)
            csvWrite.writerow([self.batchNum,binAcc,prec,recall, auc])

    #at the end of each epoch, add the validation scores to the csv
    def on_epoch_end(self, epoch, logs = None):
        auc = logs["val_auc"]
        recall = logs["val_recall"]
        prec = logs["val_precision"]
        binAcc = logs["val_binary_accuracy"]

        with open(self.epochLog, 'a', newline ='') as f:
            csvWrite = csv.writer(f)
            csvWrite.writerow([self.batchNum,binAcc,prec,recall, auc])


#read in data
print("loading data from csv")
trainCSV = pd.read_csv("data/twitter_training.csv")
testCSV = pd.read_csv("data/twitter_validation.csv")

trainNP = trainCSV.to_numpy()
testNP = testCSV.to_numpy()
print("data loaded as np array")

#get x and y train/test data
x_train = trainNP[:,3]
x_test = testNP[:,3]
y_train = trainNP[:,2]
y_test = testNP[:,2]

print("import parts of data have been captured")
#get rid of unwanted labels
x_train, y_train = pruneData(x_train, y_train)
x_test, y_test = pruneData(x_test, y_test)

#turn labels into a float array
y_train = assignLabels(y_train)
y_test = assignLabels(y_test)

print("data lengths for xtrain, xtest, ytrain, ytest", str(len(x_train)), str(len(x_test)), str(len(y_train)), str(len(y_test)))


print("making fastext, this part will take a bit to run :(")
#load pretrained fasttext model (I don't want to have to vectorize myself)
model_path = "data/wiki.en.bin"
gensimModel = gensim.models.fasttext.load_facebook_model(model_path)

print("fasttext loaded, running it now")

x_train = runfasttext(x_train)
x_test = runfasttext(x_test)

print("data has been vectorized")

#largest amount of words in a response
maxx = max(max(len(x) for x in x_train),max(len(x) for x in x_test))
x_train, x_test = padInput(x_train, x_test, maxx)

print("data has been padded")

x_train = np.array(x_train, dtype = np.float32)
x_test = np.array(x_test, dtype= np.float32)
y_train = np.array(y_train)
y_test = np.array(y_test)

print("shape of xtrain and xtest")
print(np.shape(x_train))
print(np.shape(x_test))

#so keras can use our arrays
x_train = tf.convert_to_tensor(x_train)
x_test = tf.convert_to_tensor(x_test)
y_train = tf.convert_to_tensor(y_train)
y_test = tf.convert_to_tensor(y_test)

setUpMetricLog()
print('data is ready for Network')
callbackLog = LoggerCallback("data/batchLog.csv", "data/epochLog.csv")
metric = [keras.metrics.BinaryAccuracy(), keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()]

#build our model
model = keras.models.Sequential()
model.add(keras.layers.Masking(mask_value=0., input_shape=(maxx, 300), dtype=np.float32))
model.add(keras.layers.LSTM(units=maxx, activation='tanh', return_sequences=False, dtype=np.float32))
model.add(keras.layers.Dense(units=32, activation='relu', dtype=np.float32))
model.add(keras.layers.Dense(units = 1, activation='sigmoid', dtype=np.float32))
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=keras.losses.BinaryCrossentropy(), metrics=metric)
model.summary()

print("network has compiled")

epoch = 10
numWorkers = 2

hist = model.fit(x_train, y_train, epochs=epoch, batch_size = 150, validation_data = (x_test,y_test), verbose = 1, max_queue_size=1, callbacks = [callbackLog])