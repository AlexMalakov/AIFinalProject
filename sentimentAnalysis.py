import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import gensim
import re
import csv

#I don't want to worry about 3 classes for classification
def pruneData(data, label):
    print(np.shape(data), np.shape(label))
    i = 0
    while i < len(label):
        if label[i]=="Neutral" or label[i] == "Irrelevant" or type(data[i]) != str:
            # label.pop(i)
            # data.pop(i)
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
        element = re.sub('<.*?>', '', element) #remove html
        words = element.split(' ')
        vectors = []
        for word in words:
            if "twitter/com/" in word or "t.co/" in word:
                word = "context"
            elif ".org" in word or ".com" in word or ".net" in word or ".gov" in word or "bit.ly" in word:
                word = "link"
            vectors.append(gensimModel.wv[word])
        fasttextout.append(vectors)
    return fasttextout

#padding data (may not be needed)
def padInput(dataTrain, dataTest, maxx):   
    #pad to this length
    padTrain = []
    padTest = []
    fasttextLen = len(dataTrain[0][0])
    for i in range(len(dataTrain)):
        #amount of missing sentence vectors
        fixedCol = dataTrain[i]
        
        #we are padding j times, so that len(fixedColumn) = max_seq
        for j in range(maxx - len(dataTrain[i])):
            fixedCol.append(np.zeros(fasttextLen))
        padTrain.append(fixedCol)

    for i in range(len(dataTest)):
        #amount of missing sentence vectors
        fixedCol = dataTest[i]
        
        #we are padding j times, so that len(fixedColumn) = max_seq
        for j in range(maxx - len(dataTest[i])):
            fixedCol.append(np.zeros(fasttextLen))
        padTest.append(fixedCol)
    
    
    return (padTrain, padTest)

#makes class data usable by model
def assignLabels(label):
    labels = []
    for i in range(len(label)):
        if label[i] == "Negative":
            labels.append(0)
        elif label[i] == "Positive":
            labels.append(1)
        else: 
            print("ERROR ASSIGNING LABELS")
            print("label valuie", label[i])
            return None
    return labels

def setUpMetricLog():

    with open('batchLog.csv', mode='w') as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow(['Batch Num', 'BinAcc', 'Percision', 'Recall',"AUCROC"])

class LoggerCallback(keras.callbacks.Callback):
    def __init__(self, batchLog):
        super(LoggerCallback, self).__init__()
        self.batchLog = batchLog
        self.batchNum = 0

    def on_batch_end(self, batch, logs = None):
        self.batchNum += 1
        auc = logs["auc"]
        recall = logs["recall"]
        prec = logs["precision"]
        binAcc = logs["binary_accuracy"]

        with open(self.batchLog, 'a', newline ='') as f:
            csvWrite = csv.writer(f)
            csvWrite.writerow([self.batchNum,binAcc,prec,recall, auc])



print("loading data")
trainCSV = pd.read_csv("data/twitter_training.csv")
testCSV = pd.read_csv("data/twitter_validation.csv")

trainNP = trainCSV.to_numpy()
testNP = testCSV.to_numpy()
print("data to np")
#get x and y train/test data
x_train = trainNP[:,3]
x_test = testNP[:,3]
y_train = trainNP[:,2]
y_test = testNP[:,2]

print("hopefully set up")

x_train, y_train = pruneData(x_train, y_train)
x_test, y_test = pruneData(x_test, y_test)

y_train = assignLabels(y_train)
y_test = assignLabels(y_test)

print(y_train)
print(y_test)


print("making fastext, this part will take a bit to run :(")
model_path = "data/wiki.en.bin"
gensimModel = gensim.models.fasttext.load_facebook_model(model_path)

print("fasttext loaded, running now")

x_train = runfasttext(x_train)
x_test = runfasttext(x_test)

maxx = max(max(len(x) for x in x_train),max(len(x) for x in x_test))
x_train, x_test = padInput(x_train, x_test, maxx)

print("post pad stuff")

x_train = np.array(x_train, dtype = np.float32)
x_test = np.array(x_test, dtype= np.float32)
y_train = np.array(y_train)
y_test = np.array(y_test)

print(np.shape(x_train))
print(np.shape(x_test))

x_train = tf.convert_to_tensor(x_train)
x_test = tf.convert_to_tensor(x_test)
y_train = tf.convert_to_tensor(y_train)
y_test = tf.convert_to_tensor(y_test)


print('data is ready for NN')
callbackLog = LoggerCallback("batchLog.csv")
metric = [keras.metrics.BinaryAccuracy(), keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()]

model = keras.models.Sequential()
model.add(keras.layers.Masking(mask_value=0., input_shape=(maxx, 300), dtype=np.float32))
model.add(keras.layers.LSTM(units=maxx, activation='tanh', return_sequences=False, dtype=np.float32))
model.add(keras.layers.Dense(units=32, activation='relu', dtype=np.float32))
model.add(keras.layers.Dense(units = 1, activation='sigmoid', dtype=np.float32))
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=keras.losses.BinaryCrossentropy(), metrics=metric)
model.summary()

print("network created")

epoch = 10
numWorkers = 2

hist = model.fit(x_train, y_train, epochs=epoch, batch_size = 150, validation_data = (x_test,y_test), verbose = 1, max_queue_size=1, callbacks = [callbackLog])
# model.save_weights(filepath=f'../model_weights/{model_name}/weights.h5', save_format='h5')