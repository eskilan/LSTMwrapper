#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 13:38:59 2018

@author: ilan
"""
# numpy, pandas, plot
import numpy as np
np.random.seed(1337) # for reproducibility
import pandas as pd
import matplotlib.pyplot as plt
# keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
# sklearn scaler and encoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

#%% Define object that performs LSTM training and prediction
class Model():
    def __init__(self, csvName='', dataFrame=[], nIn=1):
        self.csvName = csvName
        if csvName != '':
            self.loadDataFromFile()
        else:
            self.dataset = dataFrame
        
        # convert dataframe to float numpy array, and removing index
        self.data = self.dataset.values.astype('float32')
        
        self.scalerX = MinMaxScaler()
        self.scalerY = MinMaxScaler()
        self.nIn = nIn
        
    def loadData(self):
        self.dataset = pd.read_csv(self.csvName, header=0, index_col=0)
        print('Data Loaded:')
        print(self.dataset.head())
    
    # scaling functions create scaler object and return data in 0 to 1 range
    def scaleXData(self, unscaledX):
        return self.scalerX.fit_transform(unscaledX)
        
    def scaleYData(self, unscaledY):
        return self.scalerY.fit_transform(unscaledY)
        
    def setupOutputs(self,outputVarNamesList):
        self.outIndices = []
        for outputName in outputVarNamesList:
            for columnInd, columnName in enumerate(self.dataset.columns):
                if outputName == columnName:
                    self.outIndices.append(columnInd)
    
    # Scales and shapes the data into 3D matrices               
    def rearrangeData(self):
        nIn = self.nIn
        data = self.data
        
        arrayLength = data.shape[0]
        self.nSamples = arrayLength - nIn
        self.nFeatures = data.shape[1] 
        # initialize some 3d matrices. X for inputs, Y for outputs
        self.X3d = np.zeros((self.nSamples,nIn,self.nFeatures))
        self.Y3d = np.zeros((self.nSamples,1,len(self.outIndices)))
        
        # scale data
        xData = data
        yData = data[:,self.outIndices]
        
        scaledX = self.scaleXData(xData)
        scaledY = self.scaleYData(yData)
        
        # fill up 3d matrix with samples for autoregression
        for ind in range(0,self.nSamples):
            self.X3d[ind,:,:] = scaledX[ind:ind+nIn,:]
            self.Y3d[ind,0,:] = scaledY[ind+nIn,:]
        
        # set data shape
        self.inShape = (nIn,self.nFeatures)
        
    # splits the 3D matrices into training and validation
    # also reshapes the output as a 2d matrix since only outputing at time t
    def splitData(self,lastTimeFrame):
        self.X3dTrain = self.X3d[0:lastTimeFrame,:,:]
        self.X3dVal = self.X3d[lastTimeFrame:,:,:]
        self.Y3dTrain = self.Y3d[0:lastTimeFrame,:,:]
        self.Y3dVal = self.Y3d[lastTimeFrame:,:,:]
        
        # number of samples for training and validation
        nTrain = self.X3dTrain.shape[0]
        nVal = self.X3dVal.shape[0]
        
        # reshaping target to avoid mismatch at dense layer
        self.Y3dTrain = np.reshape(self.Y3dTrain,(nTrain,len(self.outIndices)))
        self.Y3dVal = np.reshape(self.Y3dVal,(nVal,len(self.outIndices)))
        
    def trainModel(self,numUnits,lossFunc,optimizerType,nEpochs,batchSize):
        self.model = Sequential()
        self.model.add(LSTM(numUnits, input_shape=self.inShape))
        self.model.add(Dense(units=1*len(self.outIndices)))
        self.model.compile(loss=lossFunc, optimizer=optimizerType)
        # run
        self.history = self.model.fit(self.X3dTrain, self.Y3dTrain, epochs=nEpochs, batch_size=batchSize, validation_data=(self.X3dVal, self.Y3dVal), verbose=2, shuffle=False)

    def plotHistory(self):
        # plot history
        plt.figure()
        plt.plot(self.history.history['loss'], label='train')
        plt.plot(self.history.history['val_loss'], label='test')
        plt.legend()
        plt.show()
    
    def predictNextTimeFrame(self, lastTimeFrame):
         # convert dataframe to float numpy array, and removing index
        data = self.data
        nIn = self.nIn
        
        # select data for input X
        dataX = data[lastTimeFrame-nIn:lastTimeFrame,:]
        scaledX = self.scalerX.transform(dataX)
        
        scaledXr = scaledX.reshape((1,nIn,self.nFeatures))
        
        scaledY = self.model.predict(scaledXr)
        
        Y = self.scalerY.inverse_transform(scaledY)
        return Y
    
    # We can only used predictSequence when all predictors are also targets
    def predictSequence(self,lastTimeFrame, sequenceLength):
        
        scaledY = np.zeros((sequenceLength,len(self.outIndices)))
        # convert dataframe to float numpy array, and removing index
        data = self.data
        nIn = self.nIn
        
        # select data for input X
        dataX = data[lastTimeFrame-nIn:lastTimeFrame,:]
        scaledX = self.scalerX.transform(dataX)
        
        # create new numpy matrix of inputs
        workingScaledX = scaledX
        
        for i in range(0,sequenceLength):
            # predict next step
            if i == 0:
                scaledXr = scaledX.reshape((1,nIn,self.nFeatures))  
                scaledY[0,:] = self.model.predict(scaledXr)
            else:
                # merge Y with X data and use last nIn dataframes
                # shift scaled X data "up"
                workingScaledX[0:nIn-1,:] = workingScaledX[1:nIn]
                # placing previous output as last timeframe of inputs
                workingScaledX[nIn-1,:] = scaledY[i-1,:]
                
                # reshape X
                scaledXr = workingScaledX.reshape((1,nIn,self.nFeatures))
                # 
                scaledY[i,:] = self.model.predict(scaledXr)
                
        Y = self.scalerY.inverse_transform(scaledY)
        return Y
    
    # Predicts pollution only
    def predictFirstValue(self,lastTimeFrame, sequenceLength):
        
        scaledY = np.zeros((sequenceLength,1))
        # convert dataframe to float numpy array, and removing index
        data = self.data
        nIn = self.nIn
        
        # select data for input X
        dataX = data[lastTimeFrame-nIn:lastTimeFrame,:]
        scaledX = self.scalerX.transform(dataX)
        
        # create new numpy matrix of inputs
        workingScaledX = scaledX
        
        for i in range(0,sequenceLength):
            # predict next step
            if i == 0:
                scaledXr = scaledX.reshape((1,nIn,self.nFeatures))  
                scaledY[0] = self.model.predict(scaledXr)
            else:
                # merge Y with X data and use last nIn dataframes
                # shift scaled X data "up"
                workingScaledX[0:nIn-1,:] = workingScaledX[1:nIn]
                # place true predictors as the last timeframe of inputs
                workingScaledX[nIn-1,:] = self.scalerX.transform(data[lastTimeFrame+i-1,:].reshape(1,-1))
                # place previous pollution output as last timeframe of inputs
                workingScaledX[nIn-1,0] = scaledY[i-1]
                
                
                # reshape X
                scaledXr = workingScaledX.reshape((1,nIn,self.nFeatures))
                # 
                scaledY[i] = self.model.predict(scaledXr)
                
        Y = self.scalerY.inverse_transform(scaledY)
        return Y

    # makes several "next step" predictions
    def predictMultipleNextSteps(self, lastTimeFrame,length):
        Y = np.zeros((length,len(self.outIndices)))
        for i in range(0,length):
            Y[i,:] = self.predictNextTimeFrame(lastTimeFrame)
            lastTimeFrame = lastTimeFrame + 1
        return Y
#%% Accesory function to plot sequences
def plotSequences(dataset, lastTimeFrame, Yseq):
    data = dataset.values
    sequenceLength = Yseq.shape[0]
    trueY = data[lastTimeFrame:lastTimeFrame+sequenceLength,:]
    
    numPlots = Yseq.shape[1]
    numRows = int(np.ceil(np.sqrt(numPlots)))
    numCols = numRows
    
    fig, ax = plt.subplots(nrows=numRows, ncols=numCols)

    for i in range(0,numPlots):
        plt.subplot(numRows, numCols, i+1)

        plt.plot(trueY[:,i],'r.') # plot true values
        plt.plot(Yseq[:,i],'b') # plot prediction
        plt.title(dataset.columns.get_values()[i])
    plt.show()

#%%
pollutionDF = pd.read_csv('pollution.csv',header=0, index_col=0)
pollutionReduced = pollutionDF.drop(labels=['wnd_dir','snow'],axis=1)

#%%
# create LSTM wrapping model
myNN = Model(dataFrame=pollutionReduced, nIn=12)
#outputVarNamesList = ['pollution','dew']
outputVarNamesList = pollutionReduced.columns.get_values()
# set outputs
myNN.setupOutputs(outputVarNamesList)
# create 3d dataset
myNN.rearrangeData()
# create split data sets
lastTimeFrame= 3*365*24
myNN.splitData(lastTimeFrame)
# set up hyperparammeters
numUnits = 20
lossFunc = 'mse'
optimizerType = 'adam'
nEpochs = 25
batchSize = 70
# train the LSTM NN
myNN.trainModel(numUnits,lossFunc,optimizerType,nEpochs,batchSize)
#myNN.plotHistory()

# number of steps to predict
numSteps = 12
#Y = myNN.predictNextTimeFrame(lastTimeFrame)
Yseq = myNN.predictSequence(lastTimeFrame,numSteps)
Yseq2 = myNN.predictMultipleNextSteps(lastTimeFrame,numSteps)

#%% comparing
plotSequences(pollutionReduced, lastTimeFrame, Yseq)
plotSequences(pollutionReduced, lastTimeFrame, Yseq2)
#%% predict the next step multiple times
# We create a new model that only predicts pollution
myNN2 = Model(dataFrame=pollutionReduced, nIn=24)
outputVarNamesList2 = ['pollution']
myNN2.setupOutputs(outputVarNamesList2)
myNN2.rearrangeData()
myNN2.splitData(lastTimeFrame)
# train the LSTM NN
myNN2.trainModel(numUnits,lossFunc,optimizerType,nEpochs,batchSize)
#myNN2.plotHistory()
Yseq3 = myNN2.predictFirstValue(lastTimeFrame,numSteps)
Yseq4 = myNN2.predictMultipleNextSteps(lastTimeFrame, numSteps)

plotSequences(pollutionReduced, lastTimeFrame, Yseq3)
plotSequences(pollutionReduced, lastTimeFrame, Yseq4)
