#### Imports
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import random
import names
from torch.autograd import Variable
import math
import sys
from direct.showbase.ShowBase import ShowBase
from panda3d.core import TextNode, TransparencyAttrib
from panda3d.core import LPoint3, LVector3
from direct.gui.OnscreenText import OnscreenText
from direct.task.Task import Task
from math import sin, cos, pi

from direct.interval.MetaInterval import Sequence
from direct.interval.FunctionInterval import Wait, Func
from panda3d.core import loadPrcFileData

from direct.gui.OnscreenText import OnscreenText
from direct.gui.DirectGui import *
from panda3d.core import TextNode
import time
import os
from math import log10, floor
#### Globals
SPRITE_POS = 55
configVars = """
win-size 1400 720
show-frame-rate-meter 1
fullscreen 0
window-title Evolution Simulator
"""   


class ModelCreationApp(object):
    
    def __init__(self):
        '''
        Creates the default machine learning model.

        Returns
        -------
        None.

        '''
        
        a = Settings()
        
        self.__howMuchTrainingDataShouldThereBe = a.returnHowMuchTrainingData()
        self.__batchSize = a.returnBatchSize()
        
        self.__checkerObject = GenericModel(1, self.__howMuchTrainingDataShouldThereBe, self.__batchSize)
    
    def modelCreation(self, scenarioName):
        '''
        

        Parameters
        ----------
        scenarioName : int
            The scenario identifier you want a model to be created for.

        Returns
        -------
        None.

        '''      
                
        
        if self.doesAModelExist(scenarioName) == True:
            print('A accurate model for this scenario already exists - no need to make another one')
            self.__a = GenericModel(scenarioName, self.__howMuchTrainingDataShouldThereBe,self.__batchSize)
            model, loss_fn, opt, train_dl = self.__a.modelPreparations()
            return None
        
        modelNotAccurate = True
        howManyTimes = 0
        
        while modelNotAccurate == True:
            howManyTimes += 1
            self.__a = GenericModel(scenarioName, self.__howMuchTrainingDataShouldThereBe,self.__batchSize)
            model, loss_fn, opt, train_dl = self.__a.modelPreparations()
            loss = self.__a.fit(200, model, loss_fn, opt, train_dl) 
            if loss <= 0.01:
                print('Model training concluded successfully on Trial', howManyTimes)
                accuracy = calculatePercentageOfAnyNumberFromDecimal(1-loss)
                print('The Model is', accuracy,'accurate')
                modelNotAccurate = False
            
            
        self.__a.saveModel(model, scenarioName)
    
    
    def calcPrediction(self, data):
        '''
        

        Parameters
        ----------
        data : Nested list of format [[int],[int]...]
            Parameters of a object.

        Returns
        -------
        pred : int
            Prediction of the model based on inputs.

        '''
        pred = self.__a.putInputIntoModel(data)
        return pred
    
    def doesAModelExist(self,scenarioName):
        '''
        

        Parameters
        ----------
        scenarioName : int
            The scenario identifier you want check whether a model for it has already been created.

        Returns
        -------
        boolean : boolean
            Returns True if model exists, else returns False.

        '''
        
        boolean = self.__checkerObject.doesAModelExist(scenarioName)
        return boolean
        
def calculatePercentageOfAnyNumberFromDecimal(num):
    '''
    

    Parameters
    ----------
    num : int/float
        Number you want to convert into percentage.

    Returns
    -------
    Str
        Returns the percentage as a rounded float with a % symbol attached.

    '''
    
    a = Settings()
    sig = a.returnSigFig()
    
    x =  int(num*100)
    rounded = round(x, sig-int(floor(log10(abs(x))))+4)
    
    return str(rounded) + '%'

def randomInterger(lowerBounds, UpperBounds):
    '''
    

    Parameters
    ----------
    lowerBounds : int
        Lowers number that is acceptable.
    UpperBounds : int
        Highest number that is acceptable.

    Returns
    -------
    int
        Returns random number based on lB and uB.

    '''
    
    return random.randint(lowerBounds,UpperBounds)
    
def settingsLoader():
    '''
    

    Returns
    -------
    listOfLists : list
        Returns the content of the read file as one list.

    '''
    
    a_file = open("settings.txt", "r")
    listOfLists = []
    for line in a_file:
        stripped_line = line.strip()
        line_list = stripped_line.split(' ')
        listOfLists.append(line_list)
        
    a_file.close()
    
    
    return listOfLists


def settingsWriter(theNameOfVariable, value):
    '''
    

    Parameters
    ----------
    theNameOfVariable : str
        The variable you would like to change and save.
    value : int
        The value you want the variable to have.

    Raises
    ------
    
        If the value is not an int, raise an error.

    Returns
    -------
    None.

    '''
    
    a_file = open("settings.txt", "r")
    listOfLists = []
    for line in a_file:
        stripped_line = line.strip()
        line_list = stripped_line.split(' ')
        listOfLists.append(line_list)
    
    a_file.close()
    
    
    for i in range(len(listOfLists)):
        
        if listOfLists[i][0] == theNameOfVariable:
            try:
                listOfLists[i][2] = int(value)
            except TypeError:
                raise 'Wrong Data Type, Try again'
    
    a_file = open("settings.txt", "w")
    
    for i in range(len(listOfLists)):
        for j in range(len(listOfLists[i])):
            a_file.write(str(listOfLists[i][j]) + " ")
        a_file.write("\n")
        
    a_file.close()
    

class Settings(object):
    
    
    def __init__(self):
        '''
        

        Returns
        -------
        None.

        '''
        
        settings = settingsLoader()
        
        
        self.__sig = int(settings[0][2])
        self.__howMuchTrainingDataShouldThereBe = int(settings[1][2])
        self.__batchSize = int(settings[2][2])
        self.__returnDefaultParameterValue = int(settings[3][2])
        self.__returnParameterValueOfDefaultNeanthedral = int(settings[4][2])
        self.__returnSneakingValueForNeanthedral = int(settings[5][2])
        self.__returnInitialNumberOfNeanthedrals = int(settings[6][2])
        self.__returnLimitOfSystem = int(settings[7][2])
        self.__returnInitialNumberOfHumans = int(settings[8][2])
        
        self.__returnStartingYearAsInt = int(settings[9][2])
        self.__returnYearlyValueOfAGeneration = int(settings[10][2])
        self.__returnNumberOfTimesEvolutionHappens = int(settings[11][2])
        self.__returnHowManyHumansGetTheSameAdaptationAtOnce = int(settings[12][2])
        self.__returnifPopBelowEndExtinction = int(settings[13][2])
        
    def returnifPopBelowEndExtinction(self):
        '''
        

        Returns
        -------
        int
            Returns a saved variable from the file.

        '''
        return self.__returnifPopBelowEndExtinction
    
    def returnHowManyHumansGetTheSameAdaptationAtOnce(self):
        '''
        

        Returns
        -------
        int
            Returns a saved variable from the file.

        '''
        return self.__returnHowManyHumansGetTheSameAdaptationAtOnce
    
    def returnNumberOfTimesEvolutionHappens(self):
        '''
        

        Returns
        -------
        int
            Returns a saved variable from the file.

        '''
        return self.__returnNumberOfTimesEvolutionHappens
    
    def returnYearlyValueOfAGeneration(self):
        '''
        

        Returns
        -------
        int
            Returns a saved variable from the file.

        '''
        return self.__returnYearlyValueOfAGeneration
    
    def returnStartingYearAsInt(self):
        '''
        

        Returns
        -------
        int
            Returns a saved variable from the file..

        '''
        return self.__returnStartingYearAsInt
    
    def returnInitialNumberOfHumans(self):
        '''
        

        Returns
        -------
        int
            Returns a saved variable from the file.

        '''
        return self.__returnInitialNumberOfHumans
    
    def returnLimitOfSystem(self):
        '''
        

        Returns
        -------
        int
            Returns a saved variable from the file.

        '''
        return self.__returnLimitOfSystem
    
    def returnInitialNumberOfNeanthedrals(self):
        '''
        

        Returns
        -------
        int
            Returns a saved variable from the file.

        '''
        return self.__returnInitialNumberOfNeanthedrals
        
    def returnSneakingValueForNeanthedral(self):
        '''
        

        Returns
        -------
        int
            Returns a saved variable from the file..

        '''
        return self.__returnSneakingValueForNeanthedral
    
    def returnParameterValueOfDefaultNeanthedral(self):
        '''
        

        Returns
        -------
        int
            Returns a saved variable from the file..

        '''
        return self.__returnParameterValueOfDefaultNeanthedral
    
    
    def returnBatchSize(self):
        '''
        

        Returns
        -------
        int
            Returns a saved variable from the file.

        '''
        return self.__batchSize
        
    def returnSigFig(self):
        '''
        

        Returns
        -------
        int
            Returns a saved variable from the file.

        '''
        return self.__sig
    
    def returnHowMuchTrainingData(self):
        '''
        

        Returns
        -------
        int
            Returns a saved variable from the file.

        '''
        return self.__howMuchTrainingDataShouldThereBe
    
    def returnDefaultParameterValue(self):
        '''
        

        Returns
        -------
        int
            Returns a saved variable from the file.

        '''
        return self.__returnDefaultParameterValue

class GenericModel(object):
    
    def __init__(self, scenarioType,howMuchTrainingDataShouldThereBe,batchSize):
        '''
        

        Parameters
        ----------
        scenarioType : int
            The scenario identified you want a model created for.
        howMuchTrainingDataShouldThereBe : int
            When creating training data for the model, how much of it should there be.
        batchSize : int
            How much data is the model taking at one time.

        Returns
        -------
        None.

        '''
        
        
        self.scenarioType = scenarioType
        self.__howMuchTrainingDataShouldThereBe = howMuchTrainingDataShouldThereBe
        
        self.__batchSize = batchSize 
        self.__numberOfVariables = 0
    
    def inputDataGenerator(self):
        '''
        
        
        Returns
        -------
        aList : list
             List of randomly generated parameters.

        '''

        self.__throwing = randomInterger(0,10)
        self.__thicknessofSkin = randomInterger(0,10)
        self.__quickRunning = randomInterger(0,10)
        self.__distanceRunning = randomInterger(0,10)
        self.__catching = randomInterger(0,10)
        self.__nightVision = randomInterger(0,10)
        self.__dayVision = randomInterger(0,10)
        self.__hearing = randomInterger(0,10)
        self.__smell = randomInterger(0,10)
        self.__intelligence = randomInterger(0,10)
        self.__fighting = randomInterger(0,10)
        self.__toolsLevel = randomInterger(0,10)
        self.__weaponLevel = randomInterger(0,10)
        self.__fireMaking = randomInterger(0,10)
        self.__defending = randomInterger(0,10)
        self.__clothing = randomInterger(0,10)
        self.__canSurviveHotterClimates = randomInterger(0,10)
        self.__canSurviveColderClimates = randomInterger(0,10)
        self.__height = randomInterger(0,10)
        self.__hairDensity = randomInterger(0,10)
        self.__resistanceToDisease = randomInterger(0,10)
        self.__sneaking = randomInterger(0,10)
        
        aList = [self.__throwing,
        self.__thicknessofSkin,
        self.__quickRunning,
        self.__distanceRunning,
        self.__catching,
        self.__nightVision,
        self.__dayVision,
        self.__hearing,
        self.__smell,
        self.__intelligence,
        self.__fighting,
        self.__toolsLevel,
        self.__weaponLevel,
        self.__fireMaking,
        self.__defending,
        self.__clothing,
        self.__canSurviveHotterClimates,
        self.__canSurviveColderClimates,
        self.__height,
        self.__hairDensity,
        self.__resistanceToDisease,
        self.__sneaking]
        
        self.__numberOfVariables = len(aList)
        
        return aList
    
    def targetDataGenerator(self,targetData):
        '''
        

        Parameters
        ----------
        targetData : list
            The data representing what output the model should have.

        Raises
        ------
        Exception
            If the scenario identifier passed into the model is invalid, raise an error.

        Returns
        -------
        None.

        '''
        if self.scenarioType == 1: 
            
            successRate = self.__calcSuccess1(targetData)
            

        elif self.scenarioType == 2:
            
            successRate = self.__calcSuccess2(targetData)
        elif self.scenarioType == 3:
            
            successRate = self.__calcSuccess3(targetData)
        elif self.scenarioType == 4:
            
            successRate = self.__calcSuccess4(targetData)
            
        elif self.scenarioType == 5:
            successRate = self.__calcSuccess5(targetData)
        elif self.scenarioType == 6:
            successRate = self.__calcSuccess6(targetData)
        elif self.scenarioType == 7:
            successRate = self.__calcSuccess7(targetData)
        elif self.scenarioType == 8:
            successRate = self.__calcSuccess8(targetData)
        elif self.scenarioType == 9:
            successRate = self.__calcSuccess9(targetData)
        elif self.scenarioType == 10:
            successRate = self.__calcSuccess10(targetData)
        elif self.scenarioType == 11:
            successRate = self.__calcSuccess11(targetData)
        elif self.scenarioType == 12:
            successRate = self.__calcSuccess12(targetData)

        else:
            raise Exception('Invalid Scenario Type')
        return(successRate)

    def __calcSuccess7(self, targetData): 
       
     
        # Multiplies different ratings by different weights and adds it to make a single value
        
        return ((targetData[14]*100)+(targetData[12]*80)+(targetData[13]*40)+(targetData[6]*10))/240

    def __calcSuccess8(self, targetData): 
       
     
        # Multiplies different ratings by different weights and adds it to make a single value
        
        return ((targetData[20]*100)+(targetData[1]*80)+(targetData[15]*40)+(targetData[6]*10))/240

    def __calcSuccess9(self, targetData): 
       
        # Multiplies different ratings by different weights and adds it to make a single value
        
        
        return ((targetData[20]*100)+(targetData[19]*80)+(targetData[11]*40)+(targetData[16]*10))/240

    def __calcSuccess10(self, targetData): 
       
       
        # Multiplies different ratings by different weights and adds it to make a single value
        
        return ((targetData[11]*100)+(targetData[9]*80)+(targetData[4]*40)+(targetData[0]*10))/240

    def __calcSuccess11(self, targetData): 
       
        
        # Multiplies different ratings by different weights and adds it to make a single value
        
        return ((targetData[16]*100)+(targetData[20]*80)+(targetData[3]*40)+(targetData[8]*10))/240

    def __calcSuccess12(self, targetData):
       
        
        # Multiplies different ratings by different weights and adds it to make a single value
        
        return ((targetData[17]*100)+(targetData[13]*80)+(targetData[15]*40)+(targetData[9]*10))/240

    def __calcSuccess999(self, targetData): 
       
        # Multiplies different ratings by different weights and adds it to make a single value
        
        
        return ((targetData[2]*100)+(targetData[0]*80)+(targetData[3]*40)+(targetData[9]*10))/240

    def __calcSuccess998(self, targetData):
       
        # Multiplies different ratings by different weights and adds it to make a single value
        
        
        return ((targetData[5]*100)+(targetData[8]*80)+(targetData[9]*40)+(targetData[4]*10))/240

    def __calcSuccess997(self, targetData): 
       
        # Multiplies different ratings by different weights and adds it to make a single value
        
        
        return ((targetData[6]*100)+(targetData[7]*80)+(targetData[0]*40)+(targetData[9]*10))/240

    def __calcSuccess996(self, targetData): 
       
        # Multiplies different ratings by different weights and adds it to make a single value
        
        
        return ((targetData[10]*100)+(targetData[13]*80)+(targetData[0]*40)+(targetData[9]*10))/240


    
    def __calcSuccess1(self, targetData): 
       
        # Multiplies different ratings by different weights and adds it to make a single value
        
        return ((targetData[2]*100)+(targetData[0]*80)+(targetData[3]*40)+(targetData[9]*10))/240
   
    def __calcSuccess2(self, targetData): 
       
        # Multiplies different ratings by different weights and adds it to make a single value
        
        
        return ((targetData[5]*100)+(targetData[8]*80)+(targetData[9]*40)+(targetData[4]*10))/240
     
    def __calcSuccess3(self, targetData): 
        
        # Multiplies different ratings by different weights and adds it to make a single value
        
        
        return ((targetData[6]*100)+(targetData[7]*80)+(targetData[0]*40)+(targetData[9]*10))/240
 
    def __calcSuccess4(self, targetData):
        
        # Multiplies different ratings by different weights and adds it to make a single value
        
        
        return ((targetData[10]*100)+(targetData[13]*80)+(targetData[0]*40)+(targetData[9]*10))/240
 
    def __calcSuccess5(self, targetData):
       
        # Multiplies different ratings by different weights and adds it to make a single value
        
        
        return ((targetData[2]*100)+(targetData[0]*80)+(targetData[3]*40)+(targetData[9]*10))/240
 
    def __calcSuccess6(self, targetData): 
        # Multiplies different ratings by different weights and adds it to make a single value
        
        return ((targetData[2]*100)+(targetData[0]*80)+(targetData[3]*40)+(targetData[9]*10))/240
 
    def __dataInFormat(self):
        '''
        

        Returns
        -------
        inputs : array
            Input data in correct format.
        targets : array
            Training data in correct format.

        '''
        aList = []
        bList = []
        # Creates training data
        for i in range(self.__howMuchTrainingDataShouldThereBe):
            a = self.inputDataGenerator()
            
            b = self.targetDataGenerator(a) 
            
            aList.append(a)
            bList.append([b])
        
        inputs, targets = self.__convertIntoArray(aList, bList)
        
        return inputs,targets
    
    def __convertIntoArray(self, aList, bList):
        '''
        

        Parameters
        ----------
        aList : List
            Input data.
        bList : Nested List
            Target Data.

        Returns
        -------
        inputs : Array
            Input data.
        targets : Array
            Target data.

        '''
        inputs = np.array(aList,
                         dtype='float32')
        if bList == False:
            return inputs
        targets = np.array(bList,
                             dtype='float32')
        return inputs,targets
        
    
    def modelPreparations(self):
        '''
        Returns
        -------
        model : PyTorch Deep Neural Network
            Returns the model created.
        loss_fn : PyTorch compatable type
            Returns the loss function.
        opt : PyTorch compatable type
            Returns the optimiser for the model.
        train_dl : Database
            Returns the data loader, used to group the data.

        '''
        #creates both types of data
        inputs,targets = self.__dataInFormat()
        #converts them into a tensor
        inputs,targets = self.__convertIntoTensor(inputs, targets)
        # creates a training dataset based on the inputs and targets
        train_ds = TensorDataset(inputs, targets)
        # specifies how many batches of data should the program take at one time
        batch_size = self.__batchSize
        # Creates a training data loader, allowing for batches of shuffled data to be taken into a program at one time
        # So it increases the chances of the program being able to find the 'right' correlation
        train_dl = DataLoader(train_ds, batch_size, shuffle=True)
        # Creates a linear regression model with the number of input and target variables as specified
        model = nn.Linear(self.__numberOfVariables, 1)
        # Fetches a loss function from the imported library
        loss_fn = F.mse_loss 
        # creates an optimiser
        opt = torch.optim.SGD(model.parameters(), lr=1e-5)
        
        return model, loss_fn, opt, train_dl
        
    def putInputIntoModel(self, inputData):
        '''
        

        Parameters
        ----------
        inputData : list
            Input data.

        Returns
        -------
        pred : int
            Returns the prediction of the model.

        '''
        copyInputData = inputData
        changedList = []
        # converts input data into a [int,int...] format from a [[int],[int]...]
        for i in copyInputData:
            changedList.append(i[0])
            
        copyInputData = tuple(changedList)
        
        inputData = self.__convertIntoArray(copyInputData, False)
        inputData = torch.from_numpy(inputData)
        model = nn.Linear(self.__numberOfVariables,1)
        
        
        # loads the model from a saved file
        
        model.load_state_dict(torch.load(self.returnPathOfSavedModel(self.scenarioType)))
        
        pred = model(inputData)
        
        return pred
    
    def doesAModelExist(self,scenarioName):
        '''
        

        Parameters
        ----------
        scenarioName : int
            Scenario indentifier you want to check whether a model for it already exists.

        Returns
        -------
        bool
            If model exists, return True, else False.

        '''
        
        if os.path.exists(self.returnPathOfSavedModel(scenarioName)):
            return True
        else:
            return False

        
    def returnPathOfSavedModel(self,scenarioType):
        '''
        

        Parameters
        ----------
        scenarioType : int
            Scenario indetifier.

        Returns
        -------
        Str
            Returns the path the model should save as.

        '''
        
        return str(scenarioType)+'  mnist-logistic.pth'
    
    
    def __convertIntoTensor(self, inputs, targets):
        '''
        

        Parameters
        ----------
        inputs : Array
            Input data.
        targets : Array
            Target Data.

        Returns
        -------
        inputs : Tensor
            Input data.
        targets : Tensor
            Target Data.

        '''
        
        inputs = torch.from_numpy(inputs)
        targets = torch.from_numpy(targets)
        return inputs, targets
    
    def fit(self,num_epochs, model, loss_fn, opt, train_dl):
        '''
        

        Parameters
        ----------
        num_epochs : int
            How how many epochs should the model be trained for.
        model : PyTorch Deep Neural Network
            The model created.
        loss_fn : PyTorch compatable type
            Loss function.
        opt : PyTorch Compatable type
            Optimizer.
        train_dl : Dataset
            Training data loader.

        Returns
        -------
        None.

        '''
        
        # Repeat for given number of epochs
        for epoch in range(num_epochs):
            
            # Train with batches of data
            for xb,yb in train_dl:
                
                # 1. Generate predictions
                pred = model(xb)
                
                # 2. Calculate loss
                loss = loss_fn(pred, yb)
                
                # 3. Compute gradients
                loss.backward()
                
                # 4. Update parameters using gradients
                opt.step()
                
                # 5. Reset the gradients to zero
                opt.zero_grad()
            
            # Print the progress
            if (epoch+1) % 20 == 0:
                print('Epoch [{}/{}], Loss: {:.8f}'.format(epoch+1, num_epochs, loss.item()))
    
        print('Model training has concluded')  
        
        self.__stateModel = model.state_dict()
        return loss
        
    def saveModel(self, scenarioType, model):
        '''
        

        Parameters
        ----------
        scenarioType : int
            Scenario identifier.
        model : PyTorch Deep Neural Network
            Model.

        Returns
        -------
        None.

        '''
        
        savePath = self.returnPathOfSavedModel(self.scenarioType)
        torch.save(self.__stateModel, savePath)
        


class VariableHolder(object):
    '''Represents a object which holds dictonaries used my other classes'''
    
    def __init__(self):
        ''' Initialises the VariableHolder object'''
        
        # Holds a dictionary which is in the format:
        # Name of adaptation as str : Its impact on human object variables as nested list
        self.__impactOfAdaptations = {'Sweating':[[0], [0], [1], [2], [0], [0], [0], [0], [0], [0], [1], [0], [0], [0], [0], [0], [1], [0], [0], [0], [1], [0]],
                                     'More Hair':[[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [2], [1], [0]],
                                     'Better Tools':[[1], [1], [0], [0], [0], [0], [0], [0], [0], [0], [1], [2], [1], [1], [1], [1], [0], [0], [0], [0], [0], [0]],
                                     'More Leg muscle':[[0], [0], [2], [2], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
                                     'More Arm Muscle':[[2], [0], [0], [0], [0], [0], [0], [0], [0], [0], [1], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
                                     'Better day eye sight':[[0], [0], [0], [0], [0], [0], [3], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
                                     'Better night eye sight':[[0], [0], [0], [0], [0], [3], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
                                     'More fat % (better at cold climates)':[[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [-1], [2], [0], [0], [0], [0]],
                                     'Less Fat %':[[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [2], [-1], [0], [0], [0], [0]],
                                     'Taller':[[0], [0], [0], [0], [0], [0], [0], [0], [0], [1], [0], [0], [0], [0], [0], [0], [0], [0], [2], [0], [0], [0]],
                                     'Better smell':[[0], [0], [0], [0], [0], [0], [0], [0], [2], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [1], [0]],
                                     'Passed down intelligence':[[0], [0], [0], [0], [0], [0], [0], [0], [0], [2], [0], [1], [0], [1], [0], [1], [0], [0], [0], [0], [0], [0]],
                                     'More sensetinve ears':[[0], [0], [0], [0], [0], [0], [0], [2], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
                                     'Better Reflexes':[[0], [1], [0], [0], [0], [0], [0], [0], [0], [0], [1], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
                                     'Better tools':[[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [2], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
                                     'Thicker skin':[[0], [1], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
                                     'Lysosize':[[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [1], [0]],
                                     'Stronger stomach acid':[[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [1], [0]],
                                     'Tears':[[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [1], [0]],
                                     'Quicker Reaction speed time':[[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [1], [0], [0], [0], [1], [0], [0], [0], [0], [0], [0], [0]],
                                  
                                     }
        
        # Holds a dictionary in the format
        # Name of scenario as str/character : Text assosiated with the scenario as str
        self.__scenario = {'1': ['Animals are WORD faster now', 2],
                           '2': ['WORD More Nightime Animals', 0],
                           '3': ['WORD More daytime Animals', 0],
                           '4': ['Animals have WORD thicker skin and better defences', 2],
                           '5' : ['WORD More animals attack the camp at night time', 0],
                           '6' : ['WORD More animals attack the camp at daytime', 0],
                           '7' : ['Animals become WORD more aggressive towards humans', 2],
                           '8' : ['A new type of WORD venomous snake has appeared new the human camp', 4],
                           '9' : ['An WORD dangerous unknown disease has appeared', 1],
                           '10' : ['Due to changing weather conditions, the amount of gathereable food has WORD decreased', 11],
                           '11' : ['The climate has become WORD hotter', 4],
                           '12' : ['The climate has become WORD colder', 4]
                           }
                           
        self.__normalScenarioWord = 'SLIGHTLY'
        self.__radicalScenarioWord = 'VERY'   

    def __returnImpactOfAdaptations(self,adap):
        '''Takes the name of a key as str which should be from impactOfAdaptations
        Outputs the values of parameters, mirroring the humanObject parameters
        As a nested list'''
        return self.__impactOfAdaptation[adap]
    
    
    def returnRandomScenario(self):
        '''
        

        Returns
        -------
        int
            Scenario Identifier.
        nameOfScenario : str
            The string text associated with the scenario.
        isTheScenarioRadical : Boolean
            If the scenario is a radical scenario, return True, else return False.

        '''
        
        indexOfScenario = randomInterger(1, len(self.__scenario.keys())-1)
        nameOfScenario = list(self.__scenario.keys())[indexOfScenario][0]
        isTheScenarioRadical = self.isTheScenarioRadical()
        nameOfScenario = self.__convertIntoUsableString(isTheScenarioRadical, indexOfScenario)
        
        
        
        return int(indexOfScenario), nameOfScenario, isTheScenarioRadical
        
        
    def isTheScenarioRadical(self):
        '''
        

        Returns
        -------
        bool
            Return True if scenario is radical and False else.

        '''
        
        num = randomInterger(0, 100)
        if num <= 5:
            return True
        else:
            return False
        
    def __convertIntoUsableString(self,isTheScenarioRadical, indexOfScenario):
        '''
        

        Parameters
        ----------
        isTheScenarioRadical : Bool
            True or False representing whether the scenario has escalated.
        indexOfScenario : Int
            Scenario identifier.

        Returns
        -------
        aList : List
            Returns a list with a correct strings whether it is a radical scenario or not.

        '''
        
        result = [[k, v] for k,v in self.__scenario.items()]
        
        nameOfScenario = result[indexOfScenario-1][1][0]
        nameOfScenario = nameOfScenario.split(' ')
        whereIsTheWord = result[indexOfScenario-1][1][1]
        

        
        if isTheScenarioRadical == True:
            nameOfScenario[whereIsTheWord] = self.__radicalScenarioWord
        else:
            nameOfScenario[whereIsTheWord] = self.__normalScenarioWord
        
        aList = ''
        for i in nameOfScenario:
            aList += i + ' '
        
        return aList
    
    
    def __returnScenarioText(self, scenarioNumber):
        '''Takes a scenarioNumber of the scenario dic as a str
        Returns a text of a specific scenario as str'''
        return self.__scenario[scenarioNumber]
    
    def returnRandomAdaptation(self):
        '''Returns name of a random adaptation as str
        Returns the a nested list associated with the name key as a nested key'''
        
        # Picks a random adaptation
        a = list(self.__impactOfAdaptations.keys())[random.randint(0, len(self.__impactOfAdaptations.keys()))-1]
        # Picks the parameters associated with the random adaptation
        b = self.__impactOfAdaptations[a]
        return a,b


class Human:
    '''Represents an object which is used to represent a single human'''
    
    def __init__(self):
        '''Initialises the human object
        Iniatilises 20 variables as lists with a default value
        '''
        a = Settings()
        
        self.__startingValue = a.returnDefaultParameterValue()
        
        # The variables represent ratings of 'how good a human is at...' as a list with a singluar value
        # Min value = 0
        # Max value = 10
        self.__throwing = [self.__startingValue]
        self.__thicknessofSkin = [self.__startingValue]
        self.__quickRunning = [self.__startingValue]
        self.__distanceRunning = [self.__startingValue]
        self.__catching = [self.__startingValue]
        self.__nightVision = [self.__startingValue]
        self.__dayVision = [self.__startingValue]
        self.__hearing = [self.__startingValue]
        self.__smell = [self.__startingValue]
        self.__intelligence = [self.__startingValue]
        self.__fighting = [self.__startingValue]
        self.__toolsLevel = [self.__startingValue]
        self.__weaponLevel = [self.__startingValue]
        self.__fireMaking = [self.__startingValue]
        self.__defending = [self.__startingValue]
        self.__clothing = [self.__startingValue]
        self.__canSurviveHotterClimates = [self.__startingValue]
        self.__canSurviveColderClimates = [self.__startingValue]
        self.__height = [self.__startingValue]
        self.__hairDensity = [self.__startingValue]
        self.__resistanceToDisease = [self.__startingValue]
        self.__sneaking = [self.__startingValue]
        
    
    def returnParameters(self):
        '''Returns every parameter of the humanObject as a nested list'''
        return [self.__throwing,
        self.__thicknessofSkin,
        self.__quickRunning,
        self.__distanceRunning,
        self.__catching,
        self.__nightVision,
        self.__dayVision,
        self.__hearing,
        self.__smell,
        self.__intelligence,
        self.__fighting,
        self.__toolsLevel,
        self.__weaponLevel,
        self.__fireMaking,
        self.__defending,
        self.__clothing,
        self.__canSurviveHotterClimates,
        self.__canSurviveColderClimates,
        self.__height,
        self.__hairDensity,
        self.__resistanceToDisease,
        self.__sneaking]

    def updateParameters(self,parameterLocation, change):
        '''Takes parameterLocation as an int 
        Takes change as an int
        Locates a specific parameter of humanObject and changes its value according to the change
        By default, it adds the current parameter and the change
        Prints a statement whether the update of parameters was successful'''
        
        
        parameters = self.returnParameters()
        if not((parameters[parameterLocation][0] < 0) or (parameters[parameterLocation][0] >= 10)):
            parameters[parameterLocation][0] += change
        
    
    def reproductionOfParameters(self, humanObjectParameters ):
        
        
        '''
        Takes in object variables of a human which will reproduce 
       Change the stats of the human with the better human

        '''
        for i in range(len(self.returnParameters())):
            if self.returnParameters()[i][0] != humanObjectParameters[i][0]:
                self.returnParameters()[i][0] = humanObjectParameters[i][0]   
    
    def splitParameters(self, parametersOfAdaptation):
        '''
        

        Parameters
        ----------
        parametersOfAdaptation : Nested List
            .

        Returns
        -------
        None.

        '''
        
        for i in range(len(parametersOfAdaptation)):
            
            self.updateParameters(i, parametersOfAdaptation[i][0])
                
        print('Successfully updated parameter')



class HumanObjectApplication:
    
    '''Represents an object which'''
    def __init__(self):
        '''
        

        Returns
        -------
        None.

        '''
        self.__humanNameCounter = 0
    
        a = Settings()
        self.__limitOfSystem = a.returnLimitOfSystem()
        self.__initialNumberOfHumans = a.returnInitialNumberOfHumans()
        
        self.__listOfNames = self.__names()
        self.__humanObjects = self.__humanGen()
        
        
        
    def returnAmountOfHumans(self):
        '''
        

        Returns
        -------
        int
            Returns the amount of humanObjects.

        '''
        return len(self.__humanObjects)
        
    def __names(self):
        '''
        

        Returns
        -------
        a : List of strings
            Contains list of unique names, which can be used as unique identifiers.

        '''
        a = []
        for i in range(1000): #Limit of the system self.__limitOfSystem
            name = names.get_first_name()
            while name in a:
                name = names.get_first_name()
            a.append(name)
        
        return a
    
    def __humanGen(self):
        '''
        

        Returns
        -------
        a : Dictionary
            Has associated unique name identifier is a Human() class object.

        '''
        a = {}
        for i in range(self.__initialNumberOfHumans): #Initial Number of humans
            a[self.__listOfNames[i]] = Human()
            self.__humanNameCounter += 1
        return a
    
    def __pickRandomHumanObject(self):
        '''
        

        Returns
        -------
        a : str
            Name, unique identifier of an object.
        Object
            Object associated with a.

        '''
        randomHumanDicIndex = random.randint(0,len(self.returnHumanObjects())-1)
        a = list(self.__humanObjects.keys())[randomHumanDicIndex]
        return a, self.__humanObjects[a]    
        
    
    def pickRandomHuman(self):
        '''
        

        Returns
        -------
        object
            Returns an human object.

        '''
        return self.__pickRandomHumanObject()
    
    def returnHumanObjects(self):
        '''
        

        Returns
        -------
        dict
            Returns a dic of human objects.

        '''
        return self.__humanObjects
    
    
    def newAdaptation(self,parametersOfAdaptation):
        '''
        

        Parameters
        ----------
        parametersOfAdaptation : nested List
            Parameters of an adaptation in the format [[int],[int]...].

        Returns
        -------
        name : str
            Unique identifier of a object.
        humanObject : object
            Object associated with name.

        '''
        
        name, humanObject = self.__pickRandomHumanObject()
        
        humanObject.splitParameters(parametersOfAdaptation)
        
        return name, humanObject
    

    def createSingleHuman(self): 
        '''
        

        Returns
        -------
        object
            Returns the newly created object.

        '''
        name = self.__listOfNames[self.__humanNameCounter]
        self.__humanObjects[name] = Human()
        
        self.__humanNameCounter += 1
        humanToBeCopiedName, HumanToBeCopiedObject = self.__pickRandomHumanObject()
        humanParametersToBeCopied = HumanToBeCopiedObject.returnParameters()
        self.__humanObjects[name].reproductionOfParameters(humanParametersToBeCopied)
        
        
        return self.__humanObjects[name]
    
    def deleteTheseHumanObjects(self, namesOfHumans):
        '''
        

        Parameters
        ----------
        namesOfHumans : List
            List with names of humans to be deleted.

        Returns
        -------
        None.

        '''
        keyOfObjectsToBeRemoved = []
        for name, objectLocation in self.__humanObjects.items():
            if objectLocation in namesOfHumans:
                keyOfObjectsToBeRemoved.append(name)
                
        for i in range(len(keyOfObjectsToBeRemoved)):
            del self.__humanObjects[keyOfObjectsToBeRemoved[i]]




class Neanthedral(Human):
    
    def __init__(self):
        '''
        

        Returns
        -------
        None.

        '''
        
        super().__init__()
        
        a = Settings()
        self.__startingValue = a.returnParameterValueOfDefaultNeanthedral()
        self.__SneakingValueForNeanthedral = a.returnSneakingValueForNeanthedral()
        
        
        
        self.__sneaking = [self.__SneakingValueForNeanthedral]

class NeanthedralObjectApplication(object):
    
    def __init__(self):
        '''
        

        Returns
        -------
        None.

        '''
        
        
        
        self.__humanNameCounter = 0
        

        
        
        a = Settings()
        
        self.__limitOfSystem = a.returnLimitOfSystem()
        self.__initialNumberOfHumans = a.returnInitialNumberOfNeanthedrals()
        
        self.__listOfNames = self.__names()
        self.__humanObjects = self.__humanGen()
        
    def __humanGen(self):
        '''
        

        Returns
        -------
        a : Dictionary
            Has associated unique name identifier is a Human() class object.

        '''
        a = {}
        for i in range(self.__initialNumberOfHumans): #Initial Number of Neanthedral
            a[self.__listOfNames[i]] = Neanthedral()
        return a
    
    def __names(self):
        '''
        
        Returns
        -------
        a : List of strings
            Contains list of unique names, which can be used as unique identifiers.

        '''
        a = []
        for i in range(1000): #Limit of the system
            name = 'Neanthedral' + str(i)
            a.append(name)
        
        return a  
 
    def returnAmountOfHumans(self):
        '''
        

        Returns
        -------
        int
            Returns the amount of humanObjects.

        '''
        return len(self.__humanObjects)
    
    def __pickRandomHumanObject(self):
        '''
        

        Returns
        -------
        a : str
            Name, unique identifier of an object.
        Object
            Object associated with a.

        '''
        randomHumanDicIndex = random.randint(0,len(self.returnHumanObjects())-1)
        a = list(self.__humanObjects.keys())[randomHumanDicIndex]
        return a, self.__humanObjects[a]    
        

       
    
    def pickRandomHuman(self):
        '''
        

        Returns
        -------
        object
            Returns an human object.
        '''
        return self.__pickRandomHumanObject()
    
    def returnHumanObjects(self):
        '''
        

        Returns
        -------
        dict
            Returns a dic of human objects.

        '''
        return self.__humanObjects
    
    
    def newAdaptation(self,parametersOfAdaptation):
        '''
        

        Parameters
        ----------
        parametersOfAdaptation : nested List
            Parameters of an adaptation in the format [[int],[int]...].

        Returns
        -------
        name : str
            Unique identifier of a object.
        humanObject : object
            Object associated with name.

        '''
        
        name, humanObject = self.__pickRandomHumanObject()
        
        humanObject.splitParameters(parametersOfAdaptation)
        
        return name, humanObject
    

    
    
    def createSingleHuman(self): 
        '''
        

        Returns
        -------
        object
            Returns the newly created object.

        '''
        name = self.__listOfNames[self.__humanNameCounter]
        self.__humanObjects[name] = Human()
        
        #print('Human Pop', len(self.__humanObjects))
        self.__humanNameCounter += 1
        #print('Successfully added a new human')
        humanToBeCopiedName, HumanToBeCopiedObject = self.__pickRandomHumanObject()
        humanParametersToBeCopied = HumanToBeCopiedObject.returnParameters()
        self.__humanObjects[name].reproductionOfParameters(humanParametersToBeCopied)
        
        
        return self.__humanObjects[name]
    
    def deleteTheseHumanObjects(self, namesOfHumans):
        '''
        

        Parameters
        ----------
        namesOfHumans : List
            List with names of humans to be deleted.
            
        Returns
        -------
        None.

        '''
        keyOfObjectsToBeRemoved = []
        for name, objectLocation in self.__humanObjects.items():
            if objectLocation in namesOfHumans:
                #print(objectLocation)
                keyOfObjectsToBeRemoved.append(name)
                
        for i in range(len(keyOfObjectsToBeRemoved)):
            del self.__humanObjects[keyOfObjectsToBeRemoved[i]]
    
    
class ControlSystemApp(object):
    
    def __init__(self):
        '''
        

        Returns
        -------
        None.

        '''
        a = Settings()
        ## fetches the default values from Settings() object
        self.__year = a.returnStartingYearAsInt()
        
        self.__generation = a.returnYearlyValueOfAGeneration()
        
        self.__humanApp = HumanObjectApplication()
        self.__modelApp = ModelCreationApp()
        self.__variableHolder = VariableHolder()
        
        self.__howManyTimesEvoRepeats = a.returnNumberOfTimesEvolutionHappens()
        
        self.__howManyHumansGetAdaptationsPerGen = a.returnHowManyHumansGetTheSameAdaptationAtOnce()
        
        self.__NeanthedralApp = NeanthedralObjectApplication()
        
        
        self.__state = 'Evolution'
        self.__previousYear = self.__year + 50
        
        self.__howManyTimesEvolRepeated = 0
        self.__howManyTimesExinctionRepeated = 0
        
        
        
        self.__whatToPrintOnDisplay = []
        
        self.__ifPopBelowEndExtinction = a.returnifPopBelowEndExtinction()
        
        self.__limitOfSystem = a.returnLimitOfSystem() 
        
    def __returnYear(self):
        '''
        

        Returns
        -------
        str
            Returns year as string with BC.

        '''
        return str(self.__year) +'BC'
    
    def __getNextCycle(self):
        '''
        

        Returns
        -------
        None.

        '''
        self.__previousYear = self.__year
        a = randomInterger(400, self.__generation)
        self.__year -= a
    
    def __welcomeM(self):
        '''
        

        Returns
        -------
        None.

        '''
        a = 'EVOLUTION SIM STARTS...'
        print(a,'\n')
        self.addWhatToPrintOnDisplay(a)        
        
    def __startOfProgram(self):
        '''
        

        Returns
        -------
        None.

        '''
        self.__welcomeM()
        y = self.__returnYear()
        p = len(self.__humanApp.returnHumanObjects())
        a = 'Year '+ str(y) + ' Population: ' + str(p)
        print(a)
        self.addWhatToPrintOnDisplay(a)
        
    def addWhatToPrintOnDisplay(self,string):
        '''
        

        Parameters
        ----------
        string : str
            A string of what to display on screen.

        Returns
        -------
        None.

        '''
        
        self.__whatToPrintOnDisplay.append(str(string))
    
     
    def __nextCycleMessage(self, nean=False):
        '''
        

        Parameters
        ----------
        nean : Boolean, optional
            If this function should be applied to Neantheral() object, nean equals True, else equals False. The default is False.

        Returns
        -------
        None.

        '''
        
        self.__getNextCycle()
        y = self.__returnYear()
        p = len(self.__humanApp.returnHumanObjects())
        a = 'Year ' +  str(y) + ' Population Human: ' + str(p)
        print(a)
        self.addWhatToPrintOnDisplay(a)
        
        
        if nean != False:
            Len = len(self.__NeanthedralApp.returnHumanObjects())
            a = 'Population Neanthedral: ' + str(Len)
            print(a)
            self.addWhatToPrintOnDisplay(a)
        
    def __nextCycle(self, nean=False): 
        '''
        

        Parameters
        ----------
        nean : Boolean, optional
            If this function should be applied to Neantheral() object, nean equals True, else equals False. The default is False.


        Returns
        -------
        None.

        '''
        
        if nean == False:
            a = False
        else:
            a = True
        
        self.__generalPopulationGrowth(nean=a)
        
        
    def __giveAdaptations(self, nean=False):
        '''
        

        Parameters
        ----------
        nean : Boolean, optional
            If this function should be applied to Neantheral() object, nean equals True, else equals False. The default is False.


        Returns
        -------
        None.

        '''
        
        if nean == False:
            a = self.__humanApp
        else:
            a = self.__NeanthedralApp
        
        # fetches a random adaptation, with its parameters
        nameOfAdaptation, parametersOfAdaptation = self.__variableHolder.returnRandomAdaptation()
        
        for i in range(self.__howManyHumansGetAdaptationsPerGen):
            # adds the adaptation to a random human
            nameOfHumanReceivingTheAdaptation, humanObject = a.newAdaptation(parametersOfAdaptation)
            text = nameOfHumanReceivingTheAdaptation + ' has got a adaptation: ' + nameOfAdaptation
            print(text)
            self.addWhatToPrintOnDisplay(text)
            
                
        
    def __nextScenario(self):
        '''
        

        Returns
        -------
        nameOfScenario : int
            Unique identifier of a scenario.
        textOfScenario : str
            A string associated with nameOfScenario.
        impact : boolean
            If a radical scenario, impact equals true, else false.

        '''
        nameOfScenario, textOfScenario, impact = self.__variableHolder.returnRandomScenario()
        a = 'Scenario ' + str(nameOfScenario) + ' occurs'
        print(a) 
        print(textOfScenario)
        
        self.addWhatToPrintOnDisplay(a)
        self.addWhatToPrintOnDisplay(textOfScenario)
        
        return nameOfScenario, textOfScenario, impact
    

    
    
     
    def EvolutionSimOneLoop(self):
        '''
        

        Returns
        -------
        boolean
            State of the simulation.
        List of strings
            What would you like to print on display.

        '''
        
        # if this is the first time evolution sim has been executed
        if self.__howManyTimesEvolRepeated == 0:
            self.__startOfProgram() 
        
        
        
        # if evolution sim should be finished
        if self.__howManyTimesEvolRepeated == self.__howManyTimesEvoRepeats:
            
            a = 'Finished desired amount of evolutions'
            print(a)
            self.__state = 'Extinction'
            self.addWhatToPrintOnDisplay(a)
            
            return self.__state, self.__whatToPrintOnDisplay
            ## do something here
        
        
        
        

        
        #Random get the same adaptation
        for i in range(self.__howManyHumansGetAdaptationsPerGen):
                self.__giveAdaptations()

        ## A new scenario
        nameOfScenario, textOfScenario, Impact = self.__nextScenario()
        
        
        
        
        
        ## Train model for this scenario
        self.__trainModel(nameOfScenario)
        
        
        ## Put the parameters of every human into the model and find the success rate
        
        dictionary = {}
        for i in self.__humanApp.returnHumanObjects(): # gives out the names of the objects
            a = self.__returnParametersOfCertainHuman(i)# gives parameters of each human
            b = self.__returnObjectOfHuman(i)
            ## pass these into the model
            pred = self.__calcPrediction(a)
            
            print('Prediction for ', i,'with ratings', a,' === ', pred)
            ## Put the humanObject into a dictionary with 
            dictionary[b] = pred
            ## Rank each humanObject
            
            
        dictionary = {k:v for k,v, in sorted(dictionary.items(), key=lambda item: item[1])}
        
        ## Take the worst and delete humans
        self.__delHuman(dictionary, Impact) 
        
        
        # population increase
        self.__nextCycle()
        
        self.__nextCycleMessage()
        
        self.__howManyTimesEvolRepeated += 1
        
        
        
        return self.__state, self.__whatToPrintOnDisplay
        
    
    def resetWhatToPrint(self):
        '''
        

        Returns
        -------
        None.

        '''
        
        self.__whatToPrintOnDisplay = []

    def __returnParametersOfCertainHuman(self,i,nean=False):
        '''
        

        Parameters
        ----------
        i : str
            Identifier of a certain object.
        nean : Boolean, optional
            If this function should be applied to Neantheral() object, nean equals True, else equals False. The default is False.


        Returns
        -------
        Nested List
            Nested list of parameters associated with i human.

        '''
        
        if nean == False:
            a = self.__humanApp
        else:
            a = self.__NeanthedralApp
        
        return a.returnHumanObjects()[i].returnParameters()
    
    def __returnObjectOfHuman(self,i, nean=False):
        '''
        

        Parameters
        ----------
        i : str
            Identifier of a certain object.
        nean : Boolean, optional
            If this function should be applied to Neantheral() object, nean equals True, else equals False. The default is False.


        Returns
        -------
        object
            Returns an object associated with i.

        '''
        
        if nean == False:
            a = self.__humanApp
        else:
            a = self.__NeanthedralApp
        
        return a.returnHumanObjects()[i]
    
        
    def __trainModel(self, nameOfScenario):
        '''
        

        Parameters
        ----------
        nameOfScenario : int
            Unique identifier of a scenario.

        Returns
        -------
        None.

        '''
        self.__modelApp.modelCreation(int(nameOfScenario))
        
    def __calcPrediction(self,a):
        '''
        

        Parameters
        ----------
        a : Nested list
            A list of parameters of a certain object in the format [[int],[int]...].

        Returns
        -------
        pred : float
            Prediction calculated based on a.

        '''
        pred = self.__modelApp.calcPrediction(a).item()
        return pred
        
    def __startOfExtinctionSim(self):
        '''
        

        Returns
        -------
        None.

        '''
        self.__year += 5000
        a = 'Extinction Sim...'
        print('\n', a)
        self.addWhatToPrintOnDisplay(a)
    
    def ExtinctionSimOneLoop(self):
        '''
        

        Returns
        -------
        boolean
            State of the simulation.
        List of strings
            What would you like to be displayed on screen.

        '''
        
        
        ## if this is the first time extinction has repeated
        if self.__howManyTimesExinctionRepeated == 0:
            self.__startOfExtinctionSim()   
            y = self.__returnYear()
            p = len(self.__humanApp.returnHumanObjects())
            a = 'Year '+ str(y) + ' Population: ' + str(p)
            print(a)
            
            self.addWhatToPrintOnDisplay(a)
            p = len(self.__NeanthedralApp.returnHumanObjects())            
            a = 'Year '+ str(y) + ' Population: ' + str(p)
            print(a)
            self.addWhatToPrintOnDisplay(a) 

       # if the simulation should have been finished
        if self.__isSimulationOver() == True:
            a = 'The extinction Simulator has finished'
            print(a)
            self.addWhatToPrintOnDisplay(a)
            winner, loser = self.__checkWinner()
            a = winner + ' has forced ' + loser + ' into extinction.'
            print(a)
            self.__state = 'Over'
            self.addWhatToPrintOnDisplay(a)
            
            aList = []
            for i in self.__humanApp.returnHumanObjects(): # gives out the names of the objects
                a = self.__returnParametersOfCertainHuman(i)
                # returns their parameters
                b = convertParametersIntoASingleValue(a)
                # converts them into a single value by adding them up
                aList.append(b)
            
            
            c = recursiveMergeSort(aList)
            
            # picks the biggest value
            c = c[len(c)-1]
                
            a = 'Best Of the Species has cumulative value of ' + str(c)
            
            self.addWhatToPrintOnDisplay(a)
            
            return self.__state, self.__whatToPrintOnDisplay
        

        
        # gives out the adaptations to both Humans and Neanderthals      
        for i in range(self.__howManyHumansGetAdaptationsPerGen):
                self.__giveAdaptations()
                self.__giveAdaptations(nean=True)
        

        
        nameOfScenario, textOfScenario, impact = self.__nextScenario()
        
        ## Train model for this scenario
        self.__trainModel(nameOfScenario)
        ## Put the parameters of every human into the model and find the success rate
        dictionary = {}
        
        for i in self.__humanApp.returnHumanObjects(): # gives out the names of the objects
            a = self.__returnParametersOfCertainHuman(i)# gives parameters of each human
            b = self.__returnObjectOfHuman(i)
            ## pass these into the model
            pred = self.__calcPrediction(a)
         
            ## Put the humanObject into a dictionary with 
            dictionary[b] = pred
            #print('Human dic prediction',':', pred)
        
        ## calc and add neanthedrals into the dic
        for i in self.__NeanthedralApp.returnHumanObjects(): # gives out the names of the objects
            a = self.__returnParametersOfCertainHuman(i, nean=True)# gives parameters of each human
            b = self.__returnObjectOfHuman(i, nean=True)
            ## pass these into the model
            pred = self.__calcPrediction(a)
         
            ## Put the humanObject into a dictionary with 
            dictionary[b] = pred
            #print('Nean dic prediction',':', pred)
        dictionary = {k:v for k,v, in sorted(dictionary.items(), key=lambda item: item[1])}
        
        ## Take the worst and delete of both objects
        try:
            self.__delHuman(dictionary, Impact=impact) 
        except ValueError:
            return self.__state, self.__whatToPrintOnDisplay
        
        # population growth for Neanderthal
        self.__nextCycle(nean=True)
        # population growth for humans
        self.__nextCycle()
        
        self.__howManyTimesExinctionRepeated += 1
        
        
        
        self.__nextCycleMessage(nean=True)
        
        return self.__state, self.__whatToPrintOnDisplay
    
    def __checkWinner(self):
        '''
        

        Returns
        -------
        str
            The species that isn't extinct.
        str
            The species that is extinct.

        '''
        if self.__humanApp.returnAmountOfHumans() > self.__NeanthedralApp.returnAmountOfHumans():
            return 'Humans', 'Neanthedrals'
        elif self.__NeanthedralApp.returnAmountOfHumans() > self.__humanApp.returnAmountOfHumans():
            return 'Neanthedrals', 'Humans'
        else:
            print('Its a draw')
            return 'No one', 'Humans or Neanthedrals'
                    
    def __isSimulationOver(self):
        '''
        

        Returns
        -------
        bool
            If simulation over, returns True, else False.

        '''
        # Is the population of either species below the 'end of simulation' threshold
        if (self.__humanApp.returnAmountOfHumans() <= self.__ifPopBelowEndExtinction) or (self.__NeanthedralApp.returnAmountOfHumans() <= self.__ifPopBelowEndExtinction):
            return True
        else:
            return False
    
    
    def __calcPopGrowthMagnitude(self, nean=False): 
        '''
        

        Parameters
        ----------
        nean : Boolean, optional
            If this function should be applied to Neantheral() object, nean equals True, else equals False. The default is False.


        Returns
        -------
        int
            Calculates the amount of humans that should be added to the program.

        '''
        

        if nean == False:
            org = self.__humanApp.returnAmountOfHumans()
            name = ' humans'
        else:
            org = self.__NeanthedralApp.returnAmountOfHumans()
            name = ' Neanthedral'
        
        # see logistic growth in analysis
        
        c = 200
        a = 1
        b = 0.3
        x = self.__howManyTimesEvolRepeated 
        

        
        y = c/(1+a*math.exp(-b*x))

        
        popChange = y - org
        
        a = 'Trying to add: ' + str(popChange) + name
        print(a)
        self.addWhatToPrintOnDisplay(a)
        
        
        return int(round(abs(popChange)))
        


    
 
    
    def __calcPopDecreaseMagnitute(self, impact, nean=False): 
        '''
        

        Parameters
        ----------
        impact : bool
            If scenario is a radical, equals True else False.
        nean : Boolean, optional
            If this function should be applied to Neantheral() object, nean equals True, else equals False. The default is False.

        Returns
        -------
        int
            How many objects to delete.

        '''
        
        if nean == False:
            org = self.__humanApp.returnAmountOfHumans()
            name = ' humans'
        else:
            org = self.__NeanthedralApp.returnAmountOfHumans()
            name = ' Neanthedral'
        
        # see logistic decline in analysis
        
        c = 200
        a = 1
        b = 0.15
        x = self.__howManyTimesExinctionRepeated
        
        
        if impact == True:
            b += 0.5

        y = c/(1+a*math.exp(-b*x))

        
        popChange = org - y
                
        return int(round(abs(popChange)))
    
    def __generalPopulationGrowth(self, nean=False):
        '''
        

        Parameters
        ----------
        nean : Boolean, optional
            If this function should be applied to Neantheral() object, nean equals True, else equals False. The default is False.

        Returns
        -------
        None.

        '''
        

        
        if nean == False:
            a = self.__humanApp
            b = False

        else:
            a = self.__NeanthedralApp
            b = True
        '''Takes a random human and takes their parameters'''
        
        if self.__state == 'Extinction':
            num = self.__EXTINCTIONcalcPopGrowthMagnitude(b)
        else:
            num = self.__calcPopGrowthMagnitude()
        
        # is it applied to Neanderthals or Humans
        if nean==True:
            
            for i in range(num):
                if a.returnAmountOfHumans() >= self.__limitOfSystem:
                    print('System is full')
                    return None
                name, objectHuman = a.pickRandomHuman()
                parametersOfTheHuman = objectHuman.returnParameters()
                
                humanObjectToBeReplicated = a.createSingleHuman()
        else:
            for i in range(num):
                if a.returnAmountOfHumans() >= self.__limitOfSystem:
                    print('System is full')
                    return None
                name, objectHuman = a.pickRandomHuman()
                parametersOfTheHuman = objectHuman.returnParameters()
                
                humanObjectToBeReplicated = a.createSingleHuman()
         
            
        
    
    
    def __delHuman(self, dictionary, Impact=False):
        '''
        

        Parameters
        ----------
        dictionary : dict
            Dictionary with a format of unique object indentifier and their prediciton of survival.
        Impact : bool, optional
            If the scenario is a radical, equal True else False. The default is False.

        Raises
        ------
        
            If tried to delete too many humans, raise an error.

        Returns
        -------
        None.

        '''
    
        
        
        
        humanToBeRemovedObject = []
        neanthedralToBeRemovedObject = []
        
        try:
            if self.__state == 'Extinction':
                for i in range(self.__EXTINCTIONcalcPopDecreaseMagnitute(Impact)):
                
                    if list(dictionary.keys())[i] in list(self.__humanApp.returnHumanObjects().values()): 
                        humanToBeRemovedObject.append(list(dictionary.keys())[i])
                    elif list(dictionary.keys())[i] in list(self.__NeanthedralApp.returnHumanObjects().values()):
                        neanthedralToBeRemovedObject.append(list(dictionary.keys())[i])
            else:
                for i in range(self.__calcPopDecreaseMagnitute(Impact)):
                    
                    
                    if list(dictionary.keys())[i] in list(self.__humanApp.returnHumanObjects().values()): 
                        humanToBeRemovedObject.append(list(dictionary.keys())[i])
                    elif list(dictionary.keys())[i] in list(self.__NeanthedralApp.returnHumanObjects().values()):
                        neanthedralToBeRemovedObject.append(list(dictionary.keys())[i])            
        except IndexError:
            print('Tried to delete too many humans')
            raise 'End Program'
        a = 'Trying to delete ' + str(len(humanToBeRemovedObject)) + ' humans'
        self.addWhatToPrintOnDisplay(a)
        if self.__state == 'Extinction':
            a = 'Trying to delete ' + str(len(neanthedralToBeRemovedObject)) + ' neanthedral'
            self.addWhatToPrintOnDisplay(a)

        if len(humanToBeRemovedObject) != 0:
            
            self.__humanApp.deleteTheseHumanObjects(humanToBeRemovedObject)
        if len(neanthedralToBeRemovedObject) != 0:
            self.__NeanthedralApp.deleteTheseHumanObjects(neanthedralToBeRemovedObject)

    def __EXTINCTIONcalcPopGrowthMagnitude(self, nean=False):
        
        
        
        if nean == False:
            org = self.__humanApp.returnAmountOfHumans()
            b = 'humans'
        else:
            org = self.__NeanthedralApp.returnAmountOfHumans()
            b = 'Neanthedral'

        gr = 1.001

            
        gr_hr = 20
        tH = self.__previousYear - self.__year


        print('Trying to add:', abs(int(org - org*math.pow(gr, tH // gr_hr))), b)
        return abs(int(org - org*math.pow(gr, tH // gr_hr))) 
    
    def __EXTINCTIONcalcPopDecreaseMagnitute(self, impact):
        

        
        org = self.__humanApp.returnAmountOfHumans() + self.__NeanthedralApp.returnAmountOfHumans()
        
        if self.__humanApp.returnAmountOfHumans() < 10:
            print('There is less than 10 population, so kill them all')
            return self.__humanApp.returnAmountOfHumans()
        elif self.__NeanthedralApp.returnAmountOfHumans() < 10:
            print('There is less than 10 population, so kill them all')
            return self.__NeanthedralApp.returnAmountOfHumans()
        

        gr = 1.002

        gr_hr = 20
        tH = self.__previousYear - self.__year
        
        if impact == True:
            gr += 0.01
        
        
        a = abs(int(org*math.pow(gr, tH // gr_hr) - org)) # decrease in in population
        if a < 10: ##-
            a += 10 ##-
            
        print('Trying to delete:', a,'of the whole population')
        return a
        
        
class OverallApp(object):
    
    def __init__(self):
        '''
        

        Returns
        -------
        None.

        '''
        
        self.__controlApp = ControlSystemApp()
        self.__simulationState = 'Evolution'
    
    def doOneLoopOfProgram(self):
        '''
        

        Returns
        -------
        bool
            Has the program finished, true if yes, false otherwise.
        textToPrint : list of strings
            Text to be printed on display.

        '''
        
        textToPrint = ''
        # what is the state of the overall program
        if self.__simulationState == 'Evolution':
            
            self.__simulationState, textToPrint = self.__controlApp.EvolutionSimOneLoop()
            self.__controlApp.resetWhatToPrint()
            
            return False, textToPrint
        
        elif self.__simulationState == 'Extinction':
            
            self.__simulationState, textToPrint = self.__controlApp.ExtinctionSimOneLoop()
            self.__controlApp.resetWhatToPrint()
            
            return False, textToPrint
            
        elif self.__simulationState == 'Over':
            
            return True, textToPrint 
            
    
class GUI(ShowBase):

    def __init__(self):
        '''
        

        Returns
        -------
        None.

        '''
        # Initialize the ShowBase class from which we inherit, which will
        # create a window and set up everything we need for rendering into it.
        ShowBase.__init__(self)
        
        self.__OverallApp = OverallApp()
        
        self.__helpMenuText = ['This is the guide to this program', 'Press the settings button to change assumptions of the program', 'Press start to start and finish simulation', 'Press back button to go back to previous screen', 'Press exit to end program']
        


        self.__collectionOfText = []
        self.__screenTextCounter = 0 
        self.__listOfNodesToDestroy = []

        # This code puts the standard title and instruction text on screen
        self.title = OnscreenText(text="Evolution Simulator - MW",
                                  parent=base.a2dBottomRight, scale=.07,
                                  align=TextNode.ARight, pos=(-0.1, 0.1),
                                  fg=(1, 1, 1, 1), shadow=(0, 0, 0, 0.5))
        self.escapeText = genLabelText("ESC: Quit", 0)
        self.disableMouse()
        
        self.setBackgroundColor((0, 0, 0, 1))
        self.bg = loadObject("background.png", scale=400, depth=200,
                             transparency=False)
        
        self.__collectionOfButtons = []
        
        self.accept("escape", sys.exit)
        
        queueSize = 20
        self.__queue = Queue(queueSize)
        
        self.hideMouse = False
        
        self.startMenu()
    
    def startMenu(self):
        '''
        

        Returns
        -------
        None.

        '''
        
    
        self.__b = DirectButton(text=("Start","Loading!", "Start", "Start"),
                 scale=.08, command=self.__SimulationStartScreen, text_pos=(0,4))
        self.__c = DirectButton(text=("Setting","Loading!", "Setting", "Setting"),
                 scale=.08, command=self.__settingMenu, text_pos=(0,2))
        self.__d = DirectButton(text=("Exit","Exiting...", "Exit", "Exit"),
                 scale=.08, command=self.__exitProgram, text_pos=(0,-2))
        
        self.__e = DirectButton(text=("Help","Loading...", "Help", "Help"),
                 scale=.08, command=self.__helpMenu, text_pos=(0,0))
        
        self.__collectionOfButtons.append(self.__b)
        self.__collectionOfButtons.append(self.__c)
        self.__collectionOfButtons.append(self.__d)
        self.__collectionOfButtons.append(self.__e)
    
    def startMenuNotFirstTime(self):
        '''
        

        Returns
        -------
        None.

        '''
        
        self.destroyButtons()
        self.destoryText()
        
        self.bg = loadObject("background.png", scale=400, depth=200,
                     transparency=False)


                
        self.__b = DirectButton(text=("Start","Loading!", "Start", "Start"),
                 scale=.08, command=self.__SimulationStartScreen, text_pos=(0,4))
        self.__c = DirectButton(text=("Setting","Loading!", "Setting", "Setting"),
                 scale=.08, command=self.__settingMenu, text_pos=(0,2))
        self.__d = DirectButton(text=("Exit","Exiting...", "Exit", "Exit"),
                 scale=.08, command=self.__exitProgram, text_pos=(0,-2))
        
        self.__e = DirectButton(text=("Help","Loading...", "Help", "Help"),
                 scale=.08, command=self.__helpMenu, text_pos=(0,0))
        
        self.__collectionOfButtons.append(self.__b)
        self.__collectionOfButtons.append(self.__c)
        self.__collectionOfButtons.append(self.__d)
        self.__collectionOfButtons.append(self.__e)
        
        
    def __helpMenu(self):
        
        
        self.destroyButtons()
        
        self.bg = loadObject("helpMenu.png", scale=125, depth=150,
                             transparency=False)
        
        self.__c = DirectButton(text=("Back","Backinging...", "Back", "Back"),
                 scale=.1, command=self.startMenuNotFirstTime, text_pos=(8,-8))
        
        self.__collectionOfButtons.append(self.__c)
        
        self.queueTextAppear(self.__helpMenuText)
    
    def __ToDoSimulationButtonPressed(self):
        '''
        

        Returns
        -------
        None.

        '''
        
        

        self.__stop, textToPrint = self.__OverallApp.doOneLoopOfProgram()
        self.queueTextAppear(textToPrint)
        
        if self.__stop == True:
        
            a = 'Simulation has finished'
            self.queueTextAppear([a])
            self.destroyButtons()
            self.__d = DirectButton(text=("Back","Backinging...", "Back", "Back"),
                 scale=.1, command=self.startMenuNotFirstTime, text_pos=(8,-8))
        
            self.__collectionOfButtons.append(self.__d)

    def locatorForSettingsButtons(self):
        '''
        

        Returns
        -------
        int
            Position in Y-axis of the next button.

        '''
        self.__counter += 2
        return self.__counter
        

    def __settingMenu(self):
        '''
        

        Returns
        -------
        None.

        '''
        
        self.__counter = -14.5
        self.__scale = .07
        
        
        self.destroyButtons()
        
        self.bg = loadObject("settingMenu.png", scale=125, depth=150,
                             transparency=False)
    
        self.__1 = DirectButton(text=("Batch Size","Batch Size", "Batch Size", "Batch Size"),
            scale=self.__scale, command=self.takeInput, extraArgs = ['self.__batchSize'], text_pos=(-5, self.locatorForSettingsButtons()))        
        self.__2 = DirectButton(text=("When to end Extinction","When to end Extinction", "When to end Extinction", "When to end Extinction"),
            scale=self.__scale, command=self.takeInput, extraArgs = ['self.__returnifPopBelowEndExtinction'], text_pos=(-5, self.locatorForSettingsButtons()))   
        
        self.__3 = DirectButton(text=("Significance Figure","Significance Figure", "Significance Figure", "Significance Figure"),
             scale=self.__scale, command=self.takeInput, extraArgs = ['self.__sig'], text_pos=(-5, self.locatorForSettingsButtons()))
             
        self.__4 = DirectButton(text=("Quantity of Training Data","Quantity of Training Data", "Quantity of Training Data", "Quantity of Training Data"),
             scale=self.__scale, command=self.takeInput, extraArgs = ['self.__howMuchTrainingDataShouldThereBe'], text_pos=(-5, self.locatorForSettingsButtons()))
    
        self.__5 = DirectButton(text=("Human Default Parameter Value","Human Default Parameter Value", "Human Default Parameter Value", "Human Default Parameter Value"),
             scale=self.__scale, command=self.takeInput, extraArgs = ['self.__returnDefaultParameterValue'], text_pos=(-5, self.locatorForSettingsButtons()))
             
        self.__6 = DirectButton(text=("Neanthedral Default Parameter Value","Neanthedral Default Parameter Value", "Neanthedral Default Parameter Value", "Neanthedral Default Parameter Value"),
             scale=self.__scale, command=self.takeInput, extraArgs = ['self.__returnParameterValueOfDefaultNeanthedral'], text_pos=(-5, self.locatorForSettingsButtons()))
             
        self.__7 = DirectButton(text=("Neanthedral Sneaking Value","Neanthedral Sneaking Value", "Neanthedral Sneaking Value", "Neanthedral Sneaking Value"),
             scale=self.__scale, command=self.takeInput, extraArgs = ['self.__returnSneakingValueForNeanthedral'], text_pos=(-5, self.locatorForSettingsButtons()))
             
        self.__8 = DirectButton(text=("Initial Number of Neanthedral","Initial Number of Neanthedral", "Initial Number of Neanthedral", "Initial Number of Neanthedral"),
             scale=self.__scale, command=self.takeInput, extraArgs = ['self.__returnInitialNumberOfNeanthedrals'], text_pos=(-5, self.locatorForSettingsButtons()))
        
        self.__9 = DirectButton(text=("Limit of the system","Limit of the system", "Limit of the system", "Limit of the system"),
             scale=self.__scale, command=self.takeInput, extraArgs = ['self.__returnLimitOfSystem'],text_pos=(-5, self.locatorForSettingsButtons()))
             
        self.__10 = DirectButton(text=("Initial Number of Humans","Initial Number of Humans", "Initial Number of Humans", "Initial Number of Humans"),
             scale=self.__scale, command=self.takeInput, extraArgs = ['self.__returnInitialNumberOfHumans'],text_pos=(-5, self.locatorForSettingsButtons()))
             
        self.__11 = DirectButton(text=("Starting Year as BC","Starting Year as BC", "Starting Year as BC", "Starting Year as BC"),
             scale=self.__scale, command=self.takeInput, extraArgs = ['self.__returnStartingYearAsInt'], text_pos=(-5, self.locatorForSettingsButtons()))
             
        self.__12 = DirectButton(text=("How long is one Generation","How long is one Generation", "How long is one Generation", "How long is one Generation"),
             scale=self.__scale, command=self.takeInput, extraArgs = ['self.__returnYearlyValueOfAGeneration'],text_pos=(-5, self.locatorForSettingsButtons()))
             
        self.__13 = DirectButton(text=("How many times Evolution Happens","How many times Evolution Happens", "How many times Evolution Happens", "How many times Evolution Happens"),
             scale=self.__scale, command=self.takeInput, extraArgs = ['self.__returnNumberOfTimesEvolutionHappens'],text_pos=(-5, self.locatorForSettingsButtons())) 
                
        self.__14 = DirectButton(text=("How many get the same adaptation","How many get the same adaptation", "How many get the same adaptation", "How many get the same adaptation"),
             scale=self.__scale, command=self.takeInput, extraArgs = ['self.__returnHowManyHumansGetTheSameAdaptationAtOnce'], text_pos=(-5, self.locatorForSettingsButtons()))
        
        
        self.__collectionOfButtons.append(self.__1)
        self.__collectionOfButtons.append(self.__2)
        self.__collectionOfButtons.append(self.__3)
        self.__collectionOfButtons.append(self.__4)
        self.__collectionOfButtons.append(self.__5)
        self.__collectionOfButtons.append(self.__6)
        self.__collectionOfButtons.append(self.__7)
        self.__collectionOfButtons.append(self.__8)
        self.__collectionOfButtons.append(self.__9)
        self.__collectionOfButtons.append(self.__10)
        self.__collectionOfButtons.append(self.__11)
        self.__collectionOfButtons.append(self.__12)
        self.__collectionOfButtons.append(self.__13)
        self.__collectionOfButtons.append(self.__14)
        
        
        ### Back button
        self.__c = DirectButton(text=("Back","Backinging...", "Back", "Back"),
                 scale=.1, command=self.startMenuNotFirstTime, text_pos=(8,-8))
        
        
        self.__collectionOfButtons.append(self.__c)
    
    def destroyButtons(self):
        
        if len(self.__collectionOfButtons) != 0:
            for i in self.__collectionOfButtons:
                i.destroy()
        self.__collectionOfButtons = []
    
    def destoryText(self):
        
        if len(self.__collectionOfText) != 0:
            for i in self.__collectionOfText:
                i.destroy()
        self.__collectionOfText = []
        self.__screenTextCounter = 0
            
    def __exitProgram(self):
        '''
        

        Returns
        -------
        None.

        '''
        sys.exit("Program Finished") 
        

    def __SimulationStartScreen(self):
        '''
        

        Returns
        -------
        None.

        '''
        
        self.destroyButtons()
        
        
        self.bg = loadObject("evolutionSimScreen.png", scale=125, depth=150,
                             transparency=False)
        
        self.__b = DirectButton(text=("Start","Loading!", "Start", "Start"),
                 scale=.08, command=self.__ToDoSimulationButtonPressed, text_pos=(6,-9))
        
        self.__c = DirectButton(text=("Back","Backinging...", "Back", "Back"),
                 scale=.08, command=self.startMenuNotFirstTime, text_pos=(10,-9))
            
        self.__collectionOfButtons.append(self.__b)
        self.__collectionOfButtons.append(self.__c)
        
    def takeInput(self,nameOfVariable):
        '''
        

        Parameters
        ----------
        nameOfVariable : str
            Name of the variable that needs to be changed.

        Returns
        -------
        None.

        '''
        
        stop = False
        
        while stop == False:
            Input = input('INPUT >>>')
            try:
                if int(Input) > 0:
                    settingsWriter(nameOfVariable, int(Input))
                    stop = True
                else:
                    raise ValueError
            except ValueError:
                print('Wrong Data Type')
            
         
    def queueTextAppear(self, toPrint):
        '''
        

        Parameters
        ----------
        toPrint : List of strings
            Strings to be displayed on GUI.

        Returns
        -------
        Panda3D task manager
            Tells Panda3D task manager that the task has been completed.

        '''
        

        for i in toPrint:
            self.__queue.addQueue(i)
        
        finished = False
        
        while finished == False:
            try:
                taskMgr.add(self.makeTextAppear, 'Printing Text on screen', extraArgs = [self.__queue.delQueue()])
            except ArrayException:
                print('Queue is empty')
                finished = True
                
        return Task.done
    
    def makeTextAppear(self, toPrint):
        '''
        

        Parameters
        ----------
        toPrint : List of strings
            List of strings to be displayed on GUI.

        Returns
        -------
        None.

        '''
        
        if len(self.__collectionOfButtons) != 0:
            for i in self.__collectionOfButtons:
                i['state'] = DGG.DISABLED
        
        
        
        if self.__screenTextCounter >= 1.5:
            self.__screenTextCounter = 0
            for i in self.__listOfNodesToDestroy:
                i.destroy()
            self.__listOfNodesToDestroy = []

        
            
        self.__text = OnscreenText(text=toPrint,
                                  parent=base.a2dBottomRight, scale=.04,
                                  align=TextNode.ARight, pos=(-0.4, 1.8-self.__screenTextCounter),
                                  fg=(0, 0, 0, 1), shadow=(0, 0, 0, 0.5))
        
        
        
        self.__listOfNodesToDestroy.append(self.__text)
        
        self.__collectionOfText.append(self.__text)
        
        self.__screenTextCounter += 0.05
        
        if len(self.__collectionOfButtons) != 0:
            for i in self.__collectionOfButtons:
                i['state'] = DGG.NORMAL
        
        
class Array(object):
    """A simulated Array in Python because Python does not have arrays"""
    def __init__(self, size):
        self.__size = size
        self.__array = []
        for i in range(size):
            self.__array.append(None)

    def getSize(self):
        """Returns the size of the array"""
        return self.__size

    def get(self, n):
        """Returns the value in index n"""
        if n>= self.__size or n<0:
            raise ArrayException("Index "+str(n)+" out of bounds.")
        
        return self.__array[n]
        
        
        
    def assign(self, n, value):
        """Sets element n to value"""
        if n>= self.__size or n<0:
            raise ArrayException("Index "+str(n)+" out of bounds.")
        self.__array[n] = value
        


class ArrayException(Exception):
    def __init__(self, value):
        self.value = value
    def toString(self):
        return self.value 


class Queue(object):
    
    def __init__(self,queueSize):
        
        self.__array = Array(queueSize) 
        self.__headPointer = 1
        self.__tailPointer = 1
        self.__queueSize = queueSize
    
    def addQueue(self,Input): 
        
        if not(self.__tailPointer == self.__headPointer - 1): 
            self.__array.assign(self.__tailPointer, Input)
            self.__tailPointer += 1
        else: raise ArrayException("The queue is full. Can't Add")
        
        if self.__tailPointer == self.__queueSize: self.__tailPointer = 0 
        
    def isEmpty(self):
        
        return self.__headPointer == self.__tailPointer
    
    def isFull(self):
        
        if self.__tailPointer == self.__queueSize-1: return True
        else: return False
        
    def delQueue(self):
        
        
        if not(self.isEmpty()): 
            
            item = self.__array.get(self.__headPointer)
            self.__headPointer += 1

            if self.__headPointer == self.__queueSize: self.__headPointer = 0 
                
            return item
        else:
            raise ArrayException("Can't delQueue")
        
        
        
    

def loadObject(tex=None, pos=LPoint3(0, 0), depth=SPRITE_POS, scale=1,
               transparency=True):
    '''
    

    Parameters
    ----------
    tex : Panda3D Texture, optional
        Does the object have a texture. The default is None.
    pos : Panda3D coordinates, optional
        Coordinates of where do you want the object. The default is LPoint3(0, 0).
    depth : Int, optional
        How wide do you want the object. The default is SPRITE_POS.
    scale : float, optional
        How scaled do you want the object. The default is 1.
    transparency : boolean, optional
        Do you want the object to be transparent. The default is True.

    Returns
    -------
    obj : Panda3D object
        Object generated.

    '''
    # Every object uses the plane model and is parented to the camera
    # so that it faces the screen.
    obj = loader.loadModel("models/plane")
    obj.reparentTo(camera)

    # Set the initial position and scale.
    obj.setPos(pos.getX(), depth, pos.getY())
    obj.setScale(scale)

    # This tells Panda not to worry about the order that things are drawn in
    # (ie. disable Z-testing).  This prevents an effect known as Z-fighting.
    obj.setBin("unsorted", 0)
    obj.setDepthTest(False)

    if transparency:
        # Enable transparency blending.
        obj.setTransparency(TransparencyAttrib.MAlpha)

    if tex:
        # Load and set the requested texture.
        tex = loader.loadTexture("textures/" + tex)
        obj.setTexture(tex, 1)

    return obj


def genLabelText(text, i):
    '''
    

    Parameters
    ----------
    text : str
        Text you want printed.
    i : float
        Position of the text on the screen.

    Returns
    -------
    Panda3D object
        Returns a generated text object.

    '''
    return OnscreenText(text=text, parent=base.a2dTopLeft, pos=(0.07, -.06 * i - 0.1),
                        fg=(1, 1, 1, 1), align=TextNode.ALeft, shadow=(0, 0, 0, 0.5), scale=.05)




def mergeSort(arr):
    
    if len(arr) > 1:
 
         # Finding the mid of the array
        mid = len(arr)//2
 
        # Dividing the array elements
        L = arr[:mid]
 
        # into 2 halves
        R = arr[mid:]
 
        # Sorting the first half
        mergeSort(L)
 
        # Sorting the second half
        mergeSort(R)
 
        i = j = k = 0
 
        # Copy data to temp arrays L[] and R[]
        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1
 
        # Checking if any element was left
        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1
 
        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1

def recursiveMergeSort(aList):
    
    '''Sorts a list of intergers'''
    
    copyAList = aList
    mergeSort(copyAList)
    return copyAList

def convertParametersIntoASingleValue(aList):
    
    '''Takes the ratings of an objects and add them up'''
    
    valueOfOneRow = 0
        
    for j in range(len(aList)):
            
        valueOfOneRow += int(aList[j][0])
        
        
    return valueOfOneRow


def start():
    '''
    

    Returns
    -------
    None.

    '''
    
    a = GUI()
    a.run()

start()
