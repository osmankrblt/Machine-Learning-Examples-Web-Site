import numpy as np
import pandas as pd

""" def csv2array(csvPath):

    csvFile = np.genfromtxt(csvPath,delimiter=',',dtype=np.unicode_)
    
    
    
    inputLabels = np.array(list([x  for x in csvFile[0,:] if x.isupper()]))
    outputLabels = np.setdiff1d(csvFile[0,:],inputLabels)
    
    print(csvFile[1:].T)

    stripCsv = np.char.strip(csvFile[1:].T) 

    print(stripCsv)

    data =  stripCsv.astype(np.double)

    X,y = data[0:len(inputLabels)],data[len(inputLabels):]

    return inputLabels,outputLabels,(X.T,y.T) """

def csv2array(csvPath):

    csvFile = pd.read_csv(csvPath)

    csvFile.dropna(inplace=True)

    inputLabels = np.array(list([x  for x in csvFile.columns.values if x.isupper()]))
    outputLabels = np.setdiff1d(csvFile.columns.values,inputLabels)
   


    data = csvFile.to_numpy().T



    X,y = data[0:len(inputLabels)],data[len(inputLabels):]

    

    return inputLabels,outputLabels,(X.T,y.T)
    
    
    
