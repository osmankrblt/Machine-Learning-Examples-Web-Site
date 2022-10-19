import numpy as np
import pandas as pd



def csv2array(csvPath):

    try:
    
        csvFile = pd.read_csv(csvPath)

        csvFile.dropna(inplace=True)

        inputLabels = np.array(list([x  for x in csvFile.columns.values if x[0].isupper()]))
        outputLabels = np.setdiff1d(csvFile.columns.values,inputLabels)

        print(inputLabels)
        print(len(outputLabels))

        if len(inputLabels) == 0 or len(outputLabels) == 0:
            
            raise Exception("Column names error")

        data = csvFile.to_numpy().T



        X,y = data[0:len(inputLabels)],data[len(inputLabels):]

        

        return inputLabels,outputLabels,(X.T,y.T)
    except:
        raise Exception("File didnt adapted")
    
    
    
