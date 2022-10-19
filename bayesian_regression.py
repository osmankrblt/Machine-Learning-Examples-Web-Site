from inputHelper import  csv2array
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
import pickle
import numpy as np


    

def bayesianRidgeRegressionTrain(X,y):

   
    
   
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,shuffle=True)
    
   

    regr = BayesianRidge().fit(X_train,y_train)
        
    pickle.dump(regr,open("static/lr.pkl","wb"))

    return regr.score(X_test,y_test) 

        

def bayesianRidgeRegressionPredict(inputs):

         
    
    try:
        regr = pickle.load(open("static/lr.pkl","rb"))

   
        print(inputs)
       

        result = regr.predict(np.array(inputs).reshape(-1,1))
                
      

        return result   
            
    except :
      
        return "Model yok veya yüklenemedi"

if __name__ == '__main__':

    inputLabels,outputLabels,(X,y) = csv2array("C:/Users/hosma/Downloads/New folder/train.csv")
    print(bayesianRidgeRegressionTrain(X,y))

