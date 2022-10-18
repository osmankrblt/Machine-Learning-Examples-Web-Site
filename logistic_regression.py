from inputHelper import  csv2array
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle
import numpy as np


    

def logisticRegressionTrain(X,y):

    print(X)
    print(y)
    
   
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,shuffle=True)
    
   

    regr = LogisticRegression().fit(X_train,y_train)
        
    pickle.dump(regr,open("static/lr.pkl","wb"))

    return regr.score(X_test,y_test) 

        

def logisticRegressionPredict(inputs):

         
    
    try:
        regr = pickle.load(open("static/lr.pkl","rb"))

   
        print(inputs)
       

        result = regr.predict(np.array(inputs).reshape(-1,1))
                
      

        return result   
            
    except :
      
        return "Model yok veya yüklenemedi"

if __name__ == '__main__':

    #LogisticRegression().logisticRegressionPredict([[1,2,3]])

    regr = pickle.load(open("static/lr.pkl","rb"))

   

    result = regr.predict(np.array([[200]]).reshape(-1,1))
    
    print(result)