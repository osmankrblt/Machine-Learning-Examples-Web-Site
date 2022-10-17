from inputHelper import  csv2array
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle



    

def linearRegressionTrain(X,y):
    
   
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,shuffle=True)
    
    print(y_test)

    regr = LinearRegression().fit(X_train,y_train)
        
    pickle.dump(regr,open("static/lr.pkl","wb"))

    return regr.score(X_test,y_test) 

        

def linearRegressionPredict(inputs):

         

    try:
        regr = pickle.load(open("static/lr.pkl","rb"))

        print(inputs)

        result = regr.predict(inputs)
                
        print(result)

        return result   
            
    except :
      
        return "Model yok veya yüklenemedi"

if __name__ == '__main__':

    LinearRegression().linearRegressionPredict([[1,2,3]])

    