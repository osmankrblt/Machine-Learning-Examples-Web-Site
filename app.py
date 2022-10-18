from flask import Flask,render_template,flash,request,session
import os,numpy as np
from inputHelper import csv2array
from linear_regression import *
from logistic_regression import *


ALLOWED_EXTENSIONS = {"csv", 'png', 'jpg', 'jpeg'}
UPLOADED_FILE_NAME = "uploadedImage"
STATIC_FOLDER='uploadFolder'

app = Flask(__name__,)
app.secret_key = 'BAD_SECRET_KEY'
app.config['STATIC_FOLDER'] = STATIC_FOLDER
app.config['SESSION_TYPE'] = 'memcached'



def allowed_file(filename):
  
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS 

def upload(fileType):
    if 'file' not in request.files:
        flash('No file part')
        return render_template("mainPage.html")
    file = request.files['file']
       
        
    if file.filename == '':
        flash('No selected file')
        return  render_template("mainPage.html") 

    if file and allowed_file(file.filename):
            
            
        file.save(os.path.join(app.config['STATIC_FOLDER'], UPLOADED_FILE_NAME+fileType))
        return True
    return False


def getSessionValue(valueName):
  
    if valueName in session:
        return session[valueName]
    else :
        return None
    

def setSessionValue(valueName,value):

    session[valueName]=value
        

@app.route('/')
def mainTemplate():

    return render_template('mainTemplate.html')

@app.route('/linear_regression',methods=["POST","GET"])
def linearRegression():


    
    if request.method == 'POST':
        result = None
        score = getSessionValue("score")
        if request.form.get('train') == 'train':

            if upload(".csv"):
             
                
                csvPath = os.path.join(STATIC_FOLDER, UPLOADED_FILE_NAME+".csv")

                inputLabels,outputLabels,(X,y) = csv2array(csvPath)

                

               


                score = linearRegressionTrain(X,y)
                
                setSessionValue("score",score)
                setSessionValue("inputLabels", inputLabels.tolist())
                setSessionValue("outputLabels", outputLabels.tolist())

        elif request.form.get('predict') == 'predict':
            
            csvPath = os.path.join(STATIC_FOLDER, UPLOADED_FILE_NAME+".csv")

            inputLabels,outputLabels,(X,y) = csv2array(csvPath)

            inputs = []

            for inputLabel in inputLabels:

                inputs.append(int(request.form.get(inputLabel)))
        

          
          

          
            result = str(np.squeeze(np.squeeze(linearRegressionPredict([inputs]))))
            
      
        return render_template('regression.html',methodName = "Linear Regression", inputLabels=getSessionValue("inputLabels"),outputLabels=getSessionValue("outputLabels"),score=getSessionValue("score"),result=result)
    
    return render_template('regression.html',methodName = "Linear Regression")

@app.route('/logistic_regression',methods=["POST","GET"])
def logisticRegression():


    
    if request.method == 'POST':
        result = None
        score = getSessionValue("score")
        if request.form.get('train') == 'train':

            if upload(".csv"):
             
     

                csvPath = os.path.join(STATIC_FOLDER, UPLOADED_FILE_NAME+".csv")

                inputLabels,outputLabels,(X,y) = csv2array(csvPath)
          
                

                score = logisticRegressionTrain(X,y)

                setSessionValue("score",score)
                setSessionValue("inputLabels", inputLabels.tolist())
                setSessionValue("outputLabels", outputLabels.tolist())


        elif request.form.get('predict') == 'predict':
            
            csvPath = os.path.join(STATIC_FOLDER, UPLOADED_FILE_NAME+".csv")

            inputLabels,outputLabels,(X,y) = csv2array(csvPath)

            inputs = []

            for inputLabel in inputLabels:

                inputs.append(int(request.form.get(inputLabel)))
        

            
          
            result = str(np.squeeze(np.squeeze(logisticRegressionPredict([inputs]))))
            print("Result : "+str(result))
        return render_template('regression.html',methodName = "Logistic Regression",  inputLabels=getSessionValue("inputLabels"),outputLabels=getSessionValue("outputLabels"),score=getSessionValue("score"),result=result)
    
    return render_template('regression.html',methodName = "Logistic Regression")



    



if __name__ == '__main__':
    app.run(debug=True)