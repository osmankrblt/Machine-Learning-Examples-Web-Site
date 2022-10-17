from flask import Flask,render_template,flash,request,session
import os,numpy as np
from inputHelper import csv2array
from linear_regression import *


ALLOWED_EXTENSIONS = {"csv", 'png', 'jpg', 'jpeg'}
UPLOADED_FILE_NAME = "uploadedImage"
STATIC_FOLDER='uploadFolder'

app = Flask(__name__,)

app.config['STATIC_FOLDER'] = STATIC_FOLDER




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


        
        

@app.route('/')
def mainTemplate():

    return render_template('mainTemplate.html')

@app.route('/linear_regression',methods=["POST","GET"])
def linearRegression():


    
    if request.method == 'POST':
        result = None
        score = None
        if request.form.get('train') == 'train':

            if upload(".csv"):
             
                #model = lr()

                csvPath = os.path.join(STATIC_FOLDER, UPLOADED_FILE_NAME+".csv")

                inputLabels,outputLabels,(X,y) = csv2array(csvPath)
          
                print(X.shape)
                print(y.shape)

                score = linearRegressionTrain(X,y)

               

        elif request.form.get('predict') == 'predict':
            
            csvPath = os.path.join(STATIC_FOLDER, UPLOADED_FILE_NAME+".csv")

            inputLabels,outputLabels,(X,y) = csv2array(csvPath)

            inputs = []

            for inputLabel in inputLabels:

                inputs.append(int(request.form.get(inputLabel)))
        

            
          

          
            result = np.squeeze(np.squeeze(linearRegressionPredict([inputs])))

      
        return render_template('linearRegression.html',inputLabels=inputLabels,outputLabels=outputLabels,score=score,result=result)
    
    return render_template('linearRegression.html')



    



if __name__ == '__main__':
    app.run(debug=True)