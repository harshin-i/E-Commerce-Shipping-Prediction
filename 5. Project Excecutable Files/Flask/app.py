
from flask import Flask, render_template, request
import joblib
import pickle
import numpy as np
from tensorflow.keras.models import load_model

ct = joblib.load("churnct") #loading column transformer object
sc = pickle.load(open("churnscaler.pkl","rb"))
model = load_model("churn.h5")

app = Flask(__name__)

@app.route('/')
def loadpage():
    return render_template("index.html")

@app.route('/y_predict', methods = ["POST"])
def prediction():
    
    Geography = request.form["Geography"]
    Gender = request.form["Gender"]
    IsActiveMember = request.form["IsActiveMember"]
    HasCrCard = request.form["HasCrCard"]
    NumOfProducts = request.form["NumOfProducts"]
    Age = request.form["Age"]
    Tenure = request.form["Tenure"]
    Balance = request.form["Balance"]
    EstimatedSalary = request.form["EstimatedSalary"]
    CreditScore = request.form["CreditScore"]
    
    x_test = [[float(CreditScore),Geography,Gender,float(Age),float(Tenure),float(Balance),float(NumOfProducts),float(HasCrCard),float(IsActiveMember),float(EstimatedSalary)]]
    print(x_test)
    
    p = np.array(sc.transform(ct.transform(x_test)))
    #p = p.astype(np.float32)
    
    prediction = model.predict(p)
    
    prediction = prediction > 0.5
    
    if (prediction == [[False]]):
        text = "he will stay"
    else:
        text = "he will churn"
        
   
    return render_template("index.html",prediction_text = text )

    
    
if __name__ == "__main__":
    app.run(debug = False)
