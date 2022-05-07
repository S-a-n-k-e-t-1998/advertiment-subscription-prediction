# from crypt import methods
from flask import Flask
from flask import Flask,jsonify,request
import pickle
import json
import pandas as pd
import numpy as np

app=Flask(__name__)

@app.route("/")
def Welcome():
    return jsonify({"welcome":"Welcome to Advertisment Prediction"})

@app.route("/Predict",methods=["POST"])
def Adv_predictor():
    data=request.get_json()
    df=pd.DataFrame(data)
    print(df)
    df['Gender'].replace({"Male":1,"Female":0},inplace=True)
    # print(df)
    normal_model=pickle.load(open(r"E:\10.python\project\advertiment_prediction\model_api\Normal_Scale_model.pkl","rb"))  # Normal Scaling Apply model load
    df[df.columns]=normal_model.transform(df)

    model_adv=pickle.load(open(r"E:\10.python\project\advertiment_prediction\model_api\KNN_model.pkl",'rb'))  # KNN Model load
    result=model_adv.predict(df)   #
    print(result)
    list1=[]
    dict1={}
    for i in result:
        if int(i)==0:
            list1.append("Not Subsribe")
        else:
            list1.append("Subsribe")
    for i in range(len(list1)):
        dict1.update({"Input"+str(i):{"Input":df.iloc[i,:].to_dict(),"Predicted Output":list1[i]}})
    print(dict1)   
    return jsonify(dict1)


if __name__=="__main__":
    app.run(debug=True)