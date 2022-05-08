# from crypt import methopythonds
from flask import Flask,render_template,request
import pickle
import pandas as pd

application=Flask(__name__)

normal_model=pickle.load(open('Scale_model.pkl','rb'))  
knn_model=pickle.load(open('KNN_model.pkl',"rb"))

@application.route("/")
def home():
    
    return render_template("index.html")

@application.route("/Predict",methods=["POST","GET"])
def adv_prediction():
    try:
        Gender=request.form.get("Gender")
        Age=int(request.form.get("Age"))
        Estimated_Salary=int(request.form.get("Estimated_Salary"))
        dict1={"Gender":[Gender],"Age":[Age],"Estimated_Salary":[Estimated_Salary]}
        df=pd.DataFrame(dict1)
        df['Gender'].replace({"Male":1,"Female":0},inplace=True)
        print(df)
        
        df[df.columns]=normal_model.transform(df)
        result=knn_model.predict(df)   
        for i in result:
            if int(i)==0:
                final_result1= "User May not be Subscribe"
                return  render_template("index.html",Prediction1=final_result1)
            else:
                final_result2="User May be Subscribe"
                return  render_template("index.html",Prediction2=final_result2)
    except:
        final_result3= "Enter Valid Response"
        return  render_template("index.html",Prediction3=final_result3)
        




if __name__=="__main__":
    application.run(debug=True,host="0.0.0.0", port=8080)