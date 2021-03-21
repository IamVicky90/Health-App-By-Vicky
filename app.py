#Import necessary libraries
from flask import Flask,request,render_template
from tensorflow.keras.preprocessing.image import img_to_array,load_img
from tensorflow.keras.models import load_model
import jsonify
import requests
import pickle
import numpy as np
import sklearn
import pandas as pd
import os
from werkzeug.utils import secure_filename
# Create flask instance
app = Flask(__name__)
def pred_pnemoian(img_path):
    model=load_model("Chest XRay Pnemonia xception model.h5")
    img=load_img(img_path,target_size=(224,224))
    x=img_to_array(img)/225
    x=np.expand_dims(x, axis=0)
    pred=model.predict(x)
    output=np.argmax(pred,axis=1)
    if output==0:
        return "Prediction Result: Don't worry You don't have any disease!"
    elif output==1:
        return "We found that you have Pnemonia disease please consult with the doctor"
def pred_skin(img_path):
    
    model=load_model("skin cancer vgg16 model.h5")
    img=load_img(img_path,target_size=(224,224))
    x=img_to_array(img)/225
    x=np.expand_dims(x, axis=0)
    pred=model.predict(x)
    output=np.argmax(pred,axis=1)
    if output==0:
        return "Prediction Result: Don't worry You don't have any disease!"
    elif output==1:
        return "We found that you have skin cancer, please consult with the doctor"


# render index.html page
@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('index.html')
@app.route("/heart", methods=['GET', 'POST'])
def heart():
    return render_template('heart.html')

@app.route('/heart_predict',methods=["GET","POST"])
def heart_predict():
    model = pickle.load(open('Heart_disease_ab_0.90_model.sav', 'rb'))
    print("@@ Heart Disease Model Loaded")
    if request.method == 'POST':
        age=request.form['age']
        
        sex=request.form['sex']
        cp=request.form['cp']
        trestbps=request.form['trestbps']
        chol=request.form['chol']
        fbs=request.form['fbs']
        restecg=request.form['restecg']
        thalach=request.form['thalach']
        exang=request.form['exang']
        oldpeak=request.form['oldpeak']
        slope=request.form['slope']

        ca=request.form['ca']

        thal=request.form['thal']
        values=[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]
        X=[]
        try:
            for value in values:
                X.append(np.log(float(value)+1))
            output=model.predict([X])
        except Exception as e:
            print("@@",e)
            return render_template('heart.html',prediction_text="Some unknown error occured please input the values in number or contact the develpor if it still occurs")

        if output==0:
            return render_template('heart.html',prediction_text="Prediction Result: Don't worry You don't have any disease!")
        elif output==1:
            return render_template('heart.html',prediction_text="We found something wrong with you please consult with the doctor")




    else:
        return render_template('heart.html')

@app.route("/breast", methods=['GET', 'POST'])
def breast():
    return render_template('breast.html')
@app.route("/breast_predict", methods=['GET', 'POST'])
def breast_predict():
    model = pickle.load(open(r'brest_cancer_rf_model.sav', 'rb'))
    print("@@ Breast Cancer Model Loaded")
    if request.method == 'POST':
        try:
            mean_radius=float(request.form['mean_radius'])
            mean_texture=float(request.form['mean_texture'])
            mean_perimeter=float(request.form['mean_perimeter'])
            mean_area=float(request.form['mean_area'])
            mean_smoothness=float(request.form['mean_smoothness'])
        except Exception as e:
            print("@@",e)
            return render_template('breast.html',prediction_text="Some unknown error occured please input the values in number or contact the develpor if it still occurs")
        
        output=model.predict([[mean_radius,mean_texture,mean_perimeter,mean_area,mean_smoothness]])
        if output==0:
            return render_template('breast.html',prediction_text="Prediction Result: Don't worry You don't have any disease!")
        elif output==1:
            return render_template('breast.html',prediction_text="We found something wrong with you please consult with the doctor")

    return render_template('breast.html')
@app.route("/pnemonia", methods=['GET', 'POST'])
def pnemonia():
    return render_template('pnemonia.html')
@app.route("/predict_pnemonia", methods=['GET', 'POST'])
def predict_pnemonia():
    if request.method=='POST':
        # f = request.files['file']

        # Save the file to ./uploads
        # basepath = os.path.dirname(__file__)
        # ...
        file = request.files['file'] # fet input
        filename = file.filename        
        print("@@ Input posted = ", filename)
         
        file_path = os.path.join('static/images', filename)
        file.save(file_path)
        # ... 
        # file_path = os.path.join(
        # basepath, 'uploads', secure_filename(f.filename))
        # f.save(file_path)

        # Make prediction
        preds = pred_pnemoian(file_path)
        result=preds
        return result
    
@app.route("/diabtes", methods=['GET', 'POST'])
def diabtes():
    return render_template('diabtes.html')
@app.route("/diabtes_predict", methods=['GET', 'POST'])
def diabtes_predict():
    model = pickle.load(open(r'diabetes_ada_0.75_model.sav', 'rb'))
    print("@@ Diabtes Model Loaded")
    if request.method == 'POST':
        try:
            Pregnancies=float(request.form['Pregnancies'])
            Glucose=float(request.form['Glucose'])
            BloodPressure=float(request.form['BloodPressure'])
            SkinThickness=float(request.form['SkinThickness'])
            Insulin=float(request.form['Insulin'])
            BMI=float(request.form['BMI'])
            DiabetesPedigreeFunction=float(request.form['DiabetesPedigreeFunction'])
            Age=float(request.form['Age'])
        except Exception as e:
            print("@@",e)
            return render_template('diabtes.html',prediction_text="Some unknown error occured please input the values in number or contact the develpor if it still occurs")
        df=pd.DataFrame([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]],columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
        output=model.predict(df)
        if output==0:
            return render_template('diabtes.html',prediction_text="Prediction Result: Don't worry You don't have diabtes!")
        elif output==1:
            return render_template('diabtes.html',prediction_text="We found that you have diabtes, please consult with the doctor")

    return render_template('diabtes.html')
@app.route("/skin", methods=['GET', 'POST'])
def skin():
    return render_template('skin.html')
@app.route("/predict_skin", methods=['GET', 'POST'])
def predict_skin():
    if request.method=='POST':
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
        basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = pred_skin(file_path)
        result=preds
        return result
@app.route("/kidney", methods=['GET', 'POST'])
def kidney():
    return render_template('kidney.html')
@app.route("/kidney_predict", methods=['GET', 'POST'])
def kidney_predict():
    model = pickle.load(open(r'kidney_disease_ab1_model.sav', 'rb'))
    print("@@ Kidney Model Loaded")
    if request.method=='POST':
        try:
            age=float(request.form['age'])
            bp=float(request.form['bp'])
            sg=float(request.form['sg'])
            al=float(request.form['al'])
            su=float(request.form['su'])
            bgr=float(request.form['bgr'])
            bu=float(request.form['bu'])
            sc=float(request.form['sc'])
            sod=float(request.form['sod'])
            pot=float(request.form['pot'])
            hemo=float(request.form['hemo'])
            pcv=float(request.form['pcv'])
            wc=float(request.form['wc'])
            rc=float(request.form['rc'])
        except Exception as e:
            print("@@",e)
            return render_template('kidney.html',prediction_text="Some unknown error occured please input the values in number or contact the develpor if it still occurs")
        pc_=request.form['pc']
        if pc_=='normal':
            pc_normal=1
            pc_nan=0
        elif pc_=='nan':
            pc_normal=0
            pc_nan=1
        else:
            pc_normal=0
            pc_nan=0
        pcc_=request.form['pcc']
        if pcc_=='present':
            pcc_present=1
        else:
            pcc_present=0
        ba_=request.form['ba']
        if ba_=='present':
            ba_present=1
        else:
            ba_present=0    
        htn_=request.form['htn']
        if htn_=='yes':
            htn_yes=1
        else:
            htn_yes=0
        dm_=request.form['dm']
        if dm_=='yes':
            dm_no=0
            dm_yes=1
        else:
            dm_no=0
            dm_yes=0

        cad_=request.form['cad']
        if cad_=='yes':
            cad_yes=1
        else:
            cad_yes=0
        appet_=request.form['appet']
        if appet_=='poor':
            appet_poor=1
        else:
            appet_poor=0
        pe_=request.form['pe']
        if pe_=='yes':
            pe_yes=1
        else:
            pe_yes=0
        ane_=request.form['ane']
        if ane_=='yes':
            ane_yes=1
        else:
            ane_yes=0
        rbc_=request.form['rbc']
        if rbc_=='normal':
            rbc_normal=1
            rbc_nan=0
        elif rbc_=='nan':
            rbc_normal=0
            rbc_nan=1
        else:
            rbc_normal=0
            rbc_nan=0
        X=pd.DataFrame([[age, bp, sg, al, su, bgr, bu, sc, sod, pot, hemo,pcv, wc, rc, rbc_normal, rbc_nan, pc_normal, pc_nan,pcc_present, ba_present, htn_yes, dm_no, dm_yes, cad_yes,appet_poor, pe_yes, ane_yes]],columns=['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo','pcv', 'wc', 'rc', 'rbc_normal', 'rbc_nan', 'pc_normal', 'pc_nan','pcc_present', 'ba_present', 'htn_yes', 'dm_no', 'dm_yes', 'cad_yes','appet_poor', 'pe_yes', 'ane_yes'])

        X_col=X.columns[[ True,  True, False,  True, False,  True,  True,  True,  True,True,  True,  True,True,  True,  True,  True,  True, False,False, False,  True, False,  True,  True, False, False,True]]
        output=model.predict(X[X_col])
        if output==0:
            return render_template('kidney.html',prediction_text="Result: \nPrediction Result: Don't worry You don't have any Kidney disease!")
        elif output==1:
            return render_template('kidney.html',prediction_text="Result: \nWe found something wrong with your kidney, please consult with the doctor")



if __name__ == "__main__":

    app.run(debug=True)
