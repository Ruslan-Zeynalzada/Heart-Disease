import pandas as pd
import numpy as np
import sklearn
import streamlit as st
import pickle
import tensorflow as tf


header = st.container()
description = st.container()
data = st.container()
X_test = st.container()
modeling = st.container()
btn_desc = st.sidebar.button("Description")


with header: 
    st.title("This program is to predict if patient have Heart Disease or not")
    st.markdown("* First you have to enter **inputs** on the left of the window")
    st.markdown("* Then you can press **Predict** button and see the results")
    st.markdown("* If you want to see the variables's description press **Description** button")
if btn_desc : 
    with description :
        st.header("Variables' Description")
        st.markdown("* **HeartDisease** - Respondents that have ever reported having coronary heart disease (CHD) or myocardial infarction          (MI)")
        st.markdown("* **BMI** - Body Mass Index (BMI)")
        st.markdown("* **Smoking** - Have you smoked at least 100 cigarettes in your entire life?")
        st.markdown("* **Alcohol Drinking** - Heavy drinkers (adult men having more than 14 drinks per week and adult women having more than         7 drinks per week")
        st.markdown("* **Storke** - (Ever told) (you had) a stroke?")
        st.markdown("* **PhysicalHealth** - Now thinking about your physical health, which includes physical illness and injury, for how             many days during the past 30")
        st.markdown("* **MentalHealth** - Thinking about your mental health, for how many days during the past 30 days was your mental               health not good?")
        st.markdown("* **DiffWalking** - Do you have serious difficulty walking or climbing stairs?")
        st.markdown("* **Sex** - Are you male or female?")
        st.markdown("* **Age Category** - Fourteen-level age category")
        st.markdown("* **Race** - Imputed race/ethnicity value")
        st.markdown("* **Diabetic** - (Ever told) (you had) diabetes?")
        st.markdown("* **PhysicalActivity** - Adults who reported doing physical activity or exercise during the past 30 days other than             their regular job")
        st.markdown("* **Gen Health** - Would you say that in general your health is..")
        st.markdown("* **SleepTime** - On average, how many hours of sleep do you get in a 24-hour period?")
        st.markdown("* **Asthma** - (Ever told) (you had) asthma?")
        st.markdown("* **KidneyDisease** - Not including kidney stones, bladder infection or incontinence, were you ever told you had kidney         disease?")
        st.markdown("* **SkinCancer** - (Ever told) (you had) skin cancer?")
        
with data : 
    st.header("The Heart Disease Dataset")
    df = pd.read_csv("Dataset.csv")
    st.write(df.head())
    
st.sidebar.header("Inputs Giving")
st.sidebar.markdown("**Note :** 0 = No , 1 = Yes")
BMI = st.sidebar.slider("Please choose input for BMI variable" , min_value = 12.0 , max_value = 94.8 , step = 0.1)
Smoking = st.sidebar.radio("Please choose input for Smoking variable" , options = [0,1] , index = 0)
AlcoholDrinking = st.sidebar.radio("Please choose input for Alcohol Drinking variable" , options = [0,1] , index = 0)
Storke = st.sidebar.radio("Please choose input for Storke variable" , options = [0,1] , index = 0)
PhysicalHealth = st.sidebar.slider("Please choose input for PhysicalHealth variable" , min_value = 0 , max_value = 30 , value = 0 , step =1)
MentalHealth = st.sidebar.slider("Please choose input for MentalHealth variable" , min_value = 0 , max_value = 30 , value = 0 , step = 1)
DiffWalking = st.sidebar.radio("Please choose input for DiffWalking variable" , options = [0,1] , index = 0)
st.sidebar.markdown("0 = Female , 1 = Male")
Sex = st.sidebar.radio("Please choose input for Sex variable" , options = [0,1] , index = 0)
st.sidebar.markdown("18-24 : 0 , 25-29 : 1 , 30-34 : 2 , 35-39 : 3 , 40-44 : 4 , 45-49 : 5, 50-54 : 6 , 55-59 : 7 , 60-64 : 8 , 65 : 9 , 70-74 : 10 , 75-79 : 11 , 80 or older : 12")
AgeCategory = st.sidebar.slider("Please choose input for Age Category variable" , min_value = 0.0 , max_value = 12.0 , step = 1.0)
st.sidebar.markdown("White : 0 , Black : 1 , Asian : 2 , American Indian/Alaskan Native : 3 , Other : 4 , Hispanic : 5")
Race = st.sidebar.selectbox("Please choose input for Race variable" , options = [0,2,3,4,5] , index = 0)
st.sidebar.markdown("No : 0 , Yes : 1 , No, borderline diabetes : 2 , Yes (during pregnancy) : 3")
Diabetic = st.sidebar.selectbox("Please choose input for Diabetic variable" , options = [0,1,2,3] , index = 0)
PhysicalActivity = st.sidebar.radio("Please choose input for PhysicalActivity variable" , options = [0,1] , index = 0)
st.sidebar.markdown("Poor : 0 , Fair : 1 , Good : 2 , Very good : 3 , Excellent : 4")
GenHealth = st.sidebar.selectbox("Please choose input for Gen Health variable" , options = [0,1,2,3,4] , index = 0)
SleepTime = st.sidebar.slider("Please choose input for SleepTime variable" , min_value = 1 , max_value = 24 , step = 1)
Asthma = st.sidebar.radio("Please choose input for Asthma variable" , options = [0,1] , index = 0)
KidneyDisease = st.sidebar.radio("Please choose input for KidneyDisease variable" , options = [0,1] , index = 0)
SkinCancer = st.sidebar.radio("Please choose input for SkinCancer variable" , options = [0,1] , index = 0)

with X_test :
    st.header("You have entered these inputs")
    data = pd.DataFrame(data = {"BMI" :[BMI]  , "Smoking" : [Smoking],
                                "AlcoholDrinking" :[AlcoholDrinking], "Storke" : [Storke] , "PhysicalHealth" : [PhysicalHealth],
                                "MentalHealth" : [MentalHealth] , "DiffWalking" : [DiffWalking] , "Sex" : [Sex],
                                 "AgeCategory" : [AgeCategory] , "Race" : [Race] , "Diabetic" : [Diabetic], "PhysicalActivity" :                                                [PhysicalActivity],"GenHealth" : [GenHealth] , "SleepTime" : [SleepTime] , "Asthma" : [Asthma] ,                                            "KidneyDisease" : [KidneyDisease] , "SkinCancer" : [SkinCancer]})
    st.write(data)


with modeling :
    btn_predict = st.sidebar.button("Predict")

    if btn_predict :
        st.header("Prediction")
        scaler = pickle.load(open("scaler_deep" , "rb"))
        model = tf.keras.models.load_model("Kenan kurs/Deep_Learning.h5")
        y_pred_proba = tf.squeeze(model.predict(scaler.transform(data)))
        y_pred = tf.math.round(model.predict(scaler.transform(data)))
        
        if y_pred == [0] : 
                st.markdown("The patient **doesn't** have Heart Disease and probablity is {:.0%}".format(y_pred_proba))

        elif y_pred == [1] : 
                st.markdown("The patient **has** Heart Disease and probablity is {:.0%}".format(y_pred_proba))
