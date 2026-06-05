import streamlit as st
import pandas as pd
import pickle 
import joblib
best_SVC_model = joblib.load('best_SVC_model.pkl')
columns=['Drug Dosage (mg)', 'Systolic Blood Pressure (mmHg)',
       'Heart Rate (BPM)', 'Liver Toxicity Index (U/L)',
       'Blood Glucose Level (mg/dL)']
def predict_drug_response(features):
  prediction=best_SVC_model.predict(features)
  return prediction


st.title("Drug Response Prediction")

# get user Input
DD=st.number_input("Drug Dosage in mg",value=0.0)
SBP=st.number_input("Systolic Blood Pressure",value=0.0)
HR=st.number_input("Heart Rate (BPM)",min_value=0.0,value=5)
LTI=st.number_input("Liver Toxicity Index (U/L)",value=0.0)
BGL=st.number_input("Blood Glucose Level",value=0.0)



Input_data=pd.DataFrame([[DD, SBP,HR,LTI,BGL]],columns=columns)
if st.button("Predict Drug Response"):
  prediction=predict_drug_response(Input_data)
  if prediction[0]==0:
    st.write("0 No Response")
  else:
    st.write("1 Response")
