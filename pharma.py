import streamlit as st
import pandas as pd
import pickle 
import joblib
best_SVC_model = joblib.load('best_SVC_model.pkl')
columns=['Drug Dosage (mg)', 'Systolic Blood Pressure (mmHg)',
       'Heart Rate (BPM)', 'Liver Toxicity Index (U/L)',
       'Blood Glucose Level (mg/dL)', 'Drug Response']
def predict_drug_response(features):
  prediction=best_SVC_model.predict(features)
  return prediction


st.title("Drug Response Prediciton")

# get user Input
Drug Dosage (mg)=st.number_input("Drug Dosage in mg",min_value=0.0)
Systolic Blood Pressure (mmHg)=st.number_input("Systolic Blood Pressure",min_value=0.0)
Heart Rate (BPM)=st.number_input("Heart Rate (BPM)",min_value=0.0,max_value=5)
Liver Toxicity Index (U/L)=st.number_input("Liver Toxicity Index (U/L)",min_value=0.0)
Blood Glucose Level (mg/dL)=st.number_input("Blood Glucose Level",min_value=0.0)



Input_data=pd.DataFrame([[Drug Dosage (mg), Systolic Blood Pressure (mmHg),
                          Heart Rate (BPM), Liver Toxicity Index (U/L),
                          Blood Glucose Level (mg/dL), Drug Response]],columns=columns)
if st.button("Drug Response"):
  prediction=predict_drug_response(Input_data)
  if prediction[0]==0:
    st.write("0 No Response")
  else:
    st.write("1 Response")
