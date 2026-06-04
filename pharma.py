import streamlit as st
import pandas as pd
import pickle 
import joblib
loaded_model=joblib.load('best_SVC_model.pkl')
columns=['Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition',
       'Delivery_Slot', 'Driver_Experience', 'Num_Stops', 'Vehicle_Age',
       'Road_Condition_Score', 'Package_Weight', 'Fuel_Efficiency',
       'Warehouse_Processing_Time']
def predict_delivery_delay(features):
  prediction=loaded_model.predict(features)
  return prediction


st.title("Delivery Delay Prediciton")

# get user Input
Delivery_Distance=st.number_input("Delivery distance in km",min_value = 0.0)
Traffic_Congestion=st.number_input("Traffic congestion fill 1 to 5",min_value=1,max_value=5)
Weather_Condition=st.number_input("weather condition fill 1 to 5",min_value=1,max_value=5)
Delivery_Slot=st.number_input("Delivery slot",min_value=1)
Driver_Experience=st.number_input("Driver Experience in years",min_value=0.0)
Num_Stops=st.number_input("No. of Stop",min_value=0.0)
Vehicle_Age=st.number_input("Vehicle age in yrs",min_value=0.0)
Road_Condition_Score=st.number_input("Road Condition fill 1 to 5",min_value=1,max_value=5)
Package_Weight=st.number_input("Package weight in KG",min_value=0.0)
Fuel_Efficiency=st.number_input("Fuel Efficiency in km/ltr",min_value=0.0)
Warehouse_Processing_Time=st.number_input("processing time in min",min_value=0.0)


Input_data=pd.DataFrame([[Delivery_Distance, Traffic_Congestion, Weather_Condition,
       Delivery_Slot, Driver_Experience, Num_Stops, Vehicle_Age,
       Road_Condition_Score, Package_Weight, Fuel_Efficiency,
       Warehouse_Processing_Time]],columns=columns)
if st.button("predict delivery delay"):
  prediction=predict_delivery_delay(Input_data)
  if prediction[0]==0:
    st.write("0 no significant delay expected")
  else:
    st.write("1 Delivery delay")
