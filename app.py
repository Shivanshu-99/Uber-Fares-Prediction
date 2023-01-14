import streamlit as st
import pickle
import numpy as np
import pandas as pd
import yaml
import joblib
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor

#load the details file
with open('./details.yml', 'r') as readfile:
    details_dict = yaml.load(readfile, Loader=yaml.SafeLoader)
readfile.close()


#load model 
model = joblib.load(details_dict['model_filename'])


st.title("Welcome to Uber Fares Prediction App")
st.markdown("""---""")
st.subheader("This app predicts the fares in (dollars) based on following features")
st.markdown("""
    - Passenger Count 
    - Year
    - Month
    - Day
    - Hour
    - Minutes
    - Distance Travelled
""")


# Take input from the user

st.markdown("""---""")
st.subheader("Please provide your inputs")

col1, col2, col3 = st.columns([8,2,8])

with col1:
     Passenger_count = st.selectbox("Enter No. of Passengers Travelling",[1,2,3,4,5,6])

with col3:
    Year = st.selectbox("Enter your  year of Journey",[2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022])

col4, col5, col6 = st.columns([8,2,8])

with col4:
    Month = st.selectbox("Enter the month of Journey",[1,2,3,4,5,6,7,8,9,10,11,12])

with col6:
    Day = st.slider("Enter the day of Journey", min_value=1, max_value=30)

col7, col8, col9 = st.columns([8,2,8])

with col7:
    Hour = st.number_input("Enter your hour of journey")

with col9:
    Minutes = st.number_input("Enter minutes roughly")

col10, col11 ,col12 = st.columns([8,2,8])

with col10:
    Distance_travelled = st.slider("Enter your distance of jounery(in kilometers)",min_value=1.0,max_value=100.0,step=0.5)

with col12:
    Weekend_No_Weekend =  st.selectbox("Does your journey is on weekend/weekdays ",[1,0])

col13, col14, col15 = st.columns([8,2,8])

with col13:
    Quater = st.selectbox("Quater" ,[1,2,3,4])

st.markdown("""---""")

st.write(f"Based on your selection of Passenger counts: {Passenger_count},   Year: {Year},    Month: {Month} ,    Day: {Day},Hour : {Hour},    Minutes :{Minutes} ,      Distance Travelled : {Distance_travelled},     Quater :{Quater} and Weekend : {Weekend_No_Weekend}")

st.markdown("""---""")

if (st.button("Calculate Fares")):

# fitting the model
    values = np.array([Passenger_count, Year, Month, Day, Hour, Minutes , Distance_travelled, Quater,Weekend_No_Weekend]).reshape(1,-1)

    test_df = pd.DataFrame(values, columns=details_dict['feature_list'])

    prediction = np.round(model.predict(values)[0], 2)

    st.subheader('Fares in $ for your ride would be')
    st.subheader(prediction)
