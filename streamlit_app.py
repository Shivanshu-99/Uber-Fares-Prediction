import streamlit as st
import sklearn 
import pandas as pd 
import numpy as np
import pickle

model = pickle.load(open('model.sav','rb'))


st.title('Uber Fares Prediction ')
st.sidebar.header('Please select the information of the journey')

#function for input
def user_report():
    passenger_count = st.sidebar.number_input('Passenger Count',1,10,1)
    year = st.sidebar.number_input('year of journey',min_value=2009,max_value=2050,step=1)
    month = st.sidebar.number_input('month of journey',1,12,1)
    day = st.sidebar.number_input('day of journey',1,30,1)
    day_is_weekend = st.sidebar.number_input('Is your day of journey is on weekend(yes=1,No=0)',min_value=0,max_value=1,step=1) 
    quarter = st.sidebar.number_input('quarter of journey',1,4,1)
    hour = st.sidebar.number_input('hour of journey',0,23,1)
    mins = st.sidebar.number_input('minutes of journey',0,59,1)
    distance_travelled = st.sidebar.number_input('distance  of journey',min_value=1.0,max_value=100.0,step=0.5) 


    user_report_data={
        'passenger_count':passenger_count,
        'year':year,
        'month':month,
        'day': day,
        'day_is_weekend':day_is_weekend,
        'quarter' : quarter,
        'hour': hour,
        'mins': mins,
        'distance_travelled' : distance_travelled

    }
    report_data = pd.DataFrame(user_report_data,index=[0])
    return report_data





user_data = user_report()
st.header('Please check the information provided by you')
st.write(user_data)

## fares Prediction
if (st.button("Calculate Fares")):
    fare = model.predict(user_data)
    st.subheader('Fares for the journey would be')
    st.subheader(np.round(fare[0],2))