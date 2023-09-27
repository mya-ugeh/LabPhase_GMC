import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time


#Load all datasets
heart = pd.read_csv('heart.csv')
stroke = pd.read_csv('stroke_feature.csv')
lung = pd.read_csv('lung_feature.csv')

heart_variable = pd.read_csv('heart_variables.csv')
heart_variable.drop('Unnamed: 0',axis=1,inplace=True)

#export models
heart_model = joblib.load('heart_model.pkl')
stroke_model = joblib.load('stroke_model.pkl')
lung_model = joblib.load('lung_model.pkl')


#.........................DESIGN BEGINS ............................
#to add picture from local computer
import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('5eiejmut.png') 



st.title("Chronic Disease Prediction")
st.header("Using Machine Learning")
st.button("Read More")


# Create a sidebar with service selection buttons
with st.sidebar:
    st.header("Select Service")
    selected_service = st.selectbox("Choose a service", ["", "Heart Disease Prediction", "Stroke Disease Prediction", "Lung Cancer Prediction"])

# Display variable definitions when a service is selected
if (selected_service == 'Heart Disease Prediction'):
    with st.form('my_form', clear_on_submit=True):
        st.header(f"{selected_service}")
        with st.expander("Variable Definitions"):
            st.table(heart_variable)
        Age = st.slider('Input age')
        Gender = st.selectbox('Gender',['',
                                            '0',
                                            '1',])
        chestpain = st.selectbox('ChestPain',['','Self employed', 'Government Dependent',
        'Formally employed Private', 'Informally employed',
        'Formally employed Government', 'Farming and Fishing',
        'Remittance Dependent', 'Other Income',
        'Dont Know/Refuse to answer', 'No Income'])
        country = st.selectbox('Country', ['','Kenya', 'Rwanda', 'Tanzania', 'Uganda'])
        gender = st.selectbox('Gender',['','Female','Male'])
        year = st.selectbox('Year', ['',2018, 2016, 2017])
        
        submitted = st.form_submit_button("SUBMIT")   
        if (Age and Gender and chestpain and country and gender and year):
            if submitted:
                with st.spinner(text='In progress'):
                    time.sleep(3)
                    st.write("Your Inputted Data:")
                    input_var = pd.DataFrame([{'cellphone_access' : phone,'education_level' : education,'job_type' : job,'country' : country, 'year' : year, 'gender_of_respondent' : gender}])
                    st.write(input_var) 
                    
                    from sklearn.preprocessing import LabelEncoder, StandardScaler
                    lb = LabelEncoder()
                    scaler = StandardScaler()
                    for i in input_var:
                        if input_var[i].dtypes == 'int' or input_var[i].dtypes == 'float':
                            input_var[[i]] = scaler.fit_transform(input_var[[i]])
                        else:
                            input_var[i] = lb.fit_transform(input_var[i])
                            
                    # time.sleep(2)
                    prediction = mod.predict(input_var)
                    if prediction == 0:
                        st.error('Not qualified to have bank account')
                    else:
                        st.balloons
                        st.success('You are qualified to have bank account')
                    # st.write(prediction)




