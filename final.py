import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
from streamlit import session_state

st.set_page_config(
    page_title="Chronic Disease Prediction",
    page_icon="🩺"
)



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
add_bg_from_local('lrsumclb.png') 

# to import css file into streamlit
with open('final.css') as f:
    st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)
    



col1, col2 = st.columns(2)

with col1:
    st.markdown("<h1 style = 'color:black; font-size:60px;font-family:initial;font-style:oblique;text-align:left'>Chronic Disease Prediction</h1>",unsafe_allow_html=True)
    st.markdown("<h3 style = 'color:black; font-family:cursive;text-align:left'>Using Machine Learning</h3>",unsafe_allow_html=True)
    if st.button("Read More"):
        with st.expander("Background Study"):
            st.write("Chronic diseases represent a growing global health challenge that demands innovative solutions. These long-term health conditions, such as heart disease, diabetes, cancer, and respiratory disorders, have become leading causes of morbidity and mortality worldwide. The burden of chronic diseases not only affects individuals and their families but also places significant strain on healthcare systems and economies.")
            st.write("Traditionally, healthcare has been reactive, with interventions occurring after the onset of symptoms or disease complications. However, there is a paradigm shift towards proactive and preventive healthcare. Advances in data collection, technology, and machine learning present an unprecedented opportunity to predict chronic diseases before they manifest clinically. Early detection allows for timely interventions, personalized treatment plans, and improved health outcomes")
        with st.expander("Problem Statement"):
            st.write("The problem I aim to solve is predicting the likelihood of a patient developing a chronic disease based on their health data. Chronic diseases are long-lasting conditions that may require ongoing medical attention and can significantly impact a person's quality of life. Early prediction of chronic diseases can lead to better preventive and therapeutic strategies, improving overall healthcare outcomes.")
        with st.expander("Final Objective"):
            st.write("The final objective of this machine learning project is to develop a predictive model that can effectively identify individuals at risk of developing a chronic disease based on their health data")
with col2:

    model = st.selectbox("Choose Service", ("", "Lung Cancer", "Stroke Disease", "Heart Disease"), placeholder="Select model")
   

    if model == "Heart Disease": 
        time.sleep(2)
        st.subheader("Heart Disease Prediction")
        with st.form("Form",clear_on_submit=True):
            name = st.text_input("Enter Username", key="name", value="")
            age = st.slider("Age", key="age", value=0)
            gender = st.radio("Gender(0 is Female, 1 is Male)", [0, 1], key="gender")
            chest_pain_type = st.selectbox("Chest Pain Type", ["","Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"], key="chest_pain_type")
            resting_bp = st.number_input("Resting Blood Pressure", key="resting_bp", value=0)
            cholesterol = st.number_input("Cholesterol", key="cholesterol", value=0)
            fbs = st.text_input("Fasting Blood Sugar (0 or 1)", key="fbs")
            rest_ecg = st.text_input("Rest ECG (0-2)", key="rest_ecg")
            maxheartrate = st.number_input("Max Heart Rate", key="maxheartrate")
            angina = st.selectbox("Exercise Induced Angina ",['','Yes','No'], key="angina")
            st_dep = st.number_input("STDepression", key="st_dep")
            st_seg_slope = st.selectbox("ST Segment Slope",['','Upsloping','Flat','Downsloping'], key="st_seg_slope")
            nmj = st.text_input("Num Major Vessels(0-4)", key="nmj")
            tha = st.text_input("Thalassemia(0-3)", key="tha")
            submit = st.form_submit_button("Submit")
            
            if (name and (age is not None) and (gender is not None) and chest_pain_type and (resting_bp is not None) and (cholesterol is not None) and fbs and rest_ecg and (maxheartrate is not None) and angina and (st_dep is not None) and st_seg_slope and nmj and tha):
                if submit:
                    with st.status("Loading...."):
                        time.sleep(2)
                        st.success("Sucessful")
                    tab1, tab2 = st.tabs(["Interpretation", "Result"])
                    
                    with tab1:
                        time.sleep(3)
                    st.write(f"{name},Your Inputted Data:")
                    input_var = pd.DataFrame([{'Age' : age,'Gender' : gender,'Chest Pain Type' : chest_pain_type,'RestingBP' : resting_bp, 'Cholestrol' : cholesterol, 'Fastin Blood Sugar' : fbs, 'RestECG':rest_ecg, 'MaxHeartRate':maxheartrate, 'Exercise Induced Angina':angina, 'STDepression':st_dep,'STSegmentSlope':st_seg_slope, 'NumMajorVessels':nmj, 'Thalassemia': tha}])
                    st.table(input_var)
                    
                    from sklearn.preprocessing import LabelEncoder, StandardScaler
                    lb = LabelEncoder()
                    scaler = StandardScaler()
                    for i in input_var:
                        if input_var[i].dtypes == 'int' or input_var[i].dtypes == 'float':
                            input_var[[i]] = scaler.fit_transform(input_var[[i]])
                        else:
                            input_var[i] = lb.fit_transform(input_var[i])
                        
                    with tab2:
                        st.write("Your Result")
                        heart_prediction = heart_model.predict(input_var)
                        if heart_prediction == 0:
                            st.balloons('You do not have any Sign of Heart Disease')
                        else:
                            st.error('You have signs of heart disease, Please Seek Medical Help Immediately')
                    
    if model == "Stroke Disease":
        time.sleep(2)
        st.subheader("Stroke Disease Prediction")
        with st.form("Form2",clear_on_submit=True):
            name2 = st.text_input("Enter Username", key="name2", value="")
            age2 = st.slider("Age",0,100, key="age2", value=0)
            hypertension = st.selectbox("Hypertension", ['',"Yes", "No"], key="hypertension")
            heart_disease = st.selectbox("Heart Disease", ["", "Yes", "No"], key="heart_disease")
            bmi = st.number_input("BMI", key="bmi", value=0)
            avg_glucose_level = st.number_input("Average Glucose Level", key="avg_glucose_level", value=0)
            submitted = st.form_submit_button("Submit")
            
            
            if (name2 and (age2 is not None) and hypertension and heart_disease and (bmi is not None) and (avg_glucose_level is not None)):
                if submitted:
                    with st.status("Loading...."):
                        time.sleep(2)
                        st.success("Sucessful")
                    tab3, tab4 = st.tabs(["Interpretation", "Result"])
                    
                    with tab3:
                        time.sleep(3)
                    st.write(f"{name2},Your Inputted Data:")
                    input_stroke = pd.DataFrame([{'age' : age2,'hypertension' : hypertension,'heart_disease' : heart_disease, 'bmi' : bmi, 'avg_glucose_level' : avg_glucose_level}])
                    st.table(input_stroke)
                    
                    from sklearn.preprocessing import LabelEncoder, StandardScaler
                    lb = LabelEncoder()
                    scaler = StandardScaler()
                    for i in input_stroke:
                        if input_stroke[i].dtypes == 'int' or input_stroke[i].dtypes == 'float':
                            input_stroke[[i]] = scaler.fit_transform(input_stroke[[i]])
                        else:
                            input_stroke[i] = lb.fit_transform(input_stroke[i])
                    with tab4:
                        st.write("Your Result")
                        stroke_prediction = stroke_model.predict(input_stroke)
                        if stroke_prediction == 0:
                            st.success('You do not have any sign of Stroke Disease')
                            st.balloons()
                        else:
                            st.snow()
                            st.error('You may have signs of Stroke disease, Please Seek Medical Help Immediately')
            
    if model == "Lung Cancer":
        time.sleep(2)
        st.subheader("Lung Cancer Prediction")
        with st.form("Form2",clear_on_submit=True):
            name3 = st.text_input("Enter Username", key="name3")
            Obesity = st.slider("Obesity", 1, 7,key="Obesity")
            Cob = st.slider("Coughing of Blood", 1, 9, key="Cob")
            Ps = st.slider("Passive Smoker",1, 8, key="Ps")
            Au = st.slider("Alcohol use", 1, 8, key="Au")
            Da = st.slider("Dust Allergy",1,8, key="Da")
            Gr = st.slider("Genetic Risk",1,7, key="Gr")
            
            submits = st.form_submit_button("Submit")
            
            
            if (name3 and Obesity and Cob and Ps and Au and Da and Gr):
                if submits:
                    with st.status("Loading...."):
                        time.sleep(2)
                        st.success("Sucessful")
                    tab5, tab6 = st.tabs(["Interpretation", "Result"])
                    
                    with tab5:
                        time.sleep(3)
                    st.write(f"{name3},Your Inputted Data:")
                    input_lung = pd.DataFrame([{'Obesity' : Obesity,'Coughing of Blood' : Cob,'Passive Smoker' : Ps, 'Alcohol use' : Au, 'Dust Allergy' : Da, 'Genetic Risk': Gr}])
                    st.table(input_lung)
                    
                    from sklearn.preprocessing import LabelEncoder, StandardScaler
                    lb = LabelEncoder()
                    scaler = StandardScaler()
                    for i in input_lung:
                        if input_lung[i].dtypes == 'int' or input_lung[i].dtypes == 'float':
                            input_lung[[i]] = scaler.fit_transform(input_lung[[i]])
                        else:
                            input_lung[i] = lb.fit_transform(input_lung[i])
                    with tab6:
                        st.write("Your Result")
                        lung_prediction = lung_model.predict(input_lung)
                        if lung_prediction == 0:
                            st.success('You have low risk of having Lung Cancer')
                            st.balloons()
                        elif lung_prediction == 1:
                            st.warning("You have a medium risk of having Lung Cancer")
                        else:
                            st.snow()
                            st.error('You have high risk of having Lung Cancer, Please Seek Medical Help Immediately')
            
         
    
        # if "step" not in st.session_state:
        #     st.session_state.step = 1
                
        # name = age = gender = ""
        # chest_pain_type = resting_bp = cholesterol = "" 
        # fbs = rest_ecg = maxheartrate = ""
        # angina = st_dep = st_seg_slope = "" 
        # nmj = tha = ""
        # step = st.session_state.step
        
        # if step == 1:
        #     name = st.text_input("Enter Username", key="name", value="")
        #     age = st.slider("Age", key="age", value=0)
        #     gender = st.radio("Gender", ["Female", "Male"], key="gender")

        #     if st.button("Next"):
        #         if not name or not age or not gender:
        #             st.warning("Please fill in all fields")
        #         else:
        #             st.session_state.step = 2
        #             st.experimental_rerun()
            

        # elif step == 2:
        #     chest_pain_type = st.selectbox("Chest Pain Type", ["Type 1", "Type 2", "Type 3", "Type 4"], key="chest_pain_type")
        #     resting_bp = st.number_input("Resting Blood Pressure", key="resting_bp", value=0)
        #     cholesterol = st.number_input("Cholesterol", key="cholesterol", value=0)
        #     button_col3, button_col4 = st.columns([0.5,0.5])
            
        #     with button_col3:
        #         if st.button("Previous"):
        #             st.session_state.step = 1
        #             st.experimental_rerun()
        #     with button_col4:
        #         if st.button("Next"):
        #             st.session_state.step = 3
        #             st.experimental_rerun()
            
            
        # elif step == 3:
        #     fbs = st.text_input("Fasting Blood Sugar", key="fbs")
        #     rest_ecg = st.text_input("Rest ECG", key="rest_ecg")
        #     maxheartrate = st.number_input("Max Heart Rate", key="maxheartrate")

        #     button_col5, button_col6 = st.columns([0.5,0.5])
            
        #     with button_col5:
        #         if st.button("Previous"):
        #             st.session_state.step = 2
        #             st.experimental_rerun()
        #     with button_col6:
        #         if st.button("Next"):
        #             st.session_state.step = 4
        #             st.experimental_rerun()


        # elif step == 4:
        #     angina = st.text_input("Exercise Induced Angina", key="angina")
        #     st_dep = st.text_input("STDepression", key="st_dep")
        #     st_seg_slope = st.text_input("ST Segment Slope", key="st_seg_slope")
            
        #     button_col7, button_col8 = st.columns([0.5,0.5])
            
        #     with button_col7:
        #         if st.button("Previous"):
        #             st.session_state.step = 3
        #             st.experimental_rerun()
        #     with button_col8:
        #         if st.button("Next"):
        #             st.session_state.step = 5
        #             st.experimental_rerun()

        # elif step == 5:
        #     nmj = st.text_input("Num Major Vessels", key="nmj")
        #     tha = st.text_input("Thalassemia", key="tha")
            
        #     button_col9, button_col10 = st.columns(2)
        #     with button_col10:
        #         if st.button("Submit"):
        #             with st.status("Loading...."):
        #                 time.sleep(2)
        #                 st.success("Sucessful")
        #             tab1, tab2 = st.tabs(["Interpretation", "Result"])
                    
        #             with tab1:
        #                 time.sleep(3)
        #             st.write("Your Inputted Data:")
        #             input_var = pd.DataFrame([{'Age' : age,'Gender' : gender,'Chest Pain Type' : chest_pain_type,'RestingBP' : resting_bp, 'Cholestrol' : cholesterol, 'Fastin Blood Sugar' : fbs, 'RestECG':rest_ecg, 'MaxHeartRate':maxheartrate, 'Exercise Induced Angina':angina, 'STDepression':st_dep,'STSegmentSlope':st_seg_slope, 'NumMajorVessels':nmj, 'Thalassemia': tha}])
        #             st.table(input_var)
                        
        #             with tab2:
        #                 st.write("Your Result")
   
   
   
   
   
        
    # if model == 'Stroke Disease':
        # time.sleep(2)
        # st.subheader("Stroke Disease Prediction")
        
        # name, age, gender, chest_pain_type, resting_bp, cholesterol, fbs, rest_ecg, maxheartrate, angina, st_dep, st_seg_slope, nmj, tha = (None,)*14

        # step = st.session_state.step
        
        # if step == 1:
        #     name = st.text_input("Enter Username", key="name", value="")
        #     age = st.slider("Age", key="age", value=0)
        #     gender = st.radio("Gender", ["Female", "Male"], key="gender")

        #     if st.button("Next"):
        #         if not name or not age or not gender:
        #             st.warning("Please fill in all fields")
        #         else:
        #             st.session_state.step = 2
        #             st.experimental_rerun()
            

        # elif step == 2:
        #     chest_pain_type = st.selectbox("Chest Pain Type", ["Type 1", "Type 2", "Type 3", "Type 4"], key="chest_pain_type")
        #     resting_bp = st.number_input("Resting Blood Pressure", key="resting_bp", value=0)
        #     cholesterol = st.number_input("Cholesterol", key="cholesterol", value=0)
        #     button_col3, button_col4 = st.columns([0.5,0.5])
            
        #     with button_col3:
        #         if st.button("Previous"):
        #             st.session_state.step = 1
        #             st.experimental_rerun()
        #     with button_col4:
        #         if st.button("Next"):
        #             st.session_state.step = 3
        #             st.experimental_rerun()
            
            
        # elif step == 3:
        #     fbs = st.text_input("Fasting Blood Sugar", key="fbs")
        #     rest_ecg = st.text_input("Rest ECG", key="rest_ecg")
        #     maxheartrate = st.number_input("Max Heart Rate", key="maxheartrate")

        #     button_col5, button_col6 = st.columns([0.5,0.5])
            
        #     with button_col5:
        #         if st.button("Previous"):
        #             st.session_state.step = 2
        #             st.experimental_rerun()
        #     with button_col6:
        #         if st.button("Next"):
        #             st.session_state.step = 4
        #             st.experimental_rerun()


        # elif step == 4:
        #     angina = st.text_input("Exercise Induced Angina", key="angina")
        #     st_dep = st.text_input("STDepression", key="st_dep")
        #     st_seg_slope = st.text_input("ST Segment Slope", key="st_seg_slope")
            
        #     button_col7, button_col8 = st.columns([0.5,0.5])
            
        #     with button_col7:
        #         if st.button("Previous"):
        #             st.session_state.step = 3
        #             st.experimental_rerun()
        #     with button_col8:
        #         if st.button("Next"):
        #             st.session_state.step = 5
        #             st.experimental_rerun()

        # elif step == 5:
        #     nmj = st.text_input("Num Major Vessels", key="nmj")
        #     tha = st.text_input("Thalassemia", key="tha")
            
        #     button_col9, button_col10 = st.columns(2)
        #     with button_col10:
        #         if st.button("Submit"):
        #             with st.status("Loading...."):
        #                 time.sleep(2)
        #                 st.success("Sucessful")
        #             tab1, tab2 = st.tabs(["Interpretation", "Result"])
                    
        #             with tab1:
        #                 time.sleep(3)
        #             st.write("Your Inputted Data:")
        #             input_var = pd.DataFrame([{'Age' : age,'Gender' : gender,'Chest Pain Type' : chest_pain_type,'RestingBP' : resting_bp, 'Cholestrol' : cholesterol, 'Fastin Blood Sugar' : fbs, 'RestECG':rest_ecg, 'MaxHeartRate':maxheartrate, 'Exercise Induced Angina':angina, 'STDepression':st_dep,'STSegmentSlope':st_seg_slope, 'NumMajorVessels':nmj, 'Thalassemia': tha}])
        #             st.table(input_var)
                        
        #             with tab2:
        #                 st.write("Your Result")
        
    # if model == 'Lung Cancer':
    #     time.sleep(2)
    #     st.subheader("Lung Cancer Prediction")
        
    #     name, age, gender, chest_pain_type, resting_bp, cholesterol, fbs, rest_ecg, maxheartrate, angina, st_dep, st_seg_slope, nmj, tha = (None,)*14

    #     step = st.session_state.step
        
    #     if step == 1:
    #         name = st.text_input("Enter Username", key="name", value="")
    #         age = st.slider("Age", key="age", value=0)
    #         gender = st.radio("Gender", ["Female", "Male"], key="gender")

    #         if st.button("Next"):
    #             if not name or not age or not gender:
    #                 st.warning("Please fill in all fields")
    #             else:
    #                 st.session_state.step = 2
    #                 st.experimental_rerun()
            

    #     elif step == 2:
    #         chest_pain_type = st.selectbox("Chest Pain Type", ["Type 1", "Type 2", "Type 3", "Type 4"], key="chest_pain_type")
    #         resting_bp = st.number_input("Resting Blood Pressure", key="resting_bp", value=0)
    #         cholesterol = st.number_input("Cholesterol", key="cholesterol", value=0)
    #         button_col3, button_col4 = st.columns([0.5,0.5])
            
    #         with button_col3:
    #             if st.button("Previous"):
    #                 st.session_state.step = 1
    #                 st.experimental_rerun()
    #         with button_col4:
    #             if st.button("Next"):
    #                 st.session_state.step = 3
    #                 st.experimental_rerun()
            
            
    #     elif step == 3:
    #         fbs = st.text_input("Fasting Blood Sugar", key="fbs")
    #         rest_ecg = st.text_input("Rest ECG", key="rest_ecg")
    #         maxheartrate = st.number_input("Max Heart Rate", key="maxheartrate")

    #         button_col5, button_col6 = st.columns([0.5,0.5])
            
    #         with button_col5:
    #             if st.button("Previous"):
    #                 st.session_state.step = 2
    #                 st.experimental_rerun()
    #         with button_col6:
    #             if st.button("Next"):
    #                 st.session_state.step = 4
    #                 st.experimental_rerun()


    #     elif step == 4:
    #         angina = st.text_input("Exercise Induced Angina", key="angina")
    #         st_dep = st.text_input("STDepression", key="st_dep")
    #         st_seg_slope = st.text_input("ST Segment Slope", key="st_seg_slope")
            
    #         button_col7, button_col8 = st.columns([0.5,0.5])
            
    #         with button_col7:
    #             if st.button("Previous"):
    #                 st.session_state.step = 3
    #                 st.experimental_rerun()
    #         with button_col8:
    #             if st.button("Next"):
    #                 st.session_state.step = 5
    #                 st.experimental_rerun()

    #     elif step == 5:
    #         nmj = st.text_input("Num Major Vessels", key="nmj")
    #         tha = st.text_input("Thalassemia", key="tha")
            
    #         button_col9, button_col10 = st.columns(2)
    #         with button_col10:
    #             if st.button("Submit"):
    #                 with st.status("Loading...."):
    #                     time.sleep(2)
    #                     st.success("Sucessful")
    #                 tab1, tab2 = st.tabs(["Interpretation", "Result"])
                    
    #                 with tab1:
    #                     time.sleep(3)
    #                 st.write("Your Inputted Data:")
    #                 input_var = pd.DataFrame([{'Age' : age,'Gender' : gender,'Chest Pain Type' : chest_pain_type,'RestingBP' : resting_bp, 'Cholestrol' : cholesterol, 'Fastin Blood Sugar' : fbs, 'RestECG':rest_ecg, 'MaxHeartRate':maxheartrate, 'Exercise Induced Angina':angina, 'STDepression':st_dep,'STSegmentSlope':st_seg_slope, 'NumMajorVessels':nmj, 'Thalassemia': tha}])
    #                 st.table(input_var)
                        
    #                 with tab2:
    #                     st.write("Your Result")
        
    
    
    
        

# import streamlit as st

# st.title("Multi-Step Form Example")

# # Initialize session state variables
# if "step" not in st.session_state:
#     st.session_state.step = 1
#     st.session_state.form_data = {}  # Initialize an empty dictionary for form data

# # Define a placeholder for button columns
# button_col1, button_col2 = st.columns(2)

# if st.session_state.step == 1:
#     st.header("Step 1: Personal Information")
#     name = st.text_input("Enter Name")
#     age = st.number_input("Enter Age")
#     gender = st.selectbox("Select Gender", ["Male", "Female"])
    
#     # Place the "Previous" and "Next" buttons side by side
#     with button_col1:
#         if st.button("Previous"):
#             st.session_state.step = 3  # Go to the previous step
#     with button_col2:
#         if st.button("Next"):
#             st.session_state.form_data.update({"name": name, "age": age, "gender": gender})
#             st.session_state.step = 2

# if st.session_state.step == 2:
#     st.header("Step 2: Contact Information")
#     email = st.text_input("Enter Email")
#     phone = st.text_input("Enter Phone Number")
    
#     # Place the "Previous" and "Next" buttons side by side
#     with button_col1:
#         if st.button("Previous"):
#             st.session_state.step = 1  # Go to the previous step
#     with button_col2:
#         if st.button("Next"):
#             st.session_state.form_data.update({"email": email, "phone": phone})
#             st.session_state.step = 3

# if st.session_state.step == 3:
#     st.header("Step 3: Additional Information")
#     address = st.text_input("Enter Address")
#     city = st.text_input("Enter City")
    
#     # Place the "Previous" and "Submit" buttons side by side
#     with button_col1:
#         if st.button("Previous"):
#             st.session_state.step = 2  # Go to the previous step
#     with button_col2:
#         if st.button("Submit"):
#             st.session_state.form_data.update({"address": address, "city": city})
#             st.success("Form submitted successfully!")
#             st.session_state.step = 1  # Reset to step 1 after submission
