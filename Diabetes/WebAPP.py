import numpy as np
import streamlit as st
import pickle


loaded_model = pickle.load(open("C:/Users/NIKHIL/Desktop/Machine Learning/Logistic Regression/Diabetes/trained_model.sav","rb"))


# creating a function for prediction

def diabetes_prediction(input_data):

    # changing the input data to numpy array

    input_data_numpy_array = np.asarray(input_data)

    # reshape the array as we predicted for one instance

    input_data_reshaped = input_data_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    

    if(prediction == 0):
        return "This is not diabetic person"

    else:
        return "This person is diabetic "
    



def main():

    #giving title to the page
    st.title("Diabetes Prediction Web APP")

    # taking input from the user
    # Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age

    Pregnancies = st.number_input("Number of Pregnancies")
    Glucose = st.number_input("Glucose Level")
    BloodPressure = st.number_input("Blood Pressure")
    SkinThickness = st.number_input("Skin Thickness")
    Insulin = st.number_input("Insulin")
    BMI = st.number_input("BMI")
    DiabetesPedigreeFunction = st.number_input("Diabetes")
    Age = st.number_input("Enter Your Age")
    
    
    # code for prediction
    diagnosis = ""

    # creating a button for prediction

    if st.button("Diabetes Test Result"):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])

    st.success(diagnosis)


if __name__ == "__main__":
    main()
