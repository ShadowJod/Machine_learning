import numpy as np
import pickle

# loading the save model

loaded_model = pickle.load(open("C:/Users/NIKHIL/Desktop/Machine Learning/Logistic Regression/Diabetes/trained_model.sav","rb"))
input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input data to numpy array

input_data_numpy_array = np.asarray(input_data)

# reshape the array as we predicted for one instance

input_data_reshaped = input_data_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if(prediction == 0):
    print("This is not diabetic person")

else:
    print("This person is diabetic ")