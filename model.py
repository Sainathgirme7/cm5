import pickle
import joblib
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt # visualizing data
# %matplotlib inline
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler

data = joblib.load('models/saved.joblib')
clf_load = data["model"]

# def load_data():
#     df = pd.read_csv('models/Diwali Sales Data.csv', encoding= 'unicode_escape')
#     df = df[['User_ID', 'Cust_name', 'Product_ID', 'Gender', 'Age Group', 'Age',
#        'Marital_Status', 'State', 'Zone', 'Occupation', 'Product_Category',
#        'Orders', 'Amount','Status', 'unnamed1']]
#     df = df.drop(['Status', 'unnamed1'], axis=1, inplace=True)
#     print(df)

#     return df
# df= load_data() 
    

def classify(gen_der,age,marital_status,state,occupation,amount):
    f_var= ['GenderLable', 'Age', 'Marital_Status', 'StateLable', 'OccupationLable','Amount']
    input_data = (gen_der, age, marital_status, state, occupation, amount)

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    
    # # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    # # standardize the input data
    std_data = scaler_data.transform(input_data_reshaped)

    # prediction = clf_load.predict(std_data)
    


    # product_names = [
    # 'Auto', 'Beauty', 'Books', 'Clothing & Apparel', 'Decor', 'Electronics & Gadgets',
    # 'Food', 'Footwear & Shoes', 'Furniture', 'Games & Toys', 'Hand & Power Tools',
    # 'Household items', 'Office', 'Pet Care', 'Sports Products', 'Stationery', 'Tupperware', 'Veterinary'
    # ]

    # if 0 <= prediction[0] < len(product_names):
    #     print(f'Best Product is {product_names[prediction[0]]}')
    # else:
    #     print('Best Product is Veterinary')

    # return prediction