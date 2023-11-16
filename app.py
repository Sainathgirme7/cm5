from flask import Flask, render_template, request

from model import classify   # Import the classify function from model.py
# import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt # visualizing data
# %matplotlib inline
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
# Import the classify function from model.py
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64


from flask import Flask, render_template, request
import numpy as np


app = Flask(__name__)

        # import csv file
df = pd.read_csv('models/Diwali Sales Data.csv', encoding= 'unicode_escape')
df.drop(['Status', 'unnamed1'], axis=1, inplace=True)
        # df['Amount'] = df['Amount'].astype('int')
df.dropna(inplace=True)
from sklearn import preprocessing
Lab_encode = preprocessing.LabelEncoder()
        # Assign in new variable
df['OccupationLabel'] = Lab_encode.fit_transform(df['Occupation'].values)
df['ProductCategoryLabel'] = Lab_encode.fit_transform(df['Product_Category'].values)
df['GenderLabel'] = Lab_encode.fit_transform(df['Gender'].values)
df['StateLabel'] = Lab_encode.fit_transform(df['State'].values)



@app.route('/')
def home():
    f_var= ['GenderLabel', 'Age', 'Marital_Status', 'StateLabel', 'OccupationLabel','Amount']
    x=df[f_var]
    y=df['ProductCategoryLabel']
                # Standard scalar
    from sklearn.preprocessing import StandardScaler
    scaler_data = StandardScaler()
    x_scaled_data = scaler_data.fit_transform(x)
                # Importing libraries and data splitting into test and train
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x_scaled_data, y, test_size=0.25, stratify=y)


    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


                # Create a Decision Tree Classifier
    decision_tree_classifier = DecisionTreeClassifier()

                # Train the Decision Tree Classifier on your training data
    decision_tree_classifier.fit(x_train, y_train)
    
    # Accuracy score on the training data
    y_train_predictions = decision_tree_classifier.predict(x_train)
    training_data_accuracy = accuracy_score(y_train_predictions, y_train)

    # Accuracy score on the test data
    y_test_predictions = decision_tree_classifier.predict(x_test)
    test_data_accuracy = accuracy_score(y_test_predictions, y_test)


    return render_template('index.html', training_data_accuracy=training_data_accuracy, test_data_accuracy=test_data_accuracy)




@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/predict', methods=['POST'])
def form():
    if request.method == 'POST':


 # Retrieve form inputs
        gender = int(request.form['gen_der'])
        age = int(request.form['age'])
        marital_status = int(request.form['marital_status'])
        state = int(request.form['state'])
        occupation = int(request.form['occupation'])
        amount = int(request.form['amount'])
        
        
        df = pd.read_csv('models/Diwali Sales Data.csv', encoding= 'unicode_escape')
        df.drop(['Status', 'unnamed1'], axis=1, inplace=True)
                # df['Amount'] = df['Amount'].astype('int')
        df.dropna(inplace=True)
        from sklearn import preprocessing
        Lab_encode = preprocessing.LabelEncoder()
                # Assign in new variable
        df['OccupationLabel'] = Lab_encode.fit_transform(df['Occupation'].values)
        df['ProductCategoryLabel'] = Lab_encode.fit_transform(df['Product_Category'].values)
        df['GenderLabel'] = Lab_encode.fit_transform(df['Gender'].values)
        df['StateLabel'] = Lab_encode.fit_transform(df['State'].values)

        df.drop(['User_ID', 'Cust_name', 'Product_ID','Age Group','Zone'], axis=1, inplace=True)
        f_var= ['GenderLabel', 'Age', 'Marital_Status', 'StateLabel', 'OccupationLabel','Amount']
        x=df[f_var]
        y=df['ProductCategoryLabel']
                # Standard scalar
        from sklearn.preprocessing import StandardScaler
        scaler_data = StandardScaler()
        x_scaled_data = scaler_data.fit_transform(x)
                # Importing libraries and data splitting into test and train
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x_scaled_data, y, test_size=0.25, stratify=y)


        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


                # Create a Decision Tree Classifier
        decision_tree_classifier = DecisionTreeClassifier()

                # Train the Decision Tree Classifier on your training data
        decision_tree_classifier.fit(x_train, y_train)


        # f_var= ['GenderLable', 'Age', 'Marital_Status', 'StateLable', 'OccupationLable','Amount']
        input_data = (gender, age, marital_status, state, occupation, amount)

        # changing the input_data to numpy array
        input_data_as_numpy_array = np.asarray(input_data)

        # reshape the array as we are predicting for one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

        # standardize the input data
        std_data = scaler_data.transform(input_data_reshaped)

        prediction = decision_tree_classifier.predict(std_data)

    
        product_names = [
    'Automotive', 'Cosmetics', 'Literature', 'Fashion & Apparel', 'Home Decor', 'Electronics & Gadgets',
    'Groceries', 'Footwear', 'Furniture', 'Toys & Games', 'Tools & Equipment',
    'Home Essentials', 'Office Supplies', 'Pet Supplies', 'Sporting Goods', 'Stationery', 'Kitchenware', 'Veterinary Supplies'
        ]

        # Convert the prediction to a string
        if 0 <= prediction[0] < len(product_names):
            prediction_text = f'Best Product is {product_names[prediction[0]]}'
        else:
            prediction_text = 'Best Product is Veterinary'

        return render_template('prediction.html', prediction_text=prediction_text)
    
# This is for Data Visualization
import os
@app.route('/visualization')
def visualization():
    df = pd.read_csv('models/Diwali Sales Data.csv', encoding='unicode_escape')
    df.drop(['Status', 'unnamed1'], axis=1, inplace=True)
    df.dropna(inplace=True)

    # Plotting a bar chart for Gender and its count
    ax_gender = sns.countplot(x='Gender', hue='Gender', data=df, palette={'M': 'blue', 'F': 'pink'})


    for bars in ax_gender.containers:
        ax_gender.bar_label(bars)

    # Convert the Matplotlib plot to a base64-encoded image
    image_buffer_gender = io.BytesIO()
    plt.savefig(image_buffer_gender, format='png')
    image_buffer_gender.seek(0)
    image_data_gender = base64.b64encode(image_buffer_gender.read()).decode('utf-8')
    plt.close()  # Close the plot to prevent displaying it in the Flask server logs

    # Plotting a bar chart for Age Group and its count with respect to Gender
    ax_age = sns.countplot(data=df, x='Age Group', hue='Gender')

    for bars in ax_age.containers:
        ax_age.bar_label(bars)

    # Convert the Matplotlib plot to a base64-encoded image
    image_buffer_age = io.BytesIO()
    plt.savefig(image_buffer_age, format='png')
    image_buffer_age.seek(0)
    image_data_age = base64.b64encode(image_buffer_age.read()).decode('utf-8')
    plt.close()  # Close the plot to prevent displaying it in the Flask server logs


# total number of orders from top 10 states

# Assuming 'df' is your DataFrame
    sales_state = df.groupby(['State'], as_index=False)['Orders'].sum().sort_values(by='Orders', ascending=False).head(10)

# Define a color palette for each state
    state_palette = sns.color_palette("viridis", len(sales_state))

# Set the color palette and plot the bar chart
    sns.set(rc={'figure.figsize':(15,5)})
    sns.barplot(data=sales_state, x='State', y='Orders', palette=state_palette)


    # Convert the Matplotlib plot to a base64-encoded image
    image_buffer_state = io.BytesIO()
    plt.savefig(image_buffer_state, format='png')
    image_buffer_state.seek(0)
    image_data_state = base64.b64encode(image_buffer_state.read()).decode('utf-8')
    plt.close()  # Close the plot to prevent displaying it in the Flask server logs

#Occupation
# Define a color palette for each occupation
    occupation_palette = sns.color_palette("Set3", len(df['Occupation'].unique()))

# Set the figure size and color palette
    sns.set(rc={'figure.figsize': (20, 5)})
    sns.set_palette(occupation_palette)

# Plot the countplot
    ax = sns.countplot(data=df, x='Occupation')

# Add labels to the bars
    for bars in ax.containers:
        ax.bar_label(bars)

# Convert the Matplotlib plot to a base64-encoded image
    image_buffer_occupation = io.BytesIO()
    plt.savefig(image_buffer_occupation, format='png')
    image_buffer_occupation.seek(0)
    image_data_occupation = base64.b64encode(image_buffer_occupation.read()).decode('utf-8')
    plt.close()  # Close the plot to prevent displaying it in the Flask server logs



#Product_Category
    sns.set(rc={'figure.figsize':(20,5)})
    ax = sns.countplot(data = df, x = 'Product_Category')

    for bars in ax.containers:
        ax.bar_label(bars)

        # Convert the Matplotlib plot to a base64-encoded image
    image_buffer_product = io.BytesIO()
    plt.savefig(image_buffer_product, format='png')
    image_buffer_product.seek(0)
    image_data_product = base64.b64encode(image_buffer_product.read()).decode('utf-8')
    plt.close()  # Close the plot to prevent displaying it in the Flask server logs






    # Render the HTML template with the base64-encoded image data
    return render_template('visualization.html', image_data_gender=image_data_gender, image_data_age=image_data_age, image_data_state=image_data_state
                           , image_data_occupation=image_data_occupation, image_data_product=image_data_product)





if __name__ == '__main__':
    app.run(debug=True)



