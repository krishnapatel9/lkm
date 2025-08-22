from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model and columns
model = joblib.load('churn_model.pkl')
model_columns = joblib.load('model_columns.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.form.to_dict()
        
        # Convert all values to appropriate types
        # for key, value in data.items():
        #     if key in ['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction']:
        #         data[key] = float(value) # Assuming these are numerical
        #     else:
        #         data[key] = str(value) # Assuming these are categorical

        # Define default values for all possible model columns
        # These are based on the original `train_model.ipynb` feature engineering
        default_data = {
            'CustomerID': 0, # Not used in prediction, but needed for column alignment if it was in original data
            'Age': 30,  # Default age
            'Gender_Male': 0, # Default to Female
            'Tenure': 0,
            'Usage Frequency': 15, # Default usage frequency
            'Support Calls': 2, # Default support calls
            'Payment Delay': 0,
            'Subscription Type_Standard': 0, # Default to Basic
            'Subscription Type_Premium': 0, # Default to Basic
            'Contract Length_Quarterly': 0, # Default to Monthly
            'Contract Length_Annual': 0, # Default to Monthly
            'Total Spend': 100, # Default total spend
            'Last Interaction': 5 # Default last interaction
        }

        # Update default data with actual values from the form
        # Numerical inputs
        if 'Tenure' in data: default_data['Tenure'] = float(data['Tenure'])
        if 'Payment Delay' in data: default_data['Payment Delay'] = float(data['Payment Delay'])

        # Categorical inputs - handle one-hot encoding
        if 'Gender' in data:
            if data['Gender'] == 'Male':
                default_data['Gender_Male'] = 1
            else:
                default_data['Gender_Male'] = 0
        
        if 'Subscription Type' in data:
            if data['Subscription Type'] == 'Standard':
                default_data['Subscription Type_Standard'] = 1
                default_data['Subscription Type_Premium'] = 0
            elif data['Subscription Type'] == 'Premium':
                default_data['Subscription Type_Standard'] = 0
                default_data['Subscription Type_Premium'] = 1
            else: # Basic
                default_data['Subscription Type_Standard'] = 0
                default_data['Subscription Type_Premium'] = 0

        if 'Contract Length' in data:
            if data['Contract Length'] == 'Quarterly':
                default_data['Contract Length_Quarterly'] = 1
                default_data['Contract Length_Annual'] = 0
            elif data['Contract Length'] == 'Annual':
                default_data['Contract Length_Quarterly'] = 0
                default_data['Contract Length_Annual'] = 1
            else: # Monthly
                default_data['Contract Length_Quarterly'] = 0
                default_data['Contract Length_Annual'] = 0

        # Create a DataFrame from the modified data dictionary
        # Ensure all columns from model_columns are present
        input_df = pd.DataFrame([default_data])
        input_df = input_df.reindex(columns=model_columns, fill_value=0)
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0][1] # Probability of churn

        return render_template('index.html', prediction_text=f'Customer Churn Prediction: {"Churn" if prediction == 1 else "No Churn"} (Probability: {prediction_proba:.2f})')

if __name__ == '__main__':
    app.run(debug=True)
