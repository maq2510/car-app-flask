import pickle
import pandas as pd
from flask import Flask, render_template, request

# Initialize Flask app
app = Flask(__name__)

# Load the pickled model
with open('Linear.pkl', 'rb') as file:
    model = pickle.load(file)


# Set up the Flask route
@app.route('/', methods=['GET', 'POST'])
def car_selling_price_prediction():
    result = None

    if request.method == 'POST':
        # Get the form data
        name = request.form.get('name')
        year = int(request.form.get('year'))
        km_driven = int(request.form.get('km_driven'))
        fuel = request.form.get('fuel')
        seller_type = request.form.get('seller_type')
        transmission = request.form.get('transmission')
        owner = request.form.get('owner')

        # Create a DataFrame from the form data
        user_data = {'name': name,
                     'year': year,
                     'km_driven': km_driven,
                     'fuel': fuel,
                     'seller_type': seller_type,
                     'transmission': transmission,
                     'owner': owner}

        user_df = pd.DataFrame(user_data, index=[0])

        # Make the prediction
        prediction = model.predict(user_df)

        # Prepare the result message
        result = "The predicted selling price for the car is: {:.2f}".format(prediction[0])

    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)
