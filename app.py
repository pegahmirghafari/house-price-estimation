import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, Response, render_template, jsonify

app = Flask('myApp')

# route 1: hello world
@app.route('/')
def home():
    # use flask's render_template function to display an html page
    return render_template('homepage.html')


# route 4: show a form to the user
@app.route('/form')
def form():
    # use flask's render_template function to display an html page
    return render_template('form.html')


# route 5: accept the form submission and do something fancy with it
@app.route('/submit')
def submit():
    data = request.args # form data
    # load in the form data from the incoming request
    X_test = np.array([
        int(data['OverallQual']),
        int(data['BedroomAbvGr']),
        int(data['FullBath']),
        int(data['GarageArea']),
        int(data['LotArea']),
        int(data['YearBuilt'])
    ]).reshape(1, -1) # turns [1, 2, 3, 4] into [[1, 2, 3, 4]]
    # manipulate data into a format that we pass to our model
    model = pickle.load(open('assets/model.p','rb'))
    pred = model.predict(X_test)
    pred= round(pred[0],2)

    return render_template('results.html',prediction= pred)


# Call app.run(debug=True) when python script is called
if __name__ == '__main__':
    app.run(debug=True)







