import numpy as np
from flask import Flask, request, jsonify, app,url_for,render_template
import pickle

application = Flask(__name__) #Initialize the flask App
model = pickle.load(open('random_forest_model123.pkl', 'rb'))

@application.route('/')
def home():
    return render_template('home.html')

@application.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    new_data=[list(data.values())]
    output=model.predict(new_data)[0]
    return jsonify(output)

@application.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    data = [int(x) for x in request.form.values()]
    final_features = [np.array(data)]
    print(data)

    output = model.predict(final_features)
    print(output)
    #output = round(prediction[0], 2)

# Define unique message based on the predicted output
    if int (output)==0:
        message = "Great news! You don't have diabetes."
    elif int (output)==1:
        message = "Be aware you are in borderline."
    else:
        message = "You have diabetes!"

    return render_template('home.html', prediction_text= message)

if __name__ == "__main__":
    application.run(debug = True)