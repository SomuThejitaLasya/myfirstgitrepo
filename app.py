import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


app = Flask(__name__)

# Load the trained model
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect features from form data
    followers_count=float(request.form['followers_count'])
    friends_count=float(request.form['friends_count'])
    listed_count=int(request.form['listed_count'])
    lang=int(request.form['lang'])
    default_profile=int(request.form['default_profile'])
    profile_use_background_image=int(request.form['profile_use_background_image'])
    profile_background_tile=int(request.form['profile_background_tile'])


    
    
    feature_array=[[followers_count,friends_count,listed_count,lang,default_profile,
                     profile_use_background_image,profile_background_tile]]
    #print(features_array)


    
    '''encoded_features = []
    for feature in features:
        if isinstance(feature, str):
            print("string")
            encoded_feature = le.fit_transform([feature])
            encoded_features.append(encoded_feature[0])
        else:
            encoded_features.append(feature)
    features_array = np.array(encoded_features).reshape(1, -1)
    print(features_array)
    '''
    features_array = np.array(feature_array).reshape(1, -1)
    print(feature_array)
    prediction = model.predict(features_array)[0]
    print(prediction)

    # Output the prediction
    if prediction == 1:
        output='FAKE' 
    else:
        output='REAL'

    return render_template('index.html', prediction_text='Prediction: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
