import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)
model = joblib.load('weather_prediction.pkl')


@app.route('/')
def home():
    return render_template('index_weather.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    input_features = [float(i) for i in request.form.values()]
    feature_values = np.array(input_features)
    feature_names = ['apparent_temperature', 'humidity', 'wind_speed', 'wind_bearing', 'visibility',
                     'loud_cover', 'pressure']

    df = pd.DataFrame([feature_values], columns=feature_names)
    output = model.predict(df)
    return render_template('index_weather.html', prediction_text='The temperature would be {}'.format(output))
    # if request.method == 'post':
    #     precip_type = request.form['precip_type']
    #     apparent_temperature = request.form['apparent_temperature']
    #     humidity = request.form['humidity']
    #     wind_speed = request.form['wind_speed']
    #     wind_bearing = request.form['wind_bearing']
    #     visibility = request.form['visibility']
    #     loud_cover = request.form['loud_cover']
    #     pressure = request.form['pressure']
    #
    #     df = np.array(
    #         [precip_type, apparent_temperature, humidity, wind_speed, wind_bearing, visibility, loud_cover, pressure])
    #
    #     output = model.predict(df)
    #     return render_template('index_weather.html', prediction_text=output)
    # else:
    #     return render_template('index_weather.html')


if __name__ == '__main__':
    app.run(debug=True)
