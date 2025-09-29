from flask import Flask, jsonify, request
from flasgger import Swagger
import joblib

app = Flask(__name__)
swagger = Swagger(app)

def startup():
    global model
    model = joblib.load('model/logistic_model.pkl')

@app.route('/')
def home():
    return jsonify({"message": "Iris dataset logistic regression prediction!!"})

@app.route('/predict', methods = ['POST'])
def prediction():
    """
    Predict Iris flower class
    ---
    tags:
      - Iris Prediction
    parameters:
      - name: features
        in: body
        required: true
        description: List of 4 features [sepal_length, sepal_width, petal_length, petal_width]
        schema:
          type: object
          properties:
            features:
              type: array
              items:
                type: number
              example: [5.1, 3.5, 1.4, 0.2]
    responses:
      200:
        description: Prediction result
        schema:
          type: object
          properties:
            prediction:
              type: integer
    """
    data = request.json["features"]

    prediction = model.predict([data])

    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    startup()
    app.run(host="0.0.0.0", debug=True)