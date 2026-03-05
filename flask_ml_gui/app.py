from flask import Flask, render_template, request
import numpy as np
import joblib # Replace with your machine learning model import

app = Flask(__name__)

filename = 'LinRegModel.joblib'
# Load your pre-trained machine learning model here
model = joblib.load(filename) # Replace with your model loading logic

scaler = joblib.load('scaler.joblib')
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Get user input from the form
        input_data = request.form.get("input")
        float_list = [float(i) for i in input_data.split(',')]
        float_list = np.array(float_list)
        array_2d_sample = float_list.reshape(1,-1)

        # Preprocess the input data for your model (if needed)
        # ... your data preprocessing code here ...
        preprocessed_data = scaler.transform(array_2d_sample)

        # Make prediction using your model
        prediction = model.predict(preprocessed_data) # Assuming a list input
        # Format the prediction for display
        #predicted_class = prediction[0] # Assuming single class output

        return render_template("result.html", predictions=prediction, preprocessed_data=preprocessed_data)
    else:
        return "Something went wrong. Please try again."
if __name__ == "__main__":
    app.run(debug=True)