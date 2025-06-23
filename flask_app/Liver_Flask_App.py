from flask import Flask, render_template, request
import joblib
import numpy as np
import os
import joblib
import numpy as np

scaler_path = os.path.join(os.path.dirname(__file__), '../flask_app/scaler.pkl')
scaler=joblib.load(scaler_path)  # Load the scaler from the correct path

  # Scale using training scaler

app = Flask(__name__, template_folder='templates', static_folder='static')

# Load model at startup with correct path
try:
    model_path = os.path.join(os.path.dirname(__file__), '../flask_app/ETC.pkl')
    with open(model_path, 'rb') as f:
        model = joblib.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load ML model: {str(e)}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_form')
def show_form():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])  
def predict():
    try:
        required_fields = ['age', 'gender', 'tb', 'db', 'ap', 
                         'aa1', 'aa2', 'tp', 'a', 'agr']
        if not all(field in request.form for field in required_fields):
            missing = [f for f in required_fields if f not in request.form]
            return render_template('error.html', 
                                error_message=f"Missing fields: {', '.join(missing)}")

        try:
            input_data = np.array([
                [float(request.form['age']),
                float(request.form['gender']),
                float(request.form['tb']),
                float(request.form['db']),
                float(request.form['ap']),
                float(request.form['aa1']),
                float(request.form['aa2']),
                float(request.form['tp']),
                float(request.form['a']),
                float(request.form['agr'])]
            ])
        except ValueError:
            return render_template('error.html',
                                error_message="All inputs must be numeric values")

        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict(input_data_scaled)[0]
        print(input_data_scaled)
        return render_template(
            'noChance.html' if prediction == 1 else 'chance.html',
            prediction=(
                "Our analysis indicates a potential liver disease. "
                "Please consult a healthcare professional immediately."
                if prediction == 1 else
                "No signs of liver disease detected. "
                "Maintain regular checkups for continued health."
            )
        )

    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return render_template('error.html',
                            error_message="An unexpected error occurred during prediction")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)