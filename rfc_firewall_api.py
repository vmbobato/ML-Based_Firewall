from flask import Flask, request, jsonify
import joblib
import pandas as pd


model = joblib.load("rfc_model.pkl")
feature_names = joblib.load("feature_names.pkl")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        feature_values = [data.get(f, 0) for f in feature_names]
        input_row = pd.DataFrame([feature_values], columns=feature_names)

        prediction = model.predict(input_row)[0]
        probability = model.predict_proba(input_row)[0][1]

        prediction = "Anomaly" if prediction == 1 else "Normal"
        return jsonify({
            "prediction": prediction,
            "probability": round(probability, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
