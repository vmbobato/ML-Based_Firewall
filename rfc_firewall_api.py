from flask import Flask, request, jsonify
import joblib
import pandas as pd

model = joblib.load("rfc_model.pkl")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        input_df = pd.DataFrame([data]) 

        prediction = model.predict(input_df)[0]
        if prediction == 1:
            prediction = "Anomaly"
        else:
            prediction = "Normal"
        probability = model.predict_proba(input_df)[0][1]

        return jsonify({
            "prediction": prediction,
            "probability": round(probability, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
