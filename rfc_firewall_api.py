from flask import Flask, request, jsonify
import joblib
import pandas as pd


model = joblib.load("rfc_model.pkl")
column_names = pd.read_csv("Column_Names.txt", header=None).values.flatten()

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        feature_values = [data.get(f, 0) for f in column_names]
        input_row = pd.DataFrame([feature_values], columns=column_names)

        protocol_map = {"TCP": 6, "UDP": 17, "ICMP": 1}
        input_row["Protocol"] = input_row["Protocol"].map(lambda x: protocol_map.get(x.upper(), 0))

        prediction = model.predict(input_row)[0]
        probability = model.predict_proba(input_row)[0][1]

        prediction = "Anomaly" if prediction == 1 else "Normal"
        return jsonify({
            "prediction": prediction,
            "probability": round(probability, 4)
        })

    except Exception as e:
        return jsonify({"Error": str(e)}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
