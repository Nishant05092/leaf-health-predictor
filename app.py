# app.py
import os
from flask import Flask, request, render_template
import pandas as pd
import joblib
from clip_model_loader import extract_clip_features

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your trained model
model = joblib.load('model.pkl')

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get user inputs
        image = request.files["image"]
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        date_str = request.form["date"]

        # Save image
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        image.save(image_path)

        # Extract datetime features
        dt = pd.to_datetime(date_str)
        month, day, hour = dt.month, dt.day, dt.hour

        # Extract CLIP features
        image_vec, text_vec = extract_clip_features(image_path, "a close-up of a leaf")

        # Combine all features
        final_vec = list(image_vec) + list(text_vec) + [temperature, humidity, month, day, hour]

        # Debug prints
        print("✅ Final vector length:", len(final_vec))  # should be 1029
        print("📦 Feature preview (first 5):", final_vec[:5])
        print("🔍 Raw model prediction:", model.predict([final_vec]))
        try:
            proba = model.predict_proba([final_vec])
            print("📊 Prediction probabilities:", proba)
        except:
            print("⚠️ predict_proba not supported by this model.")

        # Predict
        prediction = model.predict([final_vec])[0]
        result = "Healthy 🍃" if prediction == 1 else "Unhealthy 🍂"

        return render_template("index.html", result=result, image_path=image_path)

    return render_template("index.html", result=None)

if __name__ == '__main__':
    app.run(debug=True)


