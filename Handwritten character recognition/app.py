from flask import Flask, render_template, request
import os
from predict import predict_character

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None

    if request.method == "POST":
        file = request.files["image"]

        if file:
            path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(path)

            prediction, confidence = predict_character(path)

    return render_template("index.html",
                           prediction=prediction,
                           confidence=confidence)


if __name__ == "__main__":
    app.run(debug=True)
