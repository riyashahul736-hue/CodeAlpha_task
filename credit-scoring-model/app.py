from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load AI Model
with open('model/credit_model.pkl', 'rb') as f:
    model = pickle.load(f)


# Home Page
@app.route('/')
def home():
    return render_template('index.html')


# Prediction
@app.route('/predict', methods=['POST'])
def predict():

    age = int(request.form['age'])
    income = int(request.form['income'])
    loan = int(request.form['loan'])
    debt = int(request.form['debt'])

    data = pd.DataFrame(
        [[age, income, loan, debt]],
        columns=['age', 'income', 'loan', 'debt']
    )

    prediction = model.predict(data)[0]

    # Risk Level & Loan Approval Percentage
    if prediction == 1:
        risk = "Low Risk"
        approval = "92%"
        score = "800 / 850"
    else:
        risk = "High Risk"
        approval = "25%"
        score = "450 / 850"

    return f"""
    <html>
    <head>
        <title>Credit Scoring AI Result</title>
        <style>
            body {{
                font-family: Arial;
                text-align: center;
                background: linear-gradient(135deg,#0f172a,#2563eb);
                color: white;
                padding-top: 100px;
            }}
            .card {{
                background: white;
                color: black;
                width: 400px;
                margin: auto;
                padding: 30px;
                border-radius: 20px;
                box-shadow: 0 0 20px black;
            }}
            h1 {{
                color: #2563eb;
            }}
            a {{
                text-decoration: none;
                font-weight: bold;
            }}
        </style>
    </head>

    <body>

        <div class="card">
            <h1>🏦 AI Credit Result</h1>

            <h2>Credit Score: {score}</h2>

            <h2>Risk Level: {risk}</h2>

            <h2>Loan Approval Chance: {approval}</h2>

            <br>

            <a href="/">⬅ Check Again</a>
        </div>

    </body>
    </html>
    """


if __name__ == "__main__":
    app.run(debug=True)
