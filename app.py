from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import os

app = Flask(__name__)

# ================= LOAD DATASET =================
df = pd.read_excel("personality_dataset.xlsx")

# ================= MODEL =================
le = LabelEncoder()
y = le.fit_transform(df["Category"])

X = df[["Movie_Hours", "Sleep_Hours", "Attendance", "Physical_Activity"]]

model = DecisionTreeClassifier(max_depth=4)
model.fit(X, y)

# ================= ROUTES =================

# 🔥 Landing Page
@app.route("/")
def landing():
    return render_template("landing.html")

# 🔥 Prediction
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        movie = float(request.form["movie"])
        sleep = float(request.form["sleep"])
        attendance = float(request.form["attendance"])
        activity = int(request.form["activity"])

        # 🚨 INPUT VALIDATION (VERY IMPORTANT)
        if sleep < 3 or sleep > 12:
            return render_template("result.html", result="AADMI THEEK NHI HO TUM ❌")

        if movie < 0 or movie > 12:
            return render_template("result.html", result="AADMI THEEK NHI HO TUM  ❌")

        if attendance < 0 or attendance > 100:
            return render_template("result.html", result="AADMI THEEK NHI HO TUM  ❌")

        # 🔮 Prediction
        pred = model.predict([[movie, sleep, attendance, activity]])
        result = le.inverse_transform(pred)[0]

        return render_template("result.html", result=result)

    return render_template("form.html")

# ================= RUN =================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
