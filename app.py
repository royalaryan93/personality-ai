from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import os

app = Flask(__name__)

# ================= DATASET =================
data = {
    "Movie_Hours": [1,2,2,3,4,5,6,7,8,9,5,6,7,8,9,10],
    "Sleep_Hours": [6,7,7,6,6,5,5,4,3,3,8,8,9,9,10,10],
    "Attendance":  [90,85,88,80,75,70,65,60,55,50,78,72,68,60,50,40],
    "Physical_Activity": [1,1,1,1,1,0,0,0,0,0,1,1,0,0,0,0],
    "Category": [
        "hoshiyar","hoshiyar","hoshiyar","hoshiyar",
        "average","average",
        "needs improvement","needs improvement",
        "needs improvement","needs improvement",
        "average","average",
        "needs improvement","needs improvement",
        "needs improvement","needs improvement"
    ]
}

df = pd.DataFrame(data)

# ================= MODEL =================
le = LabelEncoder()
y = le.fit_transform(df["Category"])

X = df[["Movie_Hours", "Sleep_Hours", "Attendance", "Physical_Activity"]].values

model = DecisionTreeClassifier(max_depth=3)
model.fit(X, y)

# ================= ROUTES =================

# 🔥 Landing Page
@app.route("/")
def landing():
    return render_template("landing.html")

# 🔥 Form + Prediction
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        movie = float(request.form["movie"])
        sleep = float(request.form["sleep"])
        attendance = float(request.form["attendance"])
        activity = int(request.form["activity"])

        pred = model.predict([[movie, sleep, attendance, activity]])
        result = le.inverse_transform(pred)[0]

        return render_template("result.html", result=result)

    return render_template("form.html")

# ================= RUN =================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
