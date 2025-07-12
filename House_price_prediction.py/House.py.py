import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
from tkinter import *
from tkinter import ttk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

df = pd.read_csv("Jaipur_House_Data.csv")
df = df.dropna(subset=["location", "size", "total_sqft", "bath", "price"])
df = df[df["total_sqft"].apply(lambda x: str(x).replace("-", "").replace(".", "").isnumeric())]
df["total_sqft"] = df["total_sqft"].astype(float)
df["bhk"] = df["size"].apply(lambda x: int(x.split(" ")[0]))

data = df[["location", "total_sqft", "bath", "bhk", "price"]]
data = data[(data["bath"] < 10) & (data["bhk"] < 10) & (data["total_sqft"] < 10000)]

X = data[["location", "total_sqft", "bath", "bhk"]]
y = data["price"]

ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(handle_unknown="ignore"), ["location"])], remainder="passthrough")
X_encoded = ct.fit_transform(X)

# Train model
model = LinearRegression()
model.fit(X_encoded, y)

def predict_gui():
    loc = location_var.get()
    sqft = float(sqft_var.get())
    bath = int(bath_var.get())
    bhk = int(bhk_var.get())

    input_df = pd.DataFrame([[loc, sqft, bath, bhk]], columns=["location", "total_sqft", "bath", "bhk"])
    input_encoded = ct.transform(input_df)
    prediction = model.predict(input_encoded)[0]
    result_label.config(text=f" Estimated Price: ₹ {prediction:.2f} lakhs")

#GUI
root = Tk()
root.title("Jaipur House Price Predictor ")
root.geometry("400x400")
root.configure(bg="#B1EAF6")

Label(root, text="Jaipur House Price Predictor", font=("Helvetica", 20, "bold"), bg="#f1b9b9",fg="#ee0a28").pack(pady=10)

# Location Dropdown
location_var = StringVar()
Label(root, text="Location:", bg="#f1c80e",font=("Elephant",12)).pack()
location_menu = ttk.Combobox(root, textvariable=location_var, values=sorted(df["location"].unique()), width=40)
location_menu.pack(pady=5)
location_menu.current(0)

# Square Footage Entry
sqft_var = StringVar()
Label(root, text="Total Sqft:", bg="#caea62",font=("Elephant",12)).pack()
Entry(root, textvariable=sqft_var, width=30).pack(pady=5)

# Bathroom Entry
bath_var = StringVar()
Label(root, text="Number of Bathrooms:", bg="#d7abf2",font=("Elephant",12)).pack()
Entry(root, textvariable=bath_var, width=20).pack(pady=5)

# BHK Entry
bhk_var = StringVar()
Label(root, text="Number of BHK:", bg="#f192c6",font=("Elephant",12)).pack()
Entry(root, textvariable=bhk_var, width=20).pack(pady=5)

# Predict Button
Button(root, text="Predict Price ", command=predict_gui, bg="#10E617", fg="black", padx=10, pady=5).pack(pady=10)

# Result Label
result_label = Label(root, text="", font=("Comic Sans MS", 14), bg="#f0f0f0")
result_label.pack(pady=10)

root.mainloop()

# Load data
df = pd.read_csv("Jaipur_House_Data.csv")

df = df.dropna(subset=["location", "size", "total_sqft", "bath", "price"])
df = df[df["total_sqft"].apply(lambda x: str(x).replace("-", "").replace(".", "").isnumeric())]
df["total_sqft"] = df["total_sqft"].astype(float)
df["bhk"] = df["size"].apply(lambda x: int(x.split(' ')[0]))

data = df[["location", "total_sqft", "bath", "bhk", "price"]]
data = data[(data["bath"] < 10) & (data["bhk"] < 10) & (data["total_sqft"] < 10000)]

X = data[["location", "total_sqft", "bath", "bhk"]]
y = data["price"]

ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(handle_unknown="ignore"), ["location"])], remainder="passthrough")
X_encoded = ct.fit_transform(X)

# Train the model
model = LinearRegression()
model.fit(X_encoded, y)

# Predict Function
def predict_price(location, sqft, bath, bhk):
    input_df = pd.DataFrame([[location, sqft, bath, bhk]], columns=["location", "total_sqft", "bath", "bhk"])
    input_encoded = ct.transform(input_df)
    predicted_price = model.predict(input_encoded)[0]
    return predicted_price



