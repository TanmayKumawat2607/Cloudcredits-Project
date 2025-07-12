import tkinter as tk
from tkinter import messagebox
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset and train model
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100

# Prediction
def predict_species():
    values = [s_len.get(), s_wid.get(), p_len.get(), p_wid.get()]
    prediction = model.predict([values])[0]
    prob = model.predict_proba([values])[0][prediction] * 100
    species = target_names[prediction]
    
    msg = f"Predicted Species: {species.capitalize()}\n"
    msg += f"Confidence: {prob:.2f}%\n"
    msg += f" Model Accuracy: {accuracy:.2f}%"
    
    messagebox.showinfo("Prediction Result", msg)

# Tkinter GUI Setup
root = tk.Tk()
root.title("Iris Flower Classifier")
root.geometry("400x400")

tk.Label(root, text="Iris Flower Classifier", font=("Cooper Black", 18),bg="#e8f0fe", fg="#1a237e").pack(pady=10)

frame = tk.Frame(root)
frame.pack(pady=5)

# Sliders for Input
s_len = tk.DoubleVar()
s_wid = tk.DoubleVar()
p_len = tk.DoubleVar()
p_wid = tk.DoubleVar()

sliders = [
    ("Sepal Length (cm)", s_len, 4.0, 8.0),
    ("Sepal Width (cm)", s_wid, 2.0, 4.5),
    ("Petal Length (cm)", p_len, 1.0, 7.0),
    ("Petal Width (cm)", p_wid, 0.1, 2.5),
]

for label, var, frm, to in sliders:
    tk.Label(frame, text=label,bg="#52C8D5", fg="black",width=25).pack()
    tk.Scale(frame, variable=var, from_=frm, to=to, resolution=0.1, bg="#ACDFA2",orient=tk.HORIZONTAL).pack()

tk.Button(root, text="Predict", command=predict_species, bg="#A20ED3", fg="white", width=15).pack(pady=10)
tk.Button(root, text="Exit", command=root.quit, bg="orange", fg="white", width=15).pack()

root.mainloop()
