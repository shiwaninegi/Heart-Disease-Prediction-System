import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

hdata = pd.read_csv("/kaggle/input/heart-data-1/heart (1).csv")


x = hdata.drop(columns='target', axis=1)
y = hdata['target']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)


scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


model = LogisticRegression(class_weight='balanced', max_iter=500)
model.fit(x_train_scaled, y_train)
y_pred = model.predict(x_test_scaled)

train_accuracies = []
test_accuracies = []

for i in range(1, 11): 
    model = LogisticRegression(class_weight='balanced', max_iter=i*50)  
    model.fit(x_train_scaled, y_train)    
    
    train_accuracies.append(model.score(x_train_scaled, y_train))
    test_accuracies.append(model.score(x_test_scaled, y_test))

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")

metrics = ['accuracy','Precision', 'Recall', 'F1 Score']
values = [accuracy*100,precision * 100, recall * 100, f1 * 100]

plt.bar(metrics, values, color=['blue', 'orange', 'green','pink'])
plt.xlabel('Metrics')
plt.ylabel('Score (%)')
plt.title('Performance Metrics of Heart Disease Prediction Model')
plt.ylim([0, 100])
plt.show()

y_pred = model.predict(x_test_scaled)

def predict_heart_disease():
    print("Please enter the following details for heart disease prediction:")

    
    age = float(input("Age: "))
    sex = int(input("Sex (1 = male, 0 = female): "))
    cp = int(input("Chest pain type (1-4): "))
    trestbps = float(input("Resting blood pressure: "))
    chol = float(input("Serum cholesterol: "))
    fbs = int(input("Fasting blood sugar (> 120 mg/dl, 1 = true, 0 = false): "))
    restecg = int(input("Resting electrocardiographic results (0-2): "))
    thalach = float(input("Maximum heart rate achieved: "))
    exang = int(input("Exercise induced angina (1 = yes, 0 = no): "))
    oldpeak = float(input("Oldpeak (depression induced by exercise relative to rest): "))
    slope = int(input("Slope of the peak exercise ST segment (1-3): "))
    ca = int(input("Number of major vessels (0-3) colored by fluoroscopy: "))
    thal = int(input("Thalassemia (3 = normal, 6 = fixed defect, 7 = reversable defect): "))

    user_input = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
                              columns=x.columns)    
    user_input_scaled = scaler.transform(user_input)    
    prediction = model.predict(user_input_scaled)

    if prediction == 0:
        print("\nThe model predicts that you do not have heart disease.")
    else:
        print("\nThe model predicts that you may have heart disease.")


predict_heart_disease()