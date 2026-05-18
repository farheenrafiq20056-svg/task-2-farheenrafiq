# Project 2: Data Classification Using AI
# Dataset: Iris Flowers 🌸

# Step 1: Import Libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Step 2: Load Dataset
print("=" * 50)
print("   🌸 Iris Flower Classification AI")
print("=" * 50)

iris = load_iris()

# Step 3: Understand the Data
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['flower_type'] = iris.target
df['flower_name'] = df['flower_type'].map({
    0: 'Setosa',
    1: 'Versicolor', 
    2: 'Virginica'
})

print("\n📊 First 5 rows of dataset:")
print(df.head())

print(f"\n📌 Total flowers in dataset: {len(df)}")
print(f"🌸 Flower types: {df['flower_name'].unique()}")

# Step 4: Split Data into Training & Testing
X = iris.data   # Features (measurements)
y = iris.target # Labels (flower types)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,    # 20% for testing
    random_state=42   # For consistent results
)

print(f"\n✅ Training samples: {len(X_train)}")
print(f"✅ Testing samples:  {len(X_test)}")

# Step 5: Train the AI Model
model = KNeighborsClassifier(n_neighbors=3)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model.fit(X_train, y_train)
print("\n🤖 Model trained successfully!")

# Step 6: Test the Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n🎯 Model Accuracy: {accuracy * 100:.2f}%")

# Step 7: Detailed Report
print("\n📈 Classification Report:")
print(classification_report(y_test, y_pred, 
      target_names=iris.target_names))

# Step 8: Predict a New Flower 🌸
print("=" * 50)
print("🔍 Let's predict a new flower!")
print("=" * 50)

# Sample flower measurements
new_flower = [[5.1, 3.5, 1.4, 0.2]]  # sepal length, sepal width, petal length, petal width
prediction = model.predict(new_flower)
flower_names = ['Setosa', 'Versicolor', 'Virginica']

print(f"\n📏 Measurements: {new_flower[0]}")
print(f"🌸 Predicted Flower: {flower_names[prediction[0]]}")
print("\n✅ Project 2 Complete!")