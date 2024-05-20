import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline

# Example dataset with more samples (replace this with your actual dataset)
data = pd.DataFrame({
    'question': [
        'Solve for x in the equation x^2 + 5x + 6 = 0',
        'What is the derivative of sin(x)?',
        'Find the integral of x^2',
        'What is the value of pi?',
        'Solve the quadratic equation x^2 - 4x + 4 = 0',
        'What is the limit of x as it approaches 0?',
        'Calculate the area of a circle with radius r',
        'What is the value of the square root of 16?',
        'Integrate the function x^3',
        'Find the roots of the polynomial x^2 - x - 6',
        'What is the value of e?',
        'Derive the function f(x) = x^2 + 3x + 2',
        'What is the cosine of 90 degrees?',
        'Solve the system of equations: 2x + 3y = 6 and 4x - y = 5',
        'What is the integral of sin(x)?',
        'Find the derivative of e^x',
        'Calculate the volume of a sphere with radius r'
    ],
    'on_test': [1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0]
})

# Preprocess the text data using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['question'])
y = data['on_test']

# Create a pipeline that includes SMOTE and Logistic Regression
pipeline = make_pipeline(SMOTE(random_state=42, k_neighbors=1), LogisticRegression())

# Define the parameter grid
param_grid = {
    'logisticregression__C': [0.01, 0.1, 1, 10, 100],
    'logisticregression__solver': ['lbfgs', 'liblinear']
}

# Create the GridSearchCV object
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_weighted')

# Perform the grid search
grid_search.fit(X, y)

# Get the best model
best_model = grid_search.best_estimator_

# Train the best model on the entire dataset
best_model.fit(X, y)

# Function to preprocess and predict if a question will be on the test
def predict_question(question):
    # Transform the question using the same TF-IDF vectorizer
    question_transformed = vectorizer.transform([question])
    
    # Predict using the trained model
    prediction = best_model.predict(question_transformed)
    
    return prediction[0]

# Example usage
new_question = 'What is the derivative of 1/sin(x))?'
prediction = predict_question(new_question)
print(f"Will the question '{new_question}' be on the test? {'Yes' if prediction == 1 else 'No'}")
