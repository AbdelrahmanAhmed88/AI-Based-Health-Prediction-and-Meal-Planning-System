from flask import Flask, request, render_template
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained models
bmi_model = joblib.load('random_forest_model.pkl')  # Replace with the actual path to your BMI model file
calories_model = joblib.load('Gradient_Boosting_model_for_calories.pkl')  # Replace with the actual path to your calories model file
RFC = joblib.load('RFC_model_for_Meals.pkl')  # Load your Random Forest model for meal prediction

# Mapping for activity levels
activity_mapping = {
    "sedentary": 1.2,
    "lightly active": 1.375,
    "moderately active": 1.55,
    "very active": 1.725
}

activity_level_map = {
    'sedentary': 1,
    'lightly active': 2,
    'moderately active': 3,
    'very active': 4
}

# BMI class to numerical mapping
bmi_class_map = {
    'Extremely Weak': 0,
    'Weak': 1,
    'Normal': 2,
    'Overweight': 3,
    'Obesity': 4,
    'Extremely Obesity': 5
}

# Meal mapping with detailed meal plans
meal_mapping = {
    0: {
        "meal_name": "Meal A (Low Calorie)",
        "meals": [
            {"meal": "Breakfast: Oats with low-fat milk, fruits", "calories": 300},
            {"meal": "Lunch: Grilled chicken with steamed vegetables", "calories": 400},
            {"meal": "Dinner: Baked fish with a side of quinoa", "calories": 200},
            {"meal": "Snack: Apple with almond butter", "calories": 120}
        ]
    },
    1: {
        "meal_name": "Meal B (Medium Calorie)",
        "meals": [
            {"meal": "Breakfast: Scrambled eggs with avocado toast", "calories": 350},
            {"meal": "Lunch: Turkey sandwich with whole-grain bread and veggies", "calories": 500},
            {"meal": "Dinner: Stir-fry with tofu and brown rice", "calories": 450},
            {"meal": "Snack: Greek yogurt with berries", "calories": 200}
        ]
    },
    2: {
        "meal_name": "Meal C (High Calorie)",
        "meals": [
            {"meal": "Breakfast: Pancakes with maple syrup and bacon", "calories": 600},
            {"meal": "Lunch: Grilled steak with mashed potatoes and vegetables", "calories": 700},
            {"meal": "Dinner: Spaghetti with meatballs and garlic bread", "calories": 800},
            {"meal": "Snack: Protein bar and banana", "calories": 300}
        ]
    },
    3: {
        "meal_name": "Meal D (Flex)",
        "meals": [
            {"meal": "Breakfast: Smoothie with spinach, banana, and protein powder", "calories": 400},
            {"meal": "Lunch: Grilled chicken salad with olive oil dressing", "calories": 550},
            {"meal": "Dinner: Grilled shrimp with asparagus and couscous", "calories": 500},
            {"meal": "Snack: Mixed nuts and dried fruit", "calories": 250}
        ]
    },
}

def predict_bmr_tdee(gender, height, weight, age, activity_level):
    # Map activity level to its multiplier and value
    activity_level_multiplier = activity_mapping.get(activity_level.lower(), 1.2)  # Default to Sedentary
    activity_level_value = activity_level_map.get(activity_level.lower(), 1)  # Default to Sedentary
    
    # Prepare the input data for BMI prediction
    bmi = weight / ((height / 100) ** 2)
    bmi_input = pd.DataFrame({
        'Gender': [1 if gender.lower() == 'male' else 0],
        'Height': [height],
        'Weight': [weight],
        'BMI': [bmi]
    })

    # Make the prediction for BMI class
    bmi_prediction = bmi_model.predict(bmi_input)
    bmi_classes = ['Extremely Weak', 'Weak', 'Normal', 'Overweight', 'Obesity', 'Extremely Obesity']
    bmi_class = bmi_classes[bmi_prediction[0]]

    # Prepare the input data for calorie prediction (BMR)
    calorie_input = pd.DataFrame({
        'Gender': [1 if gender.lower() == 'male' else 0],
        'Height': [height],
        'Weight': [weight],
        'age': [age],
        'activity_level': [activity_level_value],
        'activity_multiplier': [activity_level_multiplier]
    })

    # Make the prediction for daily calorie needs (BMR)
    predicted_calories = calories_model.predict(calorie_input)

    # Calculate TDEE (BMR * activity multiplier)
    tdee_prediction = predicted_calories[0] * activity_level_multiplier

    return bmi_class, round(predicted_calories[0], 2), round(tdee_prediction, 2), activity_level_value, activity_level_multiplier


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Collect input values from the form
    gender = request.form['gender']
    height = float(request.form['height'])
    weight = float(request.form['weight'])
    age = int(request.form['age'])
    activity_level = request.form['activity_level']

    # Validate inputs
    if not all([gender, height, weight, age, activity_level]) or activity_level.lower() not in activity_mapping:
        return "Invalid input", 400  # Or render an error page with feedback

    # Call the prediction function
    bmi_class, predicted_calories, tdee_prediction, activity_level_value, activity_level_multiplier = predict_bmr_tdee(
        gender, height, weight, age, activity_level)

    # Convert gender to 1 for male and 0 for female
    gender = 1 if gender.lower() == 'male' else 0

    # Map the bmi_class to its numerical value
    bmi_class_numeric = bmi_class_map.get(bmi_class, -1)  # Return -1 if not found

    if bmi_class_numeric == -1:
        return "Error: BMI class not recognized", 400  # Handle unrecognized BMI classes

    # Adjust target calories based on BMI class
    if bmi_class_numeric == 0:  # Extremely Weak
        target_calories = tdee_prediction + 500
    elif bmi_class_numeric == 1:  # Weak
        target_calories = tdee_prediction + 300
    elif bmi_class_numeric == 2:  # Normal
        target_calories = tdee_prediction
    elif bmi_class_numeric == 3:  # Overweight
        target_calories = tdee_prediction - 300
    elif bmi_class_numeric == 4:  # Obesity
        target_calories = tdee_prediction - 400
    else:  # Extremely Obesity
        target_calories = tdee_prediction - 500

    # Get the recommended meal plan (using your meal prediction logic)
    predicted_meal_category = RFC.predict(np.array([[gender, height, weight, bmi_class_numeric, age, activity_level_value, tdee_prediction, target_calories]]))[0]

    # Fetch the corresponding meals
    predicted_meal_plan = meal_mapping.get(predicted_meal_category, {"meal_name": "Meal Not Found", "meals": []})
    meal_options = predicted_meal_plan["meals"]
    
    # Render the results back to the HTML page
    return render_template('index.html', 
                           bmi_class=bmi_class, 
                           predicted_calories=predicted_calories, 
                           tdee_prediction=tdee_prediction,
                           target_calories=target_calories,
                           predicted_meal_plan=predicted_meal_plan["meal_name"],
                           meal_options=meal_options,  # Pass meal options to template
                           meal_mapping=meal_mapping)  # Optionally pass the full meal mapping if needed


if __name__ == '__main__':
    app.run(debug=True)
