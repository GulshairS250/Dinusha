import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Configuration --- #
# Map encoded Nutri-Score back to grades
nutriscore_mapping_reverse = {0.0: 'A', 1.0: 'B', 2.0: 'C', 3.0: 'D', 4.0: 'E'}

# Expected feature columns in the order the model was trained on
# This is crucial for correct prediction
expected_feature_columns = ['nova_group', 'energy_kcal', 'fat_100g', 'saturated_fat_100g',
                            'carbs_100g', 'sugars_100g', 'fiber_100g', 'proteins_100g',
                            'salt_100g', 'sodium_100g', 'contains_gluten', 'contains_dairy',
                            'contains_nuts', 'contains_soy', 'contains_eggs', 'contains_fish',
                            'food_type_Branded/Packaged', 'has_allergens',
                            'fat_to_protein_ratio', 'sugar_to_fiber_ratio',
                            'sodium_to_energy_ratio', 'total_allergens_present']

# --- Streamlit App --- #
st.set_page_config(page_title="Nutri-Score Predictor", layout="wide")
st.title("🍎 Nutri-Score Predictor 📊")
st.markdown("--- Jardine")
st.write("Welcome to the **Nutri-Score Predictor**! \n\nInput the nutritional information of a food product using the sidebar on the left, and I'll predict its Nutri-Score grade (from A to E), providing a quick assessment of its nutritional quality.")
st.markdown("--- Jardine")

# Load the trained model pipeline
@st.cache_resource
def load_model():
    try:
        model_filename = 'tuned_nutriscore_model.pkl'
        loaded_pipeline = joblib.load(model_filename)
        return loaded_pipeline
    except FileNotFoundError:
        st.error(f"Error: Model file '{model_filename}' not found. Please ensure the model is saved.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

loaded_model_pipeline = load_model()

# --- User Input Section --- #
st.sidebar.header("Input Product Details")

with st.sidebar.expander("🍚 Nutritional Values (per 100g)", expanded=True):
    nova_group = st.slider("Nova Group (Level of Processing)", 1, 4, 3, help="1 = Unprocessed/Minimally Processed; 2 = Processed Culinary Ingredients; 3 = Processed; 4 = Ultra-Processed")
    energy_kcal = st.number_input("Energy (kcal)", min_value=0.0, max_value=900.0, value=200.0, step=1.0)
    fat_100g = st.number_input("Fat (g)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
    saturated_fat_100g = st.number_input("Saturated Fat (g)", min_value=0.0, max_value=100.0, value=3.0, step=0.1)
    carbs_100g = st.number_input("Carbohydrates (g)", min_value=0.0, max_value=100.0, value=30.0, step=0.1)
    sugars_100g = st.number_input("Sugars (g)", min_value=0.0, max_value=100.0, value=15.0, step=0.1)
    fiber_100g = st.number_input("Fiber (g)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
    proteins_100g = st.number_input("Proteins (g)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
    salt_100g = st.number_input("Salt (g)", min_value=0.0, max_value=10.0, value=0.5, step=0.01)
    sodium_100g = st.number_input("Sodium (g)", min_value=0.0, max_value=4.0, value=0.2, step=0.001)

with st.sidebar.expander("⚠️ Allergen Information", expanded=True):
    contains_gluten = st.checkbox("Contains Gluten")
    contains_dairy = st.checkbox("Contains Dairy")
    contains_nuts = st.checkbox("Contains Nuts")
    contains_soy = st.checkbox("Contains Soy")
    contains_eggs = st.checkbox("Contains Eggs")
    contains_fish = st.checkbox("Contains Fish")

# Assume food_type is Branded/Packaged as per EDA insights from training data
food_type_Branded_Packaged = True

# --- Prediction Logic --- #
if st.sidebar.button("🚀 Predict Nutri-Score"):
    st.subheader("Prediction in Progress...")
    
    # Calculate engineered features
    fat_to_protein_ratio = fat_100g / proteins_100g if proteins_100g != 0 else 0.0
    sugar_to_fiber_ratio = sugars_100g / fiber_100g if fiber_100g != 0 else 0.0
    sodium_to_energy_ratio = sodium_100g / energy_kcal if energy_kcal != 0 else 0.0

    total_allergens_present_count = int(contains_gluten) + int(contains_dairy) + int(contains_nuts) + \
                                    int(contains_soy) + int(contains_eggs) + int(contains_fish)
    has_allergens_bool = 1 if total_allergens_present_count > 0 else 0

    # Create a DataFrame from user inputs, ensuring column order matches training data
    input_data_values = [
        nova_group, energy_kcal, fat_100g, saturated_fat_100g, carbs_100g, sugars_100g, fiber_100g,
        proteins_100g, salt_100g, sodium_100g,
        contains_gluten, contains_dairy, contains_nuts, contains_soy, contains_eggs, contains_fish,
        food_type_Branded_Packaged, has_allergens_bool,
        fat_to_protein_ratio, sugar_to_fiber_ratio, sodium_to_energy_ratio, total_allergens_present_count
    ]

    input_df = pd.DataFrame([input_data_values], columns=expected_feature_columns)

    # Ensure boolean columns are of the correct type (bool or int based on model expectation)
    for col in ['contains_gluten', 'contains_dairy', 'contains_nuts', 'contains_soy', 'contains_eggs', 'contains_fish', 'food_type_Branded/Packaged']:
        input_df[col] = input_df[col].astype(bool)

    # Predict Nutri-Score
    prediction_encoded = loaded_model_pipeline.predict(input_df)[0]
    predicted_grade = nutriscore_mapping_reverse.get(prediction_encoded, "Unknown")

    st.subheader("Prediction Results:")
    grade_color = {
        'A': 'green', 'B': 'lightgreen', 'C': 'orange', 'D': 'red', 'E': 'darkred', 'Unknown': 'gray'
    }
    st.markdown(f"The predicted Nutri-Score is: <span style='color:{grade_color[predicted_grade]}; font-size: 2.5em; font-weight: bold;'>Grade {predicted_grade}</span>", unsafe_allow_html=True)
    
    if predicted_grade in ['A', 'B']:
        st.balloons()
        st.success("🎉 Great choice! This product has a good Nutri-Score.")
    elif predicted_grade in ['C']:
        st.info("👍 This product has a moderate Nutri-Score.")
    elif predicted_grade in ['D', 'E']:
        st.warning("⚠️ Consider healthier alternatives, this product has a less favorable Nutri-Score.")

    st.markdown("--- Jardine")
    st.write("**Detailed Input Data Used for Prediction:**")
    st.dataframe(input_df)

st.markdown("--- Jardine")
st.write("### How to run this app locally:")
st.code("1. Save the code above as `app.py`\n2. Make sure `tuned_nutriscore_model.pkl` is in the same directory.\n3. Open your terminal or command prompt.\n4. Navigate to the directory where you saved the files.\n5. Run the command: `streamlit run app.py`")
st.markdown("--- Jardine")