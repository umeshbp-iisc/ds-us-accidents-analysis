import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gdown
import os

# -----------------------------------------
# Load Saved Model and Required Objects
# -----------------------------------------

MODEL_URL = "https://drive.google.com/uc?id=1orxpg8nWpQEST_HcO7iIf80QocEkXk77"
MODEL_PATH = "tuned_logreg_pipeline.joblib"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model... Please wait"):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return joblib.load(MODEL_PATH)

model = load_model()

# Extract internal parts
preprocessor = model.named_steps["preprocessor"]
selector = model.named_steps["feature_selection"]
stack_clf = model.named_steps["clf"]
meta_clf = stack_clf.final_estimator_

# Get selected (post-preprocessing) feature names
processed_feature_names = preprocessor.get_feature_names_out()
selected_mask = selector.get_support()
selected_features = processed_feature_names[selected_mask]

# Get meta-learner feature names
meta_feature_names = stack_clf.get_feature_names_out()

# -----------------------------------------------------------
# 🔹 Streamlit UI
# -----------------------------------------------------------

st.title("🚗 Road Accident Severity Prediction App")
st.write("This app predicts accident **Severity**, shows **probabilities**, and explains **feature contributions**.")

st.header("Enter Feature Values")

# ------------------------------------------
# Input widget helper functions
# ------------------------------------------

def number_input(name, default):
    return st.number_input(name, value=default)

def select_input(name, options):
    return st.selectbox(name, options)

# ------------------------------------------
# Build Input Form
# ------------------------------------------

numeric_cols = ['Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)',
                'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)']

bool_cols = ['Amenity','Bump','Crossing','Give_Way','Junction','No_Exit','Railway',
             'Roundabout','Station','Stop','Traffic_Calming','Traffic_Signal','Turning_Loop']

cat_cols = ['Weather_Condition','Sunrise_Sunset','Civil_Twilight',
            'Nautical_Twilight','Astronomical_Twilight']

# Weather categories as used in training
weather_top10 = ['Fair', 'Cloudy', 'Mostly Cloudy', 'Partly Cloudy', 'Light Rain',
                 'Rain', 'Fog', 'Heavy Rain', 'Snow', 'Scattered Clouds', 'Other']

user_data = {}

st.subheader("Numeric Inputs")
for c in numeric_cols:
    user_data[c] = number_input(c, 0.0)

st.subheader("Boolean Inputs")
for c in bool_cols:
    user_data[c] = st.selectbox(c, [0, 1], index=0)

st.subheader("Categorical Inputs")
for c in cat_cols:
    if c == "Weather_Condition":
        user_data[c] = select_input(c, weather_top10)
    else:
        user_data[c] = select_input(c, ["Day", "Night", "Other"])

# Convert to DataFrame
input_df = pd.DataFrame([user_data])

st.write("### Input Data")
st.dataframe(input_df)

# -----------------------------------------------------------
# 🔹 Predict Button
# -----------------------------------------------------------

if st.button("Predict Severity"):
    
    # Predict class & probabilities
    predicted_class = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    
    st.success(f"### ✅ Predicted Severity: **{predicted_class}**")
    
    # ---------------- Probability Table ----------------
    st.subheader("📊 Prediction Probabilities")
    
    class_labels = stack_clf.classes_
    proba_df = pd.DataFrame({
        "Severity": class_labels,
        "Probability": proba
    })
    st.dataframe(proba_df.style.highlight_max(color="lightgreen", axis=0))
    
    # -----------------------------------------------------------
    # 🔥 Feature Contributions
    # -----------------------------------------------------------

    st.header("📈 Feature Contributions")

    # ---------------- Meta-Learner Contributions ----------------
    st.subheader("Meta-Learner Contributions (Stacking Level)")

    coef_matrix = meta_clf.coef_
    meta_cols = [f"Class_{c}" for c in meta_clf.classes_[:-1]]

    contrib_meta_df = pd.DataFrame(
        coef_matrix.T, index=meta_feature_names, columns=meta_cols
    )
    contrib_meta_df["mean_abs_contrib"] = contrib_meta_df.abs().mean(axis=1)
    contrib_meta_df = contrib_meta_df.sort_values("mean_abs_contrib", ascending=False)

    st.dataframe(contrib_meta_df)

    # ---------------- Logistic Regression Base Model ------------
    st.subheader("Logistic Regression Base Model Contributions")

    logreg = stack_clf.named_estimators_["logreg"]
    logreg_coef = logreg.coef_

    logreg_cols = [f"Class_{c}" for c in logreg.classes_[:-1]]

    contrib_logreg_df = pd.DataFrame(
        logreg_coef.T,
        index=selected_features,
        columns=logreg_cols
    )

    contrib_logreg_df["mean_abs_contrib"] = contrib_logreg_df.abs().mean(axis=1)
    contrib_logreg_df = contrib_logreg_df.sort_values("mean_abs_contrib", ascending=False)

    st.dataframe(contrib_logreg_df)

    st.info("Higher absolute values indicate stronger influence on the final prediction.")
