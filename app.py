
import streamlit as st
import pandas as pd
import joblib

# Load artifacts
model = joblib.load("xgb_portfolio_model.pkl")
encoder = joblib.load("portfolio_encoder.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.title("🏥 Optima Life Renewal Predictor")

st.markdown(
    "Predict customer renewal probability using demographic, behavioral, and product portfolio information."
)

# --------------------------------------------------
# DEMOGRAPHICS
# --------------------------------------------------

st.header("Customer Demographics")

age = st.slider(
    "Age",
    min_value=18,
    max_value=90,
    value=40
)

tech_comfort_score = st.slider(
    "Technology Comfort Score",
    min_value=1,
    max_value=5,
    value=3
)

income_level = st.selectbox(
    "Income Level",
    ["High", "Low", "Medium", "Very High"]
)

education = st.selectbox(
    "Education",
    ["Graduate", "High School", "Other", "Post-Graduate"]
)

device_type = st.selectbox(
    "Device Type",
    ["Desktop-only", "Mobile-only", "Multi-device"]
)

# --------------------------------------------------
# ENGAGEMENT
# --------------------------------------------------

st.header("Customer Engagement")

total_num_sessions = st.slider(
    "Total Number of Sessions",
    min_value=0,
    max_value=300,
    value=50
)

total_session_length = st.slider(
    "Total Session Length",
    min_value=0,
    max_value=20000,
    value=5000
)

active_days = st.slider(
    "Active Days",
    min_value=0,
    max_value=365,
    value=100
)

active_products = st.slider(
    "Active Products",
    min_value=1,
    max_value=5,
    value=2
)

active_quarters = st.slider(
    "Active Quarters",
    min_value=1,
    max_value=4,
    value=2
)

avg_sessions_per_active_quarter = st.slider(
    "Average Sessions Per Active Quarter",
    min_value=0.0,
    max_value=150.0,
    value=25.0
)

# --------------------------------------------------
# PRODUCTS
# --------------------------------------------------

st.header("Products Owned")

healthy_meals = st.checkbox("Healthy Meals")
daily_fitness = st.checkbox("Daily Fitness")
wellness_tracker = st.checkbox("Wellness Tracker")
mindful_living = st.checkbox("Mindful Living")
premium_health = st.checkbox("Premium Health")

products_owned = sum([
    healthy_meals,
    daily_fitness,
    wellness_tracker,
    mindful_living,
    premium_health
])

# --------------------------------------------------
# PREDICT
# --------------------------------------------------

if st.button("Predict Renewal Probability"):

    raw = pd.DataFrame([{
        "INCOME_LEVEL": income_level,
        "EDUCATION": education,
        "DEVICE_TYPE": device_type
    }])

    encoded = encoder.transform(raw)

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(
            ["INCOME_LEVEL",
             "EDUCATION",
             "DEVICE_TYPE"]
        )
    )

    input_df = pd.DataFrame([{
        "AGE": age,
        "TOTAL_NUM_SESSIONS": total_num_sessions,
        "TOTAL_SESSION_LENGTH": total_session_length,
        "ACTIVE_DAYS": active_days,
        "ACTIVE_PRODUCTS": active_products,
        "ACTIVE_QUARTERS": active_quarters,
        "AVG_SESSIONS_PER_ACTIVE_QUARTER":
            avg_sessions_per_active_quarter,
        "TECH_COMFORT_SCORE": tech_comfort_score,
        "PRODUCTS_OWNED": products_owned,

        "HAS_HEALTHY_MEALS": int(healthy_meals),
        "HAS_DAILY_FITNESS": int(daily_fitness),
        "HAS_WELLNESS_TRACKER": int(wellness_tracker),
        "HAS_MINDFUL_LIVING": int(mindful_living),
        "HAS_PREMIUM_HEALTH": int(premium_health)
    }])

    input_df = pd.concat(
        [input_df, encoded_df],
        axis=1
    )

    # Ensure same columns as training
    input_df = input_df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    probability = model.predict_proba(input_df)[0][1]

    churn_probability = 1 - probability

    if churn_probability < 0.20:
        risk = "🟢 Low Risk"
    elif churn_probability < 0.50:
        risk = "🟡 Medium Risk"
    else:
        risk = "🔴 High Risk"

    st.success(
        f"Renewal Probability: {probability:.2%}"
    )

    st.warning(
        f"Churn Probability: {churn_probability:.2%}"
    )

    st.info(
        f"Risk Category: {risk}"
    )
