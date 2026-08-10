import streamlit as st
import pandas as pd
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 30px;
}

.card {
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #ddd;
    text-align: center;
}

.result {
    font-size: 25px;
    font-weight: bold;
    text-align: center;
    padding: 15px;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


try:
    model = load_model()
except Exception as e:
    st.error("❌ Could not load model.pkl")
    st.error(f"Error: {e}")
    st.stop()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🎓 Student Placement Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload student data and predict placement status using Machine Learning'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("📌 About")

    st.write(
        """
        This application uses an **XGBoost Classifier**
        to predict student placement status.
        """
    )

    st.divider()

    st.subheader("📂 Input")

    st.write(
        """
        Upload a CSV containing student information.

        The CSV must contain the same model input
        features used during training.
        """
    )


# =========================================================
# FILE UPLOADER
# =========================================================

st.subheader("📤 Upload Student Dataset")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
    help="Upload a CSV file containing student records."
)


if uploaded_file is None:

    st.info(
        "👆 Please upload a CSV file to start making predictions."
    )

    st.stop()


# =========================================================
# READ CSV
# =========================================================

try:

    df = pd.read_csv(uploaded_file)

except Exception as e:

    st.error("❌ Unable to read the CSV file.")
    st.error(f"Error: {e}")

    st.stop()


# =========================================================
# BASIC VALIDATION
# =========================================================

if df.empty:

    st.error("❌ The uploaded CSV file is empty.")

    st.stop()


if len(df.columns) == 0:

    st.error("❌ No columns were found in the uploaded CSV.")

    st.stop()


# =========================================================
# SHOW ORIGINAL DATA
# =========================================================

st.subheader("📋 Uploaded Dataset")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Students", len(df))

with col2:
    st.metric("Total Columns", len(df.columns))

with col3:
    st.metric(
        "Missing Values",
        int(df.isna().sum().sum())
    )


with st.expander("🔍 View Uploaded Data"):

    st.dataframe(
        df,
        use_container_width=True
    )


# =========================================================
# SAME PREPROCESSING USED DURING TRAINING
# =========================================================

drop_columns = [
    "branch",
    "college_tier",
    "salary_package_lpa",
    "ml_knowledge",
    "system_design",
    "open_source_contributions",
    "extracurriculars",
    "dsa_score",
    "certifications",
    "hackathons"
]


# Make a copy so original uploaded data remains unchanged

prediction_df = df.copy()


# Remove unnecessary columns

prediction_df = prediction_df.drop(
    columns=drop_columns,
    errors="ignore"
)


# =========================================================
# REMOVE TARGET COLUMN IF PRESENT
# =========================================================

if "placement_status" in prediction_df.columns:

    prediction_df = prediction_df.drop(
        "placement_status",
        axis=1
    )


# =========================================================
# MODEL FEATURE VALIDATION
# =========================================================

try:

    # Get features expected by the trained model

    expected_features = list(model.get_booster().feature_names)

except Exception:

    expected_features = list(prediction_df.columns)


actual_features = list(prediction_df.columns)


# =========================================================
# CHECK MISSING FEATURES
# =========================================================

missing_features = [
    col for col in expected_features
    if col not in actual_features
]


# =========================================================
# CHECK EXTRA FEATURES
# =========================================================

extra_features = [
    col for col in actual_features
    if col not in expected_features
]


# =========================================================
# DISPLAY VALIDATION
# =========================================================

if missing_features:

    st.error("❌ Dataset validation failed.")

    st.write("The following required columns are missing:")

    for col in missing_features:

        st.write(f"- `{col}`")

    st.warning(
        "Please upload a CSV containing all required model features."
    )

    st.stop()


# =========================================================
# REMOVE EXTRA COLUMNS
# =========================================================

if extra_features:

    st.warning(
        f"⚠️ {len(extra_features)} extra column(s) "
        "will be ignored."
    )

    prediction_df = prediction_df.drop(
        columns=extra_features,
        errors="ignore"
    )


# =========================================================
# REORDER FEATURES
# =========================================================

prediction_df = prediction_df[
    expected_features
]


# =========================================================
# CHECK MISSING VALUES
# =========================================================

missing_count = int(
    prediction_df.isna().sum().sum()
)


if missing_count > 0:

    st.warning(
        f"⚠️ Your dataset contains {missing_count} missing values."
    )

    st.info(
        "Please handle missing values before prediction."
    )

    with st.expander("View Missing Values"):

        st.dataframe(
            prediction_df.isna().sum()
        )

    st.stop()


# =========================================================
# CHECK DATA TYPES
# =========================================================

try:

    prediction_df = prediction_df.apply(
        pd.to_numeric
    )

except Exception:

    st.error(
        "❌ Some columns contain non-numeric values."
    )

    st.write(
        "The uploaded data must contain numeric values "
        "for the model input features."
    )

    st.stop()


# =========================================================
# PREDICT BUTTON
# =========================================================

st.divider()

st.subheader("🤖 Generate Predictions")

if st.button(
    "🚀 Predict Placement",
    use_container_width=True
):

    try:

        predictions = model.predict(
            prediction_df
        )

    except Exception as e:

        st.error("❌ Prediction failed.")

        st.error(f"Error: {e}")

        st.stop()


    # =====================================================
    # ADD PREDICTION
    # =====================================================

    result_df = df.copy()

    result_df["Prediction"] = predictions


    # =====================================================
    # CONVERT 0/1 TO TEXT
    # =====================================================

    result_df["Placement Result"] = result_df[
        "Prediction"
    ].map({
        0: "Not Placed",
        1: "Placed"
    })


    # =====================================================
    # SUMMARY
    # =====================================================

    total_students = len(predictions)

    placed_count = int(
        (predictions == 1).sum()
    )

    not_placed_count = int(
        (predictions == 0).sum()
    )


    placement_percentage = (
        placed_count / total_students
    ) * 100


    # =====================================================
    # SUMMARY SECTION
    # =====================================================

    st.divider()

    st.subheader("📊 Prediction Summary")


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "👨‍🎓 Total Students",
            total_students
        )


    with col2:

        st.metric(
            "✅ Predicted Placed",
            placed_count
        )


    with col3:

        st.metric(
            "❌ Predicted Not Placed",
            not_placed_count
        )


    with col4:

        st.metric(
            "📈 Placement Rate",
            f"{placement_percentage:.2f}%"
        )


    # =====================================================
    # RESULT TABLE
    # =====================================================

    st.subheader("📋 Prediction Results")

    st.dataframe(
        result_df,
        use_container_width=True,
        height=450
    )


    # =====================================================
    # SIMPLE CHART
    # =====================================================

    st.subheader("📈 Placement Distribution")


    chart_data = pd.DataFrame({
        "Status": [
            "Placed",
            "Not Placed"
        ],
        "Students": [
            placed_count,
            not_placed_count
        ]
    })


    st.bar_chart(
        chart_data.set_index("Status")
    )


    # =====================================================
    # DOWNLOAD RESULTS
    # =====================================================

    st.subheader("📥 Download Results")


    csv_data = result_df.to_csv(
        index=False
    )


    st.download_button(
        label="⬇️ Download Prediction CSV",
        data=csv_data,
        file_name="student_placement_predictions.csv",
        mime="text/csv",
        use_container_width=True
    )


    # =====================================================
    # SUCCESS MESSAGE
    # =====================================================

    st.success(
        "🎉 Prediction completed successfully!"
    )