import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

sns.set_style("whitegrid")


# Load dataset

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    return df


# Load trained model

@st.cache_resource
def load_model():
    try:
        model = joblib.load("model.pkl")
        return model
    except Exception as e:
        st.warning(f" model.pkl not found or could not be loaded: {e}")
        return None


# Main App

def main():
    st.title("I Scale Assignment — Data Explorer & Model Predictor")
    st.markdown("Interactive app to explore the dataset and use the trained model for predictions.")

    # Load dataset
    df = load_data("Data_Analytics_Task.csv")

    # Load model
    model = load_model()

    
    # Section 1: Data Exploration
    
    st.header("Data Exploration")

    if st.checkbox("Show raw data"):
        st.dataframe(df)

    st.subheader("Dataset Info")
    st.write("Shape:", df.shape)
    st.write("Columns:", list(df.columns))
    st.write(df.describe(include="all"))

    # Visualizations
    st.subheader("Visualizations")
    plot_type = st.selectbox("Select plot type", ["Histogram", "Bar plot", "Scatter plot", "Box plot"])

    if plot_type == "Histogram":
        num_col = st.selectbox("Select numeric column", options=df.select_dtypes(include="number").columns)
        if num_col:
            fig, ax = plt.subplots()
            sns.histplot(df[num_col].dropna(), kde=True, ax=ax)
            st.pyplot(fig)

    elif plot_type == "Bar plot":
        cat_cols = df.select_dtypes(include="object").columns
        if len(cat_cols) > 0:
            cat_col = st.selectbox("Select categorical column", options=cat_cols)
            fig, ax = plt.subplots()
            df[cat_col].value_counts().plot(kind="bar", ax=ax)
            ax.set_xlabel(cat_col)
            st.pyplot(fig)
        else:
            st.warning("No categorical columns available.")

    elif plot_type == "Scatter plot":
        num_cols = df.select_dtypes(include="number").columns
        if len(num_cols) >= 2:
            x_col = st.selectbox("X axis", options=num_cols, index=0)
            y_col = st.selectbox("Y axis", options=num_cols, index=1)
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Need at least two numeric columns for scatter plot")

    elif plot_type == "Box plot":
        num_cols = df.select_dtypes(include="number").columns
        if len(num_cols) > 0:
            num_col = st.selectbox("Numeric column", options=num_cols)
            fig, ax = plt.subplots()
            sns.boxplot(x=df[num_col], ax=ax)
            st.pyplot(fig)
        else:
            st.warning("No numeric columns available.")

    
    # Section 2: Model Prediction

    st.header("Model Prediction")
    st.markdown("Enter feature values to get prediction from trained model.")

    feature_cols = df.select_dtypes(include="number").columns.tolist()

    user_input = []
    for col in feature_cols:
        val = st.number_input(
            f"Enter value for {col}",
            float(df[col].min()),
            float(df[col].max()),
            float(df[col].mean())
        )
        user_input.append(val)

    if st.button("Predict"):
        if model is not None:
            input_df = pd.DataFrame([user_input], columns=feature_cols)
            try:
                prediction = model.predict(input_df)
                st.success(f" Model Prediction: {prediction[0]}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
        else:
            st.error("Model file not found. Prediction is unavailable.")


# Run app

if __name__ == "__main__":
    main()