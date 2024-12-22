import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load dataset
@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

# Title and description
st.title("Traffic Accident Analysis Dashboard")
st.markdown("This dashboard presents insights from data preparation, exploration, supervised learning, and unsupervised learning.")

# File uploader for dataset
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = load_data(uploaded_file)

    # Handle categorical features
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))  # Convert to string to handle NaN
        label_encoders[col] = le

    # Load the saved RandomForest model
    model = joblib.load('rf_model.pkl')

        # Tabs for navigation
    tabs = st.tabs(["Data Overview", "Exploration", "Supervised Learning", "Unsupervised Learning"])

    ### Tab 1: Data Overview ###
    with tabs[0]:
        st.header("Data Overview")
        
        # Show dataset structure
        st.write("Dataset Overview:")
        st.dataframe(df.head())
        
        st.write("Summary Statistics:")
        st.write(df.describe())

    ## Tab 2: Exploration ###
    with tabs[1]:
        st.header("Data Exploration")

        # Correlation heatmap
        st.subheader("Correlation Heatmap")
        numeric_data = df.select_dtypes(include=['float64', 'int64'])
        corr = numeric_data.corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        # Distribution of numerical features
        st.subheader("Distribution of Numerical Features")
        for col in numeric_data.columns:
            fig, ax = plt.subplots()
            sns.histplot(numeric_data[col], bins=30, kde=True, ax=ax)
            ax.set_title(f'Distribution of {col}')
            st.pyplot(fig)

        # Boxplot for numerical features by Accident Severity
        if 'Accident' in df.columns:
            st.subheader("Boxplot of Numerical Features by Accident Severity")
            for col in numeric_data.columns:
                fig, ax = plt.subplots()
                sns.boxplot(x='Accident', y=col, data=df, ax=ax)
                ax.set_title(f'Boxplot of {col} by Accident Severity')
                st.pyplot(fig)

        # Countplot for categorical features
        st.subheader("Countplot for Categorical Features")
        for col in categorical_cols:
            fig, ax = plt.subplots()
            sns.countplot(x=col, data=df, ax=ax)
            ax.set_title(f'Countplot of {col}')
            st.pyplot(fig)

    ### Tab 3: Supervised Learning ###
    with tabs[2]:
        st.header("Supervised Learning")

        # Select features
        st.subheader("Select Features for Prediction")
        feature_names = df.columns[:-1]  # All columns except the last one (Accident)
        selected_features = st.multiselect("Features", feature_names, default=feature_names)

        # Visualize the data
        st.subheader("Data Visualization")
        fig, ax = plt.subplots()
        sns.countplot(x='Accident', data=df, ax=ax)
        plt.title("Accident Count")
        st.pyplot(fig)

        # Prediction Section
        st.subheader("Make a Prediction")
        st.write("Enter the conditions for prediction:")

        # Create input fields for each feature
        input_data = {}
        for feature in selected_features:
            if feature in categorical_cols:
                # For categorical features, use selectbox
                input_data[feature] = st.selectbox(feature, options=label_encoders[feature].classes_)
            else:
                # For numerical features, use number input
                input_data[feature] = st.number_input(feature)

        # Button to make prediction
        if st.button("Predict"):
            # Convert input data to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Encode categorical features
            for col in categorical_cols:
                if col in input_df.columns:
                    input_df[col] = label_encoders[col].transform(input_df[col].astype(str))
            
            # One-hot encode the input DataFrame to match the model's expected input
            input_df = pd.get_dummies(input_df, columns=categorical_cols)
            
            # Align the input DataFrame with the model's expected input
            # Use model.feature_names_in_ to get the expected feature names
            input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

            # Make prediction
            prediction = model.predict(input_df)
            prediction_proba = model.predict_proba(input_df)

            # Display prediction result
            st.write(f"Prediction: {'Accident' if prediction[0] == 1 else 'No Accident'}")
            st.write(f"Prediction Probability: {prediction_proba[0]}")

    ### Tab 4: Unsupervised Learning ###
    with tabs[3]:
        st.header("Unsupervised Learning")

        # PCA for clustering visualization
        st.subheader("Cluster Visualization with PCA")
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(numeric_data.fillna(0))

        # Apply KMeans clustering to assign cluster labels
        num_clusters = 4  # You can change the number of clusters
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(numeric_data.fillna(0))  # Assign cluster labels to each point

        # Scatter plot for PCA results with cluster labels
        fig, ax = plt.subplots()
        scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], c=cluster_labels, cmap="viridis", alpha=0.6)
        ax.set_title("Cluster Visualization")
        plt.colorbar(scatter, ax=ax, label="Cluster")
        st.pyplot(fig)
        
        st.subheader("Cluster Characteristics")
        st.write("### *Ringkasan Label:*")
        st.write("1. *Cluster 1:* 'Rural Low Traffic Cluster' – Jalan pedesaan dengan lalu lintas rendah dan kecelakaan ringan.")
        st.write("2. *Cluster 2:* 'Highway Evening Cluster' – Jalan raya di malam hari dengan kecepatan tinggi dan pengalaman pengemudi yang tinggi.")
        st.write("3. *Cluster 3:* 'Moderate Traffic Daytime Cluster' – Lalu lintas sedang pada siang hari dengan angka kecelakaan rendah.")
        st.write("4. *Cluster 4:* 'Urban Peak Hour Cluster' – Lalu lintas perkotaan pada jam sibuk dengan risiko kecelakaan rendah.")
else:
    st.write("Please upload a CSV file to proceed.")