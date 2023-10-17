import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Network Traffic Analysis and Anomaly Detection")

def preprocess(file):
    file = pd.get_dummies(file, columns=['protocol_type', 'service', 'flag'], prefix="", prefix_sep="")
    X_train_multi = np.array(file)
    X_train_multi = np.reshape(X_train_multi, (X_train_multi.shape[0], X_train_multi.shape[1], 1))
    return X_train_multi

def process_pred_data(y_pred_test, file):
    threshold = 0.5
    binary_predictions_2 = (y_pred_test > threshold).astype(int)
    binary_predictions = pd.DataFrame(binary_predictions_2)
    binary_predictions_array = binary_predictions.values
    original_categories = ["DOS", "Malware", "Port Scanning", "Unauthorized Access", "Normal"]
    category_indices = np.argmax(binary_predictions_array, axis=1)
    category_indices_as_integers = category_indices.astype(int)
    categories = [original_categories[i] for i in category_indices_as_integers]
    binary_predictions['categories'] = categories
    binary_predictions = binary_predictions.categories
    file['categories'] = categories
    return binary_predictions

def load_tf_model():
    model = tf.keras.models.load_model('MultiClassification.h5')
    return model

def vizz(binary_predictions, file):
    explode = (0.1, 0.1, 0.1, 0.1)
    explode2=(0.1,0.2,0.1)
    st.write('Anomaly Analysis Report (PIE-GRAPH)')
    fig, ax1 = plt.subplots()
    ax1.pie(binary_predictions.value_counts(), labels=binary_predictions.value_counts().index, autopct="%1.1f%%", startangle=90, explode=explode)
    st.pyplot(fig)

    unique_categories = file["categories"].unique()
    for category in unique_categories:
        category_data = file[file["categories"] == category]
        st.subheader(f"{category} Analysis")

        # Visualization 1: Pie chart for 'protocol' distribution
        st.write(f"### {category} - Protocol Distribution")
        protocol_counts = category_data["protocol_type"].value_counts()
        fig1, ax1 = plt.subplots()
        ax1.pie(protocol_counts, labels=protocol_counts.index, autopct="%1.1f%%", startangle=90,explode=explode2)
        st.pyplot(fig1)
        st.write("This pie chart shows the distribution of network protocols in the dataset.")

        # Visualization 2: Scatter plot for 'src_bytes' and 'dst_bytes'
        st.write(f"### {category} - Scatter Plot")
        fig2, ax2 = plt.subplots()
        sns.scatterplot(data=category_data, x="src_bytes", y="dst_bytes", hue="flag", ax=ax2)
        ax2.set_xlabel("src_bytes")
        ax2.set_ylabel("dst_bytes")
        st.pyplot(fig2)
        st.write("This scatter plot visualizes the relationship between 'src_bytes' and 'dst_bytes' with different flags.")

def main():
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv", "txt"])
    if uploaded_file is not None:
        file = pd.read_csv(uploaded_file)
        file.drop('Unnamed: 0', axis=1, inplace=True)
        preprocessed_data = preprocess(file)
        model = load_tf_model()
        preprocessed_data = preprocessed_data.astype(np.float32)
        y_pred_test = model.predict(preprocessed_data, batch_size=100)
        binary_predictions = process_pred_data(y_pred_test, file)
        vizz(binary_predictions, file)
        st.write('Here is the list of all the networks marked with the category of its attack')
        st.write(file)

if __name__ == "__main__":
    main()
