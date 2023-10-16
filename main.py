import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
st.title("Network Traffic Analysis and Anomaly Detection")

def preprocess(file):
    file = pd.get_dummies(file, columns=['protocol_type', 'service', 'flag'], prefix="", prefix_sep="")
    X_train_multi = np.array(file)
    X_train_multi = np.reshape(X_train_multi, (X_train_multi.shape[0], X_train_multi.shape[1], 1))
    return X_train_multi

def process_pred_data(y_pred_test):
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
    return binary_predictions

def load_tf_model():
    model = tf.keras.models.load_model('MultiClassification.h5')
    return model

def vizz(binary_predictions):
    fig, ax = plt.subplots()
    ax.pie(binary_predictions.value_counts(),labels=binary_predictions.value_counts().index,shadow=True, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

def main():
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv","txt"])
    if uploaded_file is not None:
        file = pd.read_csv(uploaded_file)
        file.drop('Unnamed: 0',axis=1,inplace=True)
        preprocessed_data = preprocess(file)
        model = load_tf_model()
        preprocessed_data = preprocessed_data.astype(np.float32)
        y_pred_test = model.predict(preprocessed_data, batch_size=100)
        binary_predictions = process_pred_data(y_pred_test)
        vizz(binary_predictions)

if __name__ == "__main__":
    main()
