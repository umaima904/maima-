
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load data
st.title("📊 Customer Response Prediction App")
uploaded_file = st.file_uploader("Upload the Customer Personality Analysis CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Cleaning
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], dayfirst=True)
    current_date = pd.to_datetime('today')
    df['Customer_Since_Days'] = (current_date - df['Dt_Customer']).dt.days
    df.drop('Dt_Customer', axis=1, inplace=True)

    # Feature Engineering
    df['Age'] = current_date.year - df['Year_Birth']
    df['Family_Size'] = df['Kidhome'] + df['Teenhome'] + 1
    spend_cols = ['MntWines', 'MntFruits', 'MntMeatProducts',
                  'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
    df['Total_Spending'] = df[spend_cols].sum(axis=1)

    # Encode categorical
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'Response' in cat_cols:
        cat_cols.remove('Response')
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # EDA
    st.subheader("📈 EDA Visuals")

    st.bar_chart(df.groupby('Age')['Response'].mean())

    st.bar_chart(df.groupby('Marital_Status')['Response'].mean())

    st.bar_chart(df.groupby('Education')['Response'].mean())

    # Modeling
    X = df.drop(['ID', 'Response'], axis=1)
    y = df['Response']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model_choice = st.selectbox("Choose Classifier", ['Logistic Regression', 'Random Forest'])

    if model_choice == 'Logistic Regression':
        model = LogisticRegression(max_iter=1000)
    else:
        model = RandomForestClassifier(n_estimators=100)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    st.write(f"### Accuracy: {acc:.2f}")
    st.write("### Confusion Matrix:")
    st.dataframe(cm)

    st.write("### Classification Report:")
    st.text(classification_report(y_test, y_pred))

    if model_choice == 'Random Forest':
        importances = model.feature_importances_
        feat_importances = pd.Series(importances, index=X.columns)
        st.bar_chart(feat_importances.sort_values(ascending=False)[:10])

else:
    st.info("👈 Please upload a dataset to get started.")
