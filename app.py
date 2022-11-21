import streamlit as st
import pandas as pd
import numpy as np 

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import neighbors, datasets
import pickle

st.title("Applikasi Web Datamining")
st.write("""
# Web Apps - Klasifikasi Ecoli Bactery Dataset
Applikasi Berbasis Web untuk Mengklasifikasi Jenis **Ecoli Bactery**
""")

st.subheader("Deskripsi Dataset")
st.write("""Dataset di bawah ini menjelaskan prekursor untuk kumpulan data dan pengembangannya memberikan 
hasil (tidak divalidasi silang) untuk klasifikasi oleh sistem pakar berbasis aturan dengan versi kumpulan 
data tersebut. Sistem Pakar untuk Memprediksi Situs Lokalisasi Protein pada Bakteri Gram Negatif, 
Kenta Nakai & Minoru Kanehisa, PROTEIN: Struktur, Fungsi, dan Genetika 11:95-110, 1991.
Sistem Pakar Prediksi Situs Lokalisasi Protein Pada Bakteri Negatif Gram”, Kenta Nakai & Minoru Kanehisa, 
PROTEIN: Struktur, Fungsi, dan Genetika 11:95-110, 1991. Basis Pengetahuan untuk Memprediksi Situs 
Lokalisasi Protein dalam Sel Eukariotik, Kenta Nakai & Minoru Kanehisa, Genomics 14:897-911, 1992
### Want to learn more?
- Dataset [kaggel.com](https://www.kaggle.com/datasets/kannanaikkal/ecoli-uci-dataset)
- Github Account [github.com](https://github.com/AliGhufron-28/datamaining)
""")

algoritma = st.sidebar.selectbox(
    "Pilih Model",
    ("KNN", "Naive Bayes", "Random Forest","Stacking")
)

data = pd.read_csv("ecoli.csv")
tab1, tab2 = st.tabs(["view dataset", "Tab2"])

with tab1:
    st.write(data)
    col = data.shape
    st.write("Jumlah Baris dan Kolom : ", col)
with tab2:
    st.write("this tab 2")


st.subheader("Parameter Inputan")

# SEQUENCE_NAME = st.selectbox("Masukkan SEQUENCE_NAME : ", ("AAT_ECOLI","ACEA_ECOLI","ACEK_ECOLI","ACKA_ECOLI",
# "ADI_ECOLI","ALKH_ECOLI","AMPD_ECOLI","AMY2_ECOLI","APT_ECOLI","ARAC_ECOLI"))
MCG = st.number_input("Masukkan MCG :")
GVH = st.number_input("Masukkan GVH :")
LIP = st.number_input("Masukkan LIP :")
CHG = st.number_input("Masukkan CHG :")
AAC = st.number_input("Masukkan AAC:")
ALM1 = st.number_input("Masukkan ALM1 :")
ALM2 = st.number_input("Masukkan ALM2 :")
hasil = st.button("cek klasifikasi")

X=data.iloc[:,1:8].values
y=data.iloc[:,8].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y, random_state=0)

if hasil:
    if algoritma == "KNN":
        model = KNeighborsClassifier(n_neighbors=3)
        filename = "KNN.pkl"
    elif algoritma == "Naive Bayes":
        model = GaussianNB()
        filename = "gaussianNB.pkl"
    elif algoritma == "Random Forest":
        model = RandomForestClassifier(n_estimators = 100)
        filename = "RandomForest.pkl"
    else:
        estimators = [
            ('rf_1', RandomForestClassifier(n_estimators=10, random_state=42)),
            ('knn_1', KNeighborsClassifier(n_neighbors=10))             
            ]
        model = StackingClassifier(estimators=estimators, final_estimator=GaussianNB())
        filename = "stacking.pkl"
    model.fit(X_train,y_train)
    Y_pred = model.predict(X_test)

    score=metrics.accuracy_score(y_test,Y_pred)
    loaded_model = pickle.load(open(filename, 'rb'))
    
    dataArray = [MCG, GVH,	LIP,	CHG,	AAC,	ALM1,	ALM2]
    pred = loaded_model.predict([dataArray])

    st.success(f"Prediksi Hasil Klasifikasi : {pred[0]}")
    st.write(f"Algoritma yang digunakan = {algoritma}")
    st.success(f"Hasil Akurasi : {score}")
    