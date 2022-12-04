import streamlit as st
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

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
Applikasi Berbasis Web untuk Mengklasifikasi Lokasi **Ecoli Bactery**,
Jadi pada web applikasi ini akan bisa membantu anda untuk mengklasifikasikan sebuah dataset ecoli bactery,
dimana nanti anda akan bisa menginputkan suatu data dari setiap fitur yang ada dalam dataset ecoli bactery ini,
dan sehingga nanti akan dapat menemukan lokasi dari suatu bactery tersebut dan juga anda dapat melihat akurasi
dari beberapa algoritma yang di sediakan dalam website ini, sehingga anda dapat melihat akurasi yang paling terbaik
dari model algoritma tersebut.
### Menu yang disediakan dapat di lihat di bawah ini :
""")

# inisialisasi data 
data = pd.read_csv("ecoli.csv")
tab1, tab2, tab3, tab4 = st.tabs(["Description Data", "Preprocessing Data", "Modeling", "Implementation"])

with tab1:

    st.subheader("Deskripsi Dataset")
    st.write("""Dataset di bawah ini menjelaskan prekursor untuk kumpulan data dan pengembangannya memberikan 
    hasil (tidak divalidasi silang) untuk klasifikasi oleh sistem pakar berbasis aturan dengan versi kumpulan 
    data tersebut. Sistem Pakar untuk Memprediksi Situs Lokalisasi Protein pada Bakteri Gram Negatif, 
    Kenta Nakai & Minoru Kanehisa, PROTEIN: Struktur, Fungsi, dan Genetika 11:95-110, 1991.
    Sistem Pakar Prediksi Situs Lokalisasi Protein Pada Bakteri Negatif Gram”, Kenta Nakai & Minoru Kanehisa, 
    PROTEIN: Struktur, Fungsi, dan Genetika 11:95-110, 1991. Basis Pengetahuan untuk Memprediksi Situs 
    Lokalisasi Protein dalam Sel Eukariotik, Kenta Nakai & Minoru Kanehisa, Genomics 14:897-911, 1992
    """)

    st.write("""
    ### Want to learn more?
    - Dataset [kaggel.com](https://www.kaggle.com/datasets/kannanaikkal/ecoli-uci-dataset)
    - Github Account [github.com](https://github.com/AliGhufron-28/datamaining)
    """)

    st.write(data)
    col = data.shape
    st.write("Jumlah Baris dan Kolom : ", col)
    st.write("""
    ### Data Understanding
    Disini di jelaskan data-data yang ada dalam dataset tersebut seperti penjelasan dari setiap fitur yang
    ada dalam dataset tersebut :
    1. Sequence Name: Nomor akses untuk database SWISS-PROT.
    2. mcg: Metode McGeoch untuk pengenalan urutan sinyal.
    3. gvh: Metode von Heijne untuk pengenalan urutan sinyal.
    4. lip: skor urutan konsensus von Heijne Signal Peptidase II. Atribut biner.
    5. chg: Kehadiran muatan di N-terminus dari lipoprotein yang diprediksi. Atribut biner.
    6. aac: Skor analisis diskriminan untuk kandungan asam amino membran luar dan protein periplasma.
    7. alm1: Skor program prediksi rentang wilayah membran ALOM.
    8. alm2: Skor program ALOM setelah mengecualikan daerah sinyal putatif yang dapat dibagi dari urutan.
    """)

with tab2:
    st.subheader("Data Preprocessing")
    st.subheader("Data Asli")
    data = pd.read_csv("ecoli.csv")
    st.write(data)

    proc = st.checkbox("Normalisasi")
    if proc:
        st.subheader("Fitur Uniqe")
        st.write("Menampilkan isi Fitur yang uniqe atau class dari data")
        unique = data.SITE.value_counts()
        st.write(unique)

        # Min_Max Normalisasi
        from sklearn.preprocessing import MinMaxScaler
        data_for_minmax_scaler=pd.DataFrame(data, columns = ["MCG","GVH","LIP","CHG","AAC","ALM1","ALM2"])
        data_for_minmax_scaler.to_numpy()
        scaler = MinMaxScaler()
        data_hasil_minmax_scaler=scaler.fit_transform(data_for_minmax_scaler)

        st.subheader("Hasil Normalisasi Min_Max")
        data_hasil_minmax_scaler = pd.DataFrame(data_hasil_minmax_scaler,columns = ["MCG","GVH","LIP","CHG","AAC","ALM1","ALM2"])
        st.write(data_hasil_minmax_scaler)

        st.subheader("Drop Data")
        data_drop_column_for_minmaxscaler=data.drop(["MCG","GVH","LIP","CHG","AAC","ALM1","ALM2"], axis=1)
        st.write(data_drop_column_for_minmaxscaler)

        st.subheader("Gabung Data")
        df_new = pd.concat([data_drop_column_for_minmaxscaler,data_hasil_minmax_scaler], axis=1)
        st.write(df_new)

        st.subheader("Drop fitur SITE")
        df_drop_site = df_new.drop(['SITE'], axis=1)
        st.write(df_drop_site)

        st.subheader("Menggabungkan Fitur SITE supaya rapi")
        st.write("Menampilkan dan menyimpan fitur SITE agar bisa di gabungkan dengan benar ke data yang sudah ternormalisasi")
        df_new_site = pd.DataFrame(data, columns = ['SITE'])
        st.write(df_new_site)

        df_without_site = df_new.drop(['SITE'], axis=1)

        st.subheader("Menggabungkan Data")
        st.write("Hasil dari data yang sudah di preprocessing ")
        df_new_data = pd.concat([df_without_site, df_new_site,], axis=1)
        st.write(df_new_data)

with tab3:

    X=data.iloc[:,1:8].values
    y=data.iloc[:,8].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y, random_state=0)

    st.subheader("Pilih Model")
    model1 = st.checkbox("KNN")
    model2 = st.checkbox("Naive Bayes")
    model3 = st.checkbox("Random Forest")
    model4 = st.checkbox("Ensamble Stacking")
    eval = st.checkbox("Grafik Model")

    if model1:
        model = KNeighborsClassifier(n_neighbors=3)
        filename = "KNN.pkl"
        model.fit(X_train,y_train)
        Y_pred = model.predict(X_test)

        score=metrics.accuracy_score(y_test,Y_pred)
        loaded_model = pickle.load(open(filename, 'rb'))
        st.write("Hasil Akurasi Algoritma KNN : ",score)
    if model2:
        model = GaussianNB()
        filename = "gaussianNB.pkl"

        model.fit(X_train,y_train)
        Y_pred = model.predict(X_test)

        score=metrics.accuracy_score(y_test,Y_pred)
        loaded_model = pickle.load(open(filename, 'rb'))
        st.write("Hasil Akurasi Algoritma Naive Bayes GaussianNB : ",score)
    if model3:
        model = RandomForestClassifier(n_estimators = 100)
        filename = "RandomForest.pkl"

        model.fit(X_train,y_train)
        Y_pred = model.predict(X_test)

        score=metrics.accuracy_score(y_test,Y_pred)
        loaded_model = pickle.load(open(filename, 'rb'))
        st.write("Hasil Akurasi Algoritma Random Forest : ",score)
    if model4:
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
        st.write("Hasil Akurasi Algoritma Ensamble Stacking : ",score)

    if eval :
        estimators = [
        ('rf_1', RandomForestClassifier(n_estimators=10, random_state=42)),
        ('knn_1', KNeighborsClassifier(n_neighbors=10))             
        ]

        model1 = KNeighborsClassifier(n_neighbors=3)
        model2 = GaussianNB()
        model3 = RandomForestClassifier(n_estimators = 100) 
        model4 = StackingClassifier(estimators=estimators, final_estimator=GaussianNB())

        # KNN
        model1.fit(X_train, y_train)
        Y_pred1 = model1.predict(X_test) 

        # gaussianNB
        model2.fit(X_train, y_train)
        Y_pred2 = model2.predict(X_test)

        # Random Forest
        model3.fit(X_train, y_train)
        Y_pred3 = model3.predict(X_test) 

        # Ensamble
        model4.fit(X_train, y_train)
        Y_pred4 = model4.predict(X_test) 

        from sklearn import metrics
        score1=metrics.accuracy_score(y_test,Y_pred1)
        score2=metrics.accuracy_score(y_test,Y_pred2)
        score3=metrics.accuracy_score(y_test,Y_pred3)
        score4=metrics.accuracy_score(y_test,Y_pred4)

        import altair as alt
        # st.snow()
        chart_data = pd.DataFrame({
            'Nilai Akurasi' : [score1,score2,score3,score4],
            'Nama Model' : ['KNN','Naive Bayes','Random Forest','Ensamble Stacking']
        })
        bar_chart = alt.Chart(chart_data).mark_bar().encode(
            y = 'Nilai Akurasi',
            x = 'Nama Model'
        )

        st.altair_chart(bar_chart,use_container_width=True)


with tab4:
    # Min_Max Normalisasi
    from sklearn.preprocessing import MinMaxScaler
    data_for_minmax_scaler=pd.DataFrame(data, columns = ["MCG","GVH","LIP","CHG","AAC","ALM1","ALM2"])
    data_for_minmax_scaler.to_numpy()
    scaler = MinMaxScaler()
    data_hasil_minmax_scaler=scaler.fit_transform(data_for_minmax_scaler)

    data_hasil_minmax_scaler = pd.DataFrame(data_hasil_minmax_scaler,columns = ["MCG","GVH","LIP","CHG","AAC","ALM1","ALM2"])

    data_drop_column_for_minmaxscaler=data.drop(["MCG","GVH","LIP","CHG","AAC","ALM1","ALM2"], axis=1)

    df_new = pd.concat([data_drop_column_for_minmaxscaler,data_hasil_minmax_scaler], axis=1)

    df_drop_site = df_new.drop(['SITE'], axis=1)
    
    df_new_site = pd.DataFrame(data, columns = ['SITE'])

    df_without_site = df_new.drop(['SITE'], axis=1)

    df_new_data = pd.concat([df_without_site, df_new_site,], axis=1)
    
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
    # inisialisasi model algoritma yang digunakan
    # algoritma = st.selectbox(
    #     "Pilih Model",
    #     ("KNN", "Naive Bayes", "Random Forest","Stacking")
    # )
    hasil = st.button("cek klasifikasi")

    # Memakai yang sudah di preprocessing
    X=df_new_data.iloc[:,1:8].values
    y=df_new_data.iloc[:,8].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y, random_state=0)

    if hasil:
        # if algoritma == "KNN":
        #     model = KNeighborsClassifier(n_neighbors=3)
        #     filename = "KNN.pkl"
        # elif algoritma == "Naive Bayes":
        #     model = GaussianNB()
        #     filename = "gaussianNB.pkl"
        # elif algoritma == "Random Forest":
        #     model = RandomForestClassifier(n_estimators = 100)
        #     filename = "RandomForest.pkl"
        # else:
        #     estimators = [
        #         ('rf_1', RandomForestClassifier(n_estimators=10, random_state=42)),
        #         ('knn_1', KNeighborsClassifier(n_neighbors=10))             
        #         ]
        #     model = StackingClassifier(estimators=estimators, final_estimator=GaussianNB())
        #     filename = "stacking.pkl"
        model = RandomForestClassifier(n_estimators = 100)
        filename = "RandomForest.pkl"

        model.fit(X_train,y_train)
        Y_pred = model.predict(X_test)

        score=metrics.accuracy_score(y_test,Y_pred)
        loaded_model = pickle.load(open(filename, 'rb'))
        
        dataArray = [MCG, GVH,	LIP,	CHG,	AAC,	ALM1,	ALM2]
        pred = loaded_model.predict([dataArray])

        st.success(f"Prediksi Hasil Klasifikasi : {pred[0]}")
        st.write(f"Algoritma yang digunakan adalah = Random Forest Algorithm")
        st.success(f"Hasil Akurasi : {score}")
