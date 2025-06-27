import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Kelulusan Mahasiswa", layout="wide")
st.title("Aplikasi Interaktif untuk Prediksi Kelulusan Mahasiswa Berbasis Web Menggunakan Streamlit")

# Upload file CSV
uploaded_file = st.file_uploader("📂 Upload Dataset Mahasiswa (.csv)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    expected_cols = {"Nama", "NIM", "Kehadiran", "UTS", "UAS", "Tugas"}
    if not expected_cols.issubset(df.columns):
        st.error("❌ Kolom tidak sesuai. Pastikan: Nama, NIM, Kehadiran, UTS, UAS, Tugas")
        st.stop()

    st.subheader("🗂️ Preview Dataset")
    st.dataframe(df.head())

    st.markdown("### ℹ️ Ringkasan Dataset")
    st.write(f"- Jumlah Mahasiswa: **{df.shape[0]}**")
    st.write(f"- Jumlah Kolom: **{df.shape[1]}**")
    st.markdown("#### Statistik Deskriptif:")
    st.dataframe(df.describe())

    # Buat label kelulusan otomatis
    df["Lulus"] = np.where(
        (df["Kehadiran"] >= 75) &
        (df["Tugas"] >= 70) &
        (df["UTS"] >= 70) &
        (df["UAS"] >= 75), 1, 0
    )

    fitur = ["Kehadiran", "UTS", "UAS", "Tugas"]
    X = df[fitur]
    y = df["Lulus"]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Model Dictionary
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "Logistic Regression": LogisticRegression(),
        "Naive Bayes": GaussianNB(),
        "K-NN": KNeighborsClassifier(n_neighbors=5)
    }

    st.sidebar.header("⚙️ Pengaturan Algoritma")
    selected_model_name = st.sidebar.selectbox("Pilih Algoritma", list(models.keys()))

    model = models[selected_model_name]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.subheader("🧪 Prediksi Mahasiswa Baru")
    with st.form("form_predict"):
        kehadiran = st.slider("Kehadiran (%)", 0, 100, 75)
        uts = st.slider("Nilai UTS", 0, 100, 70)
        uas = st.slider("Nilai UAS", 0, 100, 75)
        tugas = st.slider("Nilai Tugas", 0, 100, 75)
        threshold = st.slider("Threshold Probabilitas Kelulusan (%)", 0, 100, 50)
        submit = st.form_submit_button("🔮 Prediksi")

    if submit:
        input_data = np.array([[kehadiran, uts, uas, tugas]])
        input_scaled = scaler.transform(input_data)

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_scaled)[0][1]
        else:
            decision = model.decision_function(input_scaled)
            proba = 1 / (1 + np.exp(-decision))
            proba = proba[0] if isinstance(proba, (list, np.ndarray)) else proba

        hasil = "Lulus" if proba >= threshold / 100 else "Tidak Lulus"

        st.subheader("📈 Hasil Prediksi")
        st.metric(f"{selected_model_name}", hasil)
        st.caption(f"Probabilitas: {proba:.2f}")

        st.subheader("🧠 Penjelasan & Rekomendasi")
        if hasil == "Tidak Lulus":
            st.error("💡 Saran: Tingkatkan kehadiran dan nilai tugas/UTS/UAS.")
        else:
            st.success("👍 Tetap pertahankan prestasi.")

    st.subheader("📊 Evaluasi Model")
    st.write(f"### 📌 Evaluasi: {selected_model_name}")
    st.write(f"Akurasi: **{acc:.2f}**")

    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="YlOrBr", ax=ax)
    ax.set_title(f"Confusion Matrix - {selected_model_name}")
    st.pyplot(fig)

    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

else:
    st.info("Silakan upload file CSV dengan kolom: Nama, NIM, Kehadiran, UTS, UAS, Tugas.")
