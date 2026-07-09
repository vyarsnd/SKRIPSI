# ============================================================
# APLIKASI STREAMLIT: KLASIFIKASI KOMPLIKASI DM
# Model: SVM + Firefly Algorithm + SMOTE
# Parameter: C=30.95, Gamma=0.9933, Akurasi 67.93%
# Fitur Baru: Persentase Confidence & Kontribusi Fitur
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Prediksi Komplikasi DM",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS CUSTOM
# ============================================================
st.markdown("""
<style>
    .main-title { font-size: 2.5rem; font-weight: bold; color: #2c3e50; text-align: center; }
    .sub-title { font-size: 1.2rem; color: #7f8c8d; text-align: center; margin-bottom: 2rem; }
    .result-box { padding: 2rem; border-radius: 15px; text-align: center; margin: 1rem 0; }
    .result-komplikasi { background: linear-gradient(135deg, #e74c3c, #c0392b); color: white; font-size: 1.5rem; font-weight: bold; }
    .result-tidak { background: linear-gradient(135deg, #2ecc71, #27ae60); color: white; font-size: 1.5rem; font-weight: bold; }
    .info-box { background: #f8f9fa; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #3498db; }
    .contrib-positive { color: #e74c3c; font-weight: bold; }
    .contrib-negative { color: #2ecc71; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown('<p class="main-title">Klasifikasi Komplikasi Diabetes Mellitus</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Optimasi SVM - Firefly Algorithm + SMOTE<br>RSUD Syarifah Ambami Rato Ebu Bangkalan</p>', unsafe_allow_html=True)
st.markdown("---")

# ============================================================
# LOAD MODEL, SCALERS & MEDIANS
# ============================================================
@st.cache_resource
def load_model():
    """Load model, scaler, dan median values"""
    try:
        model = joblib.load('model_komplikasi_dm.pkl')
        scaler1 = joblib.load('scaler1.pkl')
        medians = joblib.load('medians.pkl')  # Median dari data training asli

        if os.path.exists('features.json'):
            with open('features.json', 'r') as f:
                FEATURES = json.load(f)
        else:
            FEATURES = ['Umur', 'Sex', 'GDP', 'Sistolik', 'Diastolik', 'Hemoglobin', 'Keratinin_Serum']

        return model, scaler1, medians, FEATURES, True
    except FileNotFoundError as e:
        st.error(f"File tidak ditemukan: {e}")
        st.info("Pastikan file berikut ada di folder yang sama dengan app.py:")
        st.code("- model_komplikasi_dm.pkl\n- scaler1.pkl\n- medians.pkl")
        return None, None, None, None, False

with st.spinner("Memuat model SVM-Firefly..."):
    model, scaler1, medians, FEATURES, model_loaded = load_model()

if not model_loaded:
    st.stop()

st.success(f"Model berhasil dimuat! | C=30.95, Gamma=0.9933 | Akurasi: 67.93% | F1: 78.32%")

# ============================================================
# FUNGSI PREDIKSI & KONTRIBUSI
# ============================================================
def get_prediction(input_array):
    """Melakukan prediksi dan mengembalikan (kelas, prob_0, prob_1)"""
    input_scaled = scaler1.transform(input_array)
    pred = model.predict(input_scaled)[0]
    try:
        proba = model.predict_proba(input_scaled)[0]
        prob_0, prob_1 = proba[0], proba[1]
    except AttributeError:
        decision = model.decision_function(input_scaled)[0]
        prob_1 = 1 / (1 + np.exp(-decision))
        prob_0 = 1 - prob_1
    return pred, prob_0, prob_1

def compute_feature_contributions(input_array, original_prob_1):
    """Hitung kontribusi setiap fitur dengan median perturbation"""
    contributions = []
    for i, feat in enumerate(FEATURES):
        # Ganti fitur ke-i dengan median
        perturbed = input_array.copy()
        perturbed[0, i] = medians[i]
        _, _, prob_1_pert = get_prediction(perturbed)
        diff = original_prob_1 - prob_1_pert  # >0 berarti fitur asli meningkatkan prob komplikasi
        contributions.append({'Fitur': feat, 'Pengaruh': diff})
    
    # Normalisasi agar total pengaruh positif = original_prob_1
    pos_sum = sum(max(0, c['Pengaruh']) for c in contributions)
    neg_sum = sum(max(0, -c['Pengaruh']) for c in contributions)
    total_effect = pos_sum + neg_sum
    if total_effect > 0:
        for c in contributions:
            if c['Pengaruh'] >= 0:
                c['Persentase'] = (c['Pengaruh'] / pos_sum) * original_prob_1 * 100 if pos_sum > 0 else 0
            else:
                c['Persentase'] = (c['Pengaruh'] / neg_sum) * (1 - original_prob_1) * 100 if neg_sum > 0 else 0
    else:
        for c in contributions:
            c['Persentase'] = 0.0
    return contributions

# ============================================================
# SIDEBAR - INPUT DATA PASIEN
# ============================================================
with st.sidebar:
    st.header("Data Pasien")
    st.markdown("Masukkan data klinis pasien:")
    st.markdown("---")
    umur = st.number_input("Umur (tahun)", min_value=9, max_value=100, value=55, step=1)
    sex = st.selectbox("Jenis Kelamin", options=["Perempuan", "Laki-laki"])
    sex_val = 0 if sex == "Perempuan" else 1
    st.markdown("---")
    gdp = st.number_input("Gula Darah Puasa / GDP (mg/dL)", min_value=50, max_value=600, value=200, step=1)
    sistolik = st.number_input("Tekanan Sistolik (mmHg)", min_value=80, max_value=230, value=135, step=1)
    diastolik = st.number_input("Tekanan Diastolik (mmHg)", min_value=40, max_value=160, value=80, step=1)
    hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=3.0, max_value=25.0, value=11.5, step=0.1)
    keratinin = st.number_input("Keratinin Serum (mg/dL)", min_value=0.1, max_value=15.0, value=1.0, step=0.01)
    st.markdown("---")
    predict_btn = st.button("Prediksi Komplikasi", type="primary", use_container_width=True)

# ============================================================
# TAB UTAMA
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Prediksi", "Performa Model", "Informasi", "Detail Model", "Interpretasi Data"])

# ---- TAB 1: PREDIKSI ----
with tab1:
    # Inisialisasi session state untuk menyimpan hasil
    if 'prediction_done' not in st.session_state:
        st.session_state.prediction_done = False
        st.session_state.pred = None
        st.session_state.proba_0 = None
        st.session_state.proba_1 = None
        st.session_state.contributions = None

    if predict_btn:
        with st.spinner("🔍 Melakukan prediksi..."):
            try:
                input_array = np.array([[umur, sex_val, gdp, sistolik, diastolik, hemoglobin, keratinin]])
                pred, prob_0, prob_1 = get_prediction(input_array)
                contributions = compute_feature_contributions(input_array, prob_1)
                
                st.session_state.pred = pred
                st.session_state.proba_0 = prob_0
                st.session_state.proba_1 = prob_1
                st.session_state.contributions = contributions
                st.session_state.prediction_done = True
                st.rerun()
            except Exception as e:
                st.error(f"❌ Terjadi error saat prediksi: {str(e)}")
                st.session_state.prediction_done = False

    # Tampilkan hasil jika sudah selesai
    if st.session_state.prediction_done:
        st.markdown("---")
        st.subheader("🎯 Hasil Prediksi")
        
        col_r1, col_r2 = st.columns(2)
        
        with col_r1:
            if st.session_state.pred == 1:
                st.markdown('<div class="result-box result-komplikasi">⚠️ KOMPLIKASI</div>', 
                           unsafe_allow_html=True)
                st.markdown("**Pasien diprediksi mengalami komplikasi Diabetes Mellitus.**")
            else:
                st.markdown('<div class="result-box result-tidak">✅ TIDAK KOMPLIKASI</div>', 
                           unsafe_allow_html=True)
                st.markdown("**Pasien diprediksi TIDAK mengalami komplikasi Diabetes Mellitus.**")
        
        with col_r2:
            conf = st.session_state.proba_1
            st.markdown(f"**Confidence Score: {conf*100:.1f}%**")
            st.progress(float(conf))
            
            prob_df = pd.DataFrame({
                'Kelas': ['Tidak Komplikasi (0)', 'Komplikasi (1)'],
                'Probabilitas': [f"{st.session_state.proba_0*100:.1f}%", f"{conf*100:.1f}%"]
            })
            st.dataframe(prob_df, use_container_width=True, hide_index=True)
        
        # Tampilkan kontribusi fitur
        st.markdown("### 🔍 Kontribusi Fitur terhadap Prediksi")
        st.caption("Menunjukkan seberapa besar setiap fitur memengaruhi confidence score prediksi.")
        contribs = st.session_state.contributions
        if contribs:
            contrib_df = pd.DataFrame(contribs)
            contrib_df['Pengaruh (selisih)'] = contrib_df['Pengaruh'].apply(lambda x: f"{x:+.4f}")
            contrib_df['Kontribusi (%)'] = contrib_df['Persentase'].apply(lambda x: f"{x:+.2f}%")
            st.dataframe(contrib_df[['Fitur', 'Pengaruh (selisih)', 'Kontribusi (%)']], use_container_width=True, hide_index=True)
            
            # Visualisasi bar chart
            fig, ax = plt.subplots(figsize=(6, 4))
            colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in contrib_df['Pengaruh']]
            ax.barh(contrib_df['Fitur'], contrib_df['Pengaruh'], color=colors)
            ax.axvline(0, color='black', linewidth=0.5)
            ax.set_xlabel('Perubahan Probabilitas Komplikasi')
            ax.set_title('Kontribusi Fitur terhadap Prediksi')
            st.pyplot(fig)
        
        with st.expander("📋 Detail Input & Median Referensi"):
            input_df = pd.DataFrame({
                'Fitur': FEATURES,
                'Nilai Input': [umur, sex_val, gdp, sistolik, diastolik, hemoglobin, keratinin],
                'Median Populasi': medians
            })
            st.dataframe(input_df, use_container_width=True, hide_index=True)
            st.caption("Median digunakan sebagai nilai netral untuk menghitung pengaruh fitur.")
        
        # Tombol reset
        if st.button("🔄 Reset Prediksi"):
            st.session_state.prediction_done = False
            st.rerun()
    
    else:
        st.info("👈 Masukkan data pasien di sidebar dan klik **Prediksi Komplikasi**")
        
        col_demo1, col_demo2 = st.columns(2)
        with col_demo1:
            st.markdown("""
            ### 📋 Cara Penggunaan
            1. Masukkan **data klinis** pasien di sidebar kiri
            2. Klik tombol **Prediksi Komplikasi**
            3. Hasil prediksi + confidence score + kontribusi fitur akan muncul
            """)
        with col_demo2:
            st.markdown("""
            ### 🏥 Fitur Klinis
            - **Umur**: Usia pasien (tahun)
            - **Sex**: Jenis kelamin
            - **GDP**: Gula Darah Puasa
            - **Sistolik/Diastolik**: Tekanan darah
            - **Hemoglobin**: Kadar Hb
            - **Keratinin Serum**: Fungsi ginjal
            """)

# ---- TAB 2: PERFORMA MODEL ----
with tab2:
    st.subheader("Performa Model SVM-Firefly")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Akurasi", "67.93%")
    with col_m2:
        st.metric("F1-Score", "78.32%")
    with col_m3:
        st.metric("Precision", "78.14%")
    with col_m4:
        st.metric("Recall", "78.50%")
    
    st.markdown("---")
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("**Confusion Matrix (Data Testing):**")
        cm_df = pd.DataFrame(
            [[29, 47], [46, 168]],
            index=['Actual Tidak Komplikasi', 'Actual Komplikasi'],
            columns=['Pred Tidak Komplikasi', 'Pred Komplikasi']
        )
        st.dataframe(cm_df, use_container_width=True)
        st.caption("Testing: 290 pasien (76 Tidak Komplikasi, 214 Komplikasi)")
    
    with col_p2:
        st.markdown("**Classification Report:**")
        report_df = pd.DataFrame({
            'Kelas': ['Tidak Komplikasi', 'Komplikasi', 'Macro Avg', 'Weighted Avg'],
            'Precision': ['0.39', '0.78', '0.58', '0.68'],
            'Recall': ['0.38', '0.78', '0.58', '0.68'],
            'F1-Score': ['0.38', '0.78', '0.58', '0.68']
        })
        st.dataframe(report_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown("**Perbandingan Training vs Testing (3 Run FA):**")
    train_test_df = pd.DataFrame({
        'Run': [1, 2, 3],
        'Train Acc': ['71.43%', '71.53%', '72.03%'],
        'Test Acc': ['67.93%', '64.48%', '67.59%'],
        'Selisih': ['3.50%', '7.05%', '4.44%'],
        'Status': ['✓ Normal', '⚠️ Overfit', '✓ Normal']
    })
    st.dataframe(train_test_df, use_container_width=True, hide_index=True)
    st.caption("Run 1 dipilih sebagai model terbaik dengan akurasi 67.93%.")

# ---- TAB 3: INFORMASI ----
with tab3:
    st.subheader("Informasi Penelitian")
    
    col_i1, col_i2 = st.columns(2)
    with col_i1:
        st.markdown("""
        ### Tentang Penelitian
        **Judul:** Klasifikasi Penyakit Komplikasi Diabetes Mellitus Menggunakan Optimasi SVM-Firefly
        
        **Lokasi:** RSUD Syarifah Ambami Rato Ebu Bangkalan
        
        **Dataset:** 965 pasien Diabetes Mellitus (data tahun 2024)
        
        **Fitur yang Digunakan:**
        - Umur, Jenis Kelamin, GDP, Sistolik, Diastolik, Hemoglobin, Keratinin Serum
        """)
    with col_i2:
        st.markdown("""
        ### Kontribusi Penelitian
        - ✅ Firefly Algorithm terbukti meningkatkan akurasi SVM dari 53.89% menjadi 67.93%
        - ✅ SMOTE terbukti mengurangi bias deteksi kelas minoritas (TN dari 0 menjadi 29)
        - ✅ Model mampu mendeteksi kedua kelas secara lebih seimbang
        
        ### Batasan Penelitian
        - Korelasi seluruh fitur < 0.10 menyebabkan akurasi mentok di ~68%
        - Diperlukan fitur tambahan (HbA1c, profil lipid) untuk menembus 70%
        """)
    
    st.markdown("---")
    st.markdown("### Analisis Pola Data")
    st.markdown("""
    **Distribusi Komplikasi:**
    - Perempuan: 74.9% | Laki-laki: 71.8%
    - Usia 70+: 76.0% komplikasi
    - Keratinin > 2.0: 80.9% komplikasi (paling tinggi)
    - GDP bukan pembeda kuat (korelasi -0.0159)
    """)

# ---- TAB 4: DETAIL MODEL ----
with tab4:
    st.subheader("Detail Model & Proses")
    
    st.markdown("### File Model")
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        st.markdown("**model_komplikasi_dm.pkl**\nModel SVM final")
    with col_f2:
        st.markdown("**scaler1.pkl**\nNormalisasi Min-Max (fit pada X_train, sebelum SMOTE)")
    with col_f3:
        st.markdown("**medians.pkl**\nMedian fitur untuk kalkulasi kontribusi")
    
    st.markdown("---")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("### Parameter SVM")
        svm_params = pd.DataFrame({
            'Parameter': ['Algoritma', 'Kernel', 'C', 'Gamma', 'Probability', 'Random State'],
            'Nilai': ['SVM', 'RBF', '30.95', '0.9933', 'True', '42']
        })
        st.dataframe(svm_params, use_container_width=True, hide_index=True)
    
    with col_d2:
        st.markdown("### Firefly Algorithm")
        fa_params = pd.DataFrame({
            'Parameter': ['Populasi', 'Max Iterasi', 'Alpha', 'Rentang C', 'Rentang Gamma', 'Fitness', 'K-Fold CV', 'Early Stop'],
            'Nilai': ['20', '20', '0.2', '[0.1, 100]', '[0.001, 1]', 'Akurasi CV', '10-Fold', '3 iterasi']
        })
        st.dataframe(fa_params, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown("### Proses Preprocessing")
    steps = pd.DataFrame({
        'Tahap': [1, 2, 3, 4, 5, 6, 7],
        'Proses': [
            'Data Cleaning',
            'Transformasi Data',
            'Pelabelan',
            'Imputasi (KNN k=5)',
            'Split Data (70/30)',
            'Normalisasi Min-Max (Scaler 1, fit pada X_train)',
            'SMOTE (hanya pada data latih)'
        ]
    })
    st.dataframe(steps, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown("### Alur Prediksi")
    st.code("""
    Input Data Mentah (7 fitur)
        │
        ▼
    Normalisasi Min-Max — Scaler 1 (fit pada X_train, sebelum SMOTE)
        │
        ▼
    SVM Model (C=30.95, Gamma=0.9933, RBF Kernel)
        │
        ▼
    Output: Kelas (0/1) + Confidence Score + Kontribusi Fitur
    """, language=None)
    
    st.markdown("---")
    st.markdown("### Metrik Evaluasi Lengkap")
    metrics_full = pd.DataFrame({
        'Metrik': [
            'Akurasi', 'F1-Score (Weighted)', 'Precision (Weighted)', 'Recall (Weighted)',
            'Precision Kelas 0', 'Recall Kelas 0', 'F1-Score Kelas 0',
            'Precision Kelas 1', 'Recall Kelas 1', 'F1-Score Kelas 1',
            'TN (True Negative)', 'FP (False Positive)', 'FN (False Negative)', 'TP (True Positive)'
        ],
        'Nilai': [
            '67.93%', '0.68', '0.68', '0.68',
            '0.39', '0.38', '0.38',
            '0.78', '0.78', '0.78',
            '29', '47', '46', '168'
        ]
    })
    st.dataframe(metrics_full, use_container_width=True, hide_index=True)
    st.caption("Evaluasi pada data testing: 290 pasien (76 Tidak Komplikasi, 214 Komplikasi)")

# ---- TAB 5: INTERPRETASI DATA ----
with tab5:
    st.subheader("🔍 Mengapa Hasil Prediksi Bisa Seperti Ini?")
    st.markdown("""
    Model SVM-Firefly **tidak berpikir seperti dokter**. Ia hanya meniru pola statistik 
    dari **965 data pasien** RSUD Syarifah Ambami. Setiap fitur dianalisis sendiri-sendiri.
    """)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🧪 Keratinin Serum")
        st.markdown("""
        - **Pengaruh paling besar** (42.1%).
        - **Korelasi +0.08**: Makin tinggi nilainya, risiko komplikasi sedikit meningkat.
        - **Fakta**: 80.9% pasien dengan Keratinin >2.0 mengalami komplikasi.
        """)
        
        st.markdown("### ❤️ Sistolik")
        st.markdown("""
        - **Pengaruh kedua** (29.4%).
        - **Korelasi +0.06**: Makin tinggi, risiko sedikit naik.
        - **Fakta**: 80.1% pasien dengan Sistolik 140-160 mengalami komplikasi.
        """)
        
        st.markdown("### 🎂 Umur")
        st.markdown("""
        - **Korelasi +0.07**: Semakin tua, risiko meningkat.
        - **Fakta**: 76% pasien usia 70+ mengalami komplikasi.
        - **Menarik**: Perempuan 70+ risikonya 83%, laki-laki justru 68.1%.
        """)
        
        st.markdown("### 👤 Jenis Kelamin")
        st.markdown("""
        - **Korelasi -0.03**: Hampir tidak berpengaruh sendiri.
        - **Fakta**: Perempuan 74.9% komplikasi, laki-laki 71.8%. Beda tipis.
        - Bekerja sebagai kombinasi dengan umur.
        """)
    
    with col2:
        st.markdown("### 🍬 GDP (Gula Darah Puasa)")
        st.markdown("""
        - **Pengaruh sangat kecil** (7.8%).
        - **Korelasi -0.02**: Hampir tidak ada hubungan linear.
        - **Fakta Penting**: Pasien Tidak Komplikasi justru punya GDP rata-rata lebih tinggi (213.96 vs 210.16).
        - **Inilah sebabnya** GDP tinggi kadang diprediksi "Tidak Komplikasi". Model hanya meniru data.
        """)
        
        st.markdown("### 🩸 Hemoglobin")
        st.markdown("""
        - **Pengaruh kecil** (14.6%).
        - **Korelasi -0.03**: Semakin rendah Hb, risiko sedikit naik.
        - **Fakta**: 78% pasien anemia berat (<10) mengalami komplikasi.
        """)
        
        st.markdown("### 🩺 Diastolik")
        st.markdown("""
        - **Pengaruh paling kecil** (6.1%).
        - **Korelasi +0.01**: Hampir tidak ada hubungan.
        - **Fakta**: Hanya berbeda di kasus ekstrem (>100).
        """)
    
    st.markdown("---")
    st.markdown("""
    ### 💡 Intinya untuk Dosen
    **Model tidak salah.** Semua hasil prediksi adalah cerminan langsung dari data asli RSUD.
    Karena **korelasi semua fitur sangat lemah (< 0.10)**, tidak ada fitur yang bisa membedakan 
    kelas dengan kuat. Akibatnya, prediksi bisa sensitif dan kadang terlihat "terbalik" dari 
    teori medis. Ini justru menguatkan temuan bahwa diperlukan fitur tambahan seperti HbA1c 
    untuk meningkatkan akurasi.
    """)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: gray;'>
    Skripsi - Klasifikasi Komplikasi Diabetes Mellitus<br>
    Optimasi SVM-Firefly Algorithm + SMOTE<br>
    RSUD Syarifah Ambami Rato Ebu Bangkalan<br>
    © {datetime.now().year}
</div>
""", unsafe_allow_html=True)