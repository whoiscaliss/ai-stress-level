"""
Streamlit app: Stress & Student Sleep Analysis (UAS - compliant)
File: streamlit_stress_sleep_app_uas.py
Run: streamlit run streamlit_stress_sleep_app_uas.py

This version adapts the original app to satisfy UAS requirements:
 - Topic aligned to SDG (Health & Well-being)
 - Includes >= 6 ML algorithms (from scikit-learn)
 - Uses common DS libraries (pandas, numpy, matplotlib, seaborn, plotly, scikit-learn, joblib)
 - Adds a methodology (CRISP-DM) section in the UI
 - Evaluates each model with multiple metrics (MSE, RMSE, MAE, R2)
 - Integrates all models, plots, datasets into Streamlit GUI
 - Provides export (models + metrics + dataset snapshot) as a ZIP to upload to Google Drive

Notes:
 - This file assumes scikit-learn is available. It avoids optional non-standard libs (like xgboost)
 - For reproducibility, seed is configurable in the sidebar

"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import io
import zipfile
import json
from datetime import datetime

st.set_page_config(layout="wide", page_title="Stress & Sleep — UAS Compliant")

# --------------------------- Helpers ---------------------------
@st.cache_data
def load_csv_from_path(path):
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        return None

@st.cache_data
def clean_column_names(df):
    df = df.copy()
    df.columns = [c.strip().replace(' ', '_') for c in df.columns]
    return df

def numeric_overview(df):
    desc = df.describe().T
    desc['missing'] = df.isna().sum()
    return desc

def quick_plots_numeric(df, cols, prefix=''):
    for c in cols:
        fig, ax = plt.subplots(figsize=(6,3))
        sns.histplot(df[c].dropna(), kde=True, ax=ax)
        ax.set_title(f"{prefix}{c} distribution")
        st.pyplot(fig)

        fig2, ax2 = plt.subplots(figsize=(6,2))
        sns.boxplot(x=df[c].dropna(), ax=ax2)
        ax2.set_title(f"{prefix}{c} boxplot")
        st.pyplot(fig2)

def evaluate_regression(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

def zip_results(models_dict, metrics_df, dataset_snapshot):
    """Return bytes of ZIP archive containing models (joblib), metrics (json/csv), and dataset snapshot."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as z:
        # models
        for name, model in models_dict.items():
            model_bytes = io.BytesIO()
            joblib.dump(model, model_bytes)
            model_bytes.seek(0)
            z.writestr(f"models/{name}.joblib", model_bytes.read())
        # metrics
        z.writestr('metrics/metrics.json', metrics_df.to_json(orient='records', indent=2))
        z.writestr('metrics/metrics.csv', metrics_df.to_csv(index=False))
        # dataset snapshot
        snap_bytes = dataset_snapshot.to_csv(index=False).encode('utf-8')
        z.writestr('data/dataset_snapshot.csv', snap_bytes)
        # manifest
        manifest = {
            'created_at': datetime.utcnow().isoformat() + 'Z',
            'notes': 'Streamlit UAS project export: models, metrics, dataset snapshot'
        }
        z.writestr('manifest.json', json.dumps(manifest, indent=2))
    buffer.seek(0)
    return buffer.getvalue()

# --------------------------- Load Data (UI with Color Accents) ---------------------------
# Warna tema untuk membedakan dataset
STRESS_COLOR = "#ff6b6b"  # merah coral
SLEEP_COLOR = "#4dabf7"   # biru soft

st.markdown(f"""
<style>
    .stress-box {{
        padding: 10px 15px;
        border-radius: 10px;
        background-color: {STRESS_COLOR}33;
        border-left: 6px solid {STRESS_COLOR};
        margin-bottom: 10px;
    }}
    .sleep-box {{
        padding: 10px 15px;
        border-radius: 10px;
        background-color: {SLEEP_COLOR}33;
        border-left: 6px solid {SLEEP_COLOR};
        margin-bottom: 10px;
    }}
    .section-title {{
        font-size: 22px;
        font-weight: 600;
        margin-top: 20px;
    }}
</style>
""", unsafe_allow_html=True)
st.title("📊 Stress & Student Sleep — UAS Compliant Explorer")
st.write("Aplikasi ini dimodifikasi agar memenuhi ketentuan UAS: topik SDG (kesehatan), minimal 6 ML methods, metodologi, evaluasi, dan dokumentasi." )

with st.sidebar.expander("Data input & settings"):
    st.write("App akan mencoba memuat file default jika tersedia di /mnt/data/. Kamu juga dapat upload file sendiri.")
    stress_default = 'StressLevelDataset.csv'
    sleep_default = 'student_sleep_patterns.csv'

    stress_df_try = load_csv_from_path(stress_default)
    sleep_df_try = load_csv_from_path(sleep_default)

    uploaded_stress = st.file_uploader("Upload StressLevelDataset.csv ", type=['csv'], key='up1')
    uploaded_sleep = st.file_uploader("Upload student_sleep_patterns.csv ", type=['csv'], key='up2')

    sample_n = st.number_input('Contoh sampling (0 = tidak sampling)', min_value=0, value=500, step=100)
    random_state = st.number_input("Random state (seed)", min_value=0, max_value=9999, value=42)

# determine final dataframes
if uploaded_stress is not None:
    try:
        stress_df = pd.read_csv(uploaded_stress)
    except Exception:
        st.error("Gagal membaca uploaded Stress CSV. Pastikan valid CSV.")
        stress_df = pd.DataFrame()
elif stress_df_try is not None:
    stress_df = stress_df_try
else:
    stress_df = pd.DataFrame()

if uploaded_sleep is not None:
    try:
        sleep_df = pd.read_csv(uploaded_sleep)
    except Exception:
        st.error("Gagal membaca uploaded Sleep CSV. Pastikan valid CSV.")
        sleep_df = pd.DataFrame()
elif sleep_df_try is not None:
    sleep_df = sleep_df_try
else:
    sleep_df = pd.DataFrame()

# Clean column names
if not stress_df.empty:
    stress_df = clean_column_names(stress_df)
if not sleep_df.empty:
    sleep_df = clean_column_names(sleep_df)

# --------------------------- Pages / Modes ---------------------------
mode = st.sidebar.radio("Pilih mode:", [
    "Analisis Terpisah",
    "Analisis Kolom Terpilih (Modeling)",
    "Dashboard Multi-halaman",
    "Metodologi & Dokumentasi"
])

# --------------------------- Metodologi Page ---------------------------
if mode == "Metodologi & Dokumentasi":
    st.header("🧭 Metodologi — CRISP-DM (disesuaikan untuk ML)")
    st.markdown("""
    **Langkah-langkah (dicantumkan sebagai bagian dari persyaratan UAS):

    1. Business / Problem Understanding — Definisikan tujuan: analisis hubungan stress & pola tidur (SDG: Good Health & Well-being)
    2. Data Understanding — EDA: ringkasan statistik, visualisasi, pengecekan missing
    3. Data Preparation — Cleaning, feature selection, imputation
    4. Modeling — Terapkan berbagai algoritma ML (minimal 6) dan lakukan tuning sederhana
    5. Evaluation — Gunakan metrik yang sesuai (MSE, RMSE, MAE, R2)
    6. Deployment / Reporting — Integrasi ke Streamlit dan ekspor artefak (models, metrics, dataset snapshot)

    Silakan simpan semua artefak (models, metrics, dataset, laporan) ke Google Drive sebelum submit.
    """)
    st.subheader('Checklist yang harus dipenuhi')
    st.markdown('- Topik terkait SDG: **Health** — menggunakan stress/sleep data')
    st.markdown('- Minimal **6** model ML (disediakan di mode Modeling)')
    st.markdown('- Semua output diekspor sebagai ZIP (models + metrics + dataset snapshot) agar bisa diunggah ke Google Drive')

# --------------------------- Mode 1: Analisis Terpisah ---------------------------
if mode == "Analisis Terpisah":
    st.header("🔎 Analisis Terpisah — Stress Dataset & Sleep Dataset")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("StressLevelDataset")
        if stress_df.empty:
            st.info("Stress dataset tidak tersedia. Upload atau letakkan file di /mnt/data/StressLevelDataset.csv")
        else:
            df_display = stress_df.head(100) if sample_n == 0 else stress_df.sample(min(len(stress_df), sample_n), random_state=random_state)
            st.write(f"Shape: {stress_df.shape}")
            st.dataframe(df_display)
            st.markdown("**Ringkasan numerik**")
            st.dataframe(numeric_overview(stress_df))

            num_cols = stress_df.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                st.markdown("**Distribusi numerik (hist & box)**")
                quick_plots_numeric(stress_df, num_cols[:6], prefix='Stress: ')

                st.markdown("**Korelasi (top 20 numeric columns)**")
                corr = stress_df[num_cols].corr().fillna(0)
                fig = px.imshow(corr, title='Stress dataset correlation matrix')
                st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("😴 **Student Sleep Patterns**")
        if sleep_df.empty:
            st.info("Sleep dataset tidak tersedia. Upload atau letakkan file di /mnt/data/student_sleep_patterns.csv")
        else:
            df_display = sleep_df.head(100) if sample_n == 0 else sleep_df.sample(min(len(sleep_df), sample_n), random_state=random_state)
            st.write(f"Shape: {sleep_df.shape}")
            st.dataframe(df_display)
            st.markdown("**Ringkasan numerik**")
            st.dataframe(numeric_overview(sleep_df))

            num_cols2 = sleep_df.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols2:
                st.markdown("**Distribusi numerik (hist & box)**")
                quick_plots_numeric(sleep_df, num_cols2[:6], prefix='Sleep: ')

                st.markdown("**Korelasi (top numeric columns)**")
                corr2 = sleep_df[num_cols2].corr().fillna(0)
                fig2 = px.imshow(corr2, title='Sleep dataset correlation matrix')
                st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.write("Untuk modeling, gunakan mode 'Analisis Kolom Terpilih (Modeling)'.")

# --------------------------- Mode 2: Analisis Kolom Terpilih (Modeling) ---------------------------
elif mode == "Analisis Kolom Terpilih (Modeling)":
    st.header("🧪 Analisis Kolom Terpilih — Modeling (>=6 algorithms)")
    st.write("Pilih dataset, fitur (numeric), dan target. Aplikasi akan melatih minimal 6 model ML dan menampilkan metrik serta feature importances.")

    dataset_choice = st.selectbox("Pilih dataset untuk modeling:", ["StressLevelDataset", "Student_Sleep_Patterns"]) 

    if dataset_choice == 'StressLevelDataset' and not stress_df.empty:
        df = stress_df.copy()
    elif dataset_choice == 'Student_Sleep_Patterns' and not sleep_df.empty:
        df = sleep_df.copy()
    else:
        st.warning("Dataset yang dipilih tidak tersedia. Upload data atau pilih dataset lain.")
        df = pd.DataFrame()

    if not df.empty:
        st.write(f"Shape: {df.shape}")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.error('Tidak ada kolom numeric dalam dataset yang dipilih — modeling membutuhkan numeric features')
        else:
            features = st.multiselect("Pilih fitur (numeric saja):", numeric_cols, default=numeric_cols[:6], key='feat')
            target = st.selectbox("Pilih target (numeric):", numeric_cols, index=0)
            test_size = st.slider("Test size (fraction)", 0.1, 0.5, 0.2)

            if st.button("Jalankan modeling"):
                if not features:
                    st.error("Pilih minimal 1 fitur numeric untuk modeling.")
                else:
                    X = df[features].copy()
                    y = df[target].copy()

                    # handle missing values
                    X = X.fillna(X.median())
                    y = y.fillna(y.median())

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

                    # Define at least 6 models
                    models = {
                        'LinearRegression': LinearRegression(),
                        'Lasso': Lasso(random_state=random_state),
                        'Ridge': Ridge(random_state=random_state),
                        'DecisionTree': DecisionTreeRegressor(random_state=random_state),
                        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=random_state),
                        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=random_state),
                        'KNeighbors': KNeighborsRegressor(),
                        'SVR': SVR()
                    }

                    results = []
                    trained_models = {}

                    st.subheader('Training & Evaluation')
                    for name, model in models.items():
                        with st.spinner(f'Training {name} ...'):
                            try:
                                model.fit(X_train, y_train)
                                y_pred = model.predict(X_test)
                                metrics = evaluate_regression(y_test, y_pred)
                                metrics['model'] = name
                                results.append(metrics)
                                trained_models[name] = model

                                # show metrics
                                cols = st.columns(4)
                                cols[0].metric(f"{name} MSE", f"{metrics['MSE']:.4f}")
                                cols[1].metric(f"{name} RMSE", f"{metrics['RMSE']:.4f}")
                                cols[2].metric(f"{name} MAE", f"{metrics['MAE']:.4f}")
                                cols[3].metric(f"{name} R2", f"{metrics['R2']:.4f}")

                                # If model supports feature_importances_
                                if hasattr(model, 'feature_importances_'):
                                    fi = pd.DataFrame({'feature': features, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
                                    st.write(f"Feature importances — {name}")
                                    st.dataframe(fi)

                                # scatter plot actual vs predicted
                                fig = px.scatter(x=y_test, y=y_pred, labels={'x':'Actual','y':'Predicted'}, title=f'Actual vs Predicted — {name}')
                                fig.add_shape(type='line', x0=y_test.min(), x1=y_test.max(), y0=y_test.min(), y1=y_test.max(), line=dict(dash='dash'))
                                st.plotly_chart(fig, use_container_width=True)

                            except Exception as e:
                                st.error(f'Error training {name}: {e}')

                    metrics_df = pd.DataFrame(results).sort_values('R2', ascending=False)
                    st.markdown('**Ringkasan metrik (diurutkan berdasarkan R2)**')
                    st.dataframe(metrics_df)

                    # export ZIP with models + metrics + snapshot
                    st.markdown('---')
                    st.subheader('Export artefak (models + metrics + dataset snapshot)')
                    snapshot = pd.concat([X_test, y_test.reset_index(drop=True)], axis=1)
                    zip_bytes = zip_results(trained_models, metrics_df, snapshot)
                    st.download_button('Download ZIP (untuk upload ke Google Drive)', data=zip_bytes, file_name='uas_project_artifacts.zip')

# --------------------------- Mode 3: Dashboard Multi-halaman ---------------------------
elif mode == "Dashboard Multi-halaman":
    st.header("📚 Dashboard Multi-halaman — Compare & Explore")
    tabs = st.tabs(["Overview", "Compare Distributions", "Correlation & Heatmaps", "Time-of-day Sleep Patterns"])

    with tabs[0]:
        st.subheader("Overview")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Stress dataset quick stats**")
            if stress_df.empty:
                st.info("Stress dataset kosong")
            else:
                st.metric("Rows (stress)", stress_df.shape[0])
                num_cols = stress_df.select_dtypes(include=[np.number]).columns.tolist()
                st.dataframe(stress_df[num_cols].median().to_frame('median'))

        with c2:
            st.markdown("**Sleep dataset quick stats**")
            if sleep_df.empty:
                st.info("Sleep dataset kosong")
            else:
                st.metric("Rows (sleep)", sleep_df.shape[0])
                num_cols2 = sleep_df.select_dtypes(include=[np.number]).columns.tolist()
                st.dataframe(sleep_df[num_cols2].median().to_frame('median'))

    with tabs[1]:
        st.subheader("Compare Distributions")
        numeric1 = stress_df.select_dtypes(include=[np.number]).columns.tolist() if not stress_df.empty else []
        numeric2 = sleep_df.select_dtypes(include=[np.number]).columns.tolist() if not sleep_df.empty else []

        sel1 = st.selectbox("Pilih kolom stress untuk dibandingkan", options=numeric1, key='d1') if numeric1 else None
        sel2 = st.selectbox("Pilih kolom sleep untuk dibandingkan", options=numeric2, key='d2') if numeric2 else None

        if sel1:
            fig = px.histogram(stress_df, x=sel1, nbins=30, title=f'Distribution: {sel1}')
            st.plotly_chart(fig, use_container_width=True)
        if sel2:
            fig2 = px.histogram(sleep_df, x=sel2, nbins=30, title=f'Distribution: {sel2}')
            st.plotly_chart(fig2, use_container_width=True)

    with tabs[2]:
        st.subheader("Correlation & Heatmaps")
        if not stress_df.empty:
            st.markdown("**Stress correlation (top numeric)**")
            nc = stress_df.select_dtypes(include=[np.number]).columns.tolist()
            ccols = nc[:20]
            fig = px.imshow(stress_df[ccols].corr().fillna(0), title='Stress correlation')
            st.plotly_chart(fig, use_container_width=True)
        if not sleep_df.empty:
            st.markdown("**Sleep correlation (top numeric)**")
            nc2 = sleep_df.select_dtypes(include=[np.number]).columns.tolist()
            ccols2 = nc2[:20]
            fig2 = px.imshow(sleep_df[ccols2].corr().fillna(0), title='Sleep correlation')
            st.plotly_chart(fig2, use_container_width=True)

    with tabs[3]:
        st.subheader("Time-of-day Sleep Patterns")
        if not sleep_df.empty and 'Sleep_Duration' in sleep_df.columns:
            fig = px.histogram(sleep_df, x='Sleep_Duration', nbins=30, title='Sleep Duration distribution')
            st.plotly_chart(fig, use_container_width=True)

        day_cols = [c for c in sleep_df.columns if 'Weekday' in c or 'Weekend' in c]
        if day_cols:
            st.markdown("**Weekday vs Weekend sleep times**")
            melted = sleep_df[day_cols].melt(var_name='type', value_name='time')
            fig = px.violin(melted, x='time', y='type', box=True, points='all', title='Weekday vs Weekend sleep times')
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.write("Dashboard ini memberikan ringkasan multi-halaman. Pergi ke mode Modeling untuk membuat model, atau Metodologi & Dokumentasi untuk melihat kerangka kerja yang diikuti.")

# --------------------------- Footer / Notes ---------------------------
st.sidebar.markdown("---")
st.sidebar.info("App created to explore two student datasets. Save as a single file and run with Streamlit.")
st.sidebar.write("Tips: Jika dataset besar, gunakan sample saat eksplorasi. Simpan artefak (ZIP) dan upload ke Google Drive sebagai bukti submit UAS.")

st.write("\n---\nCreated with ❤️ — by Group 2 🐍🐍 (Steven Tang, Lily, Venessya Calista)")
(apptest.py)