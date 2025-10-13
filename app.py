import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from pathlib import Path

st.set_page_config(page_title="Laptop Price Predictor", layout="centered")

# -------------------------
# Helper: load model
# -------------------------
@st.cache_resource
def load_model(path="model_pipeline.joblib"):
    path = Path(path)
    if not path.exists():
        st.error(f"Model file not found: {path.resolve()}. Pastikan model_pipeline.joblib ada di folder yang sama dengan app.py")
        return None
    return joblib.load(path)

model = load_model("model_pipeline.joblib")

# -------------------------
# Metadata (kolom & opsi)
# -------------------------
company_options = [
    "Acer","Apple","Asus","Chuwi","Dell","Fujitsu","Google","HP","Huawei","LG",
    "Lenovo","Mediacom","Microsoft","MSI","Razer","Samsung","Toshiba","Vero","Xiaomi"
]
cpu_company_options = ["AMD","Intel","Samsung"]
cpu_type_options = [
    "AMD A-Series","AMD E-Series","Atom","Celeron","Core i3","Core i5","Core i7",
    "Pentium","Ryzen","Other"
]
gpu_company_options = ["AMD","ARM","Intel","Nvidia"]
storage_type_options = ["Flash","Flash+HDD","HDD","Hybrid","SSD","SSD+HDD","SSD+Hybrid"]
memory_unit_options = ['MB','GB','TB']
opsys_options = ['Mac','No OS','Windows','Linux/Other']
resolution_options = ["4K","FullHD","HD","QHD","Other"]

# -------------------------
# GPU Performance mapping
# -------------------------
def map_gpu_to_tier(gpu_name):
    s = str(gpu_name).lower()
    if any(kw in s for kw in ['gtx','geforce','quad','firepro','radeon rx','rtx']):
        return 2
    if any(mid_kw in s for mid_kw in ['mx','940','930','920','540','530','520','r7','r5','mali']):
        return 1
    if any(low_kw in s for low_kw in ['hd graphics','uhd graphics','iris','intel hd','intel uhd']):
        return 0
    return 1

# -------------------------
# UI
# -------------------------
st.title("💻 Laptop Price Predictor")
st.markdown("Masukkan spesifikasi laptop, lalu klik **Predict** untuk melihat estimasi harga.")

with st.form("input_form"):
    st.header("Spesifikasi Laptop")
    brand = st.selectbox("Brand (Company)", company_options)
    cpu_brand = st.selectbox("Brand CPU", cpu_company_options)
    cpu_type = st.selectbox("Tipe CPU", cpu_type_options)
    cpu_freq = st.number_input("Frekuensi CPU (GHz)", min_value=0.0, max_value=5.0, value=2.5, step=0.1)
    gpu_brand = st.selectbox("Brand GPU", gpu_company_options)
    gpu_perf = st.text_input("GPU Performance", "GeForce GTX 1050")
    storage_type = st.selectbox("Tipe Storage", storage_type_options)
    memory_value = st.number_input("Memory (angka)", min_value=0.0, value=256.0)
    memory_unit = st.selectbox("Satuan Memory", memory_unit_options, index=1)
    ram_gb = st.number_input("RAM (GB)", min_value=1, max_value=128, value=8)
    opsys = st.selectbox("Sistem Operasi", opsys_options)
    weight = st.number_input("Weight (kg)", min_value=0.0, max_value=9.0, value=2.0)
    inches = st.number_input("Ukuran Layar (inci)", min_value=0.0, max_value=100.0, value=15.6)
    resolution = st.selectbox("Resolusi", resolution_options)
    ppi = st.number_input("PPI", min_value=0.0, max_value=1000.0, value=141.0)
    ips_choice = st.selectbox("IPS", ["Yes","No"], index=1)
    touchscreen_choice = st.selectbox("Touchscreen", ["Yes","No"], index=1)

    submitted = st.form_submit_button("Predict")

# -------------------------
# When submitted
# -------------------------
if submitted:
    if model is None:
        st.stop()

    # Konversi memory ke TB
    if memory_unit == "TB":
        mem_tb = memory_value
    elif memory_unit == "GB":
        mem_tb = memory_value / 1024.0
    elif memory_unit == "MB":
        mem_tb = memory_value / (1024.0**2)
    else:
        mem_tb = memory_value / 1024.0

    gpu_tier = map_gpu_to_tier(gpu_perf)
    ips = 1 if ips_choice == "Yes" else 0
    touchscreen = 1 if touchscreen_choice == "Yes" else 0

    # Buat DataFrame
    data_input = pd.DataFrame({
        'Inches':[inches],
        'CPU_Frequency (GHz)':[cpu_freq],
        'RAM (GB)':[ram_gb],
        'Weight (kg)':[weight],
        'Touchscreen':[touchscreen],
        'IPS':[ips],
        'PPI':[ppi],
        'Memory_TB':[mem_tb],
        'GPU_Performance':[gpu_tier],
        'Company':[brand],
        'CPU_Company':[cpu_brand],
        'CPU_Type':[cpu_type],
        'GPU_Company':[gpu_brand],
        'Resolution_Type':[resolution],
        'Storage_Type':[storage_type],
        'OpsSys_Grouped':[opsys]
    })

    # One-hot encode
    onehot_cols = ["Company","CPU_Company","CPU_Type","GPU_Company","Resolution_Type","Storage_Type","OpsSys_Grouped"]
    data_input = pd.get_dummies(data_input, columns=onehot_cols, drop_first=True)

    # Load kolom model dan reindex
    try:
        model_columns = pickle.load(open('model_columns.pkl', 'rb'))

        # Pastikan hanya kolom yang dikenal oleh model
        data_input = data_input.reindex(columns=model_columns, fill_value=0)
        data_input = data_input.loc[:, model_columns]  # pastikan urutannya sama persis

    except Exception as e:
        st.error(f"Gagal memuat model_columns.pkl: {e}")
        st.stop()

    st.subheader("Data Input (setelah preprocessing)")
    st.dataframe(data_input.T.rename(columns={0: "Value"}))

    # Predict
    try:
        pred_log = model.predict(data_input)[0]
        pred_euro = np.expm1(pred_log)
        st.success(f"💰 Estimasi Harga: € {pred_euro:,.2f}")
        st.caption(f"(Prediksi log-price: {pred_log:.4f})")
    except Exception as e:
        st.error(f"❌ Terjadi error saat prediksi:\n\n{e}")