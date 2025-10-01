# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
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

# --- GPU performance list (gunakan list yang kamu kirim) ---
gpu_perf_options = [
 'FirePro W4190M', 'FirePro W5130M', 'FirePro W6150M',
 'GeForce 150MX', 'GeForce 920', 'GeForce 920M', 'GeForce 920MX',
 'GeForce 930M', 'GeForce 930MX', 'GeForce 940M', 'GeForce 940MX',
 'GeForce GT 940MX', 'GeForce GTX 1050', 'GeForce GTX 1050 Ti',
 'GeForce GTX 1050M', 'GeForce GTX 1050Ti', 'GeForce GTX 1060',
 'GeForce GTX 1070', 'GeForce GTX 1070M', 'GeForce GTX 1080',
 'GeForce GTX 930MX', 'GeForce GTX 940M', 'GeForce GTX 940MX',
 'GeForce GTX 950M', 'GeForce GTX 960', 'GeForce GTX 960M',
 'GeForce GTX 965M', 'GeForce GTX 970M', 'GeForce GTX 980',
 'GeForce GTX 980 SLI', 'GeForce GTX 980M', 'GeForce GTX1050 Ti',
 'GeForce GTX1060', 'GeForce GTX1070', 'GeForce GTX1080',
 'GeForce MX130', 'GeForce MX150', 'Graphics 620', 'GTX 980 SLI',
 'HD Graphics', 'HD Graphics 400', 'HD Graphics 405',
 'HD Graphics 500', 'HD Graphics 505', 'HD Graphics 510',
 'HD Graphics 515', 'HD Graphics 520', 'HD Graphics 530',
 'HD Graphics 5300', 'HD Graphics 540', 'HD Graphics 6000',
 'HD Graphics 615', 'HD Graphics 620', 'HD Graphics 630',
 'Iris Graphics 540', 'Iris Graphics 550', 'Iris Plus Graphics 640',
 'Iris Plus Graphics 650', 'Iris Pro Graphics', 'Mali T860 MP4',
 'Quadro 3000M', 'Quadro M1000M', 'Quadro M1200',
 'Quadro M2000M', 'Quadro M2200', 'Quadro M2200M',
 'Quadro M3000M', 'Quadro M500M', 'Quadro M520M',
 'Quadro M620', 'Quadro M620M', 'R17M-M1-70', 'R4 Graphics',
 'Radeon 520', 'Radeon 530', 'Radeon 540', 'Radeon Pro 455',
 'Radeon Pro 555', 'Radeon Pro 560', 'Radeon R2',
 'Radeon R2 Graphics', 'Radeon R3', 'Radeon R4',
 'Radeon R4 Graphics', 'Radeon R5', 'Radeon R5 430',
 'Radeon R5 520', 'Radeon R5 M315', 'Radeon R5 M330',
 'Radeon R5 M420', 'Radeon R5 M420X', 'Radeon R5 M430',
 'Radeon R7', 'Radeon R7 Graphics', 'Radeon R7 M360',
 'Radeon R7 M365X', 'Radeon R7 M440', 'Radeon R7 M445',
 'Radeon R7 M460', 'Radeon R7 M465', 'Radeon R9 M385',
 'Radeon RX 540', 'Radeon RX 550', 'Radeon RX 560',
 'Radeon RX 580', 'UHD Graphics 620'
]

storage_type_options = ["Flash","Flash+HDD","HDD","Hybrid","SSD","SSD+HDD","SSD+Hybrid"]

memory_unit_options = ['MB','GB','TB']

opsys_options = ['Mac','No OS','Windows','Linux/Other']

resolution_options = ["4K","FullHD","HD","QHD","Other"]

# -------------------------
# GPU Performance mapping to tier (0 low, 1 mid, 2 high)
# (gunakan kata kunci & beberapa daftar heuristic)
# -------------------------
high_gpus = {
    'GTX 1080','GTX1080','GTX 1070','GTX1070','GTX 1060','GTX1060','GTX 1050 Ti','GTX1050 Ti',
    'GTX 1050','GTX 980','GTX980','GTX 970','GTX 965','GTX 960','GTX 980 SLI','GTX 1070M','GTX 1060',
    'Quadro','FirePro','Radeon RX 580','Radeon RX 560','Radeon RX 550'
}
mid_gpus = {
    'MX150','MX130','MX110','940MX','930MX','940M','930M','GT 940MX','GeForce 920','GeForce 920M',
    'GeForce 930MX','GeForce 940MX','Radeon 540','Radeon 530','Radeon 520','Radeon R5','Radeon R4',
    'Radeon R7','Radeon R3','Mali'
}
low_gpus = {
    'HD Graphics','UHD Graphics','Iris','Iris Plus','HD Graphics 6','HD Graphics 5','Intel HD','Intel UHD'
}

def map_gpu_to_tier(gpu_name):
    s = str(gpu_name).lower()
    # quick checks
    for kw in ['gtx','geforce','quad','firepro','radeon rx','radeon','rtx']:
        if kw in s:
            # assume high if contains GTX/Quad/FirePro/RX/RTX and numeric > 1000 or rx
            if 'gtx' in s or 'geforce gtx' in s or 'geforce' in s or 'rtx' in s or 'radeon rx' in s or 'quadro' in s or 'firepro' in s:
                # map many GeForce GTX / RTX to high
                return 2
    # mid checks
    for mid_kw in ['mx','940','930','920','540','530','520','r7','r5','mali','mx']:
        if mid_kw in s:
            return 1
    # low checks
    for low_kw in ['hd graphics','uhd graphics','iris','intel hd','intel uhd','graphics 6','graphics 5']:
        if low_kw in s:
            return 0
    # fallback to mid
    return 1

# -------------------------
# Build UI
# -------------------------
st.title("Laptop Price Predictor (Streamlit Demo)")
st.markdown("Masukkan spesifikasi laptop di form, kemudian klik **Predict**.")

with st.form("input_form"):
    st.header("Spesifikasi Dasar")
    # Display layout flexible for UX: company at top but we'll reorder later into DataFrame order
    brand = st.selectbox("Brand (Company)", company_options)
    cpu_brand = st.selectbox("Brand CPU", cpu_company_options)
    cpu_type = st.selectbox("Tipe CPU", cpu_type_options)
    cpu_freq = st.number_input("Frekuensi CPU (GHz)", min_value=0.0, max_value=5.0, value=2.5, step=0.1, format="%.2f")
    gpu_brand = st.selectbox("Brand GPU", gpu_company_options)
    gpu_perf = st.selectbox("GPU Performance (searchable)", gpu_perf_options)
    storage_type = st.selectbox("Tipe Storage", storage_type_options)
    memory_value = st.number_input("Memory (angka)", min_value=0.0, value=256.0)
    memory_unit = st.selectbox("Satuan Memory", memory_unit_options, index=1)
    ram_gb = st.number_input("RAM (GB)", min_value=1, max_value=128, value=8, step=1)
    opsys = st.selectbox("Sistem Operasi", opsys_options)
    weight = st.number_input("Weight (kg)", min_value=0.0, max_value=9.0, value=2.0, step=0.1, format="%.2f")
    inches = st.number_input("Ukuran Inci", min_value=0.0, max_value=100.0, value=15.6, step=0.1, format="%.1f")
    resolution = st.selectbox("Tipe resolusi", resolution_options)
    ppi = st.number_input("PPI", min_value=0.0, max_value=1000.0, value=141.0, step=1.0)
    ips_choice = st.selectbox("IPS", ["Yes","No"], index=1)
    touchscreen_choice = st.selectbox("Touchscreen", ["Yes","No"], index=1)

    submitted = st.form_submit_button("Predict")

# -------------------------
# When submitted: build DataFrame in exact order and predict
# -------------------------
if submitted:
    if model is None:
        st.stop()

    # Convert memory to TB
    unit = memory_unit
    mem_tb = 0.0
    if unit == "TB":
        mem_tb = float(memory_value)
    elif unit == "GB":
        mem_tb = float(memory_value) / 1024.0
    elif unit == "MB":
        mem_tb = float(memory_value) / (1024.0**2)
    else:
        mem_tb = float(memory_value) / 1024.0

    # Map GPU performance to tier
    gpu_tier = map_gpu_to_tier(gpu_perf)

    # Binary mapping
    ips = 1 if ips_choice == "Yes" else 0
    touchscreen = 1 if touchscreen_choice == "Yes" else 0

    # -------------------------
    # One-hot manual: build lists in same order as your dataset columns
    # -------------------------
    # Company columns (order must match dataset)
    company_cols = [
        "Company_Apple","Company_Asus","Company_Chuwi","Company_Dell","Company_Fujitsu",
        "Company_Google","Company_HP","Company_Huawei","Company_LG","Company_Lenovo",
        "Company_MSI","Company_Mediacom","Company_Microsoft","Company_Razer","Company_Samsung",
        "Company_Toshiba","Company_Vero","Company_Xiaomi"
    ]
    # CPU company
    cpu_company_cols = ["CPU_Company_Intel","CPU_Company_Samsung"]
    # CPU types (order must match dataset)
    cpu_type_cols = [
        "CPU_Type_AMD E-Series","CPU_Type_Atom","CPU_Type_Celeron","CPU_Type_Core i3",
        "CPU_Type_Core i5","CPU_Type_Core i7","CPU_Type_Other","CPU_Type_Pentium","CPU_Type_Ryzen"
    ]
    # GPU company
    gpu_company_cols = ["GPU_Company_ARM","GPU_Company_Intel","GPU_Company_Nvidia"]
    # Resolution type
    resolution_cols = ["Resolution_Type_FullHD","Resolution_Type_HD","Resolution_Type_Other","Resolution_Type_QHD"]
    # Storage type
    storage_cols = [
        "Storage_Type_Flash+HDD","Storage_Type_HDD","Storage_Type_Hybrid","Storage_Type_SSD",
        "Storage_Type_SSD+HDD","Storage_Type_SSD+Hybrid"
    ]
    # OpsSys grouped
    opsys_cols = ["OpsSys_Grouped_Mac","OpsSys_Grouped_No OS","OpsSys_Grouped_Windows"]

    # Build one-hot vectors
    company_vector = [1 if f"Company_{brand}" == col else 0 for col in company_cols]
    cpu_company_vector = [1 if f"CPU_Company_{cpu_brand}" == col else 0 for col in cpu_company_cols]

    # CPU type mapping: some values names in form slightly different from dataset column names:
    # Map user selection to the matching column name
    cpu_type_map = {
        "AMD E-Series": "CPU_Type_AMD E-Series",
        "Atom": "CPU_Type_Atom",
        "Celeron": "CPU_Type_Celeron",
        "Core i3": "CPU_Type_Core i3",
        "Core i5": "CPU_Type_Core i5",
        "Core i7": "CPU_Type_Core i7",
        "Other": "CPU_Type_Other",
        "Pentium": "CPU_Type_Pentium",
        "Ryzen": "CPU_Type_Ryzen",
        "AMD A-Series": "CPU_Type_Other"   # fallback mapping (adjust if you have a dedicated column)
    }
    cpu_type_col_name = cpu_type_map.get(cpu_type, "CPU_Type_Other")
    cpu_type_vector = [1 if col == cpu_type_col_name else 0 for col in cpu_type_cols]

    # GPU company vector
    gpu_company_vector = [1 if f"GPU_Company_{gpu_brand}" == col else 0 for col in gpu_company_cols]

    # Resolution mapping -> map selected resolution to columns list
    res_map = {
        "FullHD": "Resolution_Type_FullHD",
        "HD": "Resolution_Type_HD",
        "Other": "Resolution_Type_Other",
        "QHD": "Resolution_Type_QHD",
        "4K": "Resolution_Type_Other"  # if 4K not present, map to Other
    }
    res_col = res_map.get(resolution, "Resolution_Type_Other")
    resolution_vector = [1 if col == res_col else 0 for col in resolution_cols]

    # Storage mapping
    storage_map = {
        "Flash": "Storage_Type_Flash+HDD",
        "HDD": "Storage_Type_HDD",
        "Hybrid": "Storage_Type_Hybrid",
        "SSD": "Storage_Type_SSD",
        "SSD+HDD": "Storage_Type_SSD+HDD",
        "SSD+Hybrid": "Storage_Type_SSD+Hybrid",
        "Flash+HDD": "Storage_Type_Flash+HDD"
    }
    storage_col = storage_map.get(storage_type, "Storage_Type_SSD")
    storage_vector = [1 if col == storage_col else 0 for col in storage_cols]

    # OpsSys mapping
    ops_map = {
        "Mac": "OpsSys_Grouped_Mac",
        "No OS": "OpsSys_Grouped_No OS",
        "Windows": "OpsSys_Grouped_Windows",
        "Linux/Other": "OpsSys_Grouped_Windows"
    }
    ops_col = ops_map.get(opsys, "OpsSys_Grouped_Windows")
    ops_vector = [1 if col == ops_col else 0 for col in opsys_cols]

    # -------------------------
    # Assemble full feature vector in the precise order requested
    # Order:
    # 0 Inches
    # 1 CPU_Frequency (GHz)
    # 2 RAM (GB)
    # 3 Weight (kg)
    # 4 Touchscreen
    # 5 IPS
    # 6 PPI
    # 7 Memory_TB
    # 8 GPU_Performance
    # 9..26 Company_*
    # 27..28 CPU_Company_*
    # 29..37 CPU_Type_*
    # 38..40 GPU_Company_*
    # 41..44 Resolution_Type_*
    # 45..50 Storage_Type_*
    # 51..53 OpsSys_Grouped_*
    # -------------------------
    feature_values = []
    # numeric block
    feature_values += [
        float(inches),
        float(cpu_freq),
        int(ram_gb),
        float(weight),
        int(touchscreen),
        int(ips),
        float(ppi),
        float(mem_tb),
        int(gpu_tier)
    ]

    # append one-hot blocks in exact order
    feature_values += company_vector
    feature_values += cpu_company_vector
    feature_values += cpu_type_vector
    feature_values += gpu_company_vector
    feature_values += resolution_vector
    feature_values += storage_vector
    feature_values += ops_vector

    # Column names in exact order (without Price_log)
    columns_order = [
        "Inches","CPU_Frequency (GHz)","RAM (GB)","Weight (kg)","Touchscreen","IPS","PPI","Memory_TB","GPU_Performance"
    ] + company_cols + cpu_company_cols + cpu_type_cols + gpu_company_cols + resolution_cols + storage_cols + opsys_cols

    # Final checks
    if len(feature_values) != len(columns_order):
        st.error(f"Internal error: jumlah feature_values ({len(feature_values)}) != jumlah kolom ({len(columns_order)}). Cek mapping one-hot.")
    else:
        input_df = pd.DataFrame([feature_values], columns=columns_order)

        st.subheader("Input Data (sesuai urutan fitur model)")
        st.dataframe(input_df.T.rename(columns={0:"value"}))

        # Predict using pipeline (expects log target)
        try:
            pred_log = model.predict(input_df)[0]
            pred_euro = np.expm1(pred_log)
            st.success(f"Prediksi Harga: € {pred_euro:,.2f}")
            st.write(f"(Prediksi dalam log: {pred_log:.4f})")
        except Exception as e:
            st.exception(f"Error saat prediksi: {e}")