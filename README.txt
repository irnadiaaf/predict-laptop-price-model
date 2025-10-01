Cara Pakai (VSCode / Lokal)
1. Buat folder project, simpan app.py dan model_pipeline.joblib di sana.
2. Buka folder itu di VSCode.
3. Buka terminal di VSCode, buat & aktifkan virtual env:

Windows PowerShell:
python -m venv venv
.\venv\Scripts\Activate.ps1

(kalau kena execution policy error, jalankan Set-ExecutionPolicy RemoteSigned -Scope CurrentUser sebagai admin, 
atau aktifkan via cmd: .\venv\Scripts\activate.bat)

Mac/Linux:
python3 -m venv venv
source venv/bin/activate

4. Install dependency:
pip install streamlit pandas numpy scikit-learn joblib

(tambahkan library lain kalau pipeline-mu butuh)

5. Jalankan: 
streamlit run app.py

6. Buka http://localhost:8501 di browser.