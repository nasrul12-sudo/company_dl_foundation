# generate_readme.py
README_CONTENT = r"""
================================================================================
                    COMPANY DL FOUNDATION
        Modular Deep Learning Foundation for Enterprise Systems
================================================================================

Versi    : 1.0.0
Lisensi  : Internal / Proprietary
Stack    : Python 3.10+, TensorFlow 2.13+, scikit-learn, FastAPI
Author   : Tim Data Science & Engineering
Terakhir diperbarui: 2024

--------------------------------------------------------------------------------
DAFTAR ISI
--------------------------------------------------------------------------------
1.  Ringkasan Proyek
2.  Prinsip Desain
3.  Arsitektur Sistem
4.  Struktur Folder
5.  Instalasi & Persiapan
6.  Cara Menjalankan (Training, Pruning, Serving)
7.  Alur Retraining Saat Data Bertambah
8.  Integrasi ke Sistem Bisnis
9.  Cara Update Kode (Versioning & Rollback)
10. Konvensi Kode & Kontribusi
11. Troubleshooting
12. Roadmap
13. Lisensi & Kontak

================================================================================
1. RINGKASAN PROYEK
================================================================================
Company DL Foundation adalah fondasi Deep Learning modular berbasis
TensorFlow untuk kebutuhan perusahaan (ERP, inventaris, pembelian,
penjualan, dan monitoring operasional).

Prinsip utama:
  - TIDAK membangun satu model besar untuk semua masalah.
  - Membangun fondasi reusable (core) yang bisa dipakai ulang.
  - Menambahkan modul task (regression / classification / time series)
    sesuai kebutuhan bisnis.
  - Siap berkembang dari data tabular (CSV) ke computer vision (gambar)
    tanpa refactor besar.

Kemampuan saat ini:
  - MLP modular untuk regresi & klasifikasi data tabular.
  - Time series (MLP window-based, LSTM, GRU).
  - Live monitoring visualisasi training (ProLiveVisualizer).
  - Pruning (TF Model Optimization) → model ringan untuk edge / mobile.
  - Serving layer (FastAPI) + connector ke sistem bisnis.
  - Siap diperluas ke CNN (klasifikasi gambar kondisi buah).

================================================================================
2. PRINSIP DESAIN
================================================================================
  a. Modular          : setiap komponen terpisah, dapat diganti sendiri.
  b. Config-driven    : semua hyperparameter via YAML, tanpa ubah kode.
  c. Reusable Core    : build_mlp(), build_cnn(), HiddenBlock, head,
                        dipakai lintas task.
  d. Reproducible     : random seed tetap, artifact versioned.
  e. Observable       : live visualization + structured logging.
  f. Deployable       : Keras SavedModel → TFLite / FastAPI / Edge.
  g. Extensible       : tabular → image → multi-modal dengan
                        penambahan modul, bukan rewrite.

================================================================================
3. ARSITEKTUR SISTEM
================================================================================

   ┌──────────────────────────────────────────────────────────────────┐
   │                       SISTEM BISNIS (ERP, dll.)                  │
   └───────────────────────┬──────────────────────────────────────────┘
                           │  REST / queue / scheduled job
                           ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │                 SERVING LAYER (FastAPI + Predictor)              │
   │                 - /predict/weight                                │
   │                 - /predict/condition_from_image                  │
   └───────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │                     TASKS (bisnis-spesifik)                      │
   │   RegressionTask    ClassificationTask    TimeSeriesTask         │
   └───────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │               REUSABLE TENSORFLOW CORE                           │
   │   build_mlp()  |  build_cnn()  |  HiddenBlock  |  Heads          │
   │   Normalization  |  Pruning Wrapper                              │
   └───────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │                   PREPROCESSING PIPELINE                         │
   │   Validators · Encoders · Scalers · Feature Engineering          │
   │   Image Pipeline (tf.data) · Augmentations                       │
   └───────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │                  DATA (raw → processed → artifact)               │
   └──────────────────────────────────────────────────────────────────┘

   ┌──────────────────────────────────────────────────────────────────┐
   │   TRAINING LOOP                                                  │
   │   Trainer · Callbacks (EarlyStopping, Checkpoint) ·              │
   │   ProLiveVisualizer (live matplotlib) · Evaluator                │
   └──────────────────────────────────────────────────────────────────┘

Alur data ringkas:
  raw data → preprocessing → tensor input → core model (MLP/CNN)
           → task head → prediksi → serving → sistem bisnis

================================================================================
4. STRUKTUR FOLDER
================================================================================
company_dl_foundation/
│
├── configs/
│   ├── base.yaml
│   ├── regression_weight.yaml
│   ├── classification_kondisi.yaml
│   ├── timeseries_demand.yaml
│   ├── image_cls_kondisi.yaml
│   └── image_reg_weight.yaml
│
├── data/
│   ├── raw/                → data mentah (CSV, gambar)
│   ├── interim/            → data hasil cleaning
│   ├── processed/          → data siap latih
│   └── registry/           → metadata versi dataset
│
├── src/
│   ├── preprocessing/
│   │   ├── validators.py
│   │   ├── encoders.py
│   │   ├── scalers.py
│   │   ├── feature_engineering.py
│   │   ├── pipeline.py
│   │   ├── image_pipeline.py
│   │   └── augmentations.py
│   │
│   ├── core/
│   │   ├── blocks.py           → HiddenBlock, ResidualBlock
│   │   ├── conv_blocks.py      → ConvBlock, SEBlock, ResidualConvBlock
│   │   ├── mlp.py              → build_mlp()
│   │   ├── cnn.py              → build_cnn()
│   │   ├── transfer_learning.py
│   │   ├── normalization.py
│   │   ├── heads.py            → regression/classification/TS heads
│   │   └── pruning.py          → apply_pruning, finalize_pruned
│   │
│   ├── tasks/
│   │   ├── base_task.py
│   │   ├── regression_task.py
│   │   ├── classification_task.py
│   │   ├── timeseries_task.py
│   │   ├── image_classification_task.py
│   │   └── image_regression_task.py
│   │
│   ├── training/
│   │   ├── trainer.py
│   │   ├── callbacks.py
│   │   └── evaluator.py
│   │
│   ├── monitoring/
│   │   └── live_visualizer.py  → ProLiveVisualizer
│   │
│   ├── serving/
│   │   ├── predictor.py
│   │   └── api.py              → FastAPI app
│   │
│   ├── integrations/
│   │   ├── base_connector.py
│   │   ├── erp_connector.py
│   │   ├── inventory_connector.py
│   │   ├── purchasing_connector.py
│   │   ├── sales_connector.py
│   │   └── ops_monitor_connector.py
│   │
│   └── utils/
│       ├── config_loader.py
│       ├── logger.py
│       ├── io_utils.py
│       └── seed.py
│
├── scripts/
│   ├── prepare_data.py
│   ├── train_regression.py
│   ├── train_classification.py
│   ├── train_timeseries.py
│   ├── train_image_classification.py
│   ├── prune_model.py
│   ├── evaluate.py
│   └── export_for_serving.py
│
├── artifacts/              → model, preprocessor, log, report
│   ├── models/{task}/{version}/
│   ├── preprocessors/{task}/{version}/
│   ├── logs/
│   └── reports/
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_core_blocks.py
│   ├── test_tasks.py
│   └── test_predictor.py
│
├── requirements.txt
├── setup.py
├── Makefile
└── README.txt

================================================================================
5. INSTALASI & PERSIAPAN
================================================================================
Prasyarat:
  - Python 3.10+
  - pip
  - (opsional) virtualenv / conda

Langkah:
  1) Clone repository
       git clone <repo-url> company_dl_foundation
       cd company_dl_foundation

  2) Buat virtual environment
       python -m venv venv
       source venv/bin/activate        (Linux/macOS)
       venv\Scripts\activate           (Windows)

  3) Install dependencies
       pip install -r requirements.txt

  4) Siapkan .env (opsional)
       cp .env.example .env

  5) Verifikasi instalasi
       python -c "import tensorflow as tf; print(tf.__version__)"

Dependensi utama (isi requirements.txt minimal):
  tensorflow>=2.13
  tensorflow-model-optimization>=0.7
  scikit-learn>=1.3
  pandas>=2.0
  numpy>=1.24
  matplotlib>=3.7
  joblib>=1.3
  pyyaml>=6.0
  fastapi>=0.104
  uvicorn>=0.24
  python-multipart
  pytest

================================================================================
6. CARA MENJALANKAN
================================================================================

A. PREPROCESSING DATA
---------------------
  python scripts/prepare_data.py --config configs/regression_weight.yaml
  Output: data/processed/regression/v1/… dan artifacts/preprocessors/regression/v1/

B. TRAINING (TABULAR)
---------------------
  Regresi berat:
    python scripts/train_regression.py --config configs/regression_weight.yaml

  Klasifikasi kondisi:
    python scripts/train_classification.py --config configs/classification_kondisi.yaml

  Time series permintaan:
    python scripts/train_timeseries.py --config configs/timeseries_demand.yaml

  Selama training berlangsung, jendela matplotlib akan menampilkan:
    - Panel 1: kurva loss (train vs val)
    - Panel 2: metrik task (MAE / akurasi)
    - Panel 3 (opsional): scatter prediksi vs aktual

  Hasil training tersimpan di:
    artifacts/models/{task}/v1/model.keras
    artifacts/models/{task}/v1/best.keras
    artifacts/reports/{task}/v1/training_curve.png
    artifacts/reports/{task}/v1/history.json

C. TRAINING (GAMBAR)
--------------------
  python scripts/train_image_classification.py \
      --config configs/image_cls_kondisi.yaml

  Struktur folder gambar yang dibutuhkan:
    data/raw/images/
      train/
        baik/
        perlu_diperiksa/
      val/
        baik/
        perlu_diperiksa/

D. PRUNING (model ringan untuk edge)
------------------------------------
  python scripts/prune_model.py --config configs/regression_weight.yaml
  Output: artifacts/models/{task}/v1/pruned.keras

  Parameter pruning di YAML:
    pruning:
      enabled: true
      target_sparsity: 0.5
      schedule: polynomial
      begin_step: 0
      end_step: 1000

E. EVALUASI MODEL
-----------------
  python scripts/evaluate.py \
      --config configs/regression_weight.yaml \
      --model-version v1

F. SERVING (REST API)
---------------------
  Uvicorn dev server:
    uvicorn src.serving.api:app --reload --host 0.0.0.0 --port 8000

  Test endpoint:
    curl -X POST http://localhost:8000/predict/weight \
         -H "Content-Type: application/json" \
         -d '{"rows":[{"diameter_cm":7.2,"rata_rata_merah_0_255":210,
                       "rata_rata_hijau_0_255":120,
                       "rata_rata_biru_0_255":40,
                       "indikator_memar_0_1":0.1,
                       "indikator_bintik_0_1":0.2,
                       "hari_sejak_panen":4,
                       "jenis_buah":"apel"}]}'

G. EXPORT UNTUK EDGE (TFLite)
-----------------------------
  python scripts/export_for_serving.py \
      --config configs/image_cls_kondisi.yaml \
      --format tflite

H. TEST & LINT
--------------
  pytest -q
  python -m flake8 src
  black src scripts

I. MAKEFILE (shortcut)
----------------------
  make install       → pip install -r requirements.txt
  make train         → jalankan script training default
  make prune         → jalankan pruning
  make serve         → uvicorn API
  make test          → pytest

================================================================================
7. ALUR RETRAINING SAAT DATA BERTAMBAH
================================================================================
  [1] Data baru masuk dari ERP / kamera QC / sensor
      ↓
  [2] Ekspor ke data/raw/{versi}/
      ↓
  [3] Jalankan scripts/prepare_data.py (menghasilkan processed/{versi})
      ↓
  [4] Jalankan training dengan config yang sama, version="v2"
      ↓
  [5] Evaluasi & bandingkan v2 vs v1 (scripts/evaluate.py)
      ↓
  [6] Jika v2 lebih baik pada metrik bisnis:
        promote → artifacts/models/{task}/v2/
        update pointer "production_version" di configs/base.yaml
      Jika tidak:
        tetap pakai v1, catat hasil percobaan di artifacts/reports/
      ↓
  [7] Predictor hot-reload (server membaca pointer versi terbaru)
      ↓
  [8] Monitoring opsional: kirim ringkasan ke dashboard

Aturan versi:
  - Versi model dan preprocessor WAJIB sepasang (v1↔v1).
  - Model tidak boleh dipakai dengan preprocessor versi berbeda.
  - Semua model baru diuji A/B minimal terhadap 1 batch produksi
    sebelum di-promote.

================================================================================
8. INTEGRASI KE SISTEM BISNIS
================================================================================
Connector pattern tersedia di src/integrations/:

  - erp_connector.py
      kirim prediksi berat / jumlah, tarik item pending inspection
  - inventory_connector.py
      rekomendasi reorder point, alarm stok, klasifikasi risiko
  - purchasing_connector.py
      demand forecast → usulan PO
  - sales_connector.py
      tren permintaan per SKU, prediksi penjualan
  - ops_monitor_connector.py
      feed dashboard operasional, alert anomali

Kontrak antarmuka:
  - Semua connector mewarisi BaseConnector
  - Method publik: fetch(), push(), health_check()
  - Timeout & retry wajib di-set di config
  - Kredensial via env var, TIDAK hardcode

Pola pemakaian:
  from src.serving.predictor import Predictor
  from src.integrations.erp_connector import ERPConnector

  predictor = Predictor("regression")
  erp = ERPConnector(base_url=os.getenv("ERP_URL"),
                     api_key=os.getenv("ERP_KEY"))

  items = erp.fetch_items_pending_inspection()
  df = pd.DataFrame(items)
  preds = predictor.predict(df)

  for item_id, w in zip(df["id"], preds):
      erp.push_weight_prediction(item_id, float(w))

================================================================================
9. CARA UPDATE KODE
================================================================================

A. UPDATE DEPENDENSI
--------------------
  1) Ubah requirements.txt (pin versi baru).
  2) Jalankan:
       pip install -r requirements.txt
       pytest -q
  3) Update CHANGELOG.txt dengan entri versi.

B. UPDATE MODEL / ARSITEKTUR
----------------------------
  1) Buat branch baru: feat/<nama-perubahan>
  2) Jika mengubah core (src/core/):
       - Tambahkan test di tests/test_core_blocks.py
       - Pastikan model lama masih bisa di-load:
           pytest tests/test_backward_compat.py
  3) Jika mengubah preprocessing:
       - Naikkan versi preprocessor (v1 → v2)
       - Wajib melatih ulang model dengan preprocessor baru
       - Jangan pakai model lama dengan preprocessor baru
  4) Buka Pull Request, minimal 1 reviewer.

C. UPDATE CONFIG
----------------
  - Semua config ada di configs/*.yaml
  - Jika menambah field baru di config, beri default value
    di src/utils/config_loader.py agar config lama tetap valid.

D. ROLLBACK MODEL
-----------------
  Jika model v2 bermasalah di produksi:
  1) Ubah production_version di configs/base.yaml ke v1
  2) Restart service serving
  3) Catat insiden di artifacts/reports/incident_<tanggal>.txt
  4) Investigasi v2 di environment staging

E. HOT RELOAD SERVING
---------------------
  - API memantau perubahan pada pointer production_version
  - Reload otomatis tanpa restart container
  - Jika reload gagal → fallback ke versi terakhir yang berhasil

F. KONVENSI COMMIT
------------------
  <tipe>(<lingkup>): <deskripsi singkat>
  Tipe: feat | fix | refactor | docs | test | chore | perf
  Contoh:
    feat(pruning): tambah polynomial schedule
    fix(serving): tangani input NaN pada preprocessing

G. TAG VERSI
------------
  git tag -a v1.2.0 -m "Release v1.2.0"
  git push origin v1.2.0

================================================================================
10. KONVENSI KODE & KONTRIBUSI
================================================================================
  - Format     : black (line-length 100)
  - Lint       : flake8
  - Tipe       : gunakan type hints untuk fungsi publik
  - Docstring  : Google style
  - Test       : pytest, coverage minimal 70% untuk src/core/
  - Naming     :
      file        : snake_case.py
      class       : PascalCase
      fungsi      : snake_case
      konstanta   : UPPER_SNAKE
  - Log        : gunakan src.utils.logger, bukan print()
  - Seed       : gunakan src.utils.seed.set_seed()
  - Non-determinism harus didokumentasikan di docstring

Struktur PR yang baik:
  - 1 PR = 1 tujuan
  - Sertakan deskripsi: apa, mengapa, bagaimana cara uji
  - Screenshot / kurva training jika mengubah model

================================================================================
11. TROUBLESHOOTING
================================================================================
Q: Visualizer tidak muncul saat training?
A: Backend matplotlib "Agg" (headless). Set:
     export MPLBACKEND=TkAgg    (Linux/macOS)
     set MPLBACKEND=TkAgg       (Windows)
   Atau jalankan dengan flag --no-viz bila di server.

Q: Error "Cannot load model, incompatible preprocessing"?
A: Model dan preprocessor harus dari versi yang sama (v1↔v1).
   Cek meta.json di artifacts/preprocessors/{task}/{version}/.

Q: Training sangat lambat?
A: 1) Aktifkan GPU: pastikan tensorflow-gpu / CUDA terpasang.
   2) Gunakan tf.data dengan .prefetch(AUTOTUNE).
   3) Kurangi batch_size bila OOM, atau naikkan bila GPU idle.

Q: Model terlalu besar untuk edge?
A: Jalankan scripts/prune_model.py dengan target_sparsity ≥0.6,
   lalu scripts/export_for_serving.py --format tflite.

Q: Metric accuracy rendah setelah preprocessing?
A: Cek skala target, pastikan y_train dan y_val memakai
   skala yang sama. Untuk regresi berat, accuracy_pm_20g
   lebih relevan daripada MAE mentah.

Q: Data baru mengubah skema CSV (kolom hilang)?
A: Validator akan menolak. Perbarui REQUIRED_COLUMNS di
   src/preprocessing/validators.py dan naikkan versi
   preprocessor sebelum retraining.

================================================================================
12. ROADMAP
================================================================================
  [SELESAI]
   ✓ MLP modular (tabular)
   ✓ Regression / Classification / Time Series
   ✓ Live visualization
   ✓ Pruning (Dense)
   ✓ Serving API
   ✓ Connector kerangka ke sistem bisnis

  [BERJALAN]
   ▸ Transfer learning (MobileNetV3 / EfficientNetB0)
   ▸ Pruning untuk Conv2D
   ▸ Endpoint /predict/condition_from_image

  [RENCANA]
   ◻ Multi-modal model (CNN gambar + MLP tabular)
   ◻ Object detection (YOLO) untuk menghitung jumlah buah
   ◻ AutoML hyperparameter tuning (Optuna)
   ◻ Aktif learning (label baru dari feedback user)
   ◻ Deployment edge (TFLite / Coral / Jetson Nano)
   ◻ Observability produksi (Prometheus + Grafana)

================================================================================
13. LISENSI & KONTAK
================================================================================
  Lisensi       : Internal / Proprietary
  Pemilik kode  : Tim Data Science & Engineering
  Kontak        : data-science@perusahaan.co.id
  Repo internal : <url-git-internal>
  Dokumentasi   : /docs (internal wiki)

  Untuk pertanyaan, hubungi kanal #dl-foundation
  di workspace internal perusahaan.

================================================================================
                              AKHIR DOKUMEN
================================================================================
"""

def generate_readme(path: str = "README.txt") -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(README_CONTENT.strip() + "\n")
    print(f"[OK] README dibuat: {path}")
    print(f"[INFO] Ukuran: {len(README_CONTENT)} karakter")

if __name__ == "__main__":
    generate_readme()
