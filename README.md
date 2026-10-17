# TSP Learning & Optimization Application
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://aplikasimetop-7gpe34hcauak4qgfaosrot.streamlit.app/)

Aplikasi web interaktif (Streamlit) untuk **membedah, menguji, dan membandingkan metode heuristik** pada *Travelling Salesman Problem* (TSP). Setiap algoritma menampilkan riwayat iterasi, detail perhitungan tiap langkah, visualisasi rute, serta perbandingan jarak dan waktu eksekusi.

> Tugas Kuliah — Metode Optimasi (METOP) · Departemen Sains Data, Institut Teknologi Sepuluh Nopember (ITS)

## Tautan

| Keperluan | Tautan |
|---|---|
| **Aplikasi web (live)** | https://aplikasimetop-7gpe34hcauak4qgfaosrot.streamlit.app/ |
| Repositori utama (kelompok) | https://github.com/kirann41ndy/aplikasi_metop |
| Repositori deploy (fork) | https://github.com/elenafaradisa/aplikasi_metop/tree/main |

## Catatan Repositori dan Deployment

Repositori ini adalah **fork** dari repositori utama kelompok. Aplikasi web **dijalankan (di-deploy) dari fork ini**, bukan dari repositori utama.

Alasannya: Streamlit Community Cloud hanya dapat men-deploy repositori yang dapat diakses oleh akun GitHub yang dipakai untuk login Streamlit. Karena itu repositori utama di-fork ke akun pribadi, lalu fork tersebut yang dihubungkan ke Streamlit Community Cloud.

Konsekuensinya:

- Perubahan kode baru tampil di web setelah di-*push* ke **fork** (Streamlit akan men-deploy ulang otomatis).
- Perubahan yang hanya masuk ke repositori utama **belum** tampil di web sampai fork disinkronkan.

Cara menyinkronkan fork dengan repositori utama:

```bash
# sekali saja: daftarkan repositori utama sebagai "upstream"
git remote add upstream <URL-repositori-utama>

# setiap kali ingin menarik perubahan terbaru
git fetch upstream
git checkout main
git merge upstream/main
git push origin main        # push ke fork -> web ter-deploy ulang
```

Cara men-deploy dari fork ke Streamlit Community Cloud:

1. Buka https://share.streamlit.io lalu login dengan akun GitHub pemilik fork.
2. Klik **Create app**, pilih repositori **fork**, branch `main`, dan *main file path* `app.py`.
3. (Opsional) pada **Advanced settings**, pilih Python 3.12.
4. Klik **Deploy**. Dependensi dibaca dari `requirements.txt`.

> Aplikasi gratis di Streamlit Community Cloud dapat "tertidur" bila lama tidak dibuka. Klik tombol *Wake up* bila muncul.

### Anggota Kelompok

| No | Nama | NRP |
|----|------|-----|
| 1 | `Ghisele Valerin Sharent Milano` | `5052241005` |
| 2 | `Kiranna Indy` | `5052241014` |
| 3 | `Elena Miska Faradisa` | `5052241022` |
| 4 | `Alycia Zada` | `5052241034` |
| 5 | `Inessa Regina Angelica Munda` | `5052241037` |

---

## Fitur Utama

- **8 algoritma** dari tiga keluarga: Constructive, Local Search, dan Metaheuristic.
- **Empat sumber data**: dataset bawaan `small.csv` (dimuat otomatis saat aplikasi dibuka), generate acak, upload CSV/Excel, dan input manual.
- **Dua mode eksekusi**
  - **Independent** — pilih satu atau beberapa algoritma, masing-masing berjalan dengan initialization dan parameternya sendiri.
  - **Hybrid** — rute dari algoritma Constructive dipakai sebagai *initial route* algoritma Optimization.
- **Detail per iterasi** — rute sebelum dan sesudah, edge yang dibuang dan ditambah, rumus delta, serta tabel kandidat move yang dievaluasi (2-opt dan 3-opt).
- **Visualisasi** — sebaran node dan rute dengan arah panah (Plotly), bisa dilihat per iterasi.
- **Perbandingan hasil** — tabel jarak awal/akhir, persentase improvement, waktu eksekusi, plus grafik perbandingan.
- **Distance matrix** — Euclidean atau Manhattan.

## Algoritma yang Tersedia

| Kategori | Algoritma | Parameter |
|---|---|---|
| Constructive | Nearest Neighbor | start node |
| Constructive | Nearest Insertion | start node |
| Constructive | Farthest Insertion | start node |
| Constructive | Arbitrary Insertion | start node, `seed` |
| Local Search | 2-opt | `max_iter`, `strategy` (`best` / `first`) |
| Local Search | 3-opt | `max_iter`, `strategy` (`best` / `first`) |
| Metaheuristic | Simulated Annealing | `initial_temp`, `cooling_rate`, `min_temp`, `max_iter`, `seed` |
| Metaheuristic | Tabu Search | `max_iter`, `tabu_tenure` |

Catatan implementasi:

- Rute selalu berbentuk **tur tertutup** `[home, ..., home]` dengan panjang `n + 1`, dan node *home* tidak pernah dipindahkan.
- **2-opt**: membalik segmen `route[i..j]` dengan `1 ≤ i < j ≤ n-1`. **3-opt**: 7 variant penyambungan ulang, 3 di antaranya setara 2-opt.
- **Strategy** `best` memilih move dengan delta paling kecil, `first` memilih move pertama yang memperbaiki rute. Pencarian berhenti di local optimum atau saat `max_iter` tercapai.
- **Simulated Annealing** memilih satu move 2-opt acak per iterasi. Move yang lebih buruk diterima dengan peluang `exp(-Δ/T)`. Suhu dikalikan `cooling_rate` tiap iterasi, dan rute terbaik selama pencarian yang dikembalikan.
- **Tabu Search** memakai neighborhood 2-opt:
  - Move yang hanya membalik seluruh rute di antara node home (rute sama, dibaca terbalik, delta 0) tidak dijadikan kandidat, sehingga untuk `n` node ada `C(n-1, 2) - 1` kandidat per iterasi.
  - Dua edge baru yang terbentuk dari move terpilih dilarang dibongkar selama `tabu_tenure` iterasi.
  - Sebuah move berstatus tabu bila **salah satu** edge yang dibuangnya masih tabu.
  - Move tabu tetap boleh dipilih bila jaraknya lebih pendek dari solusi terbaik sejauh ini (*aspiration criterion*).
  - Bila semua kandidat tabu dan tidak ada yang memenuhi aspirasi, dipilih move *least tabu* (total sisa tenure terkecil, bila seri dipilih jarak terkecil).
  - Intensification dan diversification tidak diimplementasikan.

## Struktur Proyek

```
aplikasi_metop/
├── app.py                      # Aplikasi Streamlit (UI, sidebar, hasil, hybrid, tab tugas)
├── requirements.txt
├── algorithms/
│   ├── base.py                 # Distance matrix, jarak tur, generator & validasi tur
│   ├── constructive.py         # NN, Nearest/Farthest/Arbitrary Insertion
│   ├── local_search.py         # 2-opt dan 3-opt (+ history & detail move)
│   ├── simulated_annealing.py  # Simulated Annealing
│   └── tabu_search.py          # Tabu Search
├── utils/
│   ├── load_data.py            # Baca & validasi CSV / Excel / DataFrame
│   ├── visualization.py        # Plot node dan rute (Plotly)
│   ├── comparison.py           # Tabel & grafik perbandingan hasil
│   └── logo.png
├── dataset/
│   └── small.csv               # Dataset bawaan (29 node, kolom x,y tanpa header)
└── .streamlit/
    └── config.toml             # Tema aplikasi
```

## Instalasi dan Menjalankan

Prasyarat: **Python 3.9+** (diuji di Python 3.12).

```bash
# 1. Masuk ke folder proyek
cd aplikasi_metop

# 2. (Opsional) buat virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Pasang dependensi
pip install -r requirements.txt

# 4. Jalankan aplikasi
streamlit run app.py
```

Aplikasi terbuka di `http://localhost:8501`. Gunakan Streamlit versi terbaru (disarankan ≥ 1.40) karena UI memakai `st.pills` dan ikon `:material/...`.

## Cara Pakai

### Tab "Eksperimen"

1. **Input Data** — pilih sumber: *Dataset Small* (bawaan), *Generate* (acak, 3–500 node), *Upload* (CSV/Excel), atau *Manual* (tabel yang bisa diedit).
2. Lihat sebaran node dan **distance matrix** (Euclidean / Manhattan).
3. Di **sidebar**, pilih mode **Independent** atau **Hybrid**.
   - *Independent*: pilih algoritma, atur *Initialization* (start node, atau rute awal: Generate Random / Manual / sama dengan algoritma sebelumnya) dan *Parameters*, lalu klik **Run Independent**.
   - *Hybrid*: pilih kombinasi Constructive → Optimization, start node, dan parameter, lalu klik Run.
4. Buka hasil: jarak awal/akhir, waktu eksekusi, rute, grafik, **Tabel Iterasi**, dan **Detail Iterasi** lewat pilihan *Pilih Iterasi*.
5. Bila lebih dari satu algoritma dijalankan, bagian **Comparison** menampilkan perbandingan jarak, waktu, dan improvement.

## Format Dataset

File CSV/Excel berisi koordinat dengan minimal **3 node**, tanpa *missing value*.

```csv
x,y
10,20
50,80
90,40
```

- Kolom wajib `x` dan `y`. Kolom `node` opsional dan dibuat otomatis (0, 1, 2, ...) bila tidak ada.
- CSV **tanpa header** dengan tepat 2 kolom juga diterima (seperti `dataset/small.csv`).
- Nama node tidak boleh duplikat.

## Batasan Aplikasi

- **3-opt** dibatasi sampai **150 node** karena kompleksitasnya O(n³) per iterasi.
- Detail kandidat Tabu Search (`verbose_history`) hanya diaktifkan otomatis sampai **100 node**.
- Tabel kandidat 3-opt tidak ditampilkan bila rute lebih dari **60 node**, dan distance matrix hanya ditampilkan sampai **100 node**.
- Pada mode Hybrid, maksimal **12 kombinasi** per proses.
- Delta 2-opt/3-opt mengasumsikan distance matrix **simetris** (otomatis terpenuhi bila dibuat dari koordinat).
- Setiap algoritma dijalankan satu kali per proses. Algoritma yang melibatkan proses acak (Arbitrary Insertion, rute awal acak, Simulated Annealing) bergantung pada `seed`.

## Contoh Memakai Algoritma Langsung (tanpa UI)

```python
from algorithms.base import build_distance_matrix, generate_initial_tour
from algorithms.local_search import two_opt

coords = [(10, 20), (50, 80), (90, 40), (70, 60), (30, 90)]
dist = build_distance_matrix(coords, metric="euclidean")
route = generate_initial_tour(len(coords), home=0, seed=42)

result = two_opt(route, dist, max_iter=100, strategy="best")
print(result["route"], result["distance"])
print(len(result["history"]))   # riwayat tiap iterasi
```

`two_opt` dan `three_opt` mengembalikan dict berisi `route`, `distance`, dan `history`. Setiap item `history` memuat `iteration`, `route`, `distance`, dan `move` (tipe, posisi, edge dibuang/ditambah, delta, `distance_before`, `distance_after`).

## Teknologi

Python · Streamlit · Pandas · Plotly · OpenPyXL

## Catatan

- Detail iterasi dan tabel kandidat hanya dihitung untuk jumlah node yang tidak terlalu besar agar aplikasi tetap responsif.
- Delta 2-opt/3-opt mengasumsikan distance matrix **simetris** (otomatis terpenuhi bila dibuat dari koordinat).
