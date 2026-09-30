# Optimasi Penempatan Fasilitas Minimarket (Facility Location Problem)
### Perbandingan Algoritma Genetika (GA) dan Particle Swarm Optimization (PSO)

Proyek ini mengimplementasikan penyelesaian masalah **Facility Location** untuk mencari penempatan paling efektif minimarket pada peta wilayah kustom berukuran **2.0 km × 1.5 km** (koordinat meter). Implementasi algoritma GA dan PSO ditulis **murni dari nol (from scratch)** menggunakan pustaka standar saintifik Python (**NumPy, SciPy, Pandas, Matplotlib, Pytest**) tanpa bergantung pada pustaka optimasi siap pakai (seperti DEAP atau PySwarms), sehingga setiap langkah matematis dan algoritmik dapat dijelaskan secara transparan saat presentasi ilmiah.

---

## 1. Struktur Proyek

```text
biocomputing-7/
├── config.json               # Konfigurasi bobot fitness, parameter GA, PSO, dan batasan peta
├── requirements.txt          # Daftar dependensi library resmi
├── map_model.py              # Model peta, loading JSON, geometri (point-in-poly, jarak jalan), plot_map
├── fitness.py                # FitnessEvaluator, counter evaluasi, normalisasi [0,1], mode max/min_valid
├── ga.py                     # Algoritma Genetika real-coded (Tournament, BLX-alpha, Gaussian Mutation, Elitism)
├── pso.py                    # Particle Swarm Optimization (Inersia linear decay, c1, c2, boundary reflect/clamp)
├── stats_utils.py            # Kalkulasi evaluasi ke-95%, ringkasan deskriptif, uji Mann-Whitney U (SciPy)
├── generate_maps.py          # Generator 3 preset peta kustom (desa_ramai, hauling, campuran)
├── app_tkinter.py            # Aplikasi simulasi interaktif GUI desktop (Tkinter + Matplotlib)
├── run_batch.py              # Eksperimen 30 run independen GA vs PSO (mode max & min_valid)
├── analyze.py                # Analisis statistik batch, uji hipotesis, boxplot, konvergensi, scatter lokasi
├── test_facility.py          # Unit test (Pytest) untuk evaluasi, counter, reproduktifitas, dan geometri
├── maps/                     # Folder penyimpanan file peta JSON
│   ├── desa_ramai.json
│   ├── hauling.json
│   └── campuran.json
└── results/                  # Folder output batch (CSV, NPZ, PNG, TXT laporan)
    ├── raw_runs.csv
    ├── config_used.json
    ├── convergence_data.npz
    ├── summary_statistics.csv
    ├── statistical_test_report.txt
    ├── boxplot_fitness.png
    ├── convergence_comparison.png
    └── final_locations_scatter.png
```

> **Catatan Modularitas**: Modul inti komputasi (`map_model.py`, `fitness.py`, `ga.py`, `pso.py`, `stats_utils.py`) **tidak mengimpor matplotlib di tingkat modul**, sehingga proses optimasi dan evaluasi batch dapat berjalan cepat dan murni pada lingkungan headless/server.

---

## 2. Model Peta Kustom (Format JSON)

Peta berukuran $2000 \times 1500$ meter merepresentasikan bentang lahan nyata dengan elemen-elemen berikut:
1. **Pemukiman / Rumah**: Titik koordinat $(x, y)$ dengan bobot kepadatan penduduk ($w_i$).
2. **Fasilitas Umum**: Sekolah ($w=1.5$), Pasar ($w=2.0$), Restoran ($w=1.2$), Bengkel ($w=0.8$), dan Masjid/Kantor ($w=1.0$).
3. **Kompetitor Minimarket Eksisting**: Titik lokasi minimarket pesaing yang sudah beroperasi.
4. **Jaringan Jalan**: Polyline dengan hierarki kelas jalan (*Arteri, Kolektor, Lokal, Hauling*) dan volume lalu lintas ($0.0 - 1.0$).
5. **Zona Terlarang**: Poligon tertutup (*Sungai, Sawah Produktif, Hutan Lindung, Lahan Tambang*) yang tidak boleh dibangun toko fisik.

### Tiga Preset Peta:
- **`desa_ramai.json`**: Pemukiman padat penduduk, 2 sekolah, deretan restoran, bengkel, pasar sentral, 2 kompetitor, dilintasi sungai berliku dan persawahan.
- **`hauling.json`**: Kawasan pertambangan terpencil dengan populasi barak sangat jarang, jalan hauling industri dengan truk berat, fasilitas minim, zona pit tambang dan hutan lindung luas.
- **`campuran.json`**: Zona transisi pinggiran pedesaan menuju koridor industri hauling dengan fasilitas sedang dan 1 kompetitor di persimpangan.

---

## 3. Rumus dan Komponen Fungsi Fitness

Fungsi kesesuaian lokasi mengevaluasi kelayakan penempatan minimarket berdasarkan multi-kriteria:

$$\mathcal{F}(x, y) = w_1 \cdot \text{Populasi} + w_2 \cdot \text{AksesJalan} + w_3 \cdot \text{Fasilitas} + w_4 \cdot \text{Kompetitor} - w_5 \cdot \text{BiayaLahan} - \text{Penalti}$$

Semua fitur dinormalisasi ke dalam rentang interval $[0, 1]$:

### 1. Kepadatan Populasi ($S_{pop} \in [0, 1]$)
Menggunakan model gravitasi spasial dengan peluruhan Gaussian:
$$S_{pop}(x, y) = \frac{1}{M_{pop}} \sum_{i=1}^{N_{house}} w_i \cdot \exp\left( -\frac{d((x,y), \text{rumah}_i)^2}{2\sigma_{pop}^2} \right)$$
Semakin dekat lokasi dengan kluster pemukiman padat, semakin tinggi potensi pasar pelanggan.

### 2. Kondisi & Kelayakan Jalan ($S_{road} \in [0, 1]$) — Bagus (Proper) vs Bertanah Lumpur (Hauling)
Karena penempatan minimarket sudah dibatasi wajib berada di koridor jalan fisik ($d \le 50\text{ m}$), kriteria ini **bukan sekadar mengukur ada/tidaknya jalan**, melainkan **kondisi fisik dan kelayakan jalan** untuk aktivitas belanja retail:
- **Jalan Bagus / Proper ($0.85 - 1.0$)**: Jalan Arteri dan Kolektor Desa beraspal mulus, nyaman, dan aman dilalui motor/mobil pelanggan.
- **Jalan Lokal ($0.50 - 0.60$)**: Paving block atau aspal dusun sederhana.
- **Jalan Bertanah Lumpur / Hauling ($0.15 - 0.25$)**: Jalur hauling tambang berupa tanah merah yang becek/berlumpur saat hujan, berdebu pekat saat kemarau, dan berisiko tinggi karena dilewati truk tronton muatan batubara/mineral (sangat tidak layak untuk minimarket).

$$d_{eff} = \min_{seg} \left( \frac{\text{dist}((x,y), seg)}{\text{bobot\_kualitas\_jalan} \cdot (1 + \text{lalu\_lintas})} \right)$$
$$S_{road}(x,y) = \exp\left( -\frac{d_{eff}^2}{2\sigma_{road}^2} \right)$$

### 3. Sinergi Fasilitas Umum ($S_{fac} \in [0, 1]$)
Tarikan komersial dari titik bangkitan aktivitas (sekolah, pasar, perkantoran):
$$S_{fac}(x,y) = \frac{1}{M_{fac}} \sum_{j=1}^{N_{fac}} w_j \cdot \exp\left( -\frac{d((x,y), \text{fasilitas}_j)^2}{2\sigma_{fac}^2} \right)$$

### 4. Interaksi Kompetitor ($S_{comp} \in [0, 1]$) — Kurva Tidak Monoton
Hubungan spasial non-monotonik menggunakan fungsi kurva Ricker:
$$S_{comp}(d_c) = \left( \frac{d_c}{d_{opt}} \right) \cdot \exp\left( 1 - \frac{d_c}{d_{opt}} \right)$$
- **Jika $d_c \to 0$ (berimpit)**: Nilai $\to 0$ (kanibalisasi langsung / perang harga).
- **Pada jarak optimal $d_c = d_{opt} \approx 350\text{ m}$ (sweet spot)**: Nilai mencapai puncak **$1.0$** (memanfaatkan aglomerasi keramaian pasar tanpa kanibalisasi langsung).
- **Jika $d_c \gg d_{opt}$ (terlalu jauh)**: Nilai meluruh mendekati $0$ (berada di wilayah sepi/terisolir yang tidak memiliki aktivitas komersial).

### 5. Biaya Lahan ($S_{cost} \in [0, 1]$)
Biaya sewa/akuisisi lahan diasumsikan lebih mahal pada pusat konsentrasi kota dan pinggir jalan arteri:
$$S_{cost}(x,y) = 0.55 \cdot \exp\left( -\frac{d((x,y), \text{pusat})^2}{2\sigma_{center}^2} \right) + 0.45 \cdot S_{road}$$

### 6. Penalti Batas & Zona Terlarang
- Titik di luar koordinat peta ($x < 0, x > 2000, y < 0, y > 1500$) dikenakan penalti $-1000.0$.
- Titik di dalam poligon zona terlarang dideteksi menggunakan algoritma **Ray Casting (Even-Odd rule)** dan dikenakan penalti $-1000.0$.

### 7. Penempatan Multi-Toko ($p = 1, 2, 3$) & Kanibalisasi Internal
Untuk vektor solusi berdimensi $2p$, skor rata-rata dihitung untuk seluruh toko. Jika jarak antar dua toko baru $d_{ij} < d_{cannibal}$ (default 400 m), diterapkan penalti kanibalisasi kuadratik:
$$\text{Penalti Kanibalisasi} = \sum_{i < j} w_{cannibal} \cdot \left( 1 - \frac{d_{ij}}{d_{cannibal}} \right)^2$$

### 8. Mode Optimasi: `max` vs `min_valid`
- **Mode `max`**: Memaksimalkan nilai fitness bersih untuk menemukan lokasi toko terbaik.
- **Mode `min_valid`**: Mencari **lokasi terburuk yang tetap VALID** (tidak di zona terlarang dan tidak di luar peta). Titik di zona terlarang tetap terkena penalti besar ($-1000.0$), sehingga algoritma secara presisi berkumpul pada titik valid dengan skor terendah.
- **Invert Features**: Opsi konfigurasi `invert_features: ["kompetitor"]` untuk membalik skor fitur tertentu ($S \leftarrow 1.0 - S$).

---

## 4. Jaminan Matched Evaluation Budget

Dalam riset komparasi metaheuristik, membandingkan GA dan PSO berdasarkan jumlah generasi atau iterasi adalah keliru karena ukuran populasi dan struktur loop evaluasi dapat berbeda. 

Pada proyek ini:
1. Kedua algoritma dikendalikan oleh objek `FitnessEvaluator` yang memiliki counter internal `self.eval_count`.
2. Setiap pemanggilan fungsi evaluasi kandidat menambah `self.eval_count += 1`.
3. Algoritma GA dan PSO **dihentikan secara presisi tepat saat `eval_count >= budget`** (default: 2000 evaluasi).
4. Prekomputasi matriks heatmap 2D untuk visualisasi dijalankan secara terpisah menggunakan parameter internal tanpa menambah `eval_count`, sehingga anggaran komputasi murni digunakan untuk pencarian solusi.

---

## 5. Algoritma Optimasi (Ditulis dari Nol)

### Genetic Algorithm (GA) — `ga.py`
- **Representasi**: Kromosom riil kontinu berdimensi $2p$.
- **Seleksi**: *Tournament Selection* (ukuran turnamen $k=3$).
- **Crossover**: *BLX-$\alpha$ (Blend Crossover)* dengan $\alpha = 0.5$ untuk menjaga eksplorasi kontinu melampaui rentang kedua induk.
- **Mutasi**: *Gaussian Mutation* ($\sigma_{mut} = 40.0\text{ m}$) dengan probabilitas per-gen $p_m = 0.15$.
- **Elitisme**: $E = 2$ individu terbaik dipertahankan tanpa mutasi ke generasi berikutnya.
- **Pelacakan Riwayat**: Menyimpan koordinat seluruh populasi, indeks elit, pasangan induk-anak, dan penanda mutasi untuk animasi flash.

### Particle Swarm Optimization (PSO) — `pso.py`
- **Representasi**: Partikel $X_i \in \mathbb{R}^{2p}$ dengan kecepatan $V_i \in [-V_{max}, V_{max}]$.
- **Inersia Dinamis**: Peluruhan linier inersia dari $w_{max} = 0.9$ menuju $w_{min} = 0.4$ sepanjang iterasi untuk menyeimbangkan eksplorasi global awal dan eksploitasi lokal akhir.
- **Komponen Akselerasi**: Kognitif $c_1 = 1.494$ dan Sosial $c_2 = 1.494$.
- **Penanganan Batas**: Opsi *Reflect* (memantul dari dinding batas wilayah dengan pembalikan arah vektor kecepatan) atau *Clamp*.
- **Pelacakan Riwayat**: Menyimpan posisi partikel, vektor kecepatan (untuk quiver), lintasan gerak pendek (*motion trail*), $pbest$, dan $gbest$.

---

## 6. Cara Menjalankan Program

### Prasyarat dan Instalasi
Pastikan Python 3.9+ telah terpasang, lalu instal paket dependensi:
```bash
pip install -r requirements.txt
```

### 1. Menghasilkan File Preset Peta
Menghasilkan 3 file JSON di dalam folder `maps/`:
```bash
python generate_maps.py
```

### 2. Menjalankan Pengujian Unit (Pytest)
Memverifikasi fungsionalitas counter evaluasi, penalti zona, reproducibility seed, kanibalisasi, dan geometri:
```bash
pytest -v test_facility.py
```

### 3. Menjalankan Simulasi Interaktif (Tkinter)
Aplikasi desktop berbasis **Tkinter** terintegrasi penuh untuk simulasi dan presentasi:
```bash
python app_tkinter.py
```
* **Fitur Interaktif Tkinter**:
  * **Visualisasi Simultan**: Matplotlib live di canvas Tkinter (GA kiri, PSO kanan, Konvergensi bawah).
  * **Zoom Independen Tiap Metode**: Zoom in/out mandiri untuk subplot GA atau PSO via scroll mouse, tombol toolbar `[➕] [➖] [Desa] [⟲]`, atau preset fokus desa.
  * **Batasan Koridor Jalan**: Penempatan minimarket dibatasi di koridor jalan ($d \le 50\text{ m}$), dengan deteksi dan penalti off-road otomatis.
  * **Visualisasi Mutasi Rapi**: Individu hasil mutasi ditandai titik kuning/amber yang jelas dan rapi tanpa garis laba-laba yang membingungkan.
  * **1 Kompetitor Eksisting**: Penempatan 1 minimarket lama (K) di persimpangan jalan desa.
  * **Kontrol Pemutaran**: Play/Pause, Step (langkah per langkah), Reset, Run to End, dan Slider Delay Kecepatan (ms).
  * **Tab Papan Skor (Scoreboard)**: Live leader badge (*"GA Memimpin"* vs *"PSO Memimpin"*), perbandingan skor real-time, koordinat, evaluasi terpakai, dan bar matched budget.
  * **Tab Inspeksi Titik (Klik Peta)**: Klik titik manapun pada peta untuk melihat koordinat $(X, Y)$, legalitas (valid atau zona terlarang), status koridor jalan ($d \le 50\text{ m}$), estimasi fitness, dan rincian bar meter 5 kriteria (*Populasi, Jalan, Fasilitas, Kompetitor, Biaya*).
  * **Tab "Ini Apa & Bagaimana?"**: Penjelasan visual mendalam setiap warna, ikon, koridor jalan, dan rumus matematika.
  * **Preset & Bobot Dinamis**: Ganti peta secara instan, ubah mode `max`/`min_valid`, dan atur bobot $w_1 - w_5$ dengan kalkulasi ulang langsung.

### 4. Menjalankan Eksperimen Batch (30 Run Independen)
Mengeksekusi 30 run mandiri untuk GA dan PSO (seed 1 s/d 30) pada kedua mode:
```bash
python run_batch.py --runs 30 --budget 2000 --mode both
```
Hasil akan tersimpan di:
- `results/raw_runs.csv`: Data run individual (fitness, waktu, evaluasi ke-95%, koordinat).
- `results/convergence_data.npz`: Kurva konvergensi tiap run.
- `results/config_used.json`: Parameter konfigurasi yang digunakan.

### 5. Melakukan Analisis Statistik & Visualisasi Ilmiah
Menganalisis data batch, menghitung statistik deskriptif, menjalankan uji Mann-Whitney U, dan mencetak plot:
```bash
python analyze.py
```
Output yang dihasilkan di folder `results/`:
- `results/summary_statistics.csv`: Tabel Mean, Std, Median, IQR, Min, Max, Waktu.
- `results/statistical_test_report.txt`: Laporan uji hipotesis Mann-Whitney U komprehensif dalam Bahasa Indonesia.
- `results/boxplot_fitness.png`: Boxplot distribusi fitness GA vs PSO.
- `results/convergence_comparison.png`: Kurva konvergensi rata-rata $\pm$ std deviasi.
- `results/final_locations_scatter.png`: Sebaran koordinat akhir 30 run di atas peta.
