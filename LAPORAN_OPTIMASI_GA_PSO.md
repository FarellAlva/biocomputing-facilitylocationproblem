# LAPORAN TUGAS BESAR OPTIMASI BERBASIS LAPANGAN
# PERBANDINGAN ALGORITMA GENETIKA (GA) DAN PARTICLE SWARM OPTIMIZATION (PSO) DALAM PENENTUAN LOKASI FASILITAS KOPERASI DESA (KOPDES) TERPADU DI DALAM SATU WILAYAH DESA

---

**Mata Kuliah:** Komputasi Evolusioner / Biocomputing & Optimasi Sistem  
**Studi Kasus:** Penentuan Lokasi Pembangunan Gedung Koperasi Desa (KOPDES) Sukamaju Mandiri  
**Cakupan Wilayah:** 1 Wilayah Desa Lengkap (Pemukiman, Sentra Pertanian, Jaringan Jalan Desa, Fasilitas Publik, & Batas Sempadan)  
**Dimensi Wilayah:** 2.0 km × 1.5 km (2.000 meter × 1.500 meter = 300 Hektar)  
**Tipe Data:** Data Primer Spasial & Dummy Realistis Terkalibrasi (50 Klaster Rumah, Jaringan Jalan Heterogen, 4 Zona Batasan)  
**Anggaran Komputasi:** 2.000 Evaluasi Fungsi Fitness (Matched Budget)  
**Metode:** Genetic Algorithm (Real-Coded GA) vs Particle Swarm Optimization (Continuous PSO)  
**Penyusun:** Tim Mahasiswa Biocomputing  

---

## DAFTAR ISI

1. [Bab 1: Pernyataan Masalah dan Konteks Keputusan KOPDES](#bab-1-pernyataan-masalah-dan-konteks-keputusan-kopdes)
2. [Bab 2: Latar Belakang Masalah Lokal & Dampak Keputusan di Dalam Desa](#bab-2-latar-belakang-masalah-lokal--dampak-keputusan-di-dalam-desa)
3. [Bab 3: Metode Observasi, Kualitas Data, & Masukan Pemangku Kepentingan](#bab-3-metode-observasi-kualitas-data--masukan-pemangku-kepentingan)
4. [Bab 4: Formulasi Matematika Model Optimasi KOPDES](#bab-4-formulasi-matematika-model-optimasi-kopdes)
5. [Bab 5: Desain dan Implementasi Algoritma GA dan PSO](#bab-5-desain-dan-implementasi-algoritma-ga-dan-pso)
6. [Bab 6: Desain Eksperimen, Skenario Desa, & Uji Statistik](#bab-6-desain-eksperimen-skenario-desa--uji-statistik)
7. [Bab 7: Hasil Komputasi dan Analisis Kritis](#bab-7-hasil-komputasi-dan-analisis-kritis)
8. [Bab 8: Validasi Lapangan, Risiko Operasional, Etika, & Kesimpulan](#bab-8-validasi-lapangan-risiko-operasional-etika--kesimpulan)
9. [Lampiran A: Problem Proposal Template (Formulir Pengajuan Masalah KOPDES)](#lampiran-a-problem-proposal-template)
10. [Lampiran B: Catatan Observasi Lapangan & Cleaning Log](#lampiran-b-catatan-observasi-lapangan--cleaning-log)
11. [Lampiran C: Formulir Pengungkapan Penggunaan AI (AI Disclosure Table)](#lampiran-c-formulir-pengungkapan-penggunaan-ai)
12. [Lampiran D: Panduan Tanya-Jawab Ujian Lisan (Oral Defense Prompts)](#lampiran-d-panduan-tanya-jawab-ujian-lisan)

---

## BAB 1: PERNYATAAN MASALAH DAN KONTEKS KEPUTUSAN KOPDES

### 1.1 Identifikasi Masalah dari Observasi Nyata di Dalam Satu Desa
Penelitian ini berfokus pada permasalahan riil perencanaan tata ruang pedesaan: **menentukan lokasi pembangunan gedung Koperasi Desa (KOPDES) Sukamaju Mandiri yang paling optimal di dalam satu wilayah desa**. 

Koperasi Desa (KOPDES) memiliki peran vital sebagai urat nadi ekonomi warga desa dengan beberapa unit fungsi terpadu:
1. **Unit Distribusi Sarana Produksi Pertanian (Saprodi)**: Penyaluran pupuk bersubsidi, bibit padi unggul, obat hama, dan pakan ternak bagi kelompok tani.
2. **Unit Toko Retail Sembako & Konsumsi Warga**: Penyediaan kebutuhan pokok harian (beras, minyak, gula, tepung, telur, perlengkapan sanitasi) dengan harga terjangkau bagi anggota koperasi.
3. **Unit Simpan Pinjam & Penampungan Komoditas Hasil Bumi**: Pelayanan keuangan mikro desa dan penerimaan hasil panen dari petani untuk dijual bersama.

Saat ini, KOPDES belum memiliki gedung operasional permanen dan masih menumpang di salah satu garasi rumah pengurus lama di ujung jalan sempit pedesaan. Akibatnya, truk pengangkut pupuk tidak bisa masuk, distribusi sembako tersendat, dan anggota koperasi dari dusun seberang enggan datang karena jarak tempuh yang jauh dan berliku.

### 1.2 Pemilik Keputusan (Decision Owner)
Keputusan lokasi pendirian gedung KOPDES ini berada di bawah wewenang bersama:
- **Ketua Pengurus Koperasi Desa (KOPDES) Sukamaju Mandiri**
- **Kepala Desa Sukamaju**
- **Ketua Badan Permusyawaratan Desa (BPD)**

Pemilik keputusan bertanggung jawab mengalokasikan dana alokasi khusus desa (Dana Desa / BUMDes), menetapkan status lahan, dan mengurus izin pendirian bangunan fasilitas publik desa.

### 1.3 Kelemahan Cara Pengambilan Keputusan Saat Ini (Current Practice)
Sebelum penelitian komputasi ini dilakukan, pengurus desa sempat merencanakan pembangunan gedung KOPDES di koordinat $(X = 1400\text{ m}, Y = 920\text{ m})$ di pinggiran timur desa dekat jalan tanah perbatasan.
- **Kelemahan Fatal**: Usulan tersebut semata-mata didasarkan karena tanah kas desa di titik itu murah dan tidak dipakai. Namun, lokasinya berada di tepi jalan tanah yang sangat becek/licin saat hujan, jauh dari pemukiman warga desa ($>700\text{ meter}$), rawan debu lalu lintas kendaraan berat, dan tidak memiliki akses parkir memadai untuk sepeda motor maupun gerobak angkut petani. 

### 1.4 Kebutuhan Algoritma Optimasi Metaheuristik
Penentuan titik KOPDES di dalam 1 desa tidak dapat dipecahkan dengan rumus sederhana atau sekadar memilih tanah kosong sembarangan karena:
1. **Ruang Pencarian Spasial Kontinu**: Luas wilayah desa mencakup $2.000\text{ m} \times 1.500\text{ m}$ (300 hektar).
2. **Trade-off Multi-Kriteria yang Kompleks**: KOPDES harus sedekat mungkin dengan permukiman warga (520 KK), berada tepat di koridor jalan beraspal mulus yang kuat dilalui truk pupuk, dekat dengan pusat keramaian desa (Pasar dan Balai Desa/Sekolah), serta tidak mematikan warung kelontong tradisional milik warga lama (kompetitor K).
3. **Batasan Non-Konveks & Zona Terlarang**: Dilarang keras menggusur sawah produktif irigasi (melanggar Perda LP2B), dilarang di sempadan sungai, dilarang di kawasan hutan lindung adat, dan dilarang di luar koridor fisik jalan desa ($>50\text{ m}$).

Dua algoritma metaheuristik populer, **Genetic Algorithm (GA)** dan **Particle Swarm Optimization (PSO)**, digunakan untuk menyelesaikan model optimasi berkonstrain ini secara objektif dan matematis.

---

## BAB 2: LATAR BELAKANG MASALAH LOKAL & DAMPAK KEPUTUSAN DI DALAM DESA

### 2.1 Profil Anggota Koperasi dan Pengguna Terdampak di Dalam Desa
Wilayah 1 desa ini mencakup beberapa kelompok pengguna:
- **Keluarga Warga Dusun Desa (520 KK)**: Tersebar pada 50 klaster perumahan di sisi barat, memerlukan akses mudah untuk membeli kebutuhan pokok dan mengurus simpan pinjam desa tanpa harus keluar desa.
- **Kelompok Tani Sawah Produktif**: Petani yang menggarap hamparan persawahan irigasi seluas 60 hektar di barat daya, memerlukan kemudahan mengangkut pupuk bersubsidi menggunakan motor roda tiga (Tossa/Viar).
- **Pengurus & Pengelola Logistik KOPDES**: Memerlukan akses jalan beraspal dua arah agar truk distribusi dari distributor kabupaten (Pupuk Indonesia / Bulog) dapat bongkar muat tepat di depan gudang KOPDES.
- **Pedagang Warung Tradisional Eksisting**: Toko kelontong milik warga (toko K di koordinat $X=860, Y=800$) yang memerlukan kepastian agar KOPDES tidak dibangun persis bersebelahan yang dapat mematikan mata pencaharian mereka.

```
+-----------------------------------------------------------------------------------+
|               TATA RUANG 1 WILAYAH DESA UNTUK PENEMPATAN KOPDES                   |
|                                                                                   |
|  [HUTAN LINDUNG ADAT]            |  JL. LINTAS PROVINSI  |     [MESS PEKERJA]     |
|  (Zona Terlarang)                |  (Batas Timur Desa)   |     (Pasar Potensial)  |
|                                  |                       |                        |
|  [DUSUN PERMUKIMAN WARGA]        |                       |    [JL. TANAH HAULING] |
|  - 520 KK Rumah Warga            |                       |    (Becek/Lumpur)      |
|  - SDN Sukamaju 1 & Pasar Desa   |                       |                        |
|  - Jl. Utama Desa (Aspal Mulus)  |                       |     [ZONA TERLARANG]   |
|                                  |                       |     (Tambang/Galian)   |
|  [SAWAH IRIGASI PRODUKTIF]       |                       |                        |
|  (Zona Terlarang Dilindungi)     |                       |                        |
+-----------------------------------------------------------------------------------+
```

![Gambar 1: Peta Tata Ruang Wilayah Desa Sukamaju pada Kanvas Tkinter](results/map_study_area.png)
*Gambar 1: Peta Tata Ruang Wilayah Desa Sukamaju pada Kanvas Tkinter (2.0 km × 1.5 km = 300 Hektar) untuk Penentuan Tapak Gedung KOPDES.*

### 2.2 Dampak Konsekuensi Keputusan yang Buruk
1. **Kegagalan Finansial Koperasi**: Jika KOPDES dibangun di pinggir jalan tanah becek di timur desa, warga desa barat enggan berkunjung. Modal usaha KOPDES akan mandek dan bahan sembako kadaluarsa.
2. **Konflik Sosial Antarwarga**: Jika KOPDES dibangun menempel langsung pada toko kelontong swasta warga lama ($d < 100\text{ m}$), timbul kecemburuan sosial bahwa koperasi desa memonopoli pasar dan mematikan usaha warganya sendiri.
3. **Kerusakan Lingkungan & Pelanggaran Hukum**: Pembangunan gedung di lahan sawah irigasi melanggar Undang-Undang Perlindungan Lahan Pertanian Pangan Berkelanjutan (LP2B) dengan sanksi pembongkaran paksa.

---

## BAB 3: METODE OBSERVASI, KUALITAS DATA, & MASUKAN PEMANGKU KEPENTINGAN

### 3.1 Pemisahan Fakta, Pendapat, Asumsi, Keputusan, dan Ketidakpastian
Untuk memastikan objektivitas data dummy terkalibrasi dan observasi lapangan:

| Kategori | Definisi Metodologis | Penerapan Nyata Kasus KOPDES Desa |
|---|---|---|
| **Fakta Teramati (Observed Fact)** | Terukur atau terlihat langsung dalam pemetaan desa | 50 klaster rumah warga di barat; lebar perkerasan Jalan Utama Desa adalah 5.2 meter aspal mulus; jembatan desa bentang 100 meter; terdapat toko kelontong swasta lama (K) di dekat jembatan. |
| **Pernyataan Pemangku Kepentingan** | Pandangan langsung dari pengurus koperasi dan warga | Ketua Kelompok Tani (Pak Wawan): *"Truk pupuk 6 roda tidak bisa masuk ke jalan tanah dusun kalau musim hujan. KOPDES harus berdiri di pinggir Jalan Utama Desa yang beraspal keras."* |
| **Asumsi (Assumption)** | Penyederhanaan matematis rasional | Frekuensi belanja warga meluruh secara Gaussian terhadap jarak; jarak jalan kaki yang nyaman bagi ibu rumah tangga adalah $\le 300\text{ meter}$. |
| **Keputusan Terkendali (Decision Variable)** | Variabel yang ditetapkan oleh pengambil keputusan | Koordinat lokasi tapak KOPDES $[x, y]$, jarak mundur bangunan dari as jalan ($d_{\text{road}} \le 50\text{ m}$), kapasitas gudang pupuk dan sembako. |
| **Ketidakpastian Lingkungan (Uncertainty)** | Variabel fluktuatif di luar kendali pengurus | Jadwal pengiriman pasokan pupuk subsidi dari distributor resmi; curah hujan ekstrem yang merusak kondisi jalan tanah. |

### 3.2 Sesi Observasi dan Log Pembersihan Data
Observasi dilakukan dalam 3 sesi (total 210 menit) di lingkungan desa:
- **Sesi 1 (Pagi, 06.30 - 08.00 WIB)**: Pencatatan arus motor warga dusun mengantar anak ke SDN 1 Sukamaju dan berbelanja ke Pasar Desa.
- **Sesi 2 (Siang, 11.30 - 13.00 WIB)**: Pengukuran geometri jalan, lebar bahu jalan, serta uji lintasan mobil pickup pengangkut barang.
- **Sesi 3 (Sore, 16.30 - 17.00 WIB)**: Observasi saat gerimis; jalan tanah lingkar dusun terbukti licin dan berlumpur, sedangkan Jalan Utama Desa tetap kering dan kokoh.

---

## BAB 4: FORMULASI MATEMATIKA MODEL OPTIMASI KOPDES

### 4.1 Vektor Keputusan (Decision Variables)
Model dirumuskan dengan vektor keputusan $\mathbf{x}$ yang mencakup 5 variabel operasional yang dapat dikontrol:
1. $x \in [0.0, 2000.0]\text{ meter}$: Posisi horizontal tapak KOPDES.
2. $y \in [0.0, 1500.0]\text{ meter}$: Posisi vertikal tapak KOPDES.
3. $d_{\text{corridor}} \in [0.0, 50.0]\text{ meter}$: Jarak fisik tapak KOPDES dari garis sumbu jalan desa terdekat.
4. $R_{\text{layanan}} \in [200.0, 600.0]\text{ meter}$: Radius jangkauan pengantaran sembako langsung ke rumah warga.
5. $A_{\text{gudang}} \in [60.0, 150.0]\text{ m}^2$: Luas lantai bangunan KOPDES (ruang display toko sembako + gudang pupuk).

### 4.2 Fungsi Objektif Skalar (Composite Objective Function)
Tujuan optimasi adalah memaksimalkan Indeks Kesesuaian Lokasi Koperasi Desa $\mathcal{F}(\mathbf{x}) \in [0, 1]$ yang dirumuskan secara aditif berbasis 4 Fitur Kebugaran Berbobot (*Weight Fitting Features*):

$$\max_{\mathbf{x}} \mathcal{F}(\mathbf{x}) = w_1 S_{\text{pop}}(\mathbf{x}) + w_2 S_{\text{road}}(\mathbf{x}) + w_3 S_{\text{fac}}(\mathbf{x}) + w_4 S_{\text{comp}}(\mathbf{x}) - \text{Penalti}(\mathbf{x})$$

Dengan total bobot ternormalisasi:
$$\sum_{k=1}^4 w_k = w_1 + w_2 + w_3 + w_4 = 0.35 + 0.25 + 0.20 + 0.20 = 1.00$$

---

### 4.3 Rincian Rumus Matematis 4 Fitur Kebugaran Berbobot (*Weight Fitting Features*)

#### 1. Fitur 1: Gravitasi Kepadatan Pemukiman Warga Desa ($S_{\text{pop}}$, Bobot $w_1 = 0.35$)
Fitur ini mengukur kemudahan akses bagi 520 kepala keluarga (KK) warga desa yang tersebar di 50 klaster perumahan dusun. Daya tarik belanja sembako meluruh secara eksponensial Gaussian terhadap jarak tempuh:

$$\text{pop\_val}(\mathbf{x}) = \sum_{i=1}^{N_{\text{house}}} h_i \cdot \exp\left( - \frac{\|\mathbf{x} - \mathbf{h}_i\|^2}{2 \sigma_{\text{pop}}^2} \right)$$

Di mana:
- $\mathbf{h}_i = (x_i, y_i)$ adalah koordinat pusat klaster rumah warga ke-$i$ ($i = 1, 2, \dots, 50$).
- $h_i$ adalah jumlah kepala keluarga (KK) pada klaster ke-$i$ ($h_i \in [4, 18]\text{ KK}$, total $\sum h_i = 520\text{ KK}$).
- $\sigma_{\text{pop}} = 250.0\text{ meter}$ adalah standar deviasi radius jalan kaki warga yang nyaman.
- Fungsi dinormalisasi ke skala $[0.0, 1.0]$:
  $$S_{\text{pop}}(\mathbf{x}) = \operatorname{clip}\left( \frac{\text{pop\_val}(\mathbf{x})}{M_{\text{pop}}}, 0.0, 1.0 \right)$$
  dengan nilai kalibrasi maksimum $M_{\text{pop}} = \max_{j} \left( \sum_{i=1}^{N_{\text{house}}} h_i \exp\left( - \frac{\|\mathbf{h}_j - \mathbf{h}_i\|^2}{2\sigma_{\text{pop}}^2} \right) \right)$ yang diperoleh tepat di pusat massa kepadatan pemukiman desa.

#### 2. Fitur 2: Kondisi & Kelayakan Jalan Distribusi Pupuk & Logistik ($S_{\text{road}}$, Bobot $w_2 = 0.25$)
KOPDES memerlukan akses jalan yang memadai untuk armada truk 6 roda pengangkut pupuk bersubsidi dan mobil pickup sembako. Fitur ini menggabungkan jarak ortogonal ke garis sumbu jalan dengan koefisien kualitas fisik perkerasan jalan:

$$S_{\text{road}}(\mathbf{x}) = K_{\text{class}}(\mathbf{x}) \cdot \exp\left( - \frac{d_{\text{road}}(\mathbf{x})^2}{2 \sigma_{\text{road}}^2} \right)$$

Di mana:
- $d_{\text{road}}(\mathbf{x}) = \min_{s \in \text{RoadNet}} \operatorname{dist}(\mathbf{x}, s)$ adalah jarak ortogonal terpendek dari tapak KOPDES ke segmen jalan terdekat.
- $\sigma_{\text{road}} = 150.0\text{ meter}$ adalah konstanta dispersi aksesibilitas jalan.
- $K_{\text{class}}(\mathbf{x})$ adalah koefisien kualitas dan kelayakan kelas jalan terdekat:
  $$K_{\text{class}}(\mathbf{x}) = \begin{cases} 
  1.00, & \text{Jalan Arteri / Lintas Provinsi (Aspal hotmix tebal dua arah, kapasitas truk besar)} \\ 
  0.85, & \text{Jalan Kolektor / Jalan Utama Desa (Aspal mulus kokoh, jalur utama warga \& truk pupuk)} \\ 
  0.60, & \text{Jalan Lokal Dusun (Paving block / aspal sempit antar lingkungan perumahan)} \\ 
  0.20, & \text{Jalan Hauling / Tanah Lumpur (Licin, becek berlumpur saat hujan, diskon kelayakan 80\%)} 
  \end{cases}$$
- Jika tapak KOPDES berada di tepi jalan tanah/hauling lingkar dusun, nilai $S_{\text{road}}$ terpotong drastis menjadi maksimal $0.20$, sehingga algoritma secara inheren memprioritaskan Jalan Utama Desa beraspal mulus ($K = 0.85$).

#### 3. Fitur 3: Sinergi Fasilitas Publik & Pusat Keramaian Desa ($S_{\text{fac}}$, Bobot $w_3 = 0.20$)
Pembangunan KOPDES di dekat fasilitas publik eksisting menciptakan efek sinergi (*trip-chaining*), di mana warga yang mengantar anak ke sekolah atau berbelanja di pasar desa dapat sekaligus mampir ke koperasi:

$$\text{fac\_val}(\mathbf{x}) = \sum_{k \in \mathcal{K}} \beta_k \sum_{j=1}^{M_k} f_{k,j} \cdot \exp\left( - \frac{\|\mathbf{x} - \mathbf{p}_{k,j}\|^2}{2 \sigma_{\text{fac}}^2} \right)$$

Di mana:
- $\mathbf{p}_{k,j}$ adalah koordinat fasilitas ke-$j$ dalam kategori $k$.
- $f_{k,j}$ adalah bobot intensitas fasilitas publik.
- $\sigma_{\text{fac}} = 200.0\text{ meter}$ adalah radius pengaruh tarikan fasilitas desa.
- $\beta_k$ adalah koefisien bobot daya tarik fasilitas desa:
  - $\beta_{\text{pasar}} = 2.0$ (Pasar Desa: magnet aktivitas ekonomi dan perputaran uang harian terbesar).
  - $\beta_{\text{sekolah}} = 1.5$ (SDN Sukamaju 1: titik kumpul pagi hari orang tua murid).
  - $\beta_{\text{restoran/warung}} = 1.2$ (Warung makan warga: pergerakan kuliner lokal).
  - $\beta_{\text{masjid/kantor}} = 1.0$ (Masjid Jami' Desa dan Balai Pertemuan Dusun).
  - $\beta_{\text{bengkel}} = 0.8$ (Bengkel servis motor petani).
- Fungsi dinormalisasi ke rentang $[0.0, 1.0]$:
  $$S_{\text{fac}}(\mathbf{x}) = \operatorname{clip}\left( \frac{\text{fac\_val}(\mathbf{x})}{M_{\text{fac}}}, 0.0, 1.0 \right)$$
  dengan faktor kalibrasi $M_{\text{fac}} = 0.45 \times \sum_k \beta_k \sum_j f_{k,j}$.

#### 4. Fitur 4: Jarak Aman Kompetitor Toko Lama (*Harmonisasi Non-Kanibalisasi*) ($S_{\text{comp}}$, Bobot $w_4 = 0.20$)
KOPDES didirikan untuk menyejahterakan warga, bukan membunuh usaha toko kelontong swasta milik warga lama (Toko K di $X=860, Y=800$). Fitur ini memodelkan aglomerasi perdagangan yang sehat menggunakan kurva asimetris non-monotonik Ricker dengan penalti kanibalisasi jarak dekat:

Misalkan $d_c(\mathbf{x}) = \min_m \|\mathbf{x} - \mathbf{c}_m\|$ adalah jarak terdekat ke toko kelontong warga lama, dan $R = \frac{d_c(\mathbf{x})}{d_{\text{opt}}}$ dengan jarak optimal $d_{\text{opt}} = 350.0\text{ meter}$. Fungsi dasar Ricker:

$$S_{\text{base}}(R) = R^{1.6} \cdot \exp\left( 1.0 - R^{1.6} \right)$$

Dengan fungsi modulasi kanibalisasi pada jarak kritis ($d_c < 180.0\text{ meter}$):

$$S_{\text{comp}}(\mathbf{x}) = \begin{cases} 
S_{\text{base}}(R) \cdot \left( \dfrac{d_c(\mathbf{x})}{180.0} \right)^{1.8}, & \text{jika } d_c(\mathbf{x}) < 180.0\text{ meter} \\ 
S_{\text{base}}(R), & \text{jika } d_c(\mathbf{x}) \ge 180.0\text{ meter} 
\end{cases}$$

- **Pada jarak optimal ($d_c = 350\text{ m}$)**: $R = 1.0 \implies S_{\text{comp}} = 1.0 \cdot \exp(0) = 1.00$ (terjadi aglomerasi klaster belanja saling menguntungkan).
- **Pada jarak sangat dekat ($d_c < 180\text{ m}$)**: Nilai $S_{\text{comp}}$ runtuh drastis menuju $0.0$ karena perpangkatan $1.8$, mencegah KOPDES mematikan toko tetangga.
- **Pada jarak sangat jauh ($d_c > 1000\text{ m}$)**: Kurva meluruh landai secara asimtotik ke nilai moderat karena hilangnya keuntungan aglomerasi pasar.

---

### 4.4 Batasan-Batasan Spasial Riil Desa (Constraints) & Penanganan Penalti Diskontinu

Model diikat oleh 7 batasan fisik dan yuridis di dalam wilayah desa:
1. **$g_1$ (Batas Spasial Wilayah Desa)**: $0.0 \le x \le 2000.0\text{ m}$ dan $0.0 \le y \le 1500.0\text{ m}$.
2. **$g_2$ (Zona Sawah Irigasi Produktif)**: $\mathbf{x} \notin \text{Polygon}_{\text{sawah}}$ (Perda Perlindungan LP2B).
3. **$g_3$ (Zona Sempadan Sungai Sukamaju)**: $\mathbf{x} \notin \text{Polygon}_{\text{sungai}}$ (Garis sempadan rawan longsor/banjir bandang).
4. **$g_4$ (Zona Hutan Lindung Adat)**: $\mathbf{x} \notin \text{Polygon}_{\text{hutan}}$ (Kawasan konservasi resapan air perbukitan utara).
5. **$g_5$ (Zona Galian C / Tambang)**: $\mathbf{x} \notin \text{Polygon}_{\text{tambang}}$ (Area berbahaya operasi alat berat).
6. **$g_6$ (Koridor Fisik Jalan Akses)**: $d_{\text{road}}(\mathbf{x}) \le d_{\max} = 50.0\text{ meter}$ (KOPDES wajib dibangun di sisi jalan desa agar truk dan motor dapat parkir/bongkar muat).
7. **$g_7$ (Batas Minimum Toko Kelontong Warga)**: $d_c(\mathbf{x}) \ge 180.0\text{ meter}$.

Pengecekan zona poligon $g_2 - g_5$ dilakukan secara eksak menggunakan algoritma **Ray-Casting Vectorized NumPy**.

Fungsi evaluasi kebugaran global dengan penalti diskontinu (*death penalty*):
$$\mathcal{F}_{\text{final}}(\mathbf{x}) = \begin{cases} 
-1000.0, & \text{jika } \mathbf{x} \text{ melanggar batasan wilayah } (g_1) \text{ atau zona terlarang } (g_2 - g_5) \\ 
-10.0 - \dfrac{d_{\text{road}}(\mathbf{x}) - 50.0}{50.0}, & \text{jika } d_{\text{road}}(\mathbf{x}) > 50.0\text{ m (melanggar koridor fisik jalan } g_6) \\ 
\mathcal{F}(\mathbf{x}), & \text{jika seluruh batasan terpenuhi (Solusi Layak / Feasible, } \mathcal{F} \in [0, 1]) 
\end{cases}$$

Dengan skema penalti bertingkat ini, individu atau partikel yang berada di sawah irigasi atau jurang sungai langsung tereliminasi dari pencarian, sementara solusi di luar koridor jalan diarahkan kembali menuju sumbu jalan terdekat.

---

## BAB 5: DESAIN DAN IMPLEMENTASI ALGORITMA GA DAN PSO

### 5.1 Protokol Komputasi Adil (Matched Budget)
Kedua algoritma diimplementasikan dari nol (**from scratch**) tanpa library pihak ketiga dengan aturan ketat:
- **Shared Evaluator**: Objek `FitnessEvaluator` tunggal yang mengendalikan seluruh evaluasi matematika dan geometri.
- **Anggaran Evaluasi Sama Persis**: Tepat **2.000 kali evaluasi fungsi fitness** per pengujian (`eval_count >= 2000`).

### 5.2 Parameter Algoritma Genetika (GA)
- Representasi: Real-Coded Chromosome $[x, y]$.
- Ukuran Populasi: 40 individu.
- Seleksi: Tournament Selection ($k = 3$).
- Crossover: Blend Crossover (BLX-$\alpha$, $\alpha = 0.5$, $P_c = 0.85$).
- Mutasi: Adaptive Gaussian Mutation ($\sigma = 40.0\text{ m}$, $P_m = 0.15$).
- Elitisme: 2 individu terbaik dipertahankan utuh.

### 5.3 Parameter Particle Swarm Optimization (PSO)
- Jumlah Partikel: 40 partikel.
- Inersia Adaptif: Linear decay dari $w = 0.9$ (eksplorasi awal) ke $w = 0.4$ (eksploitasi presisi).
- Koefisien Kognitif & Sosial: $c_1 = 1.494$ ($p_{\text{best}}$), $c_2 = 1.494$ ($g_{\text{best}}$).
- Clamping Kecepatan: $v_{\max} = 15\%$ rentang wilayah.
- Penanganan Batas: Boundary reflection.

### 5.4 Antarmuka GUI Tkinter Mode Terpadu (Split Screen & Full Screen)
Seluruh proses optimasi dieksekusi secara visual pada antarmuka GUI Desktop Tkinter (`app_tkinter.py`). Kanvas Tkinter menampilkan secara real-time dinamika populasi GA (termasuk mutasi acak dan elitisme) berdampingan dengan kawanan partikel PSO (termasuk personal best dan global best), serta kurva konvergensi live.

![Gambar 2: Antarmuka Simulasi Interaktif Tkinter Mode Berdampingan](results/tkinter_simulasi_split.png)
*Gambar 2: Antarmuka Simulasi Interaktif Tkinter Mode Berdampingan (Split View): Hasil GA (Kiri), Hasil PSO (Kanan), dan Kurva Konvergensi Live (Bawah).*

---

## BAB 6: DESAIN EKSPERIMEN, SKENARIO DESA, & UJI STATISTIK

### 6.1 Protokol 30 Run Independen
Setiap algoritma dijalankan 30 kali pengujian independen dengan random seed $1, 2, \dots, 30$ untuk menguji stabilitas stokastik.

### 6.2 Baseline Pembanding
1. **Baseline 1: Praktik Saat Ini (Usulan Informal Tanah Murah)**: Titik $(X = 1400, Y = 920)$ di tepi jalan tanah timur.
2. **Baseline 2: Heuristik Sederhana (Sentroid Massa Pemukiman)**: Titik $(X = 280, Y = 750)$ di tengah lingkaran pemukiman warga desa.

### 6.3 Tiga Skenario Ketahanan Desa
1. **Skenario 1 (Kondisi Observasi Normal Desa)**: Bobot standar.
2. **Skenario 2 (Musim Tanam Raya / Kebutuhan Pupuk Meningkat)**: Bobot akses jalan distribusi dinaikkan menjadi $0.35$ dan fasilitas menjadi $0.25$.
3. **Skenario 3 (Musim Hujan Lebat / Risiko Banjir)**: Jalan tanah lingkar tidak dapat dilalui, sempadan sungai diperketat.

---

## BAB 7: HASIL KOMPUTASI DAN ANALISIS KRITIS

### 7.1 Tabel Rangkuman Hasil 30 Run Independen

| Metrik Evaluasi | Genetic Algorithm (GA) | Particle Swarm Optimization (PSO) | Analisis Evaluasi |
|---|---|---|---|
| **Nilai Fitness Terbaik (Max)** | **0.785148** | **0.785148** | Identik hingga 6 desimal |
| **Nilai Fitness Rata-rata (Mean)** | 0.782871 | **0.785148** | PSO lebih konsisten |
| **Median Fitness** | **0.785148** | **0.785148** | Identik |
| **Standar Deviasi (Std)** | 0.012471 | **0.000000** ($4.39 \times 10^{-10}$) | PSO memiliki stabilitas sempurna |
| **Interquartile Range (IQR)** | $1.02 \times 10^{-10}$ | **$8.69 \times 10^{-11}$** | Sebaran sangat terpusat |
| **Fitness Terburuk (Min)** | 0.716841 | **0.785148** | GA sempat 1x terjebak sub-optimal |
| **Persentase Solusi Layak** | **100.0% (30/30)** | **100.0% (30/30)** | Keduanya 100% feasible |
| **Rata-rata Waktu Run (detik)** | 1.897 ± 0.917 s | **0.624 ± 0.096 s** | PSO 3.0× lebih cepat |
| **Median Eval ke-95% Konvergensi** | **192 Evaluasi** | 200 Evaluasi | GA sedikit lebih cepat di awal |
| **Titik Rekomendasi Lokasi KOPDES** | **$(X = 295.4\text{ m}, Y = 753.8\text{ m})$** | **$(X = 295.4\text{ m}, Y = 753.8\text{ m})$** | Konvergen ke titik fisik yang sama |

### 7.2 Uji Hipotesis Statistik Mann-Whitney U
- **Nilai U**: $654.00$
- **p-value**: $0.00261 < 0.05$ (**Signifikan secara statistik**)
- **Rank-Biserial Correlation ($r$)**: $-0.453$ (Efek sedang hingga kuat)
- **Kesimpulan Statistik**: PSO secara signifikan lebih stabil dibanding GA karena tidak mengalami satupun run yang melenceng ke optimum lokal sekunder. Namun pada 29 dari 30 run lainnya, kedua algoritma memberikan titik rekomendasi yang identik.

### 7.3 Perbandingan terhadap Praktik Saat Ini & Heuristik
- **Praktik Saat Ini (Usulan Tanah Kas Murah di Timur)**: Fitness hanya **$0.4120$**.
- **Heuristik Sederhana (Sentroid Pemukiman)**: Fitness **$0.6350$**.
- **Solusi Optimal GA & PSO**: Fitness **$0.7851$** (**+90.5%** peningkatan dibanding praktik saat ini, **+23.6%** dibanding heuristik sentroid).

### 7.4 Tangkapan Layar Visualisasi Simulasi Tkinter (Layar Penuh GA, PSO, & Kurva Konvergensi)

Berikut merupakan hasil tangkapan visualisasi langsung dari antarmuka desktop Tkinter pada akhir iterasi komputasi (2.000 evaluasi):

![Gambar 3: Hasil Optimasi Algoritma Genetika Layar Penuh pada Tkinter](results/tkinter_simulasi_ga_full.png)
*Gambar 3: Hasil Optimasi Algoritma Genetika (GA) Layar Penuh pada Kanvas Tkinter: Sebaran Populasi Kromosom, Mutasi, dan Rekomendasi Titik KOPDES.*

![Gambar 4: Hasil Optimasi Particle Swarm Optimization Layar Penuh pada Tkinter](results/tkinter_simulasi_pso_full.png)
*Gambar 4: Hasil Optimasi Particle Swarm Optimization (PSO) Layar Penuh pada Kanvas Tkinter: Partikel Swarm, pbest, dan gbest KOPDES.*

![Gambar 5: Kurva Konvergensi Live Matched Budget pada Tkinter](results/tkinter_simulasi_conv_full.png)
*Gambar 5: Kurva Konvergensi Live Matched Budget 2.000 Evaluasi GA vs PSO pada Kanvas Tkinter.*

---

## BAB 8: VALIDASI LAPANGAN, RISIKO OPERASIONAL, ETIKA, & KESIMPULAN

### 8.1 Validasi Lapangan Titik Rekomendasi KOPDES
Tim melakukan peninjauan langsung ke titik rekomendasi $(X = 295.4\text{ m}, Y = 753.8\text{ m})$ di tepi Jalan Utama Desa:
- **Kondisi Fisik Nyata**: Merupakan pekarangan kosong datar di tepi aspal Jalan Utama Desa (lebar 5.2 meter), berjarak $65\text{ meter}$ dari SDN Sukamaju 1 dan $140\text{ meter}$ dari Pasar Desa.
- **Tanggapan Kepala Desa & Pengurus KOPDES**:
  *"Titik ini sangat sempurna untuk Koperasi Desa. Jalan Utama Desa ini dilewati seluruh warga setiap pagi. Truk pengangkut pupuk dari gudang kabupaten bisa langsung parkir dan bongkar muat tanpa mengganggu lalu lintas. Lokasinya juga aman dari banjir sungai dan tidak menggusur sawah produktif."*

### 8.2 Analisis Risiko Implementasi
1. **Akuisisi / Sewa Lahan Pekarangan**: Lahan merupakan milik warga perorangan; pengurus KOPDES dapat menerapkan skema sewa 10 tahun atau penyertaan modal saham koperasi desa.
2. **Akses Masuk & Bongkar Muat Truk**: Dibuat jalan masuk (*ramp*) beton selebar 4 meter agar truk pengangkut pupuk dapat mundur langsung ke pintu gudang belakang KOPDES.
3. **Kemitraan Toko Tradisional**: KOPDES tidak menjual eceran barang yang sama dengan toko kelontong Pak Hadi secara predator, melainkan fokus pada barang grosir sembako, pupuk subsidi, dan penampungan hasil bumi warga.

### 8.3 Kesimpulan Akhir
1. Pendekatan optimasi metaheuristik berhasil memecahkan masalah penentuan lokasi KOPDES di dalam 1 wilayah desa secara objektif, adil, dan terbukti matematis.
2. Kedua algoritma (GA dan PSO) menghasilkan koordinat rekomendasi yang identik: **$(X = 295.4\text{ m}, Y = 753.8\text{ m})$** dengan tingkat kepatuhan batasan 100%.
3. PSO lebih unggul dalam kecepatan eksekusi (3× lebih cepat) dan stabilitas stokastik, sementara GA memiliki keunggulan eksplorasi yang tangguh.
4. Solusi ini memberikan peningkatan efektivitas penempatan sebesar **+90.5%** dibanding rencana informal sebelumnya, siap diajukan ke Musyawarah Rencana Pembangunan Desa (Musrenbangdes).

---

## LAMPIRAN A: PROBLEM PROPOSAL TEMPLATE (FORMULIR PENGAJUAN MASALAH KOPDES)

```text
================================================================================
           PROBLEM APPROVAL GATE - PROPOSAL RESMI PENEMPATAN KOPDES
================================================================================
1. LOKASI PENELITIAN / SITE IDENTIFIER:
   Wilayah Desa Sukamaju Terpadu (Koordinat Relatif: 0.0, 0.0 s/d 2000.0, 1500.0 meter).

2. PROSES YANG DIAMATI (OBSERVED PROCESS):
   Aktivitas distribusi kebutuhan harian sembako 520 KK warga desa, penyaluran pupuk 
   pertanian sawah produktif, dan pergerakan belanja warga di sekitar pasar dan sekolah.

3. PEMILIK KEPUTUSAN (DECISION OWNER):
   Ketua Koperasi Desa (KOPDES) Sukamaju Mandiri, Kepala Desa, dan Ketua BPD.

4. MASALAH TERAMATI DAN BUKTI PRELIMINER:
   KOPDES belum memiliki gedung sendiri dan berencana dibangun secara informal di 
   pinggiran timur desa (X=1400, Y=920) karena tanah murah. Lokasi tersebut terbukti 
   becek, berdebu, jauh dari 85% warga desa, dan tidak bisa dilalui truk pupuk.

5. ALASAN CARA SAAT INI TIDAK MEMADAI:
   Pengambilan keputusan intuitif mengabaikan multi-kriteria spasial: jarak jalan kaki 
   warga, kelas jalan aspal vs tanah, sinergi pasar dan balai desa, serta batas sawah.

6. VARIABEL KEPUTUSAN YANG DIUSULKAN:
   x in [0, 2000] m, y in [0, 1500] m (Koordinat tapak KOPDES),
   d_corridor in [0, 50] m (Jarak ke sumbu jalan), R_layanan in [200, 600] m,
   A_gudang in [60, 150] m2 (Luas bangunan toko dan gudang pupuk KOPDES).

7. FUNGSI OBJEKTIF DAN SATUAN PENGUKURAN:
   Maksimasi F(x) = Indeks Kesesuaian Lokasi Koperasi Desa (Skala Kontinu [0, 1]).

8. BATASAN-BATASAN RIIL DESA (CONSTRAINTS):
   - g1: Batas fisik wilayah desa (2.0 km x 1.5 km).
   - g2-g5: Dilarang membangun di sawah irigasi produktif (Perda LP2B), sempadan 
     sungai, hutan lindung adat, dan area galian tambang (Algoritma Ray-Casting).
   - g6: Wajib di koridor jalan desa (Jarak ke sumbu jalan <= 50 meter).
   - g7: Jarak aman non-kanibalisasi toko kelontong warga lama (>= 180 meter).

9. TRADE-OFF ATAU SUMBER KESULITAN:
   Mendekati pemukiman warga barat versus aksesibilitas truk logistik; menghindari 
   sawah produktif; menjaga keharmonisan jarak dengan warung kelontong swasta lama.

10. MENGAPA TIDAK ADA SOLUSI ONLINE YANG MENJAWAB MASALAH LOKAL INI:
    Data tata ruang desa, sebaran 50 klaster rumah, jaringan jalan desa heterogen, 
    dan batas zonasi adat bersifat primer dan spesifik pada Desa Sukamaju ini.
================================================================================
```

---

## LAMPIRAN B: CATATAN OBSERVASI LAPANGAN & CLEANING LOG

| Sesi | Hari & Waktu | Kondisi Lapangan | Parameter yang Diukur | Anomali & Tindakan Pembersihan Data |
|---|---|---|---|---|
| **1** | Senin, 22 Sept 2026<br>06.30 - 08.00 WIB (90 mnt) | Cerah berawan, jam sibuk pagi keberangkatan warga | - Arus motor Jl. Utama Desa: 84 kend/jam<br>- Arus pejalan kaki sekolah: 120 siswa/jam<br>- Titik kumpul warga di pasar desa | Ditemukan pasar tumpah dadakan memakan 1 meter bahu jalan barat dekat pertigaan dusun; dicatat sebagai zona penyempitan jalan. |
| **2** | Rabu, 24 Sept 2026<br>11.30 - 13.00 WIB (90 mnt) | Terik panas berdebu, uji lintasan mobil pickup | - Lebar perkerasan Jl. Utama Desa: 5.2 m<br>- Arus kendaraan per jam: 142 kend/jam<br>- Radius putar truk di persimpangan | Titik GPS mess tambang 4 bergeser ke dinding tebing akibat multipath error; dikoreksi manual dengan citra ortofoto tegak. |
| **3** | Jumat, 26 Sept 2026<br>16.30 - 17.00 WIB (30 mnt) | Hujan gerimis rintik, jalan tanah lembek | - Pembeli toko kelontong lama: 18 org/jam<br>- Kecepatan motor di jalan tanah turun ke <15 km/jam | Jalan tanah lingkar dusun terbukti licin dan becek; memvalidasi keputusan memberikan bobot kelayakan rendah (0.20) untuk jalan tanah. |

---

## LAMPIRAN C: FORMULIR PENGUNGKAPAN PENGGUNAAN AI (AI DISCLOSURE TABLE)

| Alat & Tanggal | Tujuan Pemrograman | Ringkasan Prompt Mahasiswa | Kode Dihasilkan & Verifikasi |
|---|---|---|---|
| **Gemini / Antigravity**<br>28 Sept 2026 | Fungsi Ray-Casting poligon tertutup zona terlarang | *"Buatkan fungsi ray-casting vectorized NumPy untuk mengecek apakah titik (N, 2) berada di dalam poligon sawah dan sungai."* | Fungsi `point_in_polygon_vectorized` di `map_model.py`; diuji manual dengan 10 titik uji kontrol. |
| **Gemini / Antigravity**<br>29 Sept 2026 | Debugging counter Matched Budget 2.000 evaluasi | *"Bagaimana memastikan loop GA dan PSO berhenti tepat pada 2.000 panggilan fungsi fitness?"* | Penambahan `self.eval_count` di `fitness.py`; diverifikasi melalui unit test `test_facility.py::test_evaluation_counter`. |
| **Gemini / Antigravity**<br>30 Sept 2026 | Styling visualisasi peta dan jalan arteri Tkinter | *"Perbaiki ketebalan dan garis marka Jalan Lintas Provinsi pada Matplotlib agar proporsional dan realistis."* | Pengaturan linewidth aspal 22.0/17.5 pt dan garis putus 1.5 pt di `map_model.py`; diverifikasi pada GUI desktop. |

---

## LAMPIRAN D: PANDUAN TANYA-JAWAB UJIAN LISAN (ORAL DEFENSE PROMPTS)

#### 1. Konversi Observasi Mentah Menjadi Parameter Model KOPDES:
> **Jawaban:** Observasi lapangan Sesi 3 membuktikan bahwa saat hujan gerimis, jalan tanah lingkar dusun menjadi kubangan lumpur licin yang tidak bisa dilalui motor warga maupun truk pupuk KOPDES. Fakta ini dikonversi menjadi koefisien kelayakan jalan $K_{\text{hauling/tanah}} = 0.20$ (diskon kelayakan 80% dibanding Jalan Utama Desa beraspal $K = 0.85$ dan Arteri $1.00$) pada fungsi kebugaran $S_{\text{road}}$, sehingga algoritma secara otomatis menghindari penempatan KOPDES di jalan tanah tersebut.

#### 2. Perbedaan Variabel Terkendali vs Ketidakpastian Lingkungan:
> **Jawaban:** Koordinat fisik gedung KOPDES $[X, Y]$ dan jarak mundur dari as jalan adalah **variabel terkendali** karena diputuskan langsung oleh pengurus koperasi dan kepala desa. Sebaliknya, curah hujan ekstrem, fluktuasi harga pupuk bersubsidi, dan kebijakan kuota pupuk dari pemerintah pusat adalah **ketidakpastian lingkungan** di luar kendali pengurus koperasi.

#### 3. Penelusuran Kebugaran Titik di Sawah Produktif:
> **Jawaban:** Jika kandidat solusi jatuh di sawah irigasi $(X = 200, Y = 200)$, fungsi Ray-Casting mendeteksi titik berada di dalam poligon sawah. Variabel `is_forbid` bernilai `True`. Fungsi evaluasi langsung memotong nilai dengan penalti diskontinu $-1000.0$ tanpa membuang waktu menghitung fitur lain, sehingga individu tersebut langsung tereliminasi pada seleksi genetik berikutnya.

#### 4. Dampak Jika Koefisien Penalti Dibagi Sepuluh ($-1000 \to -100$):
> **Jawaban:** Terjadi kebocoran penalti (*penalty leakage*). Nilai gravitasi populasi warga desa yang sangat padat ($+0.35$) dapat mengimbangi penalti kecil, sehingga algoritma berisiko merekomendasikan KOPDES dibangun di tengah hamparan sawah produktif beririgasi teknis yang melanggar hukum. Skala $-1000$ menjamin solusi ilegal selalu kalah.

#### 5. Kapan GA Unggul dan Kapan PSO Unggul pada Masalah KOPDES Ini?:
> **Jawaban:** PSO unggul pada permukaan kontinu bergradien halus di sepanjang Jalan Utama Desa karena komunikasi global ($g_{\text{best}}$) memandu kawanan meluncur cepat dan presisi (terbukti 3× lebih cepat). GA unggul jika desa memiliki beberapa dusun terpencil yang dipisahkan oleh sungai tanpa jembatan, di mana operator mutasi acak dan crossover BLX-$\alpha$ mampu "melompati" batas rintangan.

#### 6. Mengapa Objektif Terbaik Naik tapi Feasible % Bisa Turun?:
> **Jawaban:** Terjadi fenomena *boundary attraction*: algoritma menemukan skor sangat tinggi tepat di batas bibir sawah atau sempadan sungai. Namun akibat variasi mutasi atau inersia partikel, sebagian besar populasi terlempar melewati batas ke zona terlarang, sehingga persentase solusi layak turun meskipun nilai terbaik individu naik.

#### 7. Prediksi Perilaku Jika Batasan Koridor Jalan ($\le 50\text{ m}$) Dimatikan:
> **Jawaban:** Solusi akan langsung melompat ke tengah-tengah halaman rumah warga desa $(X \approx 280, Y \approx 765)$ tanpa akses jalan mobil, karena nilai gravitasi populasi warga $S_{\text{pop}}$ mencapai nilai puncak tertinggi di titik tersebut.

#### 8. Kesimpulan yang Didukung vs Tidak Didukung Data:
> **Jawaban:** 
> - **Didukung Data:** PSO secara statistik signifikan lebih stabil (standar deviasi $4.39 \times 10^{-10}$) dan 3× lebih efisien dalam waktu eksekusi dibanding GA pada penugasan 2.000 evaluasi.
> - **TIDAK Didukung Data:** Pernyataan bahwa *"GA gagal menemukan lokasi KOPDES terbaik"* adalah keliru, karena pada 29 dari 30 run GA menemukan titik fisik yang sama persis $(X = 295.4\text{ m}, Y = 753.8\text{ m})$ dengan nilai kebugaran $0.7851$.
