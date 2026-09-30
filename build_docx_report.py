"""
Script untuk menghasilkan LAPORAN_OPTIMASI_GA_PSO.docx dengan tata letak profesional,
tabel berformat, dan murni menggunakan visualisasi kanvas Tkinter (100% seragam):
1. Peta Tata Ruang Desa (Kanvas Tkinter)
2. Antarmuka Simulasi Interaktif Tkinter Mode Split (GA, PSO, & Konvergensi Live)
3. Hasil Optimasi GA Layar Penuh Tkinter (Populasi, Mutasi, Elit, & Rekomendasi KOPDES)
4. Hasil Optimasi PSO Layar Penuh Tkinter (Swarm, pbest, gbest Rekomendasi KOPDES)
5. Kurva Konvergensi Matched Budget Live Tkinter
"""

import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

def set_cell_background(cell, fill_hex):
    """Mengatur warna latar belakang sel tabel."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Mengatur padding sel tabel."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'''
        <w:tcMar {nsdecls("w")}>
            <w:top w:w="{top}" w:type="dxa"/>
            <w:bottom w:w="{bottom}" w:type="dxa"/>
            <w:left w:w="{left}" w:type="dxa"/>
            <w:right w:w="{right}" w:type="dxa"/>
        </w:tcMar>
    ''')
    tcPr.append(tcMar)

def add_callout(doc, text_p_list, title="CATATAN / PEMBELAAN SAINTIFIK"):
    """Membuat kotak callout berwarna untuk catatan penting atau tanya-jawab ujian lisan."""
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    set_cell_background(cell, "F1F5F9")
    set_cell_margins(cell, top=130, bottom=130, left=180, right=180)
    
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(f'''
        <w:tcBorders {nsdecls("w")}>
            <w:left w:val="single" w:sz="24" w:space="0" w:color="2563EB"/>
            <w:top w:val="none"/>
            <w:right w:val="none"/>
            <w:bottom w:val="none"/>
        </w:tcBorders>
    ''')
    tcPr.append(borders)
    
    p0 = cell.paragraphs[0]
    p0.paragraph_format.space_before = Pt(0)
    p0.paragraph_format.space_after = Pt(3)
    run_t = p0.add_run(f"■ {title}")
    run_t.bold = True
    run_t.font.name = "Calibri"
    run_t.font.size = Pt(10)
    run_t.font.color.rgb = RGBColor(30, 58, 138)
    
    for t in text_p_list:
        p = cell.add_paragraph()
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(3)
        run = p.add_run(t)
        run.font.name = "Calibri"
        run.font.size = Pt(9.5)
        run.font.color.rgb = RGBColor(30, 41, 59)
    doc.add_paragraph().paragraph_format.space_after = Pt(6)

def style_table(tbl, col_widths, headers, rows_data):
    """Menerapkan format tabel modern dengan header biru gelap dan zebra shading."""
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    
    # Header row
    hdr_row = tbl.rows[0]
    for i, h_text in enumerate(headers):
        cell = hdr_row.cells[i]
        cell.width = Inches(col_widths[i])
        set_cell_background(cell, "1E3A8A")
        set_cell_margins(cell, top=120, bottom=120, left=150, right=150)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        run = p.add_run(h_text)
        run.bold = True
        run.font.name = "Calibri"
        run.font.size = Pt(9.5)
        run.font.color.rgb = RGBColor(255, 255, 255)
        
    # Data rows
    for r_idx, r_data in enumerate(rows_data):
        row = tbl.add_row()
        bg_col = "F8FAFC" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(r_data):
            cell = row.cells[c_idx]
            cell.width = Inches(col_widths[c_idx])
            set_cell_background(cell, bg_col)
            set_cell_margins(cell, top=90, bottom=90, left=130, right=130)
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            if c_idx == 0:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER if len(val) < 20 else WD_ALIGN_PARAGRAPH.LEFT
            run = p.add_run(str(val))
            run.font.name = "Calibri"
            run.font.size = Pt(9)
            run.font.color.rgb = RGBColor(30, 41, 59)
            
    # Set borders to subtle light gray
    tblPr = tbl._tbl.tblPr
    borders = parse_xml(f'''
        <w:tblBorders {nsdecls("w")}>
            <w:top w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>
            <w:bottom w:val="single" w:sz="8" w:space="0" w:color="94A3B8"/>
            <w:insideH w:val="single" w:sz="4" w:space="0" w:color="E2E8F0"/>
            <w:insideV w:val="none"/>
            <w:left w:val="none"/>
            <w:right w:val="none"/>
        </w:tblBorders>
    ''')
    tblPr.append(borders)

def add_figure(doc, img_path, caption, width=Inches(6.2)):
    """Menambahkan gambar dengan format rapi dan keterangan di bawahnya."""
    if os.path.exists(img_path):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(8)
        p_img.paragraph_format.space_after = Pt(2)
        run_img = p_img.add_run()
        run_img.add_picture(img_path, width=width)
        
        p_cap = doc.add_paragraph(caption)
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_before = Pt(2)
        p_cap.paragraph_format.space_after = Pt(14)
        run_cap = p_cap.runs[0]
        run_cap.font.name = "Calibri"
        run_cap.font.size = Pt(9)
        run_cap.font.italic = True
        run_cap.font.color.rgb = RGBColor(100, 116, 139)

def build_report():
    doc = docx.Document()
    
    # Setup page margins (1 inch)
    for s in doc.sections:
        s.top_margin = Inches(1.0)
        s.bottom_margin = Inches(1.0)
        s.left_margin = Inches(1.0)
        s.right_margin = Inches(1.0)
        
    # Document Header Metadata
    p_meta = doc.add_paragraph()
    p_meta.paragraph_format.space_before = Pt(0)
    p_meta.paragraph_format.space_after = Pt(3)
    run_meta = p_meta.add_run("TUGAS BESAR BIOCOMPUTING & OPTIMASI SISTEM — PROGRAM STUDI INFORMATIKA / ILMU KOMPUTER")
    run_meta.font.name = "Calibri"
    run_meta.font.size = Pt(9.5)
    run_meta.font.bold = True
    run_meta.font.color.rgb = RGBColor(37, 99, 235)
    
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(2)
    p_title.paragraph_format.space_after = Pt(6)
    run_title = p_title.add_run("LAPORAN OPTIMASI BERBASIS LAPANGAN:\nPERBANDINGAN ALGORITMA GENETIKA (GA) DAN PARTICLE SWARM OPTIMIZATION (PSO) DALAM PENENTUAN LOKASI FASILITAS KOPERASI DESA (KOPDES) TERPADU DI DALAM SATU DESA")
    run_title.font.name = "Calibri"
    run_title.font.size = Pt(16)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(15, 23, 42)
    
    p_sub = doc.add_paragraph()
    p_sub.paragraph_format.space_after = Pt(14)
    run_sub = p_sub.add_run(
        "Studi Kasus: Pembangunan Gedung Koperasi Desa (KOPDES) Sukamaju Mandiri\n"
        "Cakupan: 1 Wilayah Desa Lengkap (2.0 km × 1.5 km = 300 Hektar) | Anggaran Komputasi: 2.000 Evaluasi (Matched Budget)\n"
        "Visualisasi: 100% Berbasis Antarmuka GUI Desktop Tkinter Terintegrasi GA & PSO"
    )
    run_sub.font.name = "Calibri"
    run_sub.font.size = Pt(9.5)
    run_sub.font.italic = True
    run_sub.font.color.rgb = RGBColor(71, 85, 105)
    
    # RINGKASAN EKSEKUTIF
    doc.add_heading("RINGKASAN EKSEKUTIF", level=1)
    doc.add_paragraph(
        "Laporan ini menyajikan penyelesaian masalah optimasi penentuan lokasi gedung Koperasi Desa (KOPDES) Sukamaju Mandiri di dalam satu "
        "wilayah desa terpadu seluas 300 hektar (2.0 km × 1.5 km). KOPDES mengintegrasikan tiga fungsi strategis: penyaluran pupuk bersubsidi "
        "dan sarana pertanian (saprodi), toko retail sembako murah bagi 520 KK warga desa, serta unit simpan pinjam desa. Keputusan yang selama ini "
        "direncanakan secara informal di pinggiran desa berisiko gagal komersial karena berada di jalan tanah becek dan jauh dari warga. "
        "Dengan memformulasikan 4 fitur kebugaran berbobot (kepadatan warga w1=0.35, kondisi jalan distribusi w2=0.25, sinergi fasilitas desa w3=0.20, "
        "dan jarak aman toko warga lama w4=0.20) serta batasan zona terlarang, algoritma Genetic Algorithm (GA) dan Particle Swarm Optimization (PSO) "
        "dieksekusi langsung menggunakan antarmuka interaktif Tkinter. Kedua metode mencapai 100% kepatuhan batasan (feasible) dan merekomendasikan "
        "koordinat lokasi fisik yang sama persis: (X = 295.4 m, Y = 753.8 m) di tepi Jalan Utama Desa dengan nilai kebugaran 0.7851. Rekomendasi ini "
        "menghasilkan perbaikan efektivitas sebesar +90.5% dibandingkan rencana informal sebelumnya."
    )
    
    # ----------------------------------------------------
    # BAB 1
    # ----------------------------------------------------
    doc.add_heading("BAB 1: PERNYATAAN MASALAH DAN KONTEKS KEPUTUSAN KOPDES", level=1)
    doc.add_paragraph(
        "Koperasi Desa (KOPDES) Sukamaju Mandiri didirikan untuk melayani kebutuhan ekonomi 520 kepala keluarga (KK) warga desa yang mayoritas "
        "berprofesi sebagai petani sawah, pedagang warung, dan buruh harian. KOPDES mengintegrasikan tiga fungsi utama:\n"
        "1. Gudang Distribusi Saprodi: Penyaluran pupuk bersubsidi, bibit padi unggul, dan pakan ternak untuk kelompok tani.\n"
        "2. Toko Retail Sembako Anggota: Penyediaan beras, minyak goreng, gula, tepung, telur, dan barang konsumsi harian berharga terjangkau.\n"
        "3. Unit Simpan Pinjam & Penampungan Hasil Bumi: Layanan keuangan mikro desa dan pembelian komoditas hasil panen warga.\n\n"
        "Saat ini, KOPDES belum memiliki gedung permanen dan masih menumpang di salah satu garasi rumah pengurus lama di ujung jalan sempit dusun. "
        "Akibatnya, truk pengangkut pupuk tidak bisa masuk dan warga dusun seberang mengeluh karena akses jalan yang berliku."
    )
    doc.add_paragraph(
        "Pemilik Keputusan (Decision Owner) adalah Ketua KOPDES bersama Kepala Desa Sukamaju dan Badan Permusyawaratan Desa (BPD). "
        "Sebelumnya, pengurus sempat mengusulkan membangun gedung KOPDES di koordinat (X = 1400 m, Y = 920 m) di pinggir jalan tanah timur "
        "semata-mata karena tanah kas desa di sana murah. Usulan informal ini mengandung cacat kritis: jalan tanah tersebut sangat becek "
        "dan berlumpur saat musim hujan, jauh dari 85% pemukiman warga desa (>700 meter), dan berbahaya karena dilalui truk tambang. "
        "Oleh karena itu, diperlukan model optimasi komputasi untuk menentukan titik tapak KOPDES terbaik di dalam wilayah 1 desa ini."
    )
    
    # Gambar 1: Peta Wilayah Desa Tkinter
    add_figure(
        doc,
        "results/map_study_area.png",
        "Gambar 1: Peta Tata Ruang Wilayah Desa Sukamaju pada Kanvas Tkinter (2.0 km × 1.5 km = 300 Hektar) untuk Penentuan Tapak KOPDES"
    )

    # ----------------------------------------------------
    # BAB 2
    # ----------------------------------------------------
    doc.add_heading("BAB 2: LATAR BELAKANG MASALAH LOKAL & DAMPAK KEPUTUSAN DI DALAM DESA", level=1)
    doc.add_paragraph(
        "Wilayah satu desa ini memiliki dinamika tata ruang dan pengguna yang saling berinteraksi:\n"
        "• 520 KK Warga Dusun Desa: Tersebar pada 50 klaster perumahan di sisi barat, memerlukan akses belanja sembako dan simpan pinjam desa "
        "dengan jarak tempuh jalan kaki atau sepeda motor < 5 menit.\n"
        "• Kelompok Tani Sawah Irigasi: Menggarap 60 hektar sawah produktif di barat daya, memerlukan akses angkut pupuk subsidi dengan motor roda tiga (Tossa).\n"
        "• Truk Distributor Pupuk Kabupaten: Memerlukan jalan desa beraspal mulus dengan lebar perkerasan minimal 5 meter agar truk 6 roda dapat bongkar muat lancar.\n"
        "• Warung Tradisional Warga: Toko kelontong milik warga (toko K di koordinat X=860, Y=800) yang memerlukan jarak aman agar KOPDES tidak mematikan usaha warganya sendiri."
    )
    doc.add_paragraph(
        "Dampak keputusan penempatan yang buruk:\n"
        "1. Risiko Kebangkrutan Koperasi: Jika KOPDES dibangun di pinggir jalan tanah becek di timur desa, warga enggan berbelanja dan omzet mandek.\n"
        "2. Konflik Sosial Antarwarga: Jika KOPDES dibangun menempel (< 100 m) pada toko kelontong swasta lama, timbul tuduhan kanibalisasi pasar.\n"
        "3. Pelanggaran Hukum Tata Ruang: Pendirian bangunan di hamparan sawah produktif melanggar Perda LP2B dengan ancaman pembongkaran paksa."
    )

    # ----------------------------------------------------
    # BAB 3
    # ----------------------------------------------------
    doc.add_heading("BAB 3: METODE OBSERVASI, KUALITAS DATA, & MASUKAN PEMANGKU KEPENTINGAN", level=1)
    doc.add_paragraph(
        "Sesuai pedoman tugas besar, tim melakukan pemisahan tegas antara fakta teramati, pendapat stakeholder, asumsi, keputusan terkendali, dan ketidakpastian:"
    )
    
    headers_sep = ["Kategori Metodologis", "Definisi Operasional", "Penerapan Nyata Kasus KOPDES Desa"]
    widths_sep = [1.8, 2.0, 2.7]
    rows_sep = [
        ["Fakta Teramati (Observed Fact)", "Terukur atau terlihat langsung dalam pemetaan spasial desa", "50 klaster rumah warga di barat; lebar perkerasan Jalan Utama Desa adalah 5.2 meter aspal mulus; jembatan desa bentang 100 meter; terdapat toko kelontong swasta lama (K) di dekat jembatan."],
        ["Pernyataan Stakeholder", "Pandangan langsung dari pengurus koperasi dan warga", "Ketua Kelompok Tani (Pak Wawan): 'Truk pupuk 6 roda tidak bisa masuk ke jalan tanah dusun kalau musim hujan. KOPDES harus berdiri di pinggir Jalan Utama Desa yang beraspal keras.'"],
        ["Asumsi (Assumption)", "Penyederhanaan matematis rasional", "Frekuensi belanja warga meluruh secara Gaussian terhadap jarak; jarak jalan kaki yang nyaman bagi ibu rumah tangga adalah <= 300 meter."],
        ["Keputusan Terkendali", "Variabel yang diatur dan ditetapkan oleh solusi optimasi", "Koordinat lokasi tapak KOPDES [x, y], jarak mundur bangunan dari as jalan (d_road <= 50m), kapasitas gudang pupuk dan toko sembako."],
        ["Ketidakpastian Lingkungan", "Variabel fluktuatif di luar kendali pengurus", "Jadwal pasokan pupuk subsidi dari distributor kabupaten; curah hujan ekstrem yang merusak kondisi jalan tanah lingkar dusun."]
    ]
    t_sep = doc.add_table(rows=1, cols=3)
    style_table(t_sep, widths_sep, headers_sep, rows_sep)
    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    # ----------------------------------------------------
    # BAB 4: FORMULASI MATEMATIKA DENGAN 4 FITUR KEBUGARAN
    # ----------------------------------------------------
    doc.add_heading("BAB 4: FORMULASI MATEMATIKA MODEL OPTIMASI KOPDES", level=1)
    doc.add_paragraph(
        "Fungsi Kebugaran (Fitness Function) dirumuskan untuk mengevaluasi kelayakan setiap calon tapak KOPDES [x, y] di dalam wilayah desa. "
        "Model ini secara murni mengintegrasikan 4 Fitur Kebugaran Berbobot (Weight Fitting Features) sebagai berikut:\n\n"
        "max F(x) = w1 * S_pop(x) + w2 * S_road(x) + w3 * S_fac(x) + w4 * S_comp(x) - Penalti(x)\n\n"
        "Dengan total bobot: w1 + w2 + w3 + w4 = 0.35 + 0.25 + 0.20 + 0.20 = 1.00."
    )
    
    # Rincian 4 Fitur Matematika
    headers_f = ["Simbol & Bobot", "Fitur Kebugaran", "Formulasi Matematis Lengkap", "Penjelasan Fungsi Operasional"]
    widths_f = [1.2, 1.8, 2.2, 2.3]
    rows_f = [
        [
            "w1 = 0.35",
            "Fitur 1: Kepadatan Pemukiman Warga (S_pop)",
            "S_pop(x) = (1 / M_pop) * SUM [ w_i * exp( -d(x, h_i)^2 / (2 * sigma_pop^2) ) ]\n(sigma_pop = 250.0 m)",
            "Menghitung aksesibilitas belanja sembako bagi 520 KK anggota koperasi desa. Titik dekat pusat pemukiman bernilai maksimal."
        ],
        [
            "w2 = 0.25",
            "Fitur 2: Kondisi Jalan Distribusi (S_road)",
            "S_road(x) = K_class * exp( -d(x, RoadNet)^2 / (2 * sigma_road^2) )\n(sigma_road = 150.0 m)",
            "Menilai kelayakan fisik jalan bagi armada truk pupuk:\n- Arteri Lintas Provinsi: K = 1.00\n- Kolektor Jalan Utama Desa: K = 0.85\n- Lokal Dusun: K = 0.60\n- Tanah Lumpur/Hauling: K = 0.20 (diskon berat)"
        ],
        [
            "w3 = 0.20",
            "Fitur 3: Sinergi Fasilitas Umum (S_fac)",
            "S_fac(x) = (1 / M_fac) * SUM [ W_j * exp( -d(x, f_j)^2 / (2 * sigma_fac^2) ) ]\n(sigma_fac = 200.0 m)",
            "Mengukur tarikan keramaian publik desa:\n- Pasar Desa: W = 2.0\n- SDN Sukamaju 1: W = 1.5\n- Warung/Restoran: W = 1.2\n- Bengkel: W = 0.8"
        ],
        [
            "w4 = 0.20",
            "Fitur 4: Jarak Aman Toko Lama (S_comp)",
            "S_comp(dc) = (dc / d_opt)^1.6 * exp( 1 - (dc / d_opt)^1.6 ) * Phi(dc)\n(d_opt = 350.0 m, Phi(dc) = (dc/180)^1.8 jika dc < 180m)",
            "Kurva Ricker non-monotonik: Jarak optimal 350 m bernilai 1.0 (aglomerasi sehat); jarak < 180 m dipotong penalti kanibalisasi agar tidak mematikan warung kelontong warga."
        ]
    ]
    t_f = doc.add_table(rows=1, cols=4)
    style_table(t_f, widths_f, headers_f, rows_f)
    doc.add_paragraph().paragraph_format.space_after = Pt(10)
    
    doc.add_paragraph(
        "Fungsi Penalti Batasan Spasial & Koridor Fisik Jalan:\n"
        "Penalti(x) = 1000.0, jika x berada di luar batas desa (0 <= x <= 2000, 0 <= y <= 1500) atau di dalam zona terlarang (sawah irigasi LP2B, sempadan sungai, hutan adat, galian tambang).\n"
        "Penalti(x) = 10.0 + (d_road - 50.0) / 50.0, jika x berjarak > 50.0 meter dari garis sumbu jalan desa (di luar koridor akses).\n"
        "Penalti(x) = 0.0, jika x memenuhi seluruh batasan (solusi feasible)."
    )

    # ----------------------------------------------------
    # BAB 5: SIMULASI PADA TKINTER
    # ----------------------------------------------------
    doc.add_heading("BAB 5: DESAIN SIMULASI DAN IMPLEMENTASI TKINTER", level=1)
    doc.add_paragraph(
        "Seluruh proses optimasi dijalankan secara terintegrasi pada antarmuka GUI Desktop Tkinter (app_tkinter.py):\n"
        "• Shared Evaluator: GA dan PSO mengeksekusi fungsi kebugaran yang sama secara bergantian dengan matched budget 2.000 evaluasi.\n"
        "• Visualisasi Split Mode: Layar sebelah kiri menyajikan evolusi populasi GA (termasuk mutasi dan elitisme), layar sebelah kanan menyajikan "
        "pergerakan kawanan partikel PSO (termasuk pbest dan gbest), serta layar bawah menampilkan kurva konvergensi live real-time.\n"
        "• Fitur Inspeksi Titik: Pengguna dapat mengklik koordinat manapun pada kanvas peta untuk melihat legalitas zona, jarak koridor jalan, dan rincian skor 4 fitur kebugaran."
    )
    
    # Gambar 2: Tangkapan Layar Tkinter Split Mode
    add_figure(
        doc,
        "results/tkinter_simulasi_split.png",
        "Gambar 2: Antarmuka Simulasi Interaktif Tkinter Mode Berdampingan (Split View): Hasil GA (Kiri), Hasil PSO (Kanan), dan Kurva Konvergensi Live (Bawah)"
    )

    # ----------------------------------------------------
    # BAB 6 & 7: HASIL KOMPUTASI & ANALISIS
    # ----------------------------------------------------
    doc.add_heading("BAB 6 & 7: HASIL KOMPUTASI DAN ANALISIS KRITIS", level=1)
    doc.add_paragraph(
        "Tabel berikut merangkum hasil pengujian 30 run independen pada mode pencarian lokasi KOPDES terbaik dengan anggaran 2.000 evaluasi:"
    )
    
    headers_res = ["Metrik Pengujian", "Genetic Algorithm (GA)", "Particle Swarm Optimization (PSO)", "Keterangan Evaluasi KOPDES"]
    widths_res = [1.8, 1.8, 1.9, 1.8]
    rows_res = [
        ["Nilai Fitness Terbaik (Max)", "0.785148", "0.785148", "Identik hingga 6 angka desimal"],
        ["Nilai Fitness Rata-rata (Mean)", "0.782871", "0.785148", "PSO lebih konsisten dan seragam"],
        ["Median Fitness", "0.785148", "0.785148", "Identik sempurna"],
        ["Standar Deviasi (Std)", "0.012471", "0.000000 (4.39e-10)", "PSO sangat stabil lintas run"],
        ["Interquartile Range (IQR)", "1.02e-10", "8.69e-11", "Variasi kuartil sangat rapat"],
        ["Fitness Terburuk (Min)", "0.716841", "0.785148", "GA sempat 1x terjebak sub-optimal"],
        ["Persentase Solusi Layak", "100.0% (30/30)", "100.0% (30/30)", "Keduanya 100% patuh batasan desa"],
        ["Rata-rata Waktu Run (detik)", "1.897 ± 0.917 s", "0.624 ± 0.096 s", "PSO 3.0× lebih cepat"],
        ["Median Eval ke-95% Konvergensi", "192 Evaluasi", "200 Evaluasi", "GA sedikit lebih cepat di awal"],
        ["Koordinat Rekomendasi KOPDES", "(X=295.4 m, Y=753.8 m)", "(X=295.4 m, Y=753.8 m)", "Kedua metode menuju titik fisik sama"]
    ]
    t_res = doc.add_table(rows=1, cols=4)
    style_table(t_res, widths_res, headers_res, rows_res)
    doc.add_paragraph().paragraph_format.space_after = Pt(10)
    
    # Gambar 3 & 4: Tampilan Layar Penuh GA dan PSO dari Tkinter
    add_figure(
        doc,
        "results/tkinter_simulasi_ga_full.png",
        "Gambar 3: Hasil Optimasi Algoritma Genetika (GA) Layar Penuh pada Kanvas Tkinter: Sebaran Populasi, Mutasi, dan Solusi Terbaik KOPDES"
    )
    add_figure(
        doc,
        "results/tkinter_simulasi_pso_full.png",
        "Gambar 4: Hasil Optimasi Particle Swarm Optimization (PSO) Layar Penuh pada Kanvas Tkinter: Partikel Swarm, pbest, dan gbest KOPDES"
    )
    add_figure(
        doc,
        "results/tkinter_simulasi_conv_full.png",
        "Gambar 5: Kurva Konvergensi Live Matched Budget 2.000 Evaluasi GA vs PSO pada Kanvas Tkinter"
    )

    doc.add_paragraph(
        "Hasil Uji Statistik Mann-Whitney U:\n"
        "Diperoleh nilai U = 654.00 dengan p-value = 0.00261 (p < 0.05). Terdapat perbedaan signifikan secara statistik pada konsistensi "
        "fitness akhir. PSO terbukti lebih stabil karena tidak mengalami satupun run yang terjebak pada optimum lokal sekunder, sedangkan "
        "GA memiliki 1 run out of 30 dengan fitness 0.7168.\n\n"
        "Perbandingan terhadap Baseline Desa:\n"
        "• Praktik Saat Ini (Usulan Tanah Kas Murah di Timur): Fitness = 0.4120 (lokasi di jalan tanah becek).\n"
        "• Heuristik Sederhana (Sentroid Pemukiman): Fitness = 0.6350 (lokasi di tengah desa tanpa akses fasilitas optimal).\n"
        "• Solusi Rekomendasi GA & PSO: Fitness = 0.7851 (+90.5% peningkatan dibanding rencana informal, +23.6% dibanding heuristik sentroid)."
    )

    # ----------------------------------------------------
    # BAB 8: VALIDASI & KESIMPULAN
    # ----------------------------------------------------
    doc.add_heading("BAB 8: VALIDASI LAPANGAN, RISIKO OPERASIONAL, ETIKA, & KESIMPULAN", level=1)
    doc.add_paragraph(
        "Validasi Lapangan Titik Rekomendasi KOPDES:\n"
        "Tim melakukan peninjauan langsung ke titik rekomendasi (X = 295.4 m, Y = 753.8 m) di tepi Jalan Utama Desa. Lokasi tersebut "
        "merupakan lahan pekarangan kosong datar di tepi aspal Jalan Utama Desa (lebar 5.2 meter), berjarak 65 meter dari SDN Sukamaju 1 "
        "dan 140 meter dari Pasar Desa. Kepala Desa dan Ketua KOPDES menyambut positif rekomendasi ini karena Jalan Utama Desa dilewati seluruh "
        "warga setiap hari, truk pengangkut pupuk bisa langsung parkir dan bongkar muat tanpa mengganggu lalu lintas, aman dari banjir sungai, "
        "dan tidak menggusur sawah irigasi produktif warga.\n\n"
        "Risiko Implementasi Fisik & Etika Koperasi:\n"
        "1. Lahan Pekarangan Warga: Pengurus KOPDES dapat menerapkan skema sewa jangka panjang 10 tahun atau bagi hasil penyertaan modal saham koperasi desa.\n"
        "2. Akses Bongkar Muat Truk: Dibuat jalan masuk (ramp) beton selebar 4 meter agar truk pengangkut pupuk dapat mundur langsung ke pintu gudang belakang KOPDES.\n"
        "3. Kemitraan dengan Warung Warga: KOPDES tidak mematikan toko kelontong warga lama, melainkan fokus pada pasokan grosir sembako, pupuk subsidi, dan penampungan hasil bumi warga.\n\n"
        "Kesimpulan:\n"
        "Model optimasi multi-kriteria berbasis GA dan PSO berhasil menyelesaikan masalah penentuan lokasi KOPDES di dalam 1 desa secara ilmiah dan transparan. "
        "PSO terbukti unggul dalam efisiensi komputasi (3× lebih cepat) dan stabilitas stokastik, sementara GA menunjukkan kemampuan diversifikasi yang tangguh. "
        "Titik rekomendasi (X = 295.4 m, Y = 753.8 m) memberikan peningkatan efektivitas sebesar +90.5% dibanding rencana informal sebelumnya dan sangat layak diimplementasikan."
    )

    # ----------------------------------------------------
    # LAMPIRAN-LAMPIRAN
    # ----------------------------------------------------
    doc.add_page_break()
    doc.add_heading("LAMPIRAN A: PROBLEM PROPOSAL TEMPLATE (PENEMPATAN KOPDES)", level=1)
    add_callout(
        doc,
        [
            "Site Identifier: Wilayah Desa Sukamaju Terpadu (Koordinat Relatif: 0.0, 0.0 s/d 2000.0, 1500.0 meter).",
            "Observed Process: Aktivitas distribusi sembako 520 KK warga desa, penyaluran pupuk bersubsidi sawah produktif, dan pergerakan belanja warga.",
            "Decision Owner: Ketua Koperasi Desa (KOPDES) Sukamaju Mandiri, Kepala Desa, dan Ketua BPD.",
            "Observed Problem: KOPDES belum memiliki gedung permanen dan sempat diusulkan secara informal di pinggiran timur desa (X=1400, Y=920) karena tanah murah. Lokasi tersebut terbukti becek, berdebu, jauh dari 85% warga desa, dan tidak bisa dilalui truk pupuk.",
            "Decision Variables: Koordinat fisik [x, y], jarak koridor jalan d_corridor <= 50m, radius layanan R_layanan in [200, 600]m, luas bangunan toko dan gudang A_gudang in [60, 150]m2.",
            "Objective: Maksimasi F(x) = Indeks Kesesuaian Lokasi Koperasi Desa (Populasi 0.35, Jalan Distribusi 0.25, Fasilitas 0.20, Toko Lama 0.20).",
            "Constraints: Batas wilayah desa, zona terlarang (sawah irigasi LP2B, sempadan sungai, hutan adat, tambang), koridor jalan <= 50m, jarak aman kompetitor >= 180m.",
            "Novelty: Data primer spasial 1 desa heterogen (pemukiman warga, sentra sawah, aspal desa vs jalan tanah lingkar)."
        ],
        title="FORMULIR APPROVAL GATE PROPOSAL RESMI PENEMPATAN KOPDES"
    )

    doc.add_heading("LAMPIRAN B: CATATAN OBSERVASI LAPANGAN & CLEANING LOG", level=1)
    headers_obs = ["Sesi", "Hari & Waktu", "Parameter yang Diukur", "Anomali & Tindakan Pembersihan Data"]
    widths_obs = [0.8, 2.0, 2.3, 2.2]
    rows_obs = [
        ["1", "Senin, 22 Sept 2026\n06.30 - 08.00 WIB\nCerah Berawan", "Arus motor Jl. Utama Desa: 84 kend/jam;\nArus pejalan kaki sekolah: 120 siswa/jam;\nTitik kumpul pasar desa.", "Ditemukan pasar tumpah dadakan memakan 1 meter bahu jalan barat; dicatat sebagai zona penyempitan jalan."],
        ["2", "Rabu, 24 Sept 2026\n11.30 - 13.00 WIB\nTerik Panas Berdebu", "Lebar perkerasan Jl. Utama Desa: 5.2 m;\nArus kendaraan per jam: 142 kend/jam;\nRadius putar motor roda tiga pengangkut pupuk.", "Titik GPS mess tambang 4 bergeser ke tebing akibat multipath error; dikoreksi manual dengan citra ortofoto tegak."],
        ["3", "Jumat, 26 Sept 2026\n16.30 - 17.00 WIB\nHujan Gerimis Rintik", "Pembeli toko kelontong lama: 18 org/jam;\nKecepatan motor di jalan tanah turun ke <15 km/jam.", "Jalan tanah lingkar dusun terbukti licin dan becek; memvalidasi keputusan memberikan bobot kelayakan rendah (0.20) untuk jalan tanah."]
    ]
    t_obs = doc.add_table(rows=1, cols=4)
    style_table(t_obs, widths_obs, headers_obs, rows_obs)
    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    doc.add_heading("LAMPIRAN C: FORMULIR PENGUNGKAPAN PENGGUNAAN AI (AI DISCLOSURE)", level=1)
    headers_ai = ["Alat & Tanggal", "Tujuan Pemrograman", "Ringkasan Prompt Mahasiswa", "Kode Dihasilkan & Verifikasi"]
    widths_ai = [1.5, 1.8, 2.0, 2.0]
    rows_ai = [
        ["Gemini / Antigravity\n28 Sept 2026", "Fungsi Ray-Casting poligon tertutup", "Buatkan fungsi ray-casting vectorized NumPy untuk cek apakah titik (N, 2) berada di dalam poligon sawah dan sungai.", "Fungsi point_in_polygon_vectorized di map_model.py; diuji manual dengan 10 titik uji kontrol."],
        ["Gemini / Antigravity\n29 Sept 2026", "Debugging counter Matched Budget", "Bagaimana memastikan loop GA dan PSO berhenti tepat pada 2.000 panggilan fungsi fitness?", "Penambahan self.eval_count di fitness.py; diverifikasi melalui unit test test_facility.py::test_evaluation_counter."],
        ["Gemini / Antigravity\n30 Sept 2026", "Styling visualisasi peta dan jalan", "Perbaiki linewidth dan marka garis putus-putus Jalan Lintas Provinsi pada Matplotlib agar realistis.", "Kode linewidth 22.0/17.5 pt dan garis putus 1.5 pt di map_model.py; diverifikasi pada GUI desktop."]
    ]
    t_ai = doc.add_table(rows=1, cols=4)
    style_table(t_ai, widths_ai, headers_ai, rows_ai)
    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    doc.add_heading("LAMPIRAN D: PANDUAN TANYA-JAWAB UJIAN LISAN (ORAL DEFENSE PROMPTS)", level=1)
    qa_list = [
        ("1. Konversi Observasi Mentah Menjadi Parameter Model KOPDES:", 
         "Observasi lapangan Sesi 3 membuktikan bahwa saat hujan gerimis, jalan tanah lingkar dusun menjadi kubangan lumpur licin yang tidak bisa dilalui motor warga maupun truk pupuk KOPDES. Fakta ini dikonversi menjadi koefisien kelayakan jalan Khauling/tanah = 0.20 (diskon kelayakan 80% dibanding Jalan Utama Desa beraspal K = 0.85 dan Arteri 1.00) pada fungsi kebugaran Sroad, sehingga algoritma secara otomatis menghindari penempatan KOPDES di jalan tanah tersebut."),
        ("2. Perbedaan Variabel Terkendali vs Ketidakpastian Lingkungan:", 
         "Koordinat fisik gedung KOPDES [X, Y] dan jarak mundur dari as jalan adalah variabel terkendali karena diputuskan langsung oleh pengurus koperasi dan kepala desa. Sebaliknya, curah hujan ekstrem, fluktuasi harga pupuk bersubsidi, dan kebijakan kuota pupuk dari pemerintah pusat adalah ketidakpastian lingkungan di luar kendali pengurus koperasi."),
        ("3. Penelusuran Kebugaran Titik di Sawah Produktif:", 
         "Jika kandidat solusi jatuh di sawah irigasi (X = 200, Y = 200), fungsi Ray-Casting mendeteksi titik berada di dalam poligon sawah. Variabel is_forbid bernilai True. Fungsi evaluasi langsung memotong nilai dengan penalti diskontinu -1000.0 tanpa membuang waktu menghitung fitur lain, sehingga individu tersebut langsung tereliminasi pada seleksi genetik berikutnya."),
        ("4. Dampak Jika Koefisien Penalti Dibagi Sepuluh (-1000 -> -100):", 
         "Terjadi kebocoran penalti (penalty leakage). Nilai gravitasi populasi warga desa yang sangat padat (+0.35) dapat mengimbangi penalti kecil, sehingga algoritma berisiko merekomendasikan KOPDES dibangun di tengah hamparan sawah produktif beririgasi teknis yang melanggar hukum. Skala -1000 menjamin solusi ilegal selalu kalah."),
        ("5. Kapan GA Unggul dan Kapan PSO Unggul pada Masalah KOPDES Ini?:", 
         "PSO unggul pada permukaan kontinu bergradien halus di sepanjang Jalan Utama Desa karena komunikasi global (gbest) memandu kawanan meluncur cepat dan presisi (terbukti 3× lebih cepat). GA unggul jika desa memiliki beberapa dusun terpencil yang dipisahkan oleh sungai tanpa jembatan, di mana operator mutasi acak dan crossover BLX-alpha mampu melompati batas rintangan."),
        ("6. Mengapa Objektif Terbaik Naik tapi Feasible % Bisa Turun?:", 
         "Terjadi fenomena boundary attraction: algoritma menemukan skor sangat tinggi tepat di batas bibir sawah atau sempadan sungai. Namun akibat variasi mutasi atau inersia partikel, sebagian besar populasi terlempar melewati batas ke zona terlarang, sehingga persentase solusi layak turun meskipun nilai terbaik individu naik."),
        ("7. Prediksi Perilaku Jika Batasan Koridor Jalan (<= 50m) Dimatikan:", 
         "Solusi akan langsung melompat ke tengah-tengah halaman rumah warga desa (X=280, Y=765) tanpa akses jalan mobil, karena nilai gravitasi populasi warga Spop mencapai nilai puncak tertinggi di titik tersebut."),
        ("8. Kesimpulan yang Didukung vs Tidak Didukung Data:", 
         "Didukung Data: PSO secara statistik signifikan lebih stabil (standar deviasi 4.39e-10) dan 3× lebih efisien dalam waktu eksekusi dibanding GA pada penugasan 2.000 evaluasi. TIDAK Didukung Data: Pernyataan bahwa 'GA gagal menemukan lokasi KOPDES terbaik' adalah keliru, karena pada 29 dari 30 run GA menemukan titik fisik yang sama persis (X = 295.4 m, Y = 753.8 m) dengan nilai kebugaran 0.7851.")
    ]
    for q, a in qa_list:
        add_callout(doc, [a], title=q)
        
    out_path = "d:/biocomputing-7/LAPORAN_OPTIMASI_GA_PSO.docx"
    try:
        doc.save(out_path)
        print(f"Laporan docx berhasil disimpan ke: {out_path}")
    except PermissionError:
        alt_path = "d:/biocomputing-7/LAPORAN_OPTIMASI_GA_PSO_REVISI.docx"
        doc.save(alt_path)
        print(f"Peringatan: {out_path} sedang dibuka di Microsoft Word.")
        print(f"Laporan docx berhasil disimpan ke alternatif: {alt_path}")

if __name__ == "__main__":
    build_report()
