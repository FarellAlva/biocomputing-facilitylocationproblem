"""
Skrip Generator Peta Kustom (generate_maps.py)
Menghasilkan preset peta wilayah 2.0 km x 1.5 km dalam format JSON
sesuai dengan diagram referensi resmi:
'Peta Studi: Desa, Jalan Lintas, dan Kawasan Hauling'

Struktur Wilayah:
- Sisi Barat:
  * Kawasan Hutan Lindung (Barat Laut, Zona Terlarang)
  * Kawasan Persawahan (Barat Daya, Zona Terlarang)
  * Permukiman warga padat dengan Masjid di tengahnya
  * Fasilitas: 3 Sekolah (S), 1 Pasar (P), 3 Restoran (R), 2 Bengkel (B)
  * 2 Minimarket lama / Kompetitor (K) di Jl. Utama Desa sebelum jembatan
  * Titik optimal: 'Lokasi terbaik' (Pin Hijau) di selatan perumahan
- Sisi Tengah:
  * Sungai berliku dari Utara ke Selatan (Zona Terlarang)
  * Jl. Lintas Provinsi (Arteri 4 lajur) membentang vertikal (X = 1150 m)
  * Jembatan penghubung jalan desa ke jalan lintas provinsi
- Sisi Timur:
  * Jl. Akses Tambang menghubungkan arteri ke jalan hauling
  * Titik terburuk valid: 'Lokasi terburuk' (Pin Merah) di dekat belokan akses tambang
  * Jalan Hauling (jalur truk tambang batubara/mineral)
  * Mess tambang (barak pekerja) dan Bengkel alat berat
  * Lahan tambang aktif (Pit galian terbuka, Zona Terlarang)
  * Area timbunan / stockpile (Zona Terlarang)
"""

import os
import json
import numpy as np


def generate_peta_studi() -> dict:
    """
    Menghasilkan preset peta studi lengkap sesuai diagram referensi visual pengguna:
    'Peta Studi: Desa, Jalan Lintas, dan Kawasan Hauling'
    """
    rng = np.random.default_rng(101)

    houses = []
    # 1. Permukiman Warga (Kluster Barat di dalam batas lingkaran elips)
    # Centered at X ~ 300, Y ~ 780
    for _ in range(95):
        # Distribusi Gaussian terkonsentrasi di permukiman warga barat
        hx = float(np.clip(rng.normal(290, 105), 60, 520))
        hy = float(np.clip(rng.normal(780, 110), 570, 990))
        w = float(rng.uniform(1.5, 4.8))
        houses.append({"x": round(hx, 1), "y": round(hy, 1), "weight": round(w, 2)})

    # 2. Mess Tambang (Barak Pekerja di Kawasan Tambang Timur Laut)
    # Centered at X ~ 1840, Y ~ 1280
    for _ in range(16):
        hx = float(np.clip(rng.normal(1840, 45), 1740, 1940))
        hy = float(np.clip(rng.normal(1280, 50), 1180, 1380))
        houses.append({"x": round(hx, 1), "y": round(hy, 1), "weight": 1.2})

    # Fasilitas Umum sesuai ikon peta referensi
    facilities = [
        # Masjid di pusat permukiman warga
        {"type": "masjid_kantor", "x": 280.0, "y": 780.0, "weight": 1.5, "name": "Masjid Warga"},
        # 3 Sekolah (S): 2 di Jl. Lingkar Timur, 1 di Jl. Utama Desa
        {"type": "sekolah", "x": 480.0, "y": 950.0, "weight": 1.6, "name": "Sekolah 1"},
        {"type": "sekolah", "x": 560.0, "y": 950.0, "weight": 1.6, "name": "Sekolah 2"},
        {"type": "sekolah", "x": 480.0, "y": 860.0, "weight": 1.5, "name": "Sekolah 3"},
        # Pasar (P) di persimpangan komersial
        {"type": "pasar", "x": 560.0, "y": 860.0, "weight": 2.2, "name": "Pasar Desa"},
        # 3 Restoran / Warung (R, R, R) berderet di sepanjang Jl. Utama Desa
        {"type": "restoran", "x": 640.0, "y": 830.0, "weight": 1.3, "name": "Restoran 1"},
        {"type": "restoran", "x": 690.0, "y": 830.0, "weight": 1.3, "name": "Restoran 2"},
        {"type": "restoran", "x": 740.0, "y": 830.0, "weight": 1.3, "name": "Restoran 3"},
        # 2 Bengkel (B, B) di sisi selatan jalan
        {"type": "bengkel", "x": 600.0, "y": 750.0, "weight": 0.9, "name": "Bengkel Motor 1"},
        {"type": "bengkel", "x": 660.0, "y": 750.0, "weight": 0.9, "name": "Bengkel Motor 2"},
        # Bengkel Alat Berat di Kawasan Tambang
        {"type": "bengkel", "x": 1780.0, "y": 1080.0, "weight": 1.6, "name": "Bengkel Alat Berat"},
    ]

    # 1 Minimarket Lama / Kompetitor (K) di Jl. Utama Desa sebelum jembatan sungai
    competitors = [
        {"x": 860.0, "y": 800.0, "name": "Minimarket Lama (K)"},
    ]

    # Jaringan Jalan Sesuai Diagram
    roads = [
        # Jalan Lintas Provinsi (Arteri 4 Lajur Membentang Vertikal dari Selatan ke Utara)
        {
            "name": "Jl. Lintas Provinsi",
            "class": "arteri",
            "traffic": 0.90,
            "points": [
                [1150.0, 0.0],
                [1150.0, 500.0],
                [1150.0, 800.0],
                [1150.0, 1100.0],
                [1150.0, 1500.0],
            ],
        },
        # Jalan Utama Desa (Menghubungkan Permukiman Barat ke Jl. Lintas Provinsi)
        {
            "name": "Jl. Utama Desa",
            "class": "kolektor",
            "traffic": 0.65,
            "points": [
                [50.0, 720.0],
                [250.0, 750.0],
                [480.0, 800.0],
                [760.0, 810.0],
                [920.0, 800.0],
                [1150.0, 800.0],
            ],
        },
        # Jl. Lingkar Timur (Akses ke Sekolah dan Sisi Utara Desa)
        {
            "name": "Jl. Lingkar Timur",
            "class": "lokal",
            "traffic": 0.35,
            "points": [
                [480.0, 800.0],
                [480.0, 950.0],
                [560.0, 950.0],
                [580.0, 1100.0],
            ],
        },
        # Jl. Dusun Barat (Akses ke Sisi Selatan / Menuju Sawah)
        {
            "name": "Jl. Dusun Barat",
            "class": "lokal",
            "traffic": 0.30,
            "points": [
                [480.0, 800.0],
                [460.0, 680.0],
                [530.0, 600.0],
                [620.0, 540.0],
            ],
        },
        # Jl. Akses Tambang (Menghubungkan Arteri ke Kawasan Tambang & Hauling)
        {
            "name": "Jl. Akses Tambang",
            "class": "kolektor",
            "traffic": 0.70,
            "points": [
                [1150.0, 800.0],
                [1320.0, 830.0],
                [1460.0, 850.0],
                [1580.0, 900.0],
            ],
        },
        # Jalan Hauling (Jalur Tambang Khusus Truk Hauling Lebar)
        {
            "name": "Jalan Hauling",
            "class": "hauling",
            "traffic": 0.95,
            "points": [
                [1680.0, 1500.0],
                [1630.0, 1150.0],
                [1580.0, 900.0],
                [1520.0, 550.0],
                [1450.0, 0.0],
            ],
        },
        # Jalan Akses Mess Tambang & Bengkel Alat Berat
        {
            "name": "Jl. Akses Mess & Bengkel",
            "class": "lokal",
            "traffic": 0.40,
            "points": [
                [1630.0, 1150.0],
                [1780.0, 1080.0],
                [1840.0, 1280.0],
            ],
        },
        # Jl. Niaga Pasar (Akses Khusus Pasar Desa & Sekolah)
        {
            "name": "Jl. Niaga Pasar",
            "class": "lokal",
            "traffic": 0.50,
            "points": [
                [480.0, 860.0],
                [590.0, 860.0],
            ],
        },
    ]

    # Zona Terlarang Sesuai Peta Referensi
    forbidden_zones = [
        # 1. Hutan Lindung (Barat Laut)
        {
            "name": "Kawasan Hutan Lindung",
            "type": "hutan",
            "polygon": [
                [0.0, 1080.0],
                [740.0, 1080.0],
                [760.0, 1500.0],
                [0.0, 1500.0],
            ],
        },
        # 2. Kawasan Persawahan Produktif (Barat Daya)
        {
            "name": "Kawasan Pertanian Sawah",
            "type": "sawah",
            "polygon": [
                [0.0, 0.0],
                [640.0, 0.0],
                [640.0, 520.0],
                [0.0, 520.0],
            ],
        },
        # 3. Sungai Berliku (Mengalir Membelah Wilayah Barat & Tengah)
        {
            "name": "Aliran Sungai Utama",
            "type": "sungai",
            "polygon": [
                [770.0, 1500.0],
                [820.0, 1500.0],
                [800.0, 1150.0],
                [830.0, 900.0],
                [860.0, 750.0],
                [840.0, 500.0],
                [800.0, 250.0],
                [740.0, 0.0],
                [680.0, 0.0],
                [730.0, 250.0],
                [780.0, 500.0],
                [790.0, 750.0],
                [760.0, 900.0],
                [730.0, 1150.0],
            ],
        },
        # 4. Lahan Tambang Aktif (Pit Terbuka di Tenggara)
        {
            "name": "Lahan Tambang Aktif (Open Pit)",
            "type": "lahan_tambang",
            "polygon": [
                [1720.0, 420.0],
                [1950.0, 420.0],
                [2000.0, 450.0],
                [2000.0, 850.0],
                [1750.0, 850.0],
                [1720.0, 600.0],
            ],
        },
        # 5. Area Timbunan (Stockpile)
        {
            "name": "Area Timbunan Batubara (Stockpile)",
            "type": "lahan_tambang",
            "polygon": [
                [1680.0, 60.0],
                [1980.0, 60.0],
                [1980.0, 360.0],
                [1680.0, 360.0],
            ],
        },
    ]

    return {
        "name": "Peta Studi: Desa, Jalan Lintas, dan Kawasan Hauling",
        "dimensions": {"width": 2000.0, "height": 1500.0},
        "houses": houses,
        "facilities": facilities,
        "competitors": competitors,
        "roads": roads,
        "forbidden_zones": forbidden_zones,
    }


def generate_hauling() -> dict:
    """
    Preset alternatif: Fokus murni kawasan pertambangan.
    """
    base = generate_peta_studi()
    base["name"] = "Peta Koridor Hauling Tambang (Populasi Ekstrem Jarang)"
    return base


def generate_campuran() -> dict:
    """
    Preset alternatif: Wilayah transisi pinggiran ke industri.
    """
    base = generate_peta_studi()
    base["name"] = "Peta Campuran (Transisi Pedesaan ke Koridor Industri)"
    return base


def main():
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "maps")
    os.makedirs(output_dir, exist_ok=True)

    study_map = generate_peta_studi()

    presets = [
        ("peta_studi.json", study_map),
    ]

    print("=" * 70)
    print("GENERATOR PETA SESUAI DIAGRAM: 'Desa, Jalan Lintas, dan Kawasan Hauling'")
    print("=" * 70)

    for filename, map_data in presets:
        target_path = os.path.join(output_dir, filename)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(map_data, f, indent=2, ensure_ascii=False)
        print(f"[OK] Berhasil membuat: {target_path}")
        print(f"     Nama: {map_data['name']}")
        print(f"     Rumah: {len(map_data['houses'])}, Fasilitas: {len(map_data['facilities'])}, "
              f"Kompetitor: {len(map_data['competitors'])}, Jalan: {len(map_data['roads'])}, "
              f"Zona Terlarang: {len(map_data['forbidden_zones'])}")

    print("\nPeta studi berhasil dibuat sesuai dengan gambar visual pengguna.")


if __name__ == "__main__":
    main()
