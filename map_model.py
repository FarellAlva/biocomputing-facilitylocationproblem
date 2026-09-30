"""
Modul Model Peta (map_model.py)
Menyimpan struktur data peta 2x1.5 km, membaca format JSON,
menghitung geometri (point-in-polygon, jarak ke segmen jalan),
dan menyediakan fungsi visualisasi peta (plot_map) secara modular.

Catatan Desain: Modul inti ini tidak mengimpor matplotlib di tingkat modul
agar komputasi tanpa GUI tetap murni dan mandiri.
"""

from typing import Dict, List, Tuple, Any, Optional
import json
import numpy as np


class MapModel:
    """
    Kelas representasi peta wilayah untuk optimasi penempatan fasilitas.
    Koordinat dalam satuan meter (default: 2000m x 1500m).
    """

    def __init__(self, data: Dict[str, Any]):
        """
        Inisialisasi MapModel dari dictionary yang dimuat dari JSON.
        """
        self.name: str = data.get("name", "Peta Kustom")
        dims = data.get("dimensions", {"width": 2000.0, "height": 1500.0})
        self.width: float = float(dims.get("width", 2000.0))
        self.height: float = float(dims.get("height", 1500.0))

        # 1. Rumah / Pemukiman: array [x, y, weight]
        houses_raw = data.get("houses", [])
        if houses_raw:
            self.houses = np.array(
                [[h["x"], h["y"], h.get("weight", 1.0)] for h in houses_raw],
                dtype=np.float64,
            )
        else:
            self.houses = np.empty((0, 3), dtype=np.float64)

        # 2. Fasilitas Umum: list dict & dictionary array per jenis
        self.facilities_raw: List[Dict[str, Any]] = data.get("facilities", [])
        self.facilities_by_type: Dict[str, np.ndarray] = {}
        for fac in self.facilities_raw:
            ftype = fac["type"]
            coord_w = [fac["x"], fac["y"], fac.get("weight", 1.0)]
            if ftype not in self.facilities_by_type:
                self.facilities_by_type[ftype] = []
            self.facilities_by_type[ftype].append(coord_w)
        for ftype in self.facilities_by_type:
            self.facilities_by_type[ftype] = np.array(
                self.facilities_by_type[ftype], dtype=np.float64
            )

        # 3. Kompetitor: array [x, y] dan daftar nama
        comp_raw = data.get("competitors", [])
        if comp_raw:
            self.competitors = np.array(
                [[c["x"], c["y"]] for c in comp_raw], dtype=np.float64
            )
            self.competitor_names = [c.get("name", f"Kompetitor {i+1}") for i, c in enumerate(comp_raw)]
        else:
            self.competitors = np.empty((0, 2), dtype=np.float64)
            self.competitor_names = []

        # 4. Jalan: prekomputasi menjadi daftar segmen garis [A, B] beserta kelas dan lalu lintas
        self.roads_raw: List[Dict[str, Any]] = data.get("roads", [])
        self._build_road_segments()

        # 5. Zona Terlarang: daftar poligon numpy array
        self.forbidden_zones: List[Dict[str, Any]] = []
        for zone in data.get("forbidden_zones", []):
            poly_pts = np.array(zone.get("polygon", []), dtype=np.float64)
            self.forbidden_zones.append({
                "name": zone.get("name", "Zona Terlarang"),
                "type": zone.get("type", "zona"),
                "polygon": poly_pts,
            })

    def _build_road_segments(self) -> None:
        """
        Mengonversi polyline jalan menjadi pasangan segmen garis vektor [seg_start, seg_end]
        lengkap dengan kelas jalan dan nilai lalu lintas untuk kalkulasi jarak cepat.
        """
        seg_starts = []
        seg_ends = []
        seg_classes = []
        seg_traffics = []
        seg_names = []

        for road in self.roads_raw:
            pts = np.array(road.get("points", []), dtype=np.float64)
            r_class = road.get("class", "lokal")
            traffic = float(road.get("traffic", 0.5))
            r_name = str(road.get("name", "Jalan"))

            if len(pts) >= 2:
                for i in range(len(pts) - 1):
                    seg_starts.append(pts[i])
                    seg_ends.append(pts[i + 1])
                    seg_classes.append(r_class)
                    seg_traffics.append(traffic)
                    seg_names.append(r_name)

        if seg_starts:
            self.road_seg_starts = np.array(seg_starts, dtype=np.float64)
            self.road_seg_ends = np.array(seg_ends, dtype=np.float64)
            self.road_seg_classes = seg_classes
            self.road_seg_traffics = np.array(seg_traffics, dtype=np.float64)
            self.road_seg_names = seg_names
        else:
            self.road_seg_starts = np.empty((0, 2), dtype=np.float64)
            self.road_seg_ends = np.empty((0, 2), dtype=np.float64)
            self.road_seg_classes = []
            self.road_seg_traffics = np.empty((0,), dtype=np.float64)
            self.road_seg_names = []

    @classmethod
    def load_from_json(cls, file_path: str) -> "MapModel":
        """
        Membuat objek MapModel dari file JSON kustom.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data)

    def is_out_of_bounds(self, points: np.ndarray) -> np.ndarray:
        """
        Memeriksa apakah koordinat titik berada di luar batas peta [0, width] x [0, height].
        Input: points dengan shape (N, 2) atau (2,)
        Output: boolean array (N,) bertuliskan True jika di luar batas.
        """
        pts = np.atleast_2d(points)
        x = pts[:, 0]
        y = pts[:, 1]
        out_mask = (x < 0.0) | (x > self.width) | (y < 0.0) | (y > self.height)
        if points.ndim == 1:
            return out_mask[0]
        return out_mask

    def is_in_forbidden_zone(self, points: np.ndarray) -> np.ndarray:
        """
        Memeriksa apakah titik berada di dalam salah satu poligon zona terlarang
        menggunakan algoritma Ray Casting (Even-Odd rule).
        Input: points dengan shape (N, 2) atau (2,)
        Output: boolean array (N,) bernilai True jika berada dalam zona terlarang.
        """
        pts = np.atleast_2d(points)
        num_pts = pts.shape[0]
        in_zone = np.zeros(num_pts, dtype=bool)

        for zone in self.forbidden_zones:
            poly = zone["polygon"]
            if len(poly) < 3:
                continue
            # Evaluasi ray casting untuk setiap titik terhadap poligon ini
            mask = point_in_polygon_vectorized(pts, poly)
            in_zone = in_zone | mask
            if np.all(in_zone):
                break

        if points.ndim == 1:
            return in_zone[0]
        return in_zone

    def distance_to_roads(
        self, points: np.ndarray, class_weights: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Menghitung jarak efektif terdekat dari titik ke jaringan jalan.
        Jarak dibagi dengan bobot kelas jalan dan faktor lalu lintas:
        d_efektif = min_seg ( jarak_segmen / (bobot_kelas * (1 + lalu_lintas)) )
        
        Input: points shape (N, 2) atau (2,)
        Output: d_min array (N,)
        """
        if class_weights is None:
            class_weights = {"arteri": 1.0, "kolektor": 0.8, "lokal": 0.5, "hauling": 0.3}

        pts = np.atleast_2d(points)
        num_pts = len(pts)

        if len(self.road_seg_starts) == 0:
            res = np.full(num_pts, 10000.0, dtype=np.float64)
            return res[0] if points.ndim == 1 else res

        # Vektor bobot pengali untuk tiap segmen jalan
        factors = np.array(
            [
                class_weights.get(cls_name, 0.5) * (1.0 + trf)
                for cls_name, trf in zip(self.road_seg_classes, self.road_seg_traffics)
            ],
            dtype=np.float64,
        )  # shape: (S,)

        # Hitung jarak Euclidean titik ke semua segmen
        # pts: (N, 1, 2), seg_starts: (1, S, 2), seg_ends: (1, S, 2)
        P = pts[:, np.newaxis, :]  # (N, 1, 2)
        A = self.road_seg_starts[np.newaxis, :, :]  # (1, S, 2)
        B = self.road_seg_ends[np.newaxis, :, :]  # (1, S, 2)

        AB = B - A  # (1, S, 2)
        AP = P - A  # (N, S, 2)

        AB_lensq = np.sum(AB ** 2, axis=2)  # (1, S)
        # Cegah pembagian dengan nol untuk titik yang berimpit
        AB_lensq = np.maximum(AB_lensq, 1e-12)

        # Proyeksi skalar t = (AP . AB) / |AB|^2
        t = np.sum(AP * AB, axis=2) / AB_lensq  # (N, S)
        t = np.clip(t, 0.0, 1.0)  # Batasi pada batas segmen [0, 1]

        # Titik terdekat pada segmen: C = A + t * AB
        closest_pts = A + t[:, :, np.newaxis] * AB  # (N, S, 2)

        # Jarak geometris: |P - C|
        dists = np.sqrt(np.sum((P - closest_pts) ** 2, axis=2))  # (N, S)

        # Jarak efektif terbobot: dist / faktor
        effective_dists = dists / factors[np.newaxis, :]  # (N, S)

        # Ambil nilai minimum per titik
        min_eff_dists = np.min(effective_dists, axis=1)  # (N,)

        if points.ndim == 1:
            return min_eff_dists[0]
        return min_eff_dists

    def geometric_distance_to_roads(self, points: np.ndarray) -> np.ndarray:
        """
        Menghitung jarak geometris murni (meter Euclidean) dari titik ke segmen jalan terdekat,
        tanpa pembagian faktor kelas jalan.
        Digunakan untuk memastikan penempatan fasilitas berada dalam koridor jalan fisik.
        
        Input: points shape (N, 2) atau (2,)
        Output: d_min array (N,) atau float jika 1 titik
        """
        pts = np.atleast_2d(points)
        if len(self.road_seg_starts) == 0:
            res = np.full(len(pts), 10000.0, dtype=np.float64)
            return res[0] if points.ndim == 1 else res

        P = pts[:, np.newaxis, :]  # (N, 1, 2)
        A = self.road_seg_starts[np.newaxis, :, :]  # (1, S, 2)
        B = self.road_seg_ends[np.newaxis, :, :]  # (1, S, 2)

        AB = B - A
        AP = P - A
        AB_lensq = np.maximum(np.sum(AB ** 2, axis=2), 1e-12)
        t = np.clip(np.sum(AP * AB, axis=2) / AB_lensq, 0.0, 1.0)
        closest_pts = A + t[:, :, np.newaxis] * AB
        dists = np.sqrt(np.sum((P - closest_pts) ** 2, axis=2))
        min_dists = np.min(dists, axis=1)

        if points.ndim == 1:
            return float(min_dists[0])
        return min_dists

    def get_nearest_road_info(self, point: np.ndarray) -> Tuple[float, str, str]:
        """
        Mengembalikan (jarak_geometris, kelas_jalan, nama_jalan) untuk titik (x, y) terhadap segmen terdekat.
        """
        pt = np.asarray(point, dtype=np.float64).flatten()[:2]
        if len(self.road_seg_starts) == 0:
            return 10000.0, "lokal", "Tidak Ada Jalan"

        P = pt[np.newaxis, :]  # (1, 2)
        A = self.road_seg_starts  # (S, 2)
        B = self.road_seg_ends    # (S, 2)

        AB = B - A
        AP = P - A
        AB_lensq = np.maximum(np.sum(AB ** 2, axis=1), 1e-12)
        t = np.clip(np.sum(AP * AB, axis=1) / AB_lensq, 0.0, 1.0)
        closest_pts = A + t[:, np.newaxis] * AB
        dists = np.sqrt(np.sum((P - closest_pts) ** 2, axis=1))

        best_idx = int(np.argmin(dists))
        best_dist = float(dists[best_idx])
        best_class = self.road_seg_classes[best_idx] if best_idx < len(self.road_seg_classes) else "lokal"
        best_name = self.road_seg_names[best_idx] if best_idx < len(self.road_seg_names) else "Jalan"

        return best_dist, best_class, best_name

    def get_forbidden_zone_name(self, point: np.ndarray) -> Optional[str]:
        """
        Mengembalikan nama zona terlarang jika titik berada di dalamnya, atau None jika valid.
        """
        pt_2d = np.asarray(point, dtype=np.float64).reshape((1, 2))
        for zone in self.forbidden_zones:
            poly = np.array(zone.get("polygon", []), dtype=np.float64)
            if len(poly) >= 3:
                if point_in_polygon_vectorized(pt_2d, poly)[0]:
                    return zone.get("name", zone.get("type", "Zona Terlarang"))
        return None


def point_in_polygon_vectorized(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """
    Algoritma Ray Casting untuk mengecek apakah kumpulan titik berada di dalam poligon.
    points: array shape (N, 2)
    polygon: array shape (V, 2)
    Output: boolean array shape (N,)
    """
    x = points[:, 0]
    y = points[:, 1]
    n_pts = len(points)
    n_vert = len(polygon)
    inside = np.zeros(n_pts, dtype=bool)

    # Iterasi setiap sisi poligon (A -> B)
    p1 = polygon[0]
    for i in range(1, n_vert + 1):
        p2 = polygon[i % n_vert]
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]

        # Sinar horizontal dari titik ke arah kanan (+x)
        # Cek apakah segmen (y1, y2) memotong garis horizontal y
        cond_y = (y1 > y) != (y2 > y)

        if np.any(cond_y):
            # Hitung titik potong x dari sinar horizontal dengan garis segmen
            # x_intersect = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            dx = x2 - x1
            dy = y2 - y1
            denom = dy if dy != 0 else 1e-12
            x_intersect = x1 + (y - y1) * dx / denom
            cond_x = x < x_intersect
            intersect = cond_y & cond_x
            inside = inside ^ intersect

        p1 = p2

    return inside


def plot_map(
    map_model: MapModel,
    ax=None,
    title: str = "Peta Wilayah Fasilitas",
    show_legend: bool = True,
    show_heatmap: bool = False,
    heatmap_grid: Optional[np.ndarray] = None,
    heatmap_extent: Optional[Tuple[float, float, float, float]] = None,
    heatmap_cmap: str = "viridis",
    heatmap_alpha: float = 0.55,
):
    """
    Fungsi visualisasi peta menggunakan Matplotlib sesuai diagram referensi visual:
    'Peta Studi: Desa, Jalan Lintas, dan Kawasan Hauling'.
    Legenda ditempatkan di bawah sumbu peta agar tidak tumpang tindih.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    if ax is None:
        fig, ax = plt.subplots(figsize=(10.5, 7.8))

    # 1. Warna Latar Dasar Wilayah (Pedesaan Hijau di Barat, Kawasan Tambang Pasir Hangat di Timur)
    is_study_map = "Desa" in map_model.name or "Studi" in map_model.name or "Hauling" in map_model.name
    if is_study_map:
        ax.add_patch(patches.Rectangle((0, 0), 1150, map_model.height, facecolor="#f1f8ed", edgecolor="none", zorder=0))
        ax.add_patch(patches.Rectangle((1150, 0), map_model.width - 1150, map_model.height, facecolor="#faefe3", edgecolor="none", zorder=0))

    # 2. Tampilkan Heatmap jika diminta
    if show_heatmap and heatmap_grid is not None and heatmap_extent is not None:
        im = ax.imshow(
            heatmap_grid,
            origin="lower",
            extent=heatmap_extent,
            cmap=heatmap_cmap,
            alpha=heatmap_alpha,
            aspect="auto",
            zorder=1,
        )

    # 3. Gambar Zona Terlarang
    color_map_zone = {
        "sungai": ("#38bdf8", "#0284c7", 0.90, "Sungai"),
        "sawah": ("#b7e4c7", "#74c69d", 0.85, "Sawah"),
        "hutan": ("#2d6a4f", "#1b4332", 0.85, "Hutan"),
        "lahan_tambang": ("#b08968", "#78350f", 0.85, "Tambang"),
    }
    drawn_zone_labels = set()

    for zone in map_model.forbidden_zones:
        poly_pts = zone["polygon"]
        ztype = zone.get("type", "zona")
        fcolor, ecolor, alpha_val, default_label = color_map_zone.get(
            ztype, ("#ff7f7f", "#b91c1c", 0.7, "Zona Terlarang")
        )
        label = default_label if default_label not in drawn_zone_labels else None
        if label:
            drawn_zone_labels.add(default_label)

        poly_patch = patches.Polygon(
            poly_pts,
            closed=True,
            facecolor=fcolor,
            edgecolor=ecolor,
            linestyle="--" if ztype == "lahan_tambang" else "-",
            linewidth=1.2,
            alpha=alpha_val,
            zorder=2,
            label=label,
        )
        ax.add_patch(poly_patch)

        # Pola kisi sawah
        if ztype == "sawah" and is_study_map:
            for gx in range(60, 640, 60):
                ax.plot([gx, gx], [0, 520], color="#74c69d", lw=0.6, alpha=0.45, zorder=2)
            for gy in range(50, 520, 50):
                ax.plot([0, 640], [gy, gy], color="#74c69d", lw=0.6, alpha=0.45, zorder=2)

    # Lingkaran Dotted Permukiman Warga
    if is_study_map:
        ellipse = patches.Ellipse(
            (290, 780), width=480, height=420,
            facecolor="#e2f3e5", edgecolor="#0284c7",
            linestyle="--", linewidth=1.5, alpha=0.6, zorder=3,
            label="Permukiman warga"
        )
        ax.add_patch(ellipse)

    # 4. Gambar Jalan sesuai kelasnya
    drawn_road_labels = set()
    for road in map_model.roads_raw:
        pts = np.array(road.get("points", []), dtype=np.float64)
        r_class = road.get("class", "lokal")
        r_name = road.get("name", "Jalan")

        if r_class == "arteri":
            # 4 lajur gelap dengan marka putus-putus putih
            lbl = "Jalan arteri" if "Jalan arteri" not in drawn_road_labels else None
            drawn_road_labels.add("Jalan arteri")
            ax.plot(pts[:, 0], pts[:, 1], color="#1e293b", linewidth=6.5, zorder=4, label=lbl)
            ax.plot(pts[:, 0], pts[:, 1], color="#ffffff", linewidth=1.1, linestyle="--", zorder=5)
        elif r_class == "hauling":
            # Jalur tambang lebar dengan jejak debu
            lbl = "Jalan hauling" if "Jalan hauling" not in drawn_road_labels else None
            drawn_road_labels.add("Jalan hauling")
            ax.plot(pts[:, 0], pts[:, 1], color="#a3704c", linewidth=6.0, zorder=4, label=lbl)
            ax.plot(pts[:, 0], pts[:, 1], color="#c99d75", linewidth=2.5, linestyle=":", zorder=5)
        elif r_class == "kolektor":
            lbl = "Jalan kolektor" if "Jalan kolektor" not in drawn_road_labels else None
            drawn_road_labels.add("Jalan kolektor")
            ax.plot(pts[:, 0], pts[:, 1], color="#ea580c", linewidth=3.0, zorder=4, label=lbl)
        else:
            lbl = "Jalan lokal" if "Jalan lokal" not in drawn_road_labels else None
            drawn_road_labels.add("Jalan lokal")
            ax.plot(pts[:, 0], pts[:, 1], color="#64748b", linewidth=1.8, zorder=4, label=lbl)

    # 5. Gambar Rumah / Pemukiman (Atap Merah)
    if len(map_model.houses) > 0:
        ax.scatter(
            map_model.houses[:, 0],
            map_model.houses[:, 1],
            s=22,
            c="#e11d48",
            alpha=0.65,
            marker="s",
            edgecolors="#9f1239",
            linewidths=0.5,
            zorder=6,
            label="Rumah",
        )

    # 6. Gambar Fasilitas Umum (Dengan Ikon Huruf S, P, R, B, M)
    facility_markers = {
        "sekolah": ("#1d4ed8", "S", 100, "Sekolah"),
        "pasar": ("#16a34a", "P", 110, "Pasar"),
        "restoran": ("#ea580c", "R", 95, "Restoran / Warung"),
        "bengkel": ("#9333ea", "B", 100, "Bengkel"),
        "masjid_kantor": ("#059669", "M", 105, "Masjid"),
    }
    drawn_fac_labels = set()

    for fac in map_model.facilities_raw:
        ftype = fac.get("type", "fasilitas")
        color, letter, size, f_label = facility_markers.get(
            ftype, ("#0284c7", "*", 80, ftype.capitalize())
        )
        lbl = f_label if f_label not in drawn_fac_labels else None
        if lbl:
            drawn_fac_labels.add(f_label)

        # Gambar lingkaran latar belakang
        ax.scatter(
            fac["x"],
            fac["y"],
            c=color,
            marker="o",
            s=size,
            edgecolors="#ffffff",
            linewidths=1.2,
            zorder=7,
            label=lbl,
        )
        # Gambar huruf di tengah lingkaran
        ax.text(
            fac["x"],
            fac["y"],
            letter,
            color="#ffffff",
            fontsize=7,
            weight="bold",
            ha="center",
            va="center",
            clip_on=True,
            zorder=8,
        )

    # 7. Gambar Kompetitor (Minimarket Lama K)
    if len(map_model.competitors) > 0:
        ax.scatter(
            map_model.competitors[:, 0],
            map_model.competitors[:, 1],
            c="#dc2626",
            marker="o",
            s=115,
            edgecolors="#ffffff",
            linewidths=1.3,
            zorder=7,
            label="Minimarket lama",
        )
        for i, (cx, cy) in enumerate(map_model.competitors):
            ax.text(
                cx,
                cy,
                "K",
                color="#ffffff",
                fontsize=7.5,
                weight="bold",
                ha="center",
                va="center",
                clip_on=True,
                zorder=8,
            )

    # 8. Anotasi Label & Badge Spasial (Peta Studi)
    if is_study_map:
        bbox_dark = dict(boxstyle="round,pad=0.25", facecolor="#0f172a", edgecolor="none", alpha=0.82)
        bbox_white = dict(boxstyle="round,pad=0.25", facecolor="#ffffff", edgecolor="#cbd5e1", lw=1.0, alpha=0.92)
        bbox_blue = dict(boxstyle="round,pad=0.25", facecolor="#0284c7", edgecolor="none", alpha=0.88)
        bbox_brown = dict(boxstyle="round,pad=0.25", facecolor="#78350f", edgecolor="none", alpha=0.82)

        # Label Zona
        ax.text(90, 1370, "Hutan", color="#ffffff", fontsize=7.5, weight="bold", ha="center", bbox=bbox_dark, clip_on=True, zorder=10)
        ax.text(110, 260, "Sawah", color="#0f172a", fontsize=7.5, weight="bold", ha="center", bbox=bbox_white, clip_on=True, zorder=10)
        ax.text(800, 360, "Sungai", color="#ffffff", fontsize=7.5, weight="bold", ha="center", bbox=bbox_blue, clip_on=True, zorder=10)
        ax.text(200, 1005, "Permukiman warga", color="#ffffff", fontsize=7, weight="bold", ha="center", bbox=bbox_blue, clip_on=True, zorder=10)

        # Label Jalan
        ax.text(210, 720, "Jl. Utama Desa", color="#ffffff", fontsize=6.5, weight="bold", bbox=bbox_dark, clip_on=True, zorder=10)
        ax.text(340, 975, "Jl. Lingkar Timur", color="#ffffff", fontsize=6.5, weight="bold", bbox=bbox_dark, clip_on=True, zorder=10)
        ax.text(400, 610, "Jl. Dusun Barat", color="#ffffff", fontsize=6.5, weight="bold", bbox=bbox_dark, clip_on=True, zorder=10)
        ax.text(1150, 1260, "Jl. Lintas Provinsi", color="#ffffff", fontsize=7, weight="bold", ha="center", bbox=bbox_dark, clip_on=True, zorder=10)
        ax.text(1420, 810, "Jl. Akses Tambang", color="#ffffff", fontsize=6.5, weight="bold", ha="center", bbox=bbox_dark, clip_on=True, zorder=10)
        ax.text(1640, 1440, "Jalan Hauling", color="#ffffff", fontsize=7, weight="bold", ha="center", bbox=bbox_brown, clip_on=True, zorder=10)

        # Fasilitas Kawasan Tambang
        ax.text(1840, 1370, "Mess tambang", color="#ffffff", fontsize=7, weight="bold", ha="center", bbox=bbox_brown, clip_on=True, zorder=10)
        ax.text(1840, 1140, "Bengkel alat berat", color="#ffffff", fontsize=7, weight="bold", ha="center", bbox=bbox_brown, clip_on=True, zorder=10)
        ax.text(1820, 620, "Lahan tambang aktif", color="#ffffff", fontsize=7, weight="bold", ha="center", bbox=bbox_dark, clip_on=True, zorder=10)
        ax.text(1830, 210, "Area timbunan (stockpile)", color="#ffffff", fontsize=7, weight="bold", ha="center", bbox=bbox_dark, clip_on=True, zorder=10)

    # Format Tampilan & Batas Sumbu
    ax.set_xlim(0, map_model.width)
    ax.set_ylim(0, map_model.height)
    ax.set_aspect("equal")
    ax.set_xlabel("Koordinat X (meter)", fontsize=8)
    ax.set_ylabel("Koordinat Y (meter)", fontsize=8)
    ax.set_title(title, fontsize=10, weight="bold")
    ax.grid(True, linestyle=":", alpha=0.4, color="#94a3b8")

    # Legenda ditaruh di BAWAH agar tidak tumpang tindih dengan peta
    if show_legend:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=5,
            fontsize=7.5,
            framealpha=0.95,
            edgecolor="#cbd5e1",
        )

    return ax

