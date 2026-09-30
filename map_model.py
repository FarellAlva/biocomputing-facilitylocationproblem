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
    heatmap_alpha: float = 0.50,
    show_banner: Optional[bool] = None,
):
    """
    Fungsi visualisasi peta kartografis realistis dan terilustrasi tinggi
    menggunakan Matplotlib sesuai diagram referensi visual:
    'Peta Studi: Desa, Jalan Lintas, dan Kawasan Hauling'.
    Menyajikan lanskap pedesaan hijau, sawah berpetak, hutan berkanopi awan,
    sungai berliku dengan jembatan beton, jalan hauling tanah merah dengan truk tambang kuning,
    serta terasering tambang terbuka (open pit) dan stockpile batubara.
    """
    import math
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.lines as lines

    if ax is None:
        fig, ax = plt.subplots(figsize=(14.0, 10.2), dpi=150)

    if show_banner is None:
        show_banner = show_legend

    is_study_map = "Desa" in map_model.name or "Studi" in map_model.name or "Hauling" in map_model.name

    # ----------------------------------------------------
    # 1. LATAR BELAKANG WILAYAH (TERRAIN BACKDROP)
    # ----------------------------------------------------
    if is_study_map:
        # Sisi Barat: Padang rumput hijau subur (#cfe8a9 / #d4eab3)
        ax.add_patch(patches.Rectangle((0, 0), 1150, map_model.height, facecolor="#cfe8a9", edgecolor="none", zorder=0))
        # Gradasi/variasi lembut padang rumput
        ax.add_patch(patches.Polygon([[0, 800], [500, 1100], [700, 700], [0, 500]], closed=True, facecolor="#c7e39f", edgecolor="none", alpha=0.5, zorder=0))
        ax.add_patch(patches.Polygon([[400, 0], [900, 300], [1150, 200], [1150, 0]], closed=True, facecolor="#c4e199", edgecolor="none", alpha=0.4, zorder=0))

        # Sisi Timur: Lahan tambang & tanah liat/pasir hangat (#e6c8a2 / #dfbe95)
        ax.add_patch(patches.Rectangle((1150, 0), map_model.width - 1150, map_model.height, facecolor="#e6c8a2", edgecolor="none", zorder=0))
        ax.add_patch(patches.Polygon([[1150, 600], [1600, 800], [1800, 400], [1150, 300]], closed=True, facecolor="#deb68b", edgecolor="none", alpha=0.45, zorder=0))

        # Pepohonan / Semak Hiasan Tersebar di Padang Pedesaan
        tree_clump_centers = [
            (140, 1040), (230, 1020), (380, 1120), (510, 1160), (630, 1220), (680, 1050),
            (450, 410), (530, 440), (650, 310), (620, 160), (680, 80), (320, 560), (430, 580),
            (710, 1340), (730, 1200), (715, 1080), (840, 1380), (870, 1180), (950, 1300),
            (920, 1050), (960, 900), (940, 650), (910, 400), (980, 250), (890, 120)
        ]
        for tx, ty in tree_clump_centers:
            # Bayangan lembut pohon
            ax.add_patch(patches.Ellipse((tx + 2, ty - 3), 32, 20, facecolor="#b2d488", edgecolor="none", alpha=0.6, zorder=1))
            # Lingkaran tajuk pohon bertumpuk
            offsets = [(-8, -3, 13, "#2d5a27"), (8, -2, 14, "#387d3a"), (0, 6, 15, "#439247"), (-2, 2, 11, "#4cae50")]
            for ox, oy, r, col in offsets:
                ax.add_patch(patches.Circle((tx + ox, ty + oy), r, facecolor=col, edgecolor="#20461b", linewidth=0.5, zorder=1.1))

        # Bebatuan / Semak Gersang Tersebar di Sisi Tambang
        arid_clumps = [
            (1280, 1320), (1360, 1120), (1440, 1260), (1530, 1360),
            (1310, 420), (1390, 260), (1580, 220), (1420, 650),
            (1260, 950), (1320, 890), (1240, 680), (1280, 520), (1220, 180),
            (1720, 1450), (1750, 1280), (1740, 960), (1680, 780), (1650, 480)
        ]
        for ax_c, ay_c in arid_clumps:
            ax.add_patch(patches.Ellipse((ax_c + 2, ay_c - 2), 24, 15, facecolor="#cca67a", edgecolor="none", alpha=0.5, zorder=1))
            ax.add_patch(patches.Circle((ax_c - 4, ay_c), 8, facecolor="#9e6b40", edgecolor="#7a4e28", linewidth=0.5, zorder=1.1))
            ax.add_patch(patches.Circle((ax_c + 4, ay_c + 2), 7, facecolor="#b38054", edgecolor="#7a4e28", linewidth=0.5, zorder=1.1))
            ax.add_patch(patches.Circle((ax_c, ay_c - 3), 6, facecolor="#80512c", edgecolor="#593416", linewidth=0.5, zorder=1.1))
    else:
        ax.add_patch(patches.Rectangle((0, 0), map_model.width, map_model.height, facecolor="#f8fafc", edgecolor="none", zorder=0))

    # ----------------------------------------------------
    # 2. TAMPILKAN HEATMAP JIKA DIMINTA
    # ----------------------------------------------------
    if show_heatmap and heatmap_grid is not None and heatmap_extent is not None:
        ax.imshow(
            heatmap_grid,
            origin="lower",
            extent=heatmap_extent,
            cmap=heatmap_cmap,
            alpha=heatmap_alpha,
            aspect="auto",
            zorder=1.5,
        )

    # ----------------------------------------------------
    # 3. ZONA TERLARANG (FOREST, PADDY, RIVER, MINES)
    # ----------------------------------------------------
    # 3A. HUTAN LINDUNG (NW) DENGAN KANOPI REALISTIS
    hutan_zone = next((z for z in map_model.forbidden_zones if z.get("type") == "hutan"), None)
    if hutan_zone is not None:
        poly_pts = hutan_zone["polygon"]
        ax.add_patch(patches.Polygon(poly_pts, closed=True, facecolor="#235327", edgecolor="#183f1d", linewidth=1.5, zorder=2))

        if is_study_map:
            # Lapisan tajuk-tajuk pohon rimbun berkubah
            rng = np.random.RandomState(101)
            gx_vals = np.linspace(20, 730, 26)
            gy_vals = np.linspace(1100, 1490, 15)
            canopy_colors = ["#183f1d", "#235327", "#2e6a32", "#3b853e", "#499f4d", "#1f4b23", "#347a38"]
            for gx in gx_vals:
                for gy in gy_vals:
                    jx = gx + rng.uniform(-14, 14)
                    jy = gy + rng.uniform(-14, 14)
                    if jx < 750 and jy > 1070:
                        cr = rng.uniform(32, 52)
                        ccol = canopy_colors[rng.randint(0, len(canopy_colors))]
                        ax.add_patch(patches.Circle((jx, jy), cr, facecolor=ccol, edgecolor="#143618", linewidth=0.6, alpha=0.95, zorder=2.1))
                        ax.add_patch(patches.Circle((jx - cr * 0.22, jy + cr * 0.22), cr * 0.55, facecolor="#57b85c", edgecolor="none", alpha=0.25, zorder=2.2))

            for ex in np.linspace(15, 745, 28):
                er = rng.uniform(34, 48)
                ecol = rng.choice(["#235327", "#2e6a32", "#3b853e"])
                ax.add_patch(patches.Circle((ex, 1080), er, facecolor=ecol, edgecolor="#183f1d", linewidth=0.7, zorder=2.3))

            for ey in np.linspace(1080, 1500, 18):
                er = rng.uniform(34, 48)
                ecol = rng.choice(["#235327", "#2e6a32", "#3b853e"])
                ax.add_patch(patches.Circle((740, ey), er, facecolor=ecol, edgecolor="#183f1d", linewidth=0.7, zorder=2.3))

    # 3B. SAWAH (SW) DENGAN KISI-KISI PETAK PERTANIAN
    sawah_zone = next((z for z in map_model.forbidden_zones if z.get("type") == "sawah"), None)
    if sawah_zone is not None:
        poly_pts = sawah_zone["polygon"]
        ax.add_patch(patches.Polygon(poly_pts, closed=True, facecolor="#8ebf4b", edgecolor="#5e842f", linewidth=1.5, zorder=2))

        if is_study_map:
            x_cuts = [0, 85, 175, 270, 365, 460, 550, 640]
            y_cuts = [0, 85, 170, 255, 345, 430, 520]
            paddy_palette = [
                "#a3c959", "#8ebf4b", "#b8d96e", "#9bbe47", "#c8dd7b",
                "#7fa73d", "#92bd44", "#a9cc5f", "#bcdb72", "#a6cb58"
            ]
            p_idx = 0
            for i in range(len(x_cuts) - 1):
                for j in range(len(y_cuts) - 1):
                    x0, x1 = x_cuts[i], x_cuts[i + 1]
                    y0, y1 = y_cuts[j], y_cuts[j + 1]
                    col = paddy_palette[(p_idx * 7 + (i * 3 + j * 5)) % len(paddy_palette)]
                    p_idx += 1
                    ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=col, edgecolor="#5c822e", linewidth=1.2, zorder=2.1))
                    if (i + j) % 2 == 0:
                        for row_y in np.linspace(y0 + 15, y1 - 15, 3):
                            ax.plot([x0 + 8, x1 - 8], [row_y, row_y], color="#6f9338", linewidth=0.6, alpha=0.45, zorder=2.2)
                    else:
                        for col_x in np.linspace(x0 + 15, x1 - 15, 3):
                            ax.plot([col_x, col_x], [y0 + 8, y1 - 8], color="#6f9338", linewidth=0.6, alpha=0.45, zorder=2.2)

    # 3C. SUNGAI UTAMA (MEANDERING RIVER)
    sungai_zone = next((z for z in map_model.forbidden_zones if z.get("type") == "sungai"), None)
    if sungai_zone is not None:
        poly_pts = sungai_zone["polygon"]
        ax.add_patch(patches.Polygon(poly_pts, closed=True, facecolor="#3ea4e8", edgecolor="#1a71b3", linewidth=2.2, zorder=2.5))

        if is_study_map:
            river_mid_pts = [
                (795, 1500), (765, 1150), (795, 900), (825, 750),
                (810, 500), (765, 250), (710, 0)
            ]
            rmp = np.array(river_mid_pts)
            ax.plot(rmp[:, 0], rmp[:, 1], color="#75c8f9", linewidth=2.0, alpha=0.7, zorder=2.6)
            ax.plot(rmp[:, 0] - 14, rmp[:, 1], color="#ffffff", linewidth=1.1, alpha=0.5, linestyle=(0, (8, 12)), zorder=2.6)
            ax.plot(rmp[:, 0] + 14, rmp[:, 1], color="#ffffff", linewidth=1.1, alpha=0.5, linestyle=(0, (8, 12)), zorder=2.6)

    # 3D. LAHAN TAMBANG AKTIF (OPEN PIT TERRACES)
    if is_study_map:
        pit_center = (1860, 640)
        pit_steps = [
            (270, 210, "#d9a67a", "#aa7044"),
            (225, 175, "#c59163", "#945930"),
            (180, 140, "#ab7243", "#7d421d"),
            (135, 105, "#92522c", "#652f11"),
            (95, 75, "#753c18", "#4e210a"),
            (60, 45, "#54250c", "#381504"),
        ]
        for w, h, fcol, ecol in pit_steps:
            ax.add_patch(patches.Ellipse(pit_center, w, h, angle=-10, facecolor=fcol, edgecolor=ecol, linewidth=1.2, zorder=2.1))
        # Jalan spiral akses tambang menurun
        spiral_pts = [
            (1730, 600), (1790, 710), (1920, 710), (1960, 620), (1910, 560),
            (1820, 570), (1800, 650), (1890, 660), (1880, 620), (1860, 630)
        ]
        sp = np.array(spiral_pts)
        ax.plot(sp[:, 0], sp[:, 1], color="#bf8354", linewidth=3.0, zorder=2.2)
        ax.plot(sp[:, 0], sp[:, 1], color="#5c2b0d", linewidth=0.9, linestyle="--", zorder=2.3)

    # 3E. AREA TIMBUNAN BATUBARA (STOCKPILE)
    if is_study_map:
        sp_center = (1830, 210)
        sp_steps = [
            (290, 170, "#8d6e63", "#5d4037"),
            (230, 130, "#6d4c41", "#4e342e"),
            (170, 95, "#4e342e", "#3e2723"),
            (110, 60, "#2d1c16", "#1a0f0b"),
        ]
        for w, h, fcol, ecol in sp_steps:
            ax.add_patch(patches.Ellipse(sp_center, w, h, angle=5, facecolor=fcol, edgecolor=ecol, linewidth=1.2, zorder=2.1))

    # 3F. FASILITAS MESS TAMBANG & BENGKEL ALAT BERAT
    if is_study_map:
        # Barak Mess Tambang (Bangunan Atap Biru)
        mess_coords = [
            (1820, 1340), (1860, 1340), (1900, 1340),
            (1820, 1300), (1860, 1300), (1900, 1300),
            (1820, 1260), (1860, 1260), (1900, 1260),
        ]
        for mx, my in mess_coords:
            ax.add_patch(patches.Rectangle((mx - 15, my - 10), 30, 20, facecolor="#cca67a", edgecolor="none", zorder=3.8))
            ax.add_patch(patches.Rectangle((mx - 16, my - 9), 32, 18, facecolor="#2980b9", edgecolor="#1a5276", linewidth=0.8, zorder=4.0))
            ax.plot([mx - 16, mx + 16], [my, my], color="#aed6f1", linewidth=0.8, zorder=4.1)

        # Bengkel Alat Berat (Hanggar Besar)
        bx, by = 1780, 1080
        ax.add_patch(patches.Rectangle((bx - 30, by - 22), 60, 44, facecolor="#78909c", edgecolor="#455a64", linewidth=1.2, zorder=4.0))
        ax.add_patch(patches.Rectangle((bx - 20, by - 22), 40, 16, facecolor="#37474f", edgecolor="none", zorder=4.1))
        ax.plot([bx - 30, bx + 30], [by, by], color="#b0bec5", linewidth=1.2, zorder=4.2)
        # Lencana Bengkel Alat Berat (Lingkaran Emas dengan Ikon Kunci Pas Silang)
        ax.add_patch(patches.Circle((bx - 34, by + 18), 16, facecolor="#f39c12", edgecolor="#ffffff", linewidth=1.5, zorder=6.8))
        ax.plot([bx - 40, bx - 28], [by + 12, by + 24], color="#ffffff", linewidth=2.4, solid_capstyle="round", zorder=6.9)
        ax.plot([bx - 28, bx - 40], [by + 12, by + 24], color="#ffffff", linewidth=2.4, solid_capstyle="round", zorder=6.9)
        ax.add_patch(patches.Circle((bx - 34, by + 18), 4, facecolor="#f39c12", edgecolor="#ffffff", linewidth=1.2, zorder=7.0))

    # ----------------------------------------------------
    # 4. PERMUKIMAN WARGA (SETTLEMENT) & RUMAH ILUSTRASI
    # ----------------------------------------------------
    if is_study_map:
        # Lingkaran Dotted Permukiman Warga (Glow Biru Lembut)
        ellipse = patches.Ellipse(
            (280, 765), width=490, height=430,
            facecolor="#dcf0fd", edgecolor="#258cdb",
            linestyle="--", linewidth=1.8, alpha=0.35, zorder=3,
        )
        ax.add_patch(ellipse)

    # RUMAH-RUMAH WARGA TERILUSTRASI (Terracotta Cottages)
    if len(map_model.houses) > 0:
        for h in map_model.houses:
            hx, hy = h[0], h[1]
            w, h_dim = 24, 16
            ax.add_patch(patches.Rectangle((hx - w/2 + 2, hy - h_dim/2 - 2), w, h_dim, facecolor="#b4d588" if is_study_map else "#cbd5e1", edgecolor="none", alpha=0.7, zorder=3.8))
            ax.add_patch(patches.Rectangle((hx - w/2, hy - h_dim/2), w, h_dim * 0.65, facecolor="#f8fafc", edgecolor="#94a3b8", linewidth=0.5, zorder=3.9))
            ax.add_patch(patches.Rectangle((hx - 2.5, hy - h_dim/2), 5, 5, facecolor="#64748b", edgecolor="none", zorder=3.95))
            ax.add_patch(patches.Rectangle((hx - 8, hy - h_dim/2 + 2), 3.5, 3.5, facecolor="#38bdf8", edgecolor="#64748b", linewidth=0.4, zorder=3.95))

            roof_palette = [("#e67e22", "#c0392b"), ("#d35400", "#a93226"), ("#e74c3c", "#922b21")]
            c_light, c_dark = roof_palette[int(hx + hy) % len(roof_palette)]
            poly_roof_l = [[hx - w/2 - 2, hy - 1], [hx, hy + h_dim/2], [hx, hy - 1]]
            ax.add_patch(patches.Polygon(poly_roof_l, closed=True, facecolor=c_light, edgecolor="#922b21", linewidth=0.5, zorder=4.0))
            poly_roof_r = [[hx + w/2 + 2, hy - 1], [hx, hy + h_dim/2], [hx, hy - 1]]
            ax.add_patch(patches.Polygon(poly_roof_r, closed=True, facecolor=c_dark, edgecolor="#922b21", linewidth=0.5, zorder=4.0))
            ax.plot([hx, hx], [hy - 1, hy + h_dim/2], color="#ffffff", linewidth=0.8, zorder=4.1)

    # ----------------------------------------------------
    # 5. JARINGAN JALAN & JEMBATAN
    # ----------------------------------------------------
    # 5A. JEMBATAN PADA SUNGAI (Jl. Utama Desa menyeberang sungai)
    if is_study_map:
        ax.add_patch(patches.Rectangle((750, 792), 100, 26, facecolor="#94a3b8", edgecolor="#334155", linewidth=1.5, zorder=4.2))
        ax.plot([750, 850], [805 + 13, 805 + 13], color="#1e293b", linewidth=2.5, zorder=4.3)
        ax.plot([750, 850], [805 - 13, 805 - 13], color="#1e293b", linewidth=2.5, zorder=4.3)

    # 5B. GAMBAR JALAN BERDASARKAN KELAS
    for road in map_model.roads_raw:
        pts = np.array(road.get("points", []), dtype=np.float64)
        r_class = road.get("class", "lokal")

        if r_class == "arteri":
            # JL. LINTAS PROVINSI (4 Lajur Aspal Gelap)
            ax.plot(pts[:, 0], pts[:, 1], color="#202428", linewidth=13.0, solid_capstyle="butt", zorder=4.5)
            ax.plot(pts[:, 0], pts[:, 1], color="#33383e", linewidth=10.5, solid_capstyle="butt", zorder=4.6)
            ax.plot(pts[:, 0] - 8, pts[:, 1], color="#e2e8f0", linewidth=1.2, zorder=4.7)
            ax.plot(pts[:, 0] + 8, pts[:, 1], color="#e2e8f0", linewidth=1.2, zorder=4.7)
            ax.plot(pts[:, 0], pts[:, 1], color="#ffffff", linewidth=1.8, linestyle=(0, (6, 6)), zorder=4.8)

        elif r_class == "hauling":
            # JALAN HAULING (Jalur Lebar Tanah Merah / Clay dengan Jejak Ban Truk)
            ax.plot(pts[:, 0], pts[:, 1], color="#dfaa85", linewidth=18.0, alpha=0.45, solid_capstyle="round", zorder=4.4)
            ax.plot(pts[:, 0], pts[:, 1], color="#bf7952", linewidth=13.5, solid_capstyle="round", zorder=4.5)
            ax.plot(pts[:, 0] - 3.0, pts[:, 1], color="#8a4928", linewidth=1.6, linestyle="-", zorder=4.6)
            ax.plot(pts[:, 0] + 3.0, pts[:, 1], color="#8a4928", linewidth=1.6, linestyle="-", zorder=4.6)

        elif r_class == "kolektor":
            # JALAN KOLEKTOR (Jl. Utama Desa, Jl. Akses Tambang)
            ax.plot(pts[:, 0], pts[:, 1], color="#64748b", linewidth=7.2, solid_capstyle="round", zorder=4.4)
            ax.plot(pts[:, 0], pts[:, 1], color="#8a95a5", linewidth=5.2, solid_capstyle="round", zorder=4.5)

        else:
            # JALAN LOKAL
            ax.plot(pts[:, 0], pts[:, 1], color="#94a3b8", linewidth=4.8, solid_capstyle="round", zorder=4.4)
            ax.plot(pts[:, 0], pts[:, 1], color="#cbd5e1", linewidth=3.4, solid_capstyle="round", zorder=4.5)

    # 5C. TRUK TAMBANG KUNING REALISTIS DI JALAN HAULING (DUMP TRUCKS)
    if is_study_map:
        truck_positions = [
            (1655, 1320, -100),
            (1615, 1050, -105),
            (1590, 840, -100),
            (1545, 630, -102),
            (1495, 340, -100),
            (1460, 120, -100),
        ]
        for tx, ty, rot_deg in truck_positions:
            rad = math.radians(rot_deg)
            cos_r, sin_r = math.cos(rad), math.sin(rad)
            wheel_w, wheel_h = 8, 4.5

            def rot(dx, dy):
                return (tx + dx * cos_r - dy * sin_r, ty + dx * sin_r + dy * cos_r)

            for wdx, wdy in [(-11, -9), (-11, 9), (11, -9), (11, 9)]:
                wx, wy = rot(wdx, wdy)
                ax.add_patch(patches.Rectangle((wx - wheel_w/2, wy - wheel_h/2), wheel_w, wheel_h, facecolor="#1e293b", edgecolor="#0f172a", linewidth=0.5, zorder=5.0))

            dump_pts = [rot(-14, -8), rot(7, -8), rot(7, 8), rot(-14, 8)]
            ax.add_patch(patches.Polygon(dump_pts, closed=True, facecolor="#f1c40f", edgecolor="#b7950b", linewidth=0.8, zorder=5.1))
            load_pts = [rot(-12, -6), rot(5, -6), rot(5, 6), rot(-12, 6)]
            ax.add_patch(patches.Polygon(load_pts, closed=True, facecolor="#3e2723", edgecolor="none", zorder=5.2))

            cab_pts = [rot(7, -7), rot(15, -7), rot(15, 7), rot(7, 7)]
            ax.add_patch(patches.Polygon(cab_pts, closed=True, facecolor="#f39c12", edgecolor="#b7950b", linewidth=0.8, zorder=5.3))
            windshield = [rot(9, -5), rot(14, -5), rot(14, 5), rot(9, 5)]
            ax.add_patch(patches.Polygon(windshield, closed=True, facecolor="#2c3e50", edgecolor="none", zorder=5.4))

    # ----------------------------------------------------
    # 6. FASILITAS UMUM (LINGKARAN BADGE BERIKON)
    # ----------------------------------------------------
    facility_markers = {
        "sekolah": ("#1d4ed8", "S", 135),
        "pasar": ("#16a34a", "P", 145),
        "restoran": ("#ea580c", "R", 130),
        "bengkel": ("#9333ea", "B", 135),
        "masjid_kantor": ("#059669", "M", 155),
    }

    for fac in map_model.facilities_raw:
        ftype = fac.get("type", "fasilitas")
        color, letter, size = facility_markers.get(ftype, ("#0284c7", "*", 95))
        fx, fy = fac["x"], fac["y"]

        # Bayangan badge
        ax.scatter(fx + 2, fy - 2, c="#0f172a", marker="o", s=size, alpha=0.35, edgecolors="none", zorder=6.8)
        # Lingkaran badge utama
        ax.scatter(fx, fy, c=color, marker="o", s=size, edgecolors="#ffffff", linewidths=2.0, zorder=7.0)

        if ftype == "masjid_kantor":
            dome_poly = [
                [fx - 7, fy - 3], [fx - 5, fy + 4], [fx, fy + 8], [fx + 5, fy + 4], [fx + 7, fy - 3]
            ]
            ax.add_patch(patches.Polygon(dome_poly, closed=True, facecolor="#ffffff", edgecolor="none", zorder=7.2))
            ax.plot([fx, fx], [fy + 8, fy + 11], color="#ffffff", linewidth=1.2, zorder=7.3)
            ax.text(fx, fy - 1, "M", color="#059669", fontsize=6.5, weight="bold", ha="center", va="center", zorder=7.4)
        else:
            ax.text(fx, fy, letter, color="#ffffff", fontsize=9.0, weight="bold", ha="center", va="center", zorder=7.2)

    # ----------------------------------------------------
    # 7. KOMPETITOR (MINIMARKET LAMA K)
    # ----------------------------------------------------
    if len(map_model.competitors) > 0:
        for cx, cy in map_model.competitors:
            ax.scatter(cx + 2, cy - 2, c="#0f172a", marker="o", s=150, alpha=0.35, edgecolors="none", zorder=6.8)
            ax.scatter(cx, cy, c="#dc2626", marker="o", s=150, edgecolors="#ffffff", linewidths=2.2, zorder=7.0)
            ax.text(cx, cy, "K", color="#ffffff", fontsize=9.5, weight="bold", ha="center", va="center", zorder=7.2)

    # ----------------------------------------------------
    # 8. LABEL & BADGE PILL SPASIAL SESUAI GAMBAR REFERENSI
    # ----------------------------------------------------
    if is_study_map:
        def pill_label(x, y, text, fcol, text_col="#ffffff", font_size=7.5, pad=0.35):
            ax.text(x + 2, y - 2, text, color="#0f172a", fontsize=font_size, weight="bold", ha="center", va="center",
                    bbox=dict(boxstyle=f"round,pad={pad}", facecolor="#0f172a", edgecolor="none", alpha=0.3), zorder=6.0)
            ax.text(x, y, text, color=text_col, fontsize=font_size, weight="bold", ha="center", va="center",
                    bbox=dict(boxstyle=f"round,pad={pad}", facecolor=fcol, edgecolor="#ffffff", linewidth=0.8, alpha=0.96), zorder=6.2)

        # Label Wilayah / Zona
        pill_label(90, 1370, "Hutan", "#143818", pad=0.4, font_size=8.5)
        pill_label(110, 260, "Sawah", "#244517", pad=0.4, font_size=8.5)
        pill_label(815, 480, "Sungai", "#1565c0", pad=0.4, font_size=8.5)
        pill_label(135, 1025, "Permukiman warga", "#1565c0", pad=0.4, font_size=8.2)

        # Label Jalan
        pill_label(210, 680, "Jl. Utama Desa", "#1e293b", font_size=7.2)
        pill_label(335, 1060, "Jl. Lingkar Timur", "#1e293b", font_size=7.2)
        pill_label(340, 540, "Jl. Dusun Barat", "#1e293b", font_size=7.2)
        pill_label(1150, 1260, "Jl. Lintas Provinsi", "#1e293b", font_size=8.0)
        pill_label(1350, 770, "Jl. Akses Tambang", "#1e293b", font_size=7.2)
        pill_label(1660, 1440, "Jalan Hauling", "#78350f", font_size=8.0)

        # Label Fasilitas Tambang
        pill_label(1840, 1375, "Mess tambang", "#78350f", font_size=7.5)
        pill_label(1850, 1140, "Bengkel alat berat", "#78350f", font_size=7.5)
        pill_label(1820, 620, "Lahan tambang aktif", "#2c1b12", font_size=8.0)
        pill_label(1830, 210, "Area timbunan\n(stockpile)", "#2c1b12", font_size=7.2)

    # ----------------------------------------------------
    # 9. BANNER JUDUL TOP HEADER & BATAS SUMBU
    # ----------------------------------------------------
    ax.set_xlim(0, map_model.width)
    ax.set_ylim(0, map_model.height)
    ax.set_aspect("equal")

    # Banner Header Biru Modern seperti di image.png
    if is_study_map and show_banner:
        header_text = "Peta Studi: Desa, Jalan Lintas, dan Kawasan Hauling"
        font_sz = 11.5 if show_legend else 9.5
        ax.text(
            25, map_model.height - 35,
            f" {header_text} ",
            color="#ffffff",
            fontsize=font_sz,
            weight="bold",
            ha="left",
            va="top",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#163b65", edgecolor="#ffffff", linewidth=1.2, alpha=0.96),
            zorder=6.5,
        )

    for spine in ax.spines.values():
        spine.set_color("#64748b")
        spine.set_linewidth(1.2)

    ax.set_xticks([])
    ax.set_yticks([])

    # ----------------------------------------------------
    # 10. LEGENDA BAWAH SEPERTI DI GAMBAR REFERENSI
    # ----------------------------------------------------
    if show_legend:
        legend_elements = [
            # Ikon Bangunan & Fasilitas
            patches.Patch(facecolor="#e11d48", edgecolor="#9f1239", label="Rumah"),
            lines.Line2D([0], [0], marker="o", color="w", markerfacecolor="#1d4ed8", markersize=8.5, label="Sekolah"),
            lines.Line2D([0], [0], marker="o", color="w", markerfacecolor="#16a34a", markersize=8.5, label="Pasar"),
            lines.Line2D([0], [0], marker="o", color="w", markerfacecolor="#ea580c", markersize=8.5, label="Restoran / Warung"),
            lines.Line2D([0], [0], marker="o", color="w", markerfacecolor="#9333ea", markersize=8.5, label="Bengkel"),
            lines.Line2D([0], [0], marker="o", color="w", markerfacecolor="#dc2626", markersize=8.5, label="Minimarket lama"),
            # Jalan
            lines.Line2D([0], [0], color="#94a3b8", lw=3.0, label="Jalan lokal"),
            lines.Line2D([0], [0], color="#64748b", lw=4.0, label="Jalan kolektor"),
            lines.Line2D([0], [0], color="#202428", lw=5.0, linestyle="--", label="Jalan arteri"),
            lines.Line2D([0], [0], color="#bf7952", lw=5.0, label="Jalan hauling"),
            lines.Line2D([0], [0], color="#258cdb", lw=2.0, linestyle="--", label="Permukiman warga"),
            # Zona
            patches.Patch(facecolor="#235327", edgecolor="#143618", label="Hutan"),
            patches.Patch(facecolor="#8ebf4b", edgecolor="#5e842f", label="Sawah"),
            patches.Patch(facecolor="#3ea4e8", edgecolor="#1a71b3", label="Sungai"),
            patches.Patch(facecolor="#c59163", edgecolor="#7d421d", label="Tambang"),
        ]
        leg = ax.legend(
            handles=legend_elements,
            loc="upper center",
            bbox_to_anchor=(0.44, -0.02),
            ncol=5,
            fontsize=8.2,
            frameon=True,
            facecolor="#ffffff",
            edgecolor="#94a3b8",
            framealpha=0.98,
        )
        leg.get_frame().set_linewidth(1.2)

        # Kompas Arah Utara (North Arrow) & Skala 1 km persis image.png
        ax.annotate(
            "N", xy=(1920, -55), xytext=(1920, -115),
            arrowprops=dict(facecolor="#1e293b", edgecolor="#0f172a", width=2.5, headwidth=9),
            ha="center", va="center", fontsize=9, weight="bold", color="#1e293b",
            annotation_clip=False, zorder=20
        )
        ax.plot([1780, 1980], [-135, -135], color="#1e293b", linewidth=2.0, clip_on=False, zorder=20)
        ax.plot([1780, 1780], [-130, -140], color="#1e293b", linewidth=2.0, clip_on=False, zorder=20)
        ax.plot([1980, 1980], [-130, -140], color="#1e293b", linewidth=2.0, clip_on=False, zorder=20)
        ax.text(1880, -150, "1 km", ha="center", va="top", fontsize=8.5, weight="bold", color="#1e293b", clip_on=False, zorder=20)

    return ax



