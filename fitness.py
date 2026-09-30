"""
Modul Fungsi Fitness (fitness.py)
Menghitung fungsi kesesuaian (fitness) multi-kriteria untuk penempatan minimarket:
F(x,y) = w1*Populasi + w2*AksesJalan + w3*Fasilitas + w4*Kompetitor - Penalti

Mendukung:
- Normalisasi fitur dalam rentang [0, 1]
- Mode optimasi: 'max' (lokasi terbaik) dan 'min_valid' (lokasi terburuk yang tetap valid)
- Pembalikan fitur (invert_features) per komponen
- Penempatan p = 1, 2, 3 toko dengan penalti kanibalisasi antar toko baru
- Counter evaluasi internal untuk memastikan matched evaluation budget
- Prekomputasi grid heatmap tanpa memengaruhi counter evaluasi algoritma

Catatan Desain: Modul inti ini murni komputasi matematika dengan NumPy,
bebas dari impor matplotlib.
"""

from typing import Dict, List, Tuple, Any, Optional, Union
import json
import numpy as np

from map_model import MapModel


class FitnessEvaluator:
    """
    Kelas evaluator fungsi fitness dengan penghitung evaluasi (eval_count)
    dan vektorisasi NumPy efisien.
    """

    def __init__(
        self,
        map_model: MapModel,
        config: Optional[Dict[str, Any]] = None,
        config_file: Optional[str] = None,
        mode: str = "max",
        invert_features: Optional[List[str]] = None,
        num_stores: int = 1,
    ):
        """
        Inisialisasi FitnessEvaluator.
        """
        self.map = map_model

        # Muat konfigurasi dari file jika diberikan
        if config is None and config_file is not None:
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
        elif config is None:
            config = {}

        self.config = config
        self.mode = mode.lower()
        if self.mode not in ("max", "min_valid"):
            raise ValueError(f"Mode tidak dikenal: '{mode}'. Gunakan 'max' atau 'min_valid'.")

        self.invert_features = set(invert_features or config.get("optimization", {}).get("invert_features", []))
        self.num_stores = int(num_stores or config.get("facility", {}).get("num_stores", 1))

        # Ekstraksi bobot kriteria
        weights = config.get("fitness_weights", {})
        self.w_pop = float(weights.get("w1_populasi", 0.35))
        self.w_road = float(weights.get("w2_akses_jalan", 0.25))
        self.w_fac = float(weights.get("w3_fasilitas", 0.20))
        self.w_comp = float(weights.get("w4_kompetitor", 0.20))

        # Ekstraksi parameter kriteria
        params = config.get("fitness_params", {})
        self.sigma_pop = float(params.get("sigma_populasi", 250.0))
        self.sigma_road = float(params.get("sigma_jalan", 150.0))
        self.sigma_fac = float(params.get("sigma_fasilitas", 200.0))
        self.comp_d_opt = float(params.get("competitor_d_opt", 350.0))

        self.penalty_forbidden = float(params.get("penalty_forbidden", 1000.0))
        self.penalty_out_of_bounds = float(params.get("penalty_out_of_bounds", 1000.0))

        # Parameter kanibalisasi toko
        fac_cfg = config.get("facility", {})
        self.cannibal_dist = float(fac_cfg.get("cannibalization_distance", 400.0))
        self.cannibal_weight = float(fac_cfg.get("cannibalization_weight", 0.5))

        # Bobot per jenis fasilitas dan jalan
        self.facility_weights = params.get(
            "facility_weights",
            {"sekolah": 1.5, "pasar": 2.0, "restoran": 1.2, "bengkel": 0.8, "masjid_kantor": 1.0},
        )
        self.road_class_weights = params.get(
            "road_class_weights",
            {"arteri": 1.0, "kolektor": 0.85, "lokal": 0.6, "hauling": 0.2},
        )
        # Batasan koridor fisik jalan (titik terjauh penempatan minimarket adalah di kawasan jalan)
        self.max_road_corridor = float(params.get("max_road_corridor", 50.0))

        # Faktor normalisasi skala populasi & fasilitas agar bernilai [0, 1]
        self._calibrate_normalization_factors()

        # Counter evaluasi fitness
        self.eval_count: int = 0

    def _calibrate_normalization_factors(self) -> None:
        """
        Menentukan faktor pembagi agar fitur populasi dan fasilitas terkalibrasi ke skala [0, 1].
        """
        # Kalibrasi populasi: estimasi nilai maksimum di pusat konsentrasi pemukiman
        if len(self.map.houses) > 0:
            sample_pts = self.map.houses[:, :2]
            # Hitung densitas sampel
            dists_sq = np.sum((sample_pts[:, None, :] - sample_pts[None, :, :]) ** 2, axis=2)
            weights = self.map.houses[:, 2]
            densities = np.sum(weights[None, :] * np.exp(-dists_sq / (2.0 * (self.sigma_pop ** 2))), axis=1)
            self.max_pop_score = float(np.max(densities)) if len(densities) > 0 else 1.0
            self.max_pop_score = max(self.max_pop_score, 1.0)
        else:
            self.max_pop_score = 1.0

        # Kalibrasi fasilitas
        total_fac_weight = 0.0
        for ftype, arr in self.map.facilities_by_type.items():
            base_w = self.facility_weights.get(ftype, 1.0)
            if len(arr) > 0:
                total_fac_weight += base_w * np.sum(arr[:, 2])
        self.max_fac_score = max(total_fac_weight * 0.45, 1.0)

    def reset_counter(self) -> None:
        """
        Mereset penghitung evaluasi ke 0.
        """
        self.eval_count = 0

    def evaluate_features_single_point(self, point: np.ndarray) -> Dict[str, Union[float, bool]]:
        """
        Menghitung nilai masing-masing fitur [0, 1] untuk satu koordinat toko (x, y).
        point: array-like shape (2,)
        """
        x, y = float(point[0]), float(point[1])
        pt_2d = np.array([[x, y]], dtype=np.float64)

        # 1. Cek Validitas Spasial (Luar Batas Peta & Zona Terlarang)
        is_oob = bool(self.map.is_out_of_bounds(pt_2d)[0])
        is_forbid = bool(self.map.is_in_forbidden_zone(pt_2d)[0])
        forbid_name = self.map.get_forbidden_zone_name(pt_2d) if is_forbid else None

        # 2. Cek Koridor Fisik Jalan & Kualitas Jalan Terdekat
        d_road_geo, nearest_road_class, nearest_road_name = self.map.get_nearest_road_info(pt_2d)
        is_in_road_corridor = d_road_geo <= self.max_road_corridor
        off_dist = max(0.0, d_road_geo - self.max_road_corridor)

        is_valid = (not is_oob) and (not is_forbid) and is_in_road_corridor

        # 3. Fitur Populasi: Peluruhan Gaussian terhadap perumahan
        if len(self.map.houses) > 0:
            h_coords = self.map.houses[:, :2]
            h_weights = self.map.houses[:, 2]
            d_sq = np.sum((h_coords - pt_2d) ** 2, axis=1)
            pop_val = np.sum(h_weights * np.exp(-d_sq / (2.0 * (self.sigma_pop ** 2))))
            s_pop = float(np.clip(pop_val / self.max_pop_score, 0.0, 1.0))
        else:
            s_pop = 0.0

        # 4. Fitur Kondisi Jalan: Proper (Aspal Mulus 0.85-1.0) vs Buruk (Tanah Lumpur Hauling 0.20)
        base_quality = self.road_class_weights.get(nearest_road_class, 0.50)
        s_road = base_quality * float(np.exp(-(d_road_geo ** 2) / (2.0 * (self.sigma_road ** 2))))
        s_road = float(np.clip(s_road, 0.0, 1.0))

        # 5. Fitur Fasilitas Umum: Peluruhan Gaussian terhadap fasilitas terdekat
        fac_sum = 0.0
        for ftype, arr in self.map.facilities_by_type.items():
            if len(arr) > 0:
                f_coords = arr[:, :2]
                f_weights = arr[:, 2]
                base_w = self.facility_weights.get(ftype, 1.0)
                d_sq = np.sum((f_coords - pt_2d) ** 2, axis=1)
                fac_sum += base_w * np.sum(f_weights * np.exp(-d_sq / (2.0 * (self.sigma_fac ** 2))))
        s_fac = float(np.clip(fac_sum / self.max_fac_score, 0.0, 1.0))

        # 6. Fitur Kompetitor: Kurva Ricker Tajam & Kanibalisasi Dekat
        if len(self.map.competitors) > 0:
            d_comp_sq = np.sum((self.map.competitors - pt_2d) ** 2, axis=1)
            d_comp_min = float(np.sqrt(np.min(d_comp_sq)))
            ratio = d_comp_min / self.comp_d_opt
            s_comp = (ratio ** 1.6) * np.exp(1.0 - (ratio ** 1.6))
            # Kanibalisasi langsung jika sangat dekat (< 180m)
            if d_comp_min < 180.0:
                s_comp *= (d_comp_min / 180.0) ** 1.8
            s_comp = float(np.clip(s_comp, 0.0, 1.0))
        else:
            s_comp = 0.8
            d_comp_min = 9999.0

        # 7. Penerapan Opsi Invert per-fitur jika dikonfigurasi
        if "populasi" in self.invert_features:
            s_pop = 1.0 - s_pop
        if "akses_jalan" in self.invert_features or "kondisi_jalan" in self.invert_features:
            s_road = 1.0 - s_road
        if "fasilitas" in self.invert_features:
            s_fac = 1.0 - s_fac
        if "kompetitor" in self.invert_features:
            s_comp = 1.0 - s_comp

        # Nilai dasar kecocokan lokasi (raw score [0, 1])
        if is_oob or is_forbid:
            raw_score = -self.penalty_forbidden
        elif not is_in_road_corridor:
            raw_score = -10.0 - (off_dist / 50.0)
        else:
            raw_score = (
                self.w_pop * s_pop
                + self.w_road * s_road
                + self.w_fac * s_fac
                + self.w_comp * s_comp
            )

        return {
            "populasi": s_pop,
            "akses_jalan": s_road,
            "kondisi_jalan": s_road,
            "fasilitas": s_fac,
            "kompetitor": s_comp,
            "is_valid": is_valid,
            "is_oob": is_oob,
            "is_forbidden": is_forbid,
            "forbidden_zone_name": forbid_name,
            "is_in_road_corridor": is_in_road_corridor,
            "d_road_geo": d_road_geo,
            "nearest_road_class": nearest_road_class,
            "nearest_road_name": nearest_road_name,
            "d_comp": d_comp_min,
            "off_road_dist": off_dist,
            "max_road_corridor": self.max_road_corridor,
            "raw_score": float(raw_score),
        }

    def evaluate_candidate(self, solution: np.ndarray, count_evaluation: bool = True) -> float:
        """
        Mengevaluasi fitness satu vektor solusi p-toko [x1, y1, ..., xp, yp].
        Jika count_evaluation=True, self.eval_count bertambah 1.
        Penempatan toko dibatasi di koridor jalan (d <= max_road_corridor).
        """
        if count_evaluation:
            self.eval_count += 1

        sol = np.asarray(solution, dtype=np.float64).flatten()
        p = len(sol) // 2

        if p < 1:
            return -self.penalty_out_of_bounds

        stores = sol.reshape((p, 2))
        store_scores = []
        max_off_dist = 0.0

        for k in range(p):
            pt_2d = stores[k : k + 1]
            # Cek batas peta dan zona terlarang
            if self.map.is_out_of_bounds(pt_2d)[0] or self.map.is_in_forbidden_zone(pt_2d)[0]:
                return -self.penalty_forbidden

            # Cek koridor fisik jalan
            geo_d = float(self.map.geometric_distance_to_roads(pt_2d)[0])
            if geo_d > self.max_road_corridor:
                off_d = geo_d - self.max_road_corridor
                max_off_dist = max(max_off_dist, off_d)

            res = self.evaluate_features_single_point(stores[k])
            store_scores.append(res["raw_score"])

        # Jika ada toko di luar koridor jalan (off-road), berikan penalti negatif tergradien
        # Penalti ini SELALU bernilai negatif di KEDUA mode (max dan min_valid)
        # agar partikel/individu selalu tertarik masuk ke koridor jalan
        if max_off_dist > 0.0:
            return float(-10.0 - (max_off_dist / 50.0))

        # Rata-rata skor semua toko yang valid di koridor jalan
        avg_score = float(np.mean(store_scores))

        # Penalti kanibalisasi jika p > 1 dan jarak antar toko terlalu dekat
        cannibal_penalty = 0.0
        if p > 1:
            for i in range(p):
                for j in range(i + 1, p):
                    dist_ij = float(np.linalg.norm(stores[i] - stores[j]))
                    if dist_ij < self.cannibal_dist:
                        # Penalti kuadratik semakin dekat jaraknya
                        overlap = 1.0 - (dist_ij / self.cannibal_dist)
                        cannibal_penalty += self.cannibal_weight * (overlap ** 2)

        # Penyesuaian berdasarkan mode:
        # 'max': maksimalkan skor bersih di koridor jalan
        # 'min_valid': cari lokasi terburuk yang VALID di koridor jalan (skor valid paling kecil).
        if self.mode == "max":
            fitness = avg_score - cannibal_penalty
        else:  # mode == 'min_valid'
            fitness = (1.0 - avg_score) - cannibal_penalty

        return float(fitness)

    def evaluate(self, solution: np.ndarray) -> float:
        """
        Alias praktis untuk evaluate_candidate dengan pencatatan evaluasi otomatis.
        """
        return self.evaluate_candidate(solution, count_evaluation=True)

    def evaluate_batch(self, population: np.ndarray) -> np.ndarray:
        """
        Mengevaluasi populasi/kumpulan partikel (M individu) sekaligus.
        population shape: (M, 2*p)
        Menambah eval_count sebanyak M.
        """
        pop = np.asarray(population, dtype=np.float64)
        m = len(pop)
        fitnesses = np.empty(m, dtype=np.float64)

        for i in range(m):
            fitnesses[i] = self.evaluate_candidate(pop[i], count_evaluation=True)

        return fitnesses

    def compute_grid_heatmap(
        self, resolution_x: int = 80, resolution_y: int = 60
    ) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        """
        Melakukan prekomputasi matriks heatmap kesesuaian lokasi 2D untuk visualisasi.
        CATATAN PENTING: Metode ini TIDAK menambah self.eval_count agar visualisasi
        tidak mengacaukan anggaran evaluasi GA/PSO.
        Titik di luar koridor jalan dan zona terlarang diset NaN agar heatmap hanya menyorot koridor jalan.

        Returns:
            heatmap_grid: array 2D shape (resolution_y, resolution_x)
            extent: tuple (0, width, 0, height) untuk ax.imshow
        """
        xs = np.linspace(0, self.map.width, resolution_x)
        ys = np.linspace(0, self.map.height, resolution_y)
        grid = np.zeros((resolution_y, resolution_x), dtype=np.float64)

        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                pt = np.array([x, y])
                val = self.evaluate_candidate(pt, count_evaluation=False)
                # Hanya warnai area koridor jalan yang valid (val >= 0.0)
                if val < 0.0:
                    val = np.nan
                grid[j, i] = val

        extent = (0.0, self.map.width, 0.0, self.map.height)
        return grid, extent
