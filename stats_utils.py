"""
Modul Utilitas Statistik (stats_utils.py)
Menghitung metrik performa algoritma optimasi:
- Evaluasi menuju 95% fitness akhir (evaluasi_ke_95persen)
- Ringkasan statistik deskriptif (mean, std, median, IQR, min, max, waktu)
- Uji hipotesis non-parametrik Mann-Whitney U test menggunakan scipy.stats
- Penyusunan kesimpulan komparatif otomatis dalam Bahasa Indonesia

Catatan Desain: Modul ini murni komputasi statistik (numpy, scipy, pandas),
tidak bergantung pada matplotlib.
"""

from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats


def compute_eval_to_95_percent(
    evaluations: np.ndarray, fitness_history: np.ndarray
) -> int:
    """
    Menghitung jumlah evaluasi yang dibutuhkan algoritma untuk mencapai
    setidaknya 95% dari peningkatan fitness akhir:
        F_target = F_awal + 0.95 * (F_akhir - F_awal)

    Jika tidak ada peningkatan berarti, mengembalikan evaluasi pertama.
    """
    evals = np.asarray(evaluations, dtype=np.int64)
    fits = np.asarray(fitness_history, dtype=np.float64)

    if len(evals) == 0 or len(fits) == 0:
        return 0

    f_start = fits[0]
    f_final = fits[-1]
    delta = f_final - f_start

    if delta <= 1e-6:
        return int(evals[0])

    target = f_start + 0.95 * delta
    reached_indices = np.where(fits >= target)[0]

    if len(reached_indices) > 0:
        return int(evals[reached_indices[0]])
    return int(evals[-1])


def summarize_runs(df: pd.DataFrame, group_col: str = "algoritma") -> pd.DataFrame:
    """
    Menghasilkan ringkasan statistik deskriptif dari DataFrame hasil run.
    Metrik: Mean, Std, Median, IQR, Min (Worst), Max (Best), Waktu Rata-rata.
    """
    summary_rows = []

    for group_name, group_data in df.groupby(group_col):
        fits = group_data["fitness"].to_numpy(dtype=np.float64)
        times = group_data["waktu_detik"].to_numpy(dtype=np.float64)
        evals_95 = group_data["evaluasi_ke_95persen"].to_numpy(dtype=np.float64)

        q25, q50, q75 = np.percentile(fits, [25, 50, 75])
        iqr = q75 - q25

        summary_rows.append({
            "Algoritma": group_name,
            "Jumlah Run": len(fits),
            "Mean Fitness": np.mean(fits),
            "Std Fitness": np.std(fits, ddof=1) if len(fits) > 1 else 0.0,
            "Median Fitness": q50,
            "IQR Fitness": iqr,
            "Worst (Min)": np.min(fits),
            "Best (Max)": np.max(fits),
            "Mean Waktu (detik)": np.mean(times),
            "Std Waktu (detik)": np.std(times, ddof=1) if len(times) > 1 else 0.0,
            "Mean Eval ke-95%": np.mean(evals_95),
            "Median Eval ke-95%": np.median(evals_95),
        })

    summary_df = pd.DataFrame(summary_rows)
    return summary_df


def perform_mann_whitney_u_test(
    data_ga: np.ndarray,
    data_pso: np.ndarray,
    metric_name: str = "Nilai Fitness Akhir",
    higher_is_better: bool = True,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Melakukan uji non-parametrik Mann-Whitney U test antara GA dan PSO.
    Menghasilkan nilai statistik U, p-value, dan teks interpretasi kesimpulan ilmiah.
    """
    arr_ga = np.asarray(data_ga, dtype=np.float64)
    arr_pso = np.asarray(data_pso, dtype=np.float64)

    # Uji dua arah (two-sided)
    u_stat, p_val = stats.mannwhitneyu(arr_ga, arr_pso, alternative="two-sided")

    n1, n2 = len(arr_ga), len(arr_pso)
    # Ukuran efek Rank-Biserial correlation: r = 1 - (2*U / (n1*n2))
    rank_biserial = 1.0 - (2.0 * u_stat / (n1 * n2))

    med_ga = float(np.median(arr_ga))
    med_pso = float(np.median(arr_pso))
    mean_ga = float(np.mean(arr_ga))
    mean_pso = float(np.mean(arr_pso))

    is_significant = p_val < alpha

    # Penyusunan teks kesimpulan komparatif dalam Bahasa Indonesia
    if is_significant:
        if higher_is_better:
            winner = "GA" if med_ga > med_pso else "PSO"
        else:
            winner = "GA" if med_ga < med_pso else "PSO"

        kesimpulan = (
            f"Terdapat perbedaan yang SIGNIFIKAN secara statistik antara GA dan PSO "
            f"pada metrik {metric_name} (Mann-Whitney U = {u_stat:.2f}, p-value = {p_val:.5e} < {alpha}). "
            f"Algoritma {winner} mengungguli kompetitornya dengan median {winner} = "
            f"{(med_ga if winner == 'GA' else med_pso):.4f} berbanding "
            f"{(med_pso if winner == 'GA' else med_ga):.4f} (Rank-Biserial r = {rank_biserial:.3f})."
        )
    else:
        kesimpulan = (
            f"TIDAK terdapat perbedaan signifikan secara statistik antara GA dan PSO "
            f"pada metrik {metric_name} (Mann-Whitney U = {u_stat:.2f}, p-value = {p_val:.5f} >= {alpha}). "
            f"Kedua algoritma menunjukkan performa yang sebanding (Median GA = {med_ga:.4f}, Median PSO = {med_pso:.4f})."
        )

    return {
        "metric_name": metric_name,
        "u_statistic": float(u_stat),
        "p_value": float(p_val),
        "alpha": alpha,
        "is_significant": is_significant,
        "rank_biserial": float(rank_biserial),
        "median_ga": med_ga,
        "median_pso": med_pso,
        "mean_ga": mean_ga,
        "mean_pso": mean_pso,
        "kesimpulan": kesimpulan,
    }
