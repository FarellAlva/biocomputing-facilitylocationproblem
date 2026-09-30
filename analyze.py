"""
Modul Analisis Statistik Eksperimen (analyze.py)
Memproses luaran dari run_batch.py (results/raw_runs.csv & convergence_data.npz):
1. Menghitung tabel ringkasan statistik (mean, std, median, best, worst, mean waktu)
2. Melakukan uji hipotesis non-parametrik Mann-Whitney U test (scipy.stats)
   beserta narasi kesimpulan komparatif dalam Bahasa Indonesia
3. Menghasilkan visualisasi saintifik profesional:
   - results/boxplot_fitness.png: Boxplot distribusi fitness akhir GA vs PSO
   - results/convergence_comparison.png: Kurva konvergensi rata-rata ± std
   - results/final_locations_scatter.png: Scatter sebaran titik akhir 30 run di atas peta
4. Menyimpan tabel ke results/summary_statistics.csv dan laporan teks ke results/statistical_test_report.txt
"""

import os
import json
import argparse
import warnings
warnings.filterwarnings("ignore")
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from map_model import MapModel, plot_map
from fitness import FitnessEvaluator
from stats_utils import summarize_runs, perform_mann_whitney_u_test


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Analisis Statistik dan Visualisasi Hasil Batch GA vs PSO"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Folder penyimpanan data hasil run batch (default: results)",
    )
    parser.add_argument(
        "--map",
        type=str,
        default="maps/peta_studi.json",
        help="Path ke file peta kustom JSON (default: maps/peta_studi.json)",
    )
    return parser.parse_args()


def plot_boxplot_fitness(df: pd.DataFrame, output_path: str):
    """
    Membuat grafik Boxplot perbandingan distribusi fitness akhir GA vs PSO.
    """
    modes = df["mode"].unique()
    num_modes = len(modes)

    fig, axes = plt.subplots(1, num_modes, figsize=(6 * num_modes, 5.5), squeeze=False)
    fig.patch.set_facecolor("#f8fafc")

    colors = {"GA": "#2563eb", "PSO": "#dc2626"}

    for idx, mode in enumerate(modes):
        ax = axes[0, idx]
        ax.set_facecolor("#ffffff")
        mode_data = df[df["mode"] == mode]

        ga_fits = mode_data[mode_data["algoritma"] == "GA"]["fitness"].values
        pso_fits = mode_data[mode_data["algoritma"] == "PSO"]["fitness"].values

        data_to_plot = [ga_fits, pso_fits]
        labels = [f"GA\n(n={len(ga_fits)})", f"PSO\n(n={len(pso_fits)})"]

        box = ax.boxplot(
            data_to_plot,
            patch_artist=True,
            widths=0.45,
            tick_labels=labels,
            medianprops=dict(color="#0f172a", linewidth=2.0),
            whiskerprops=dict(color="#475569", linewidth=1.2),
            capprops=dict(color="#475569", linewidth=1.2),
            flierprops=dict(marker="o", markerfacecolor="#ef4444", markeredgecolor="#991b1b", markersize=6),
        )

        # Beri warna isi boxplot
        box["boxes"][0].set_facecolor("#93c5fd")
        box["boxes"][0].set_edgecolor("#1d4ed8")
        box["boxes"][1].set_facecolor("#fca5a5")
        box["boxes"][1].set_edgecolor("#b91c1c")

        # Jitter scatter plot titik individu
        for i, vals in enumerate(data_to_plot, start=1):
            jitter = np.random.default_rng(42).normal(0, 0.04, size=len(vals))
            col = colors["GA"] if i == 1 else colors["PSO"]
            ax.scatter(i + jitter, vals, color=col, alpha=0.6, s=28, edgecolors="#1e293b", linewidths=0.5, zorder=3)

        mode_title = "Maksimasi (Lokasi Terbaik)" if mode == "max" else "Minimasi Valid (Lokasi Terburuk Valid)"
        ax.set_title(f"Distribusi Fitness - Mode {mode_title}", fontsize=11, weight="bold", color="#1e293b")
        ax.set_ylabel("Nilai Fitness Akhir", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[PLOT] Boxplot fitness berhasil disimpan ke: {output_path}")


def plot_convergence_comparison(
    npz_data: Any, df: pd.DataFrame, output_path: str, budget: int = 2000
):
    """
    Membuat grafik kurva konvergensi rata-rata ± 1 standar deviasi untuk GA vs PSO.
    """
    modes = df["mode"].unique()
    num_modes = len(modes)

    fig, axes = plt.subplots(1, num_modes, figsize=(7 * num_modes, 5.5), squeeze=False)
    fig.patch.set_facecolor("#f8fafc")

    colors = {"GA": "#2563eb", "PSO": "#dc2626"}
    fill_colors = {"GA": "#bfdbfe", "PSO": "#fecaca"}

    # Grid interpolasi evaluasi seragam (0 sampai budget)
    interp_evals = np.linspace(40, budget, 100)

    for idx, mode in enumerate(modes):
        ax = axes[0, idx]
        ax.set_facecolor("#ffffff")

        for algo in ["GA", "PSO"]:
            algo_runs = df[(df["mode"] == mode) & (df["algoritma"] == algo)]["run"].values
            interpolated_curves = []

            for r in algo_runs:
                key_evals = f"{mode}_{algo}_run{r}_evals"
                key_fits = f"{mode}_{algo}_run{r}_fits"

                if key_evals in npz_data and key_fits in npz_data:
                    ev = npz_data[key_evals]
                    ft = npz_data[key_fits]

                    # Interpolasi kurva langkah
                    interp_f = np.interp(interp_evals, ev, ft)
                    interpolated_curves.append(interp_f)

            if len(interpolated_curves) > 0:
                curves_mat = np.array(interpolated_curves)  # shape: (runs, 100)
                mean_curve = np.mean(curves_mat, axis=0)
                std_curve = np.std(curves_mat, axis=0, ddof=1)

                ax.plot(
                    interp_evals,
                    mean_curve,
                    color=colors[algo],
                    linewidth=2.4,
                    label=f"{algo} (Mean Akhir: {mean_curve[-1]:.4f})",
                )
                ax.fill_between(
                    interp_evals,
                    mean_curve - std_curve,
                    mean_curve + std_curve,
                    color=fill_colors[algo],
                    alpha=0.45,
                    label=f"±1 Std Dev {algo}",
                )

        mode_title = "Maksimasi (Terbaik)" if mode == "max" else "Minimasi Valid (Terburuk Valid)"
        ax.set_title(f"Kurva Konvergensi Rata-rata - Mode {mode_title}", fontsize=11, weight="bold")
        ax.set_xlabel("Jumlah Evaluasi Fitness", fontsize=10)
        ax.set_ylabel("Nilai Fitness", fontsize=10)
        ax.set_xlim(0, budget)
        ax.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1")
        ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[PLOT] Kurva konvergensi berhasil disimpan ke: {output_path}")


def plot_final_locations_scatter(
    df: pd.DataFrame, map_model: MapModel, output_path: str, config: Dict[str, Any]
):
    """
    Membuat scatter plot sebaran lokasi akhir 30 run GA dan PSO di atas peta wilayah.
    """
    modes = df["mode"].unique()
    num_modes = len(modes)

    fig, axes = plt.subplots(1, num_modes, figsize=(8.5 * num_modes, 7.0), squeeze=False)
    fig.patch.set_facecolor("#f8fafc")

    for idx, mode in enumerate(modes):
        ax = axes[0, idx]

        # Buat evaluator untuk heatmap latar belakang
        evaluator = FitnessEvaluator(map_model, config=config, mode=mode)
        grid, extent = evaluator.compute_grid_heatmap(resolution_x=60, resolution_y=45)
        cmap_name = "RdYlGn" if mode == "max" else "RdYlGn_r"

        # Gambar peta dan heatmap
        plot_map(
            map_model,
            ax=ax,
            title=f"Sebaran 30 Lokasi Akhir - Mode {mode.upper()}",
            show_legend=False,
            show_heatmap=True,
            heatmap_grid=grid,
            heatmap_extent=extent,
            heatmap_cmap=cmap_name,
            heatmap_alpha=0.45,
        )

        mode_df = df[df["mode"] == mode]
        ga_data = mode_df[mode_df["algoritma"] == "GA"]
        pso_data = mode_df[mode_df["algoritma"] == "PSO"]

        # Scatter GA: Lingkaran Biru
        ax.scatter(
            ga_data["x"],
            ga_data["y"],
            c="#2563eb",
            marker="o",
            s=80,
            edgecolors="#ffffff",
            linewidths=1.2,
            alpha=0.85,
            zorder=10,
            label=f"GA Final (n={len(ga_data)})",
        )

        # Scatter PSO: Kotak / Segitiga Merah
        ax.scatter(
            pso_data["x"],
            pso_data["y"],
            c="#dc2626",
            marker="s",
            s=80,
            edgecolors="#ffffff",
            linewidths=1.2,
            alpha=0.85,
            zorder=11,
            label=f"PSO Final (n={len(pso_data)})",
        )

        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=8.5, framealpha=0.92, edgecolor="#cbd5e1")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Scatter lokasi akhir berhasil disimpan ke: {output_path}")


def analyze_results(results_dir: str = "results", map_file: str = "maps/peta_studi.json"):
    """
    Fungsi orkestrasi utama analisis statistik dan plotting.
    """
    csv_path = os.path.join(results_dir, "raw_runs.csv")
    npz_path = os.path.join(results_dir, "convergence_data.npz")
    cfg_path = os.path.join(results_dir, "config_used.json")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File hasil '{csv_path}' tidak ditemukan. Jalankan run_batch.py terlebih dahulu.")

    df = pd.read_csv(csv_path)

    config_data = {}
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)

    budget = config_data.get("optimization", {}).get("evaluation_budget", 2000)
    map_model = MapModel.load_from_json(map_file)

    # 1. Ringkasan Statistik Deskriptif per Mode & Algoritma
    summary_list = []
    for mode in df["mode"].unique():
        sub_df = df[df["mode"] == mode]
        sum_sub = summarize_runs(sub_df, group_col="algoritma")
        sum_sub.insert(0, "Mode", mode)
        summary_list.append(sum_sub)

    final_summary_df = pd.concat(summary_list, ignore_index=True)
    summary_csv_path = os.path.join(results_dir, "summary_statistics.csv")
    final_summary_df.to_csv(summary_csv_path, index=False)
    print(f"[DATA] Tabel ringkasan statistik disimpan ke: {summary_csv_path}")

    # 2. Uji Hipotesis Mann-Whitney U Test
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("LAPORAN UJI HIPOTESIS STATISTIK: PERBANDINGAN GA vs PSO")
    report_lines.append(f"Peta: {map_model.name} | Anggaran: {budget} Evaluasi | 30 Run Independen")
    report_lines.append("=" * 80 + "\n")

    for mode in df["mode"].unique():
        report_lines.append(f"--- ANALISIS MODE: {mode.upper()} ---")
        mode_df = df[df["mode"] == mode]
        ga_fits = mode_df[mode_df["algoritma"] == "GA"]["fitness"].values
        pso_fits = mode_df[mode_df["algoritma"] == "PSO"]["fitness"].values

        # Uji pada Fitness Akhir
        test_fit = perform_mann_whitney_u_test(
            ga_fits, pso_fits, metric_name=f"Fitness Akhir (Mode {mode})", higher_is_better=True
        )
        report_lines.append("1. Uji Fitness Akhir:")
        report_lines.append(f"   * GA  -> Mean: {test_fit['mean_ga']:.4f}, Median: {test_fit['median_ga']:.4f}")
        report_lines.append(f"   * PSO -> Mean: {test_fit['mean_pso']:.4f}, Median: {test_fit['median_pso']:.4f}")
        report_lines.append(f"   * Mann-Whitney U: {test_fit['u_statistic']:.2f}")
        report_lines.append(f"   * p-value: {test_fit['p_value']:.5e} (Signifikan: {test_fit['is_significant']})")
        report_lines.append(f"   * Kesimpulan: {test_fit['kesimpulan']}\n")

        # Uji pada Kecepatan Konvergensi (evaluasi ke 95%)
        ga_eval95 = mode_df[mode_df["algoritma"] == "GA"]["evaluasi_ke_95persen"].values
        pso_eval95 = mode_df[mode_df["algoritma"] == "PSO"]["evaluasi_ke_95persen"].values
        test_speed = perform_mann_whitney_u_test(
            ga_eval95, pso_eval95, metric_name="Evaluasi ke 95% (Kecepatan)", higher_is_better=False
        )
        report_lines.append("2. Uji Kecepatan Konvergensi (Evaluasi ke 95%):")
        report_lines.append(f"   * GA  -> Median Evaluasi: {test_speed['median_ga']:.1f}")
        report_lines.append(f"   * PSO -> Median Evaluasi: {test_speed['median_pso']:.1f}")
        report_lines.append(f"   * Mann-Whitney U: {test_speed['u_statistic']:.2f}, p-value: {test_speed['p_value']:.5e}")
        report_lines.append(f"   * Kesimpulan: {test_speed['kesimpulan']}\n")

    report_text = "\n".join(report_lines)
    report_txt_path = os.path.join(results_dir, "statistical_test_report.txt")
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"[DATA] Laporan uji hipotesis disimpan ke: {report_txt_path}")

    # Cetak laporan ke terminal
    print("\n" + report_text)

    # 3. Buat Grafik
    print("\nMenghasilkan visualisasi grafik...")
    boxplot_path = os.path.join(results_dir, "boxplot_fitness.png")
    plot_boxplot_fitness(df, boxplot_path)

    if os.path.exists(npz_path):
        npz_data = np.load(npz_path)
        conv_path = os.path.join(results_dir, "convergence_comparison.png")
        plot_convergence_comparison(npz_data, df, conv_path, budget=budget)

    scatter_path = os.path.join(results_dir, "final_locations_scatter.png")
    plot_final_locations_scatter(df, map_model, scatter_path, config=config_data)

    print("\n" + "=" * 80)
    print("ANALISIS DAN VISUALISASI SELESAI")
    print(f"Seluruh luaran tersimpan lengkap di direktori: '{results_dir}/'")
    print("=" * 80)


def main():
    args = parse_arguments()
    analyze_results(results_dir=args.results_dir, map_file=args.map)


if __name__ == "__main__":
    main()
