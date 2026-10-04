import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from map_model import MapModel, plot_map

def test_render():
    map_model = MapModel.load_from_json("d:/biocomputing-7/maps/peta_studi.json")
    
    # -------------------------------------------------------------
    # 1. TEST FULL SCREEN PSO (MAKSIMASI)
    # -------------------------------------------------------------
    fig_pso_max = plt.figure(figsize=(10.5, 8.2), facecolor="#f8fafc")
    fig_pso_max.subplots_adjust(left=0.07, right=0.97, top=0.93, bottom=0.13)
    ax = fig_pso_max.add_subplot(1, 1, 1)
    
    plot_map(
        map_model,
        ax=ax,
        title="PSO: Particle Swarm Optimization — Pencarian Solusi Optimum Terbaik",
        show_legend=False,
        show_banner=True,
    )
    
    # Fake particles
    sc_swarm = ax.scatter([500, 600, 700], [800, 850, 900], c="#0284c7", s=46, edgecolors="#0f172a", linewidths=0.8, alpha=0.85, label="Partikel Swarm")
    sc_pbest = ax.scatter([520, 610, 710], [810, 860, 910], c="#f97316", marker=".", s=60, alpha=0.75, label="pbest (Terbaik Pribadi)")
    sc_gbest = ax.scatter([650], [880], c="#16a34a", marker="P", s=220, edgecolors="#ffffff", linewidths=1.8, label="+ gbest (Solusi Optimum Terbaik)")
    
    # Floating badge on map
    ax.annotate(
        "★ Solusi Optimum Terbaik",
        xy=(650, 880), xytext=(650, 930),
        ha="center", va="bottom", fontsize=8.5, weight="bold", color="#166534",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#dcfce7", edgecolor="#16a34a", linewidth=1.2, alpha=0.95),
        arrowprops=dict(arrowstyle="->", color="#16a34a", lw=1.5),
        zorder=15
    )
    
    # Legend with title above items
    leg = ax.legend(
        [sc_swarm, sc_pbest, sc_gbest],
        ["Partikel Swarm", "pbest (Terbaik Pribadi)", "+ gbest: Solusi Optimum Terbaik"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.035),
        ncol=3,
        fontsize=9,
        title="🎯 Solusi Optimum Terbaik (Maksimasi)",
        title_fontsize=10.5,
        framealpha=0.96,
        facecolor="#f8fafc",
        edgecolor="#94a3b8",
    )
    if leg.get_title():
        leg.get_title().set_fontweight("bold")
        leg.get_title().set_color("#15803d")
        
    # Detail below legend
    detail_txt = "Keterangan Objek:   [S] Sekolah   •   [P] Pasar   •   [B] Bengkel   •   [R] Restoran   •   [K] Kompetitor   •   [M] Masjid"
    ax.text(
        0.5, -0.115, detail_txt,
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=8.5, weight="bold", color="#1e293b",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f1f5f9", edgecolor="#cbd5e1", linewidth=1.0, alpha=0.95),
        zorder=20
    )
    
    fig_pso_max.savefig("d:/biocomputing-7/scratch/preview_pso_max.png", dpi=100)
    plt.close(fig_pso_max)

    # -------------------------------------------------------------
    # 2. TEST FULL SCREEN PSO (MINIMASI)
    # -------------------------------------------------------------
    fig_pso_min = plt.figure(figsize=(10.5, 8.2), facecolor="#f8fafc")
    fig_pso_min.subplots_adjust(left=0.07, right=0.97, top=0.93, bottom=0.13)
    ax_min = fig_pso_min.add_subplot(1, 1, 1)
    
    plot_map(
        map_model,
        ax=ax_min,
        title="PSO: Particle Swarm Optimization — Titik Terburuk Legal (Layar Penuh)",
        show_legend=False,
        show_banner=True,
    )
    
    sc_swarm_m = ax_min.scatter([500, 600, 700], [800, 850, 900], c="#0284c7", s=46, edgecolors="#0f172a", linewidths=0.8, alpha=0.85)
    sc_pbest_m = ax_min.scatter([520, 610, 710], [810, 860, 910], c="#f97316", marker=".", s=60, alpha=0.75)
    
    # Target double reticle ring around the worst point
    gx, gy = 1700, 500
    ax_min.scatter([gx], [gy], marker="o", s=650, facecolors="none", edgecolors="#dc2626", linewidths=3.0, alpha=0.95, zorder=24)
    ax_min.scatter([gx], [gy], marker="o", s=420, facecolors="none", edgecolors="#ffffff", linewidths=1.8, alpha=0.95, zorder=24)
    sc_gbest_m = ax_min.scatter([gx], [gy], c="#dc2626", marker="P", s=340, edgecolors="#ffffff", linewidths=2.5, zorder=25)
    
    # High-contrast floating badge
    ax_min.text(
        gx, gy + 48, "▲ Titik Terburuk",
        fontsize=9.0, weight="bold", color="#ffffff", ha="center", va="bottom",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#dc2626", edgecolor="#ffffff", linewidth=1.8, alpha=0.98),
        zorder=30
    )
    
    leg_min = ax_min.legend(
        [sc_swarm_m, sc_pbest_m, sc_gbest_m],
        ["Partikel Swarm", "pbest (Terbaik Pribadi)", "+ gbest: Titik Terburuk Legal"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.045),
        ncol=3,
        fontsize=9,
        title="▲ TITIK TERBURUK LEGAL (MINIMASI)",
        title_fontsize=10.5,
        framealpha=0.96,
        facecolor="#f8fafc",
        edgecolor="#94a3b8",
    )
    if leg_min.get_title():
        leg_min.get_title().set_fontweight("bold")
        leg_min.get_title().set_color("#dc2626")
        
    detail_txt = "Keterangan Objek:   [S] Sekolah   •   [P] Pasar   •   [B] Bengkel   •   [R] Restoran   •   [K] Kompetitor   •   [M] Masjid"
    ax_min.text(
        0.5, -0.125, detail_txt,
        transform=ax_min.transAxes,
        ha="center", va="top",
        fontsize=8.5, weight="bold", color="#1e293b",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f1f5f9", edgecolor="#cbd5e1", linewidth=0.9, alpha=0.95),
        zorder=20
    )
    
    fig_pso_min.savefig("d:/biocomputing-7/scratch/preview_pso_min_reticle.png", dpi=100)
    plt.close(fig_pso_min)
        
    # -------------------------------------------------------------
    # 3. TEST SPLIT VIEW (MAKSIMASI)
    # -------------------------------------------------------------
    fig_split = plt.figure(figsize=(10.5, 8.2), facecolor="#f8fafc")
    gs = gridspec.GridSpec(
        2, 2, height_ratios=[3.3, 1.8], hspace=0.48, wspace=0.16,
        left=0.06, right=0.98, top=0.94, bottom=0.08
    )
    ax_ga = fig_split.add_subplot(gs[0, 0])
    ax_pso = fig_split.add_subplot(gs[0, 1])
    ax_cv = fig_split.add_subplot(gs[1, :])

    plot_map(map_model, ax=ax_ga, title="GA: Genetic Algorithm — Populasi & Elit", show_legend=False, show_banner=False)
    plot_map(map_model, ax=ax_pso, title="PSO: Particle Swarm — Swarm & Quiver", show_legend=False, show_banner=False)

    sc_pop = ax_ga.scatter([500], [800], c="#1d4ed8", s=34)
    sc_mut = ax_ga.scatter([520], [820], c="#f59e0b", s=46)
    sc_el = ax_ga.scatter([550], [850], c="#f59e0b", marker="*", s=140)
    sc_best_ga = ax_ga.scatter([650], [880], c="#16a34a", marker="P", s=180)

    leg_ga = ax_ga.legend(
        [sc_pop, sc_mut, sc_el, sc_best_ga],
        ["Populasi", "Termutasi", "★ Elit", "+ Solusi Optimum"],
        loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=4, fontsize=7.2,
        title="★ Solusi Optimum Terbaik (GA)", title_fontsize=8.5,
        framealpha=0.96, facecolor="#f8fafc", edgecolor="#94a3b8"
    )
    if leg_ga.get_title():
        leg_ga.get_title().set_fontweight("bold")
        leg_ga.get_title().set_color("#15803d")

    ax_ga.text(
        0.5, -0.21, "[S] Sekolah  [P] Pasar  [B] Bengkel  [R] Restoran  [K] Kompetitor",
        transform=ax_ga.transAxes, ha="center", va="top",
        fontsize=6.8, weight="bold", color="#1e293b",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#f1f5f9", edgecolor="#cbd5e1", linewidth=0.8, alpha=0.95)
    )

    sc_sw = ax_pso.scatter([500], [800], c="#0284c7", s=36)
    sc_pb = ax_pso.scatter([520], [820], c="#f97316", marker=".", s=45)
    sc_gb = ax_pso.scatter([650], [880], c="#16a34a", marker="P", s=180)

    leg_pso = ax_pso.legend(
        [sc_sw, sc_pb, sc_gb],
        ["Partikel", "pbest", "+ gbest: Solusi Optimum"],
        loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3, fontsize=7.5,
        title="★ Solusi Optimum Terbaik (PSO)", title_fontsize=8.5,
        framealpha=0.96, facecolor="#f8fafc", edgecolor="#94a3b8"
    )
    if leg_pso.get_title():
        leg_pso.get_title().set_fontweight("bold")
        leg_pso.get_title().set_color("#15803d")

    ax_pso.text(
        0.5, -0.21, "[S] Sekolah  [P] Pasar  [B] Bengkel  [R] Restoran  [K] Kompetitor",
        transform=ax_pso.transAxes, ha="center", va="top",
        fontsize=6.8, weight="bold", color="#1e293b",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#f1f5f9", edgecolor="#cbd5e1", linewidth=0.8, alpha=0.95)
    )

    ax_cv.set_title("Kurva Konvergensi Live", fontsize=9.5, weight="bold")
    ax_cv.plot([0, 100], [0.5, 0.9], color="#2563eb", label="GA")
    ax_cv.plot([0, 100], [0.4, 0.88], color="#dc2626", label="PSO")
    ax_cv.legend(loc="lower right", fontsize=8)

    fig_split.savefig("d:/biocomputing-7/scratch/preview_split.png", dpi=100)
    plt.close(fig_split)
    
    print("Rendered test previews successfully!")

if __name__ == "__main__":
    test_render()
