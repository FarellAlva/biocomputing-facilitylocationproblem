"""
Aplikasi GUI Interaktif Facility Location dengan Tkinter (app_tkinter.py)
Menyediakan visualisasi interaktif penuh untuk membandingkan
Genetic Algorithm (GA) dan Particle Swarm Optimization (PSO) dalam
penempatan minimarket pada peta wilayah 2.0 km x 1.5 km.

Fitur Utama:
1. Tampilan Grafis Matplotlib terintegrasi dalam Tkinter:
   - Subplot Kiri: Algoritma Genetika (Populasi, Elit, Mutasi Flash, Hubungan Induk-Anak)
   - Subplot Kanan: Particle Swarm Optimization (Swarm, Quiver Kecepatan, Motion Trail, pbest, gbest)
   - Subplot Bawah: Kurva Konvergensi Live (Best Fitness vs Jumlah Evaluasi Nyata)
2. Kontrol Simulasi Lengkap:
   - Play (Mulai), Pause (Jeda), Step (Langkah per Langkah), Reset, Run to End (Selesaikan Langsung)
   - Slider kecepatan simulasi (delay ms per frame)
   - Pilihan preset peta (Desa Ramai, Hauling Tambang, Campuran)
   - Pilihan mode optimasi ('max' dan 'min_valid')
   - Slider bobot multi-kriteria w1 - w4 dengan pembaruan live heatmap
3. Fitur Interaktif "Ini Apa & Bagaimana?":
   - Tab penjelasan visual komprehensif menjelaskan setiap simbol, warna, dan cara kerja algoritma
   - Papan skor (Scoreboard) live pembanding metrik GA vs PSO
   - Fitur Inspeksi Titik: klik titik manapun pada peta untuk melihat koordinat, status legalitas,
     serta rincian nilai 4 kriteria pembentuk fitness!

Jalankan dengan:
    python app_tkinter.py
"""

import sys
import os
import json
import time
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox

# Matplotlib embedding dalam Tkinter
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from map_model import MapModel, plot_map
from fitness import FitnessEvaluator
from ga import GeneticAlgorithm, GAGenerationRecord
from pso import ParticleSwarmOptimization, PSOIterationRecord


class FacilityLocationTkApp:
    """
    Kelas utama antarmuka pengguna Tkinter untuk simulasi optimasi penempatan minimarket.
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Simulasi Interaktif Facility Location: GA vs PSO (Minimarket)")
        self.root.geometry("1420x900")
        self.root.minsize(1100, 720)

        # Konfigurasi gaya visual Tkinter
        self._setup_styles()

        # Status & Data Simulasi
        self.config_path = "config.json"
        self._load_config()

        self.available_maps = {
            "Peta Studi (Desa, Jalan Lintas & Hauling)": "maps/peta_studi.json",
        }
        self.current_map_name = "Peta Studi (Desa, Jalan Lintas & Hauling)"
        self.current_map_file = self.available_maps[self.current_map_name]
        self.map_model = MapModel.load_from_json(self.current_map_file)

        # Status Tampilan Layar & Inspeksi
        self.view_mode = "split"  # 'split', 'ga', 'pso', 'conv'
        self.last_inspected_pt: Optional[Tuple[float, float]] = None

        # Variabel kontrol
        self.mode_var = tk.StringVar(value="max")
        self.stores_var = tk.IntVar(value=1)
        self.budget_var = tk.IntVar(value=2000)
        self.speed_var = tk.IntVar(value=80)  # delay dalam milidetik

        # Bobot Kriteria
        self.w1_var = tk.DoubleVar(value=self.config.get("fitness_weights", {}).get("w1_populasi", 0.35))
        self.w2_var = tk.DoubleVar(value=self.config.get("fitness_weights", {}).get("w2_akses_jalan", 0.25))
        self.w3_var = tk.DoubleVar(value=self.config.get("fitness_weights", {}).get("w3_fasilitas", 0.20))
        self.w4_var = tk.DoubleVar(value=self.config.get("fitness_weights", {}).get("w4_kompetitor", 0.20))

        # Status playback simulasi
        self.is_running = False
        self.current_frame = 0
        self.total_frames = 0
        self.timer_id = None

        # Data riwayat hasil optimasi
        self.ga_history: List[GAGenerationRecord] = []
        self.pso_history: List[PSOIterationRecord] = []
        self.ga_best_sol = None
        self.ga_best_fit = -np.inf
        self.pso_best_sol = None
        self.pso_best_fit = -np.inf

        # Bangun Layout UI
        self._build_header()
        self._build_main_interface()

        # Inisialisasi komputasi dan render kanvas awal
        self._recompute_optimization(reset_playback=True)

    def _setup_styles(self):
        """Mengonfigurasi tema warna dan font ttk."""
        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        # Warna tema modern
        self.style.configure("Header.TFrame", background="#0f172a")
        self.style.configure("HeaderTitle.TLabel", background="#0f172a", foreground="#f8fafc", font=("Helvetica", 14, "bold"))
        self.style.configure("HeaderSubtitle.TLabel", background="#0f172a", foreground="#94a3b8", font=("Helvetica", 9))
        self.style.configure("StatusBadge.TLabel", background="#1e293b", foreground="#38bdf8", font=("Helvetica", 10, "bold"), padding=5)

        self.style.configure("Card.TFrame", background="#ffffff", relief="solid", borderwidth=1)
        self.style.configure("CardHeader.TLabel", background="#f8fafc", foreground="#1e293b", font=("Helvetica", 10, "bold"), padding=4)
        
        self.style.configure("TNotebook", background="#f1f5f9", tabmargins=[2, 5, 2, 0])
        self.style.configure("TNotebook.Tab", background="#e2e8f0", foreground="#334155", font=("Helvetica", 9, "bold"), padding=[10, 4])
        self.style.map("TNotebook.Tab", background=[("selected", "#ffffff")], foreground=[("selected", "#0284c7")])

        # Tombol Khusus
        self.style.configure("Play.TButton", font=("Helvetica", 9, "bold"), foreground="#15803d", padding=4)
        self.style.configure("Pause.TButton", font=("Helvetica", 9, "bold"), foreground="#c2410c", padding=4)
        self.style.configure("Action.TButton", font=("Helvetica", 9), padding=3)

    def _load_config(self):
        """Memuat file konfigurasi JSON."""
        if os.path.exists(self.config_path):
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        else:
            self.config = {}

    def _build_header(self):
        """Header atas dengan judul dan status sistem."""
        header_frame = ttk.Frame(self.root, style="Header.TFrame", padding=(15, 10))
        header_frame.pack(side=tk.TOP, fill=tk.X)

        title_box = ttk.Frame(header_frame, style="Header.TFrame")
        title_box.pack(side=tk.LEFT, fill=tk.Y)

        title_lbl = ttk.Label(
            title_box,
            text="OPTIMASI PENEMPATAN MINIMARKET (FACILITY LOCATION)",
            style="HeaderTitle.TLabel",
        )
        title_lbl.pack(anchor="w")

        sub_lbl = ttk.Label(
            title_box,
            text="Perbandingan Head-to-Head: Genetic Algorithm (GA) vs Particle Swarm Optimization (PSO) dari Nol",
            style="HeaderSubtitle.TLabel",
        )
        sub_lbl.pack(anchor="w")

        # Status badge di kanan
        status_box = ttk.Frame(header_frame, style="Header.TFrame")
        status_box.pack(side=tk.RIGHT, fill=tk.Y)

        self.status_badge = ttk.Label(
            status_box,
            text="STATUS: SIAP",
            style="StatusBadge.TLabel",
        )
        self.status_badge.pack(anchor="e", pady=2)

    def _build_main_interface(self):
        """Membangun area kerja utama: Panel Visualisasi Matplotlib (kiri) & Panel Kontrol Interaktif (kanan)."""
        main_paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, bg="#cbd5e1", sashwidth=5, sashrelief=tk.RAISED)
        main_paned.pack(fill=tk.BOTH, expand=True)

        # 1. Panel Visualisasi Kiri
        vis_frame = ttk.Frame(main_paned, padding=5)
        main_paned.add(vis_frame, minsize=700, stretch="always")

        self._setup_matplotlib_canvas(vis_frame)

        # 2. Panel Kontrol Kanan (Notebook Tabs)
        side_frame = ttk.Frame(main_paned, width=440, padding=5)
        main_paned.add(side_frame, minsize=380, stretch="never")

        self.notebook = ttk.Notebook(side_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Kontrol & Parameter
        tab_control = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(tab_control, text="🎮 Kontrol Simulasi")
        self._build_tab_control(tab_control)

        # Tab 2: Scoreboard & Live Metrik
        tab_scoreboard = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(tab_scoreboard, text="📊 Papan Skor")
        self._build_tab_scoreboard(tab_scoreboard)

        # Tab 3: "Ini Apa & Bagaimana?" (Panduan & Legenda)
        tab_guide = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(tab_guide, text="💡 Ini Apa?")
        self._build_tab_guide(tab_guide)

        # Tab 4: Inspeksi Titik (Klik Peta)
        tab_inspect = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(tab_inspect, text="🔍 Inspeksi Titik")
        self._build_tab_inspect(tab_inspect)

    def _setup_matplotlib_canvas(self, parent):
        """Membuat kanvas Matplotlib dengan pilihan mode tampilan: Berdampingan atau Layar Penuh per metode."""
        # 1. Bilah Pilihan Mode Tampilan (Switcher Toolbar)
        view_bar = ttk.Frame(parent, style="Card.TFrame", padding=(6, 4))
        view_bar.pack(fill=tk.X, side=tk.TOP, pady=(0, 4))

        ttk.Label(view_bar, text="🖥️ Mode Tampilan:", font=("Helvetica", 9, "bold"), foreground="#1e293b").pack(side=tk.LEFT, padx=(4, 6))

        self.btn_view_split = ttk.Button(view_bar, text="● ⚖️ Berdampingan", command=lambda: self._set_view_mode("split"))
        self.btn_view_split.pack(side=tk.LEFT, padx=3)

        self.btn_view_ga = ttk.Button(view_bar, text="  🧬 Layar Penuh GA", command=lambda: self._set_view_mode("ga"))
        self.btn_view_ga.pack(side=tk.LEFT, padx=3)

        self.btn_view_pso = ttk.Button(view_bar, text="  🚀 Layar Penuh PSO", command=lambda: self._set_view_mode("pso"))
        self.btn_view_pso.pack(side=tk.LEFT, padx=3)

        self.btn_view_conv = ttk.Button(view_bar, text="  📈 Layar Penuh Konvergensi", command=lambda: self._set_view_mode("conv"))
        self.btn_view_conv.pack(side=tk.LEFT, padx=3)

        # 2. Kanvas Figure Matplotlib
        self.fig = plt.figure(figsize=(10.5, 8.2), facecolor="#f8fafc")
        self.ax_ga = None
        self.ax_pso = None
        self.ax_conv = None

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        # 3. Toolbar navigasi bawaan Matplotlib (Zoom, Pan, Save Image)
        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.pack(fill=tk.X, side=tk.BOTTOM)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

        # 4. Bilah Kendali Zoom Khusus per Metode (GA & PSO)
        zoom_bar = ttk.Frame(parent, padding=(4, 3))
        zoom_bar.pack(fill=tk.X, side=tk.BOTTOM)

        # Zoom GA
        ttk.Label(zoom_bar, text="🔍 Zoom GA:", font=("Helvetica", 8, "bold"), foreground="#1d4ed8").pack(side=tk.LEFT, padx=(4, 2))
        ttk.Button(zoom_bar, text="➕", width=3, command=self._zoom_in_ga).pack(side=tk.LEFT, padx=1)
        ttk.Button(zoom_bar, text="➖", width=3, command=self._zoom_out_ga).pack(side=tk.LEFT, padx=1)
        ttk.Button(zoom_bar, text="🏘️ Desa", command=self._focus_village_ga).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_bar, text="⟲ Reset", command=self._reset_zoom_ga).pack(side=tk.LEFT, padx=2)

        ttk.Separator(zoom_bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        # Zoom PSO
        ttk.Label(zoom_bar, text="🔍 Zoom PSO:", font=("Helvetica", 8, "bold"), foreground="#0284c7").pack(side=tk.LEFT, padx=(4, 2))
        ttk.Button(zoom_bar, text="➕", width=3, command=self._zoom_in_pso).pack(side=tk.LEFT, padx=1)
        ttk.Button(zoom_bar, text="➖", width=3, command=self._zoom_out_pso).pack(side=tk.LEFT, padx=1)
        ttk.Button(zoom_bar, text="🏘️ Desa", command=self._focus_village_pso).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_bar, text="⟲ Reset", command=self._reset_zoom_pso).pack(side=tk.LEFT, padx=2)

        ttk.Separator(zoom_bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        # Aksi Bersama
        ttk.Button(zoom_bar, text="🏘️ Fokus Desa (Keduanya)", command=self._focus_village_both).pack(side=tk.LEFT, padx=3)
        ttk.Button(zoom_bar, text="🗺️ Peta Penuh", command=self._reset_zoom_both).pack(side=tk.LEFT, padx=3)

        # Event interaksi mouse di atas peta (Klik, Hover, dan Scroll Zoom)
        self.canvas.mpl_connect("button_press_event", self._on_map_click)
        self.canvas.mpl_connect("motion_notify_event", self._on_map_hover)
        self.canvas.mpl_connect("scroll_event", self._on_map_scroll)

    def _set_view_mode(self, mode: str):
        """Mengganti mode tampilan antara Berdampingan, Layar Penuh GA, Layar Penuh PSO, atau Konvergensi."""
        if self.view_mode == mode:
            return
        self.view_mode = mode
        self._update_view_mode_buttons()
        self._init_canvas_plots()
        self._render_frame(self.current_frame)

    def _update_view_mode_buttons(self):
        """Memperbarui teks/indikator tombol mode tampilan aktif."""
        modes = {
            "split": ("⚖️ Berdampingan", self.btn_view_split),
            "ga": ("🧬 Layar Penuh GA", self.btn_view_ga),
            "pso": ("🚀 Layar Penuh PSO", self.btn_view_pso),
            "conv": ("📈 Layar Penuh Konvergensi", self.btn_view_conv),
        }
        for m_key, (label, btn) in modes.items():
            if self.view_mode == m_key:
                btn.config(text=f"● {label}")
            else:
                btn.config(text=f"  {label}")

    def _build_tab_control(self, parent):
        """Membangun tab kontrol parameter dan pemutaran simulasi."""
        # Scrollable container jika resolusi vertikal kecil
        canvas = tk.Canvas(parent, bg="#ffffff", highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scroll_content = ttk.Frame(canvas, padding=5)

        scroll_content.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_content, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 1. Pilihan Peta & Mode
        grp_setup = ttk.LabelFrame(scroll_content, text=" 🗺️ Pengaturan Wilayah & Target ", padding=8)
        grp_setup.pack(fill=tk.X, pady=4)

        ttk.Label(grp_setup, text="Peta Wilayah Studi:", font=("Helvetica", 8, "bold")).pack(anchor="w")
        lbl_peta_info = ttk.Label(
            grp_setup,
            text="📍 Peta Studi (Desa, Jalan Lintas & Hauling)\n"
                 "Dimensi: 2.0 km x 1.5 km | Terfokus Kluster Desa, Jl. Lintas & Hauling",
            font=("Helvetica", 8),
            foreground="#0369a1",
            justify=tk.LEFT,
        )
        lbl_peta_info.pack(anchor="w", pady=(2, 6))

        ttk.Label(grp_setup, text="Mode Optimasi:").pack(anchor="w")
        mode_box = ttk.Frame(grp_setup)
        mode_box.pack(fill=tk.X, pady=2)
        ttk.Radiobutton(mode_box, text="Maksimasi (Cari Lokasi Terbaik)", variable=self.mode_var, value="max", command=self._on_param_change).pack(anchor="w")
        ttk.Radiobutton(mode_box, text="Minimasi Valid (Cari Lokasi Terburuk Legal)", variable=self.mode_var, value="min_valid", command=self._on_param_change).pack(anchor="w")

        store_box = ttk.Frame(grp_setup)
        store_box.pack(fill=tk.X, pady=4)
        ttk.Label(store_box, text="Jumlah Minimarket (p):").pack(side=tk.LEFT)
        for p_val in [1, 2, 3]:
            ttk.Radiobutton(store_box, text=f"{p_val} Toko", variable=self.stores_var, value=p_val, command=self._on_param_change).pack(side=tk.LEFT, padx=6)

        # 2. Tombol Pemutaran Simulasi
        grp_sim = ttk.LabelFrame(scroll_content, text=" ⏯️ Kendali Simulasi Real-Time ", padding=8)
        grp_sim.pack(fill=tk.X, pady=6)

        btn_row1 = ttk.Frame(grp_sim)
        btn_row1.pack(fill=tk.X, pady=3)
        self.btn_play = ttk.Button(btn_row1, text="▶ Mulai (Play)", style="Play.TButton", command=self._toggle_simulation)
        self.btn_play.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        self.btn_step = ttk.Button(btn_row1, text="⏭ Langkah (Step)", style="Action.TButton", command=self._step_simulation)
        self.btn_step.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        btn_row2 = ttk.Frame(grp_sim)
        btn_row2.pack(fill=tk.X, pady=3)
        self.btn_reset = ttk.Button(btn_row2, text="🔄 Reset", style="Action.TButton", command=self._reset_simulation)
        self.btn_reset.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        self.btn_finish = ttk.Button(btn_row2, text="⚡ Ke Akhir", style="Action.TButton", command=self._run_to_end)
        self.btn_finish.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        # Slider Kecepatan
        ttk.Label(grp_sim, text="Kecepatan Animasi (Delay ms antar frame):").pack(anchor="w", pady=(6, 1))
        speed_slider = ttk.Scale(grp_sim, from_=10, to=300, variable=self.speed_var, orient=tk.HORIZONTAL)
        speed_slider.pack(fill=tk.X)
        self.lbl_speed_val = ttk.Label(grp_sim, text=f"{self.speed_var.get()} ms")
        self.lbl_speed_val.pack(anchor="e")
        speed_slider.bind("<Motion>", lambda e: self.lbl_speed_val.config(text=f"{int(self.speed_var.get())} ms"))

        # 3. Slider Bobot Multi-Kriteria (w1 - w4)
        grp_weights = ttk.LabelFrame(scroll_content, text=" ⚖️ Bobot Kriteria Fitness (Fungsi Tujuan) ", padding=8)
        grp_weights.pack(fill=tk.X, pady=6)

        self._create_weight_slider(grp_weights, "w1 Populasi (Pemukiman)", self.w1_var, "#10b981")
        self._create_weight_slider(grp_weights, "w2 Kondisi Jalan (Proper vs Lumpur)", self.w2_var, "#f59e0b")
        self._create_weight_slider(grp_weights, "w3 Sinergi Fasilitas Umum", self.w3_var, "#6366f1")
        self._create_weight_slider(grp_weights, "w4 Pengaruh Kompetitor", self.w4_var, "#ec4899")

        ttk.Button(
            grp_weights,
            text="🔄 Terapkan Bobot & Hitung Ulang",
            style="Action.TButton",
            command=self._on_param_change,
        ).pack(fill=tk.X, pady=6)

        # 4. Kontrol Zoom & Tampilan Wilayah
        grp_zoom = ttk.LabelFrame(scroll_content, text=" 🔍 Kontrol Zoom & Fokus Wilayah ", padding=8)
        grp_zoom.pack(fill=tk.X, pady=6)

        z_row_both = ttk.Frame(grp_zoom)
        z_row_both.pack(fill=tk.X, pady=2)
        ttk.Button(z_row_both, text="🏘️ Fokus Area Desa (Keduanya)", command=self._focus_village_both).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(z_row_both, text="🗺️ Reset Peta Penuh", command=self._reset_zoom_both).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        z_row_ga = ttk.Frame(grp_zoom)
        z_row_ga.pack(fill=tk.X, pady=3)
        ttk.Label(z_row_ga, text="Zoom GA:", font=("Helvetica", 8, "bold"), foreground="#1d4ed8").pack(side=tk.LEFT)
        ttk.Button(z_row_ga, text="➕ In", width=5, command=self._zoom_in_ga).pack(side=tk.LEFT, padx=2)
        ttk.Button(z_row_ga, text="➖ Out", width=5, command=self._zoom_out_ga).pack(side=tk.LEFT, padx=2)
        ttk.Button(z_row_ga, text="Desa", width=5, command=self._focus_village_ga).pack(side=tk.LEFT, padx=2)
        ttk.Button(z_row_ga, text="Reset", width=5, command=self._reset_zoom_ga).pack(side=tk.LEFT, padx=2)

        z_row_pso = ttk.Frame(grp_zoom)
        z_row_pso.pack(fill=tk.X, pady=3)
        ttk.Label(z_row_pso, text="Zoom PSO:", font=("Helvetica", 8, "bold"), foreground="#0284c7").pack(side=tk.LEFT)
        ttk.Button(z_row_pso, text="➕ In", width=5, command=self._zoom_in_pso).pack(side=tk.LEFT, padx=2)
        ttk.Button(z_row_pso, text="➖ Out", width=5, command=self._zoom_out_pso).pack(side=tk.LEFT, padx=2)
        ttk.Button(z_row_pso, text="Desa", width=5, command=self._focus_village_pso).pack(side=tk.LEFT, padx=2)
        ttk.Button(z_row_pso, text="Reset", width=5, command=self._reset_zoom_pso).pack(side=tk.LEFT, padx=2)

        ttk.Label(
            grp_zoom,
            text="💡 Tips: Anda juga dapat menggunakan roda scroll mouse langsung di atas peta untuk zoom in / out!",
            font=("Helvetica", 7, "italic"),
            foreground="#64748b",
            wraplength=350,
            justify=tk.LEFT,
        ).pack(anchor="w", pady=(4, 0))

    def _create_weight_slider(self, parent, label_text, var, color):
        """Helper membuat slider bobot kriteria."""
        box = ttk.Frame(parent)
        box.pack(fill=tk.X, pady=2)
        lbl_title = ttk.Label(box, text=label_text, font=("Helvetica", 8, "bold"))
        lbl_title.pack(side=tk.LEFT)
        lbl_val = ttk.Label(box, text=f"{var.get():.2f}", font=("Helvetica", 8, "bold"), foreground=color)
        lbl_val.pack(side=tk.RIGHT)

        slider = ttk.Scale(
            parent,
            from_=0.0,
            to=1.0,
            variable=var,
            orient=tk.HORIZONTAL,
            command=lambda v: lbl_val.config(text=f"{float(v):.2f}"),
        )
        slider.pack(fill=tk.X, pady=(0, 4))

    def _build_tab_scoreboard(self, parent):
        """Membangun tab scoreboard pembanding real-time metrik GA vs PSO."""
        # Leader Banner
        self.banner_leader = tk.Label(
            parent,
            text="🏁 MEMULAI SIMULASI...",
            font=("Helvetica", 11, "bold"),
            bg="#3b82f6",
            fg="#ffffff",
            pady=8,
            relief=tk.FLAT,
        )
        self.banner_leader.pack(fill=tk.X, pady=(0, 10))

        # Panel GA vs PSO
        score_grid = ttk.Frame(parent)
        score_grid.pack(fill=tk.X)

        # Kartu GA
        card_ga = ttk.LabelFrame(score_grid, text=" 🧬 Algoritma Genetika (GA) ", padding=8)
        card_ga.pack(fill=tk.X, pady=4)

        self.lbl_ga_fit = ttk.Label(card_ga, text="Best Fitness: 0.0000", font=("Helvetica", 10, "bold"), foreground="#2563eb")
        self.lbl_ga_fit.pack(anchor="w", pady=1)

        self.lbl_ga_coord = ttk.Label(card_ga, text="Koordinat: (0.0, 0.0)", font=("Helvetica", 8))
        self.lbl_ga_coord.pack(anchor="w", pady=1)

        self.lbl_ga_eval = ttk.Label(card_ga, text="Evaluasi: 0 / 2000", font=("Helvetica", 8))
        self.lbl_ga_eval.pack(anchor="w", pady=1)

        self.lbl_ga_status = ttk.Label(card_ga, text="Status Titik: Menunggu", font=("Helvetica", 8))
        self.lbl_ga_status.pack(anchor="w", pady=1)

        # Kartu PSO
        card_pso = ttk.LabelFrame(score_grid, text=" 🚀 Particle Swarm Optimization (PSO) ", padding=8)
        card_pso.pack(fill=tk.X, pady=4)

        self.lbl_pso_fit = ttk.Label(card_pso, text="gbest Fitness: 0.0000", font=("Helvetica", 10, "bold"), foreground="#dc2626")
        self.lbl_pso_fit.pack(anchor="w", pady=1)

        self.lbl_pso_coord = ttk.Label(card_pso, text="Koordinat gbest: (0.0, 0.0)", font=("Helvetica", 8))
        self.lbl_pso_coord.pack(anchor="w", pady=1)

        self.lbl_pso_eval = ttk.Label(card_pso, text="Evaluasi: 0 / 2000", font=("Helvetica", 8))
        self.lbl_pso_eval.pack(anchor="w", pady=1)

        self.lbl_pso_status = ttk.Label(card_pso, text="Status Titik: Menunggu", font=("Helvetica", 8))
        self.lbl_pso_status.pack(anchor="w", pady=1)

        # Selisih / Gap
        card_gap = ttk.LabelFrame(parent, text=" 📈 Analisis Komparatif Live ", padding=8)
        card_gap.pack(fill=tk.X, pady=8)

        self.lbl_gap_text = ttk.Label(
            card_gap,
            text="Selisih Nilai Fitness: 0.0000\nKeduanya bergerak menuju titik optimal wilayah.",
            font=("Helvetica", 8),
            justify=tk.LEFT,
        )
        self.lbl_gap_text.pack(anchor="w")

        # Progres Evaluasi Bar
        ttk.Label(parent, text="Progres Anggaran Evaluasi Matched Budget:").pack(anchor="w", pady=(10, 2))
        self.progress_eval = ttk.Progressbar(parent, orient="horizontal", mode="determinate")
        self.progress_eval.pack(fill=tk.X)
        self.lbl_eval_counter = ttk.Label(parent, text="0 / 2000 Evaluasi (0.0%)", font=("Helvetica", 8))
        self.lbl_eval_counter.pack(anchor="e", pady=2)

    def _build_tab_guide(self, parent):
        """Membangun tab panduan visual 'Ini Apa & Bagaimana?'."""
        canvas = tk.Canvas(parent, bg="#ffffff", highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scroll_content = ttk.Frame(canvas, padding=5)

        scroll_content.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_content, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        guide_items = [
            ("🗺️ Legenda Objek & Wilayah Peta (Peta Studi)",
             "• RUMAH (Kotak Merah): Permukiman penduduk konsumen minimarket.\n"
             "• PERMUKIMAN WARGA (Elips Biru Putus-putus): Kluster permukiman barat padat penduduk.\n"
             "• FASILITAS UMUM:\n"
             "   - M (Hijau): Masjid di tengah desa\n"
             "   - S (Biru): Sekolah (3 titik)\n"
             "   - P (Hijau): Pasar desa\n"
             "   - R (Oranye): Restoran / Warung (3 titik)\n"
             "   - B (Ungu): Bengkel motor & bengkel alat berat\n"
             "• KOMPETITOR (K, Lingkaran Merah): Minimarket lama eksisting di dekat jembatan.\n"
             "• JARINGAN JALAN:\n"
             "   - Arteri (Hitam 4 Lajur): Jl. Lintas Provinsi membentang vertikal (X=1150m)\n"
             "   - Kolektor (Oranye): Jl. Utama Desa & Jl. Akses Tambang\n"
             "   - Lokal (Abu-abu): Jl. Lingkar Timur & Jl. Dusun Barat\n"
             "   - Hauling (Cokelat Lebar): Jalur khusus truk tambang batubara/mineral\n"
             "• ZONA TERLARANG (Penalti -1000):\n"
             "   - Hutan Lindung (Barat Laut)\n"
             "   - Sawah Pertanian (Barat Daya)\n"
             "   - Sungai Utama (Tengah)\n"
             "   - Lahan Tambang Aktif (Open Pit) & Stockpile (Tenggara)\n"
             "• BATASAN KORIDOR JALAN:\n"
             "   - Penempatan minimarket dibatasi hanya di koridor jalan (jarak <= 50 meter).\n"
             "   - Titik di luar koridor jalan dikenakan penalti off-road (tidak valid)."),

            ("🗺️ Warna Latar Heatmap", 
             "• HIJAU PEKAT: Lokasi bernilai fitness tinggi (potensi pasar & akses jalan prima).\n"
             "• KUNING-ORANYE: Lokasi bernilai sedang.\n"
             "• MERAH: Lokasi bernilai buruk (jauh dari jalan atau pemukiman warga).\n"
             "• PUTIH BERGARIS PUTUS-PUTUS: Zona Terlarang (Sungai, Sawah, Tambang, Hutan). Titik di sini dikenakan penalti -1000."),
            
            ("🧬 Elemen Visual GA (Kiri)",
             "• TITIK BIRU (Individu): Solusi kandidat dalam populasi generasi saat ini.\n"
             "• TITIK ORANYE (Individu Termutasi): Individu yang mengalami pergeseran acak (Gaussian Mutation) untuk mengeksplorasi area baru dan keluar dari optimum lokal.\n"
             "• BINTANG EMAS (★ Elit): 2 individu terbaik yang dilestarikan langsung ke generasi berikutnya.\n"
             "• TANDA TAMBAH HIJAU (+ Solusi GA): Posisi lokasi terpilih yang ditemukan algoritma GA."),

            ("🛣️ Batasan Koridor Jalan (Facility Location)",
             "• Penempatan minimarket HANYA dapat dibangun di kawasan yang memiliki akses jalan langsung.\n"
             "• Titik terjauh dibatasi pada koridor jalan (jarak <= 50 meter dari garis tengah jalan).\n"
             "• Titik di luar koridor jalan dikenakan penalti off-road agar minimarket selalu berada di tepi jalan yang dapat dilalui pelanggan."),

            ("🚀 Elemen Visual PSO (Kanan)",
             "• TITIK CYAN: Posisi partikel kawanan (swarm) di ruang pencarian.\n"
             "• PANAH BIRU (Quiver): Vektor kecepatan (V) partikel yang diarahkan oleh inersia, pbest, dan gbest.\n"
             "• GARIS BIRU TEBAL (Trail): Jejak lintasan gerak partikel selama 3 iterasi terakhir.\n"
             "• TITIK ORANYE KECIL: Posisi terbaik pribadi masing-masing partikel (pbest).\n"
             "• TANDA TAMBAH MERAH (+ gbest PSO): Posisi lokasi terpilih global kawanan PSO."),

            ("📊 Kurva Konvergensi (Bawah)",
             "• Menunjukkan nilai fitness terbaik terhadap JUMLAH EVALUASI NYATA (Matched Budget), bukan generasi.\n"
             "• Garis Biru = Progres GA, Garis Merah = Progres PSO.\n"
             "• Titik lingkaran/kotak menunjukkan langkah komputasi saat ini."),

            ("🛣️ Kondisi Jalan: Bagus (Proper) vs Bertanah Lumpur",
             "• Penempatan toko sudah diwajibkan berada di koridor jalan (jarak <= 50m). Karena itu, kriteria ini BUKAN sekadar ada/tidaknya jalan, melainkan KONDISI & KELAYAKAN JALANNYA:\n"
             "• JALAN BAGUS / PROPER (Skor 0.85 - 1.0): Jalan Arteri & Kolektor Desa beraspal mulus, nyaman dan aman dilalui kendaraan pembeli.\n"
             "• JALAN LOKAL (Skor 0.50 - 0.60): Paving block atau aspal dusun sederhana.\n"
             "• JALAN BERTANAH LUMPUR (Skor 0.15 - 0.25): Jalan Hauling tambang berupa tanah merah yang becek/lumpur saat hujan, berdebu pekat, dan banyak truk tronton raksasa, sehingga sangat tidak layak bagi konsumen belanja minimarket."),

            ("🔍 Mengapa Kurva Kompetitor Tidak Monoton?",
             "• Terlalu Dekat (<150m): Skor anjlok mendekati 0 karena perang harga dan perebutan pasar langsung.\n"
             "• Jarak Optimal (~350m Sweet Spot): Skor mencapai 1.0 karena memanfaatkan aglomerasi keramaian pasar tanpa saling mematikan.\n"
             "• Terlalu Jauh (>1000m): Skor turun mendekati 0 karena berada di tempat sepi yang jauh dari aktivitas komersial."),

            ("🛡️ Mengapa Mode 'min_valid' Tidak Memilih Sungai/Sawah?",
             "• Mode 'min_valid' mencari lokasi TERBURUK yang TETAP VALID (bukan zona terlarang).\n"
             "• Titik sungai/sawah tetap dikenakan penalti -1000, sehingga algoritma berkumpul di tanah kering/sudut peta yang sepi pelanggan."),
        ]

        for title, desc in guide_items:
            card = ttk.LabelFrame(scroll_content, text=f" {title} ", padding=6)
            card.pack(fill=tk.X, pady=4)
            lbl = ttk.Label(card, text=desc, font=("Helvetica", 8), justify=tk.LEFT, wraplength=350)
            lbl.pack(anchor="w")

    def _build_tab_inspect(self, parent):
        """Membangun tab inspeksi interaktif ketika pengguna mengklik titik di peta."""
        canvas = tk.Canvas(parent, bg="#ffffff", highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scroll_content = ttk.Frame(canvas, padding=5)

        scroll_content.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_content, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Label(
            scroll_content,
            text="👉 Klik titik manapun pada peta di sebelah kiri untuk menginspeksi nilai fitur, status koridor jalan, kompetitor, dan zonanya!",
            font=("Helvetica", 8, "italic"),
            foreground="#0369a1",
            wraplength=350,
            justify=tk.LEFT,
        ).pack(anchor="w", pady=(0, 6))

        # Status Wilayah Badge
        self.inspect_badge = tk.Label(
            scroll_content,
            text="KLIK PETA UNTUK INSPEKSI",
            font=("Helvetica", 9, "bold"),
            bg="#e2e8f0",
            fg="#334155",
            pady=6,
        )
        self.inspect_badge.pack(fill=tk.X, pady=2)

        self.lbl_coord_click = ttk.Label(scroll_content, text="Koordinat Terpilih: X = --- m, Y = --- m", font=("Helvetica", 9, "bold"))
        self.lbl_coord_click.pack(anchor="w", pady=3)

        self.lbl_total_fit = ttk.Label(scroll_content, text="Estimasi Total Fitness: ---", font=("Helvetica", 10, "bold"), foreground="#15803d")
        self.lbl_total_fit.pack(anchor="w", pady=2)

        # Kartu Akses & Kondisi Fisik Jalan
        card_road = ttk.LabelFrame(scroll_content, text=" 🛣️ Status Akses & Kondisi Fisik Jalan ", padding=6)
        card_road.pack(fill=tk.X, pady=4)

        self.lbl_inspect_road_name = ttk.Label(card_road, text="Jalan Terdekat: ---", font=("Helvetica", 8, "bold"), foreground="#0f172a")
        self.lbl_inspect_road_name.pack(anchor="w", pady=1)

        self.lbl_inspect_road_dist = ttk.Label(card_road, text="Jarak ke Jalan: --- (Koridor maks: 50m)", font=("Helvetica", 8))
        self.lbl_inspect_road_dist.pack(anchor="w", pady=1)

        self.lbl_inspect_road_cond = ttk.Label(card_road, text="Kondisi Fisik: ---", font=("Helvetica", 8))
        self.lbl_inspect_road_cond.pack(anchor="w", pady=1)

        # Kartu Pengaruh Kompetitor
        card_comp = ttk.LabelFrame(scroll_content, text=" 🏪 Analisis Pengaruh Kompetitor ", padding=6)
        card_comp.pack(fill=tk.X, pady=4)

        self.lbl_inspect_comp_dist = ttk.Label(card_comp, text="Jarak ke Kompetitor: ---", font=("Helvetica", 8, "bold"), foreground="#0f172a")
        self.lbl_inspect_comp_dist.pack(anchor="w", pady=1)

        self.lbl_inspect_comp_status = ttk.Label(
            card_comp,
            text="Dampak Persaingan: ---",
            font=("Helvetica", 8),
            wraplength=340,
            justify=tk.LEFT,
        )
        self.lbl_inspect_comp_status.pack(anchor="w", pady=1)

        # Kartu Nilai 4 Kriteria
        card_features = ttk.LabelFrame(scroll_content, text=" 🔬 Rincian Nilai 4 Fitur [0, 1] ", padding=8)
        card_features.pack(fill=tk.X, pady=4)

        self.prog_pop, self.lbl_pop_val = self._create_inspect_bar(card_features, "Populasi Pemukiman:", "#10b981")
        self.prog_road, self.lbl_road_val = self._create_inspect_bar(card_features, "Kondisi Jalan (Proper vs Lumpur):", "#f59e0b")
        self.prog_fac, self.lbl_fac_val = self._create_inspect_bar(card_features, "Tarikan Fasilitas Umum:", "#6366f1")
        self.prog_comp, self.lbl_comp_val = self._create_inspect_bar(card_features, "Faktor Kompetitor (Ricker):", "#ec4899")

        # Catatan Analisis Spasial
        card_notes = ttk.LabelFrame(scroll_content, text=" 📝 Analisis Spasial & Rekomendasi ", padding=6)
        card_notes.pack(fill=tk.X, pady=4)

        self.lbl_inspect_notes = ttk.Label(
            card_notes,
            text="Belum ada titik yang dipilih. Silakan klik titik manapun pada peta.",
            font=("Helvetica", 8),
            foreground="#475569",
            wraplength=340,
            justify=tk.LEFT,
        )
        self.lbl_inspect_notes.pack(anchor="w", pady=2)

    def _create_inspect_bar(self, parent, label_text, color):
        """Helper membuat progress bar inspeksi kriteria."""
        box = ttk.Frame(parent)
        box.pack(fill=tk.X, pady=2)
        ttk.Label(box, text=label_text, font=("Helvetica", 8)).pack(side=tk.LEFT)
        lbl_v = ttk.Label(box, text="0.00", font=("Helvetica", 8, "bold"), foreground=color)
        lbl_v.pack(side=tk.RIGHT)

        bar = ttk.Progressbar(parent, orient="horizontal", mode="determinate", maximum=1.0)
        bar.pack(fill=tk.X, pady=(0, 4))
        return bar, lbl_v

    def _on_map_selected(self, event):
        """Handler saat pengguna memilih preset peta baru dari dropdown."""
        self.current_map_name = self.cmb_map.get()
        self.current_map_file = self.available_maps[self.current_map_name]
        self.map_model = MapModel.load_from_json(self.current_map_file)
        self._recompute_optimization(reset_playback=True)

    def _on_param_change(self):
        """Handler saat mode atau bobot diubah pengguna."""
        self._recompute_optimization(reset_playback=True)

    def _recompute_optimization(self, reset_playback: bool = True):
        """Menghitung ulang optimasi GA dan PSO serta merefresh kanvas."""
        if self.timer_id is not None:
            self.root.after_cancel(self.timer_id)
            self.timer_id = None
        self.is_running = False
        self.btn_play.config(text="▶ Mulai (Play)", style="Play.TButton")
        self.status_badge.config(text="STATUS: MENGHITUNG OPTIMASI...", background="#1e293b", foreground="#facc15")
        self.root.update_idletasks()

        # Konfigurasi aktif
        cfg_copy = json.loads(json.dumps(self.config))
        cfg_copy["fitness_weights"] = {
            "w1_populasi": self.w1_var.get(),
            "w2_akses_jalan": self.w2_var.get(),
            "w3_fasilitas": self.w3_var.get(),
            "w4_kompetitor": self.w4_var.get(),
        }
        num_stores = self.stores_var.get()
        cfg_copy["facility"]["num_stores"] = num_stores

        self.evaluator = FitnessEvaluator(
            self.map_model,
            config=cfg_copy,
            mode=self.mode_var.get(),
            num_stores=num_stores,
        )

        # Prekomputasi matriks heatmap
        self.heatmap_grid, self.heatmap_extent = self.evaluator.compute_grid_heatmap(
            resolution_x=65, resolution_y=50
        )

        # Eksekusi GA dari nol
        ga_cfg = cfg_copy.get("ga", {})
        pop_size = cfg_copy.get("optimization", {}).get("population_size", 40)
        self.ga = GeneticAlgorithm(
            self.evaluator,
            num_stores=num_stores,
            pop_size=pop_size,
            crossover_rate=ga_cfg.get("crossover_rate", 0.85),
            mutation_rate=ga_cfg.get("mutation_rate", 0.15),
            tournament_size=ga_cfg.get("tournament_size", 3),
            blx_alpha=ga_cfg.get("blx_alpha", 0.5),
            mutation_sigma=ga_cfg.get("gaussian_mutation_sigma", 40.0),
            elitism_count=ga_cfg.get("elitism_count", 2),
            seed=42,
        )
        self.ga_best_sol, self.ga_best_fit, self.ga_history = self.ga.optimize(max_evaluations=self.budget_var.get())

        # Eksekusi PSO dari nol
        pso_cfg = cfg_copy.get("pso", {})
        self.pso = ParticleSwarmOptimization(
            self.evaluator,
            num_stores=num_stores,
            num_particles=pop_size,
            inertia_weight=pso_cfg.get("inertia_weight", 0.729),
            inertia_decay=pso_cfg.get("inertia_decay", True),
            inertia_max=pso_cfg.get("inertia_max", 0.9),
            inertia_min=pso_cfg.get("inertia_min", 0.4),
            c1_cognitive=pso_cfg.get("c1_cognitive", 1.494),
            c2_social=pso_cfg.get("c2_social", 1.494),
            v_max_fraction=pso_cfg.get("v_max_fraction", 0.15),
            boundary_handling=pso_cfg.get("boundary_handling", "reflect"),
            seed=42,
        )
        self.pso_best_sol, self.pso_best_fit, self.pso_history = self.pso.optimize(max_evaluations=self.budget_var.get())

        self.total_frames = max(len(self.ga_history), len(self.pso_history))
        if reset_playback:
            self.current_frame = 0

        # Inisialisasi gambar kanvas dasar
        self._init_canvas_plots()
        self._render_frame(self.current_frame)

        self.status_badge.config(text="STATUS: SIAP", background="#1e293b", foreground="#38bdf8")

    def _init_canvas_plots(self):
        """Menggambar latar peta, heatmap, dan elemen dinamis sesuai mode tampilan aktif."""
        self.fig.clf()

        if self.view_mode == "split":
            gs = gridspec.GridSpec(
                2, 2, height_ratios=[3.3, 1.8], hspace=0.38, wspace=0.16,
                left=0.06, right=0.98, top=0.94, bottom=0.08
            )
            self.ax_ga = self.fig.add_subplot(gs[0, 0])
            self.ax_pso = self.fig.add_subplot(gs[0, 1])
            self.ax_conv = self.fig.add_subplot(gs[1, :])

            self._draw_map_ga(self.ax_ga, is_full=False)
            self._draw_map_pso(self.ax_pso, is_full=False)
            self._draw_conv(self.ax_conv, is_full=False)

        elif self.view_mode == "ga":
            self.fig.subplots_adjust(left=0.07, right=0.97, top=0.94, bottom=0.08)
            self.ax_ga = self.fig.add_subplot(1, 1, 1)
            self.ax_pso = None
            self.ax_conv = None

            self._draw_map_ga(self.ax_ga, is_full=True)

        elif self.view_mode == "pso":
            self.fig.subplots_adjust(left=0.07, right=0.97, top=0.94, bottom=0.08)
            self.ax_ga = None
            self.ax_pso = self.fig.add_subplot(1, 1, 1)
            self.ax_conv = None

            self._draw_map_pso(self.ax_pso, is_full=True)

        elif self.view_mode == "conv":
            self.fig.subplots_adjust(left=0.08, right=0.96, top=0.92, bottom=0.12)
            self.ax_ga = None
            self.ax_pso = None
            self.ax_conv = self.fig.add_subplot(1, 1, 1)

            self._draw_conv(self.ax_conv, is_full=True)

        self.canvas.draw_idle()

    def _draw_map_ga(self, ax, is_full: bool = False):
        """Menggambar peta, heatmap, dan elemen dinamis GA pada aksis ax."""
        ax.clear()
        cmap_name = "RdYlGn" if self.mode_var.get() == "max" else "RdYlGn_r"
        title_txt = "🧬 Genetic Algorithm (GA) — Tampilan Layar Penuh" if is_full else "Genetic Algorithm (GA) — Populasi & Elit"

        plot_map(
            self.map_model,
            ax=ax,
            title=title_txt,
            show_legend=False,
            show_heatmap=True,
            heatmap_grid=self.heatmap_grid,
            heatmap_extent=self.heatmap_extent,
            heatmap_cmap=cmap_name,
            heatmap_alpha=0.50,
        )

        s_pop = 46 if is_full else 34
        s_mut = 60 if is_full else 46
        s_elite = 180 if is_full else 140
        s_best = 220 if is_full else 180

        self.ga_scatter_pop = ax.scatter([], [], c="#1d4ed8", s=s_pop, edgecolors="#0f172a", linewidths=0.6, alpha=0.85, zorder=8, label="Individu")
        self.ga_scatter_mut = ax.scatter([], [], c="#f59e0b", s=s_mut, marker="o", edgecolors="#78350f", linewidths=1.2, alpha=0.90, zorder=9, label="Individu Termutasi")
        self.ga_scatter_elite = ax.scatter([], [], c="#f59e0b", marker="*", s=s_elite, edgecolors="#78350f", linewidths=1.2, zorder=10, label="Elit")
        self.ga_scatter_best = ax.scatter([], [], c="#16a34a", marker="P", s=s_best, edgecolors="#ffffff", linewidths=1.8, zorder=11, label="Solusi GA")

        # Retikel target crosshair untuk inspeksi titik
        self.inspect_marker_ga_circle = ax.scatter([], [], marker="o", s=240 if is_full else 170, facecolors="none", edgecolors="#e11d48", linewidths=2.5, zorder=20)
        self.inspect_marker_ga_cross = ax.scatter([], [], marker="+", s=170 if is_full else 130, c="#e11d48", linewidths=2.2, zorder=21)

        if self.last_inspected_pt is not None:
            self.inspect_marker_ga_circle.set_offsets([self.last_inspected_pt])
            self.inspect_marker_ga_cross.set_offsets([self.last_inspected_pt])

        leg_y = -0.06 if is_full else -0.12
        leg_fs = 9 if is_full else 8
        ax.legend(
            [self.ga_scatter_pop, self.ga_scatter_mut, self.ga_scatter_elite, self.ga_scatter_best],
            ["Individu (Populasi)", "Individu Termutasi", "★ Elit Terpilih", "+ Solusi Terbaik GA"],
            loc="upper center",
            bbox_to_anchor=(0.5, leg_y),
            ncol=4,
            fontsize=leg_fs,
            framealpha=0.95,
            edgecolor="#cbd5e1",
        )

    def _draw_map_pso(self, ax, is_full: bool = False):
        """Menggambar peta, heatmap, dan elemen dinamis PSO pada aksis ax."""
        ax.clear()
        cmap_name = "RdYlGn" if self.mode_var.get() == "max" else "RdYlGn_r"
        title_txt = "🚀 Particle Swarm Optimization (PSO) — Tampilan Layar Penuh" if is_full else "Particle Swarm Optimization (PSO) — Swarm & Quiver"

        plot_map(
            self.map_model,
            ax=ax,
            title=title_txt,
            show_legend=False,
            show_heatmap=True,
            heatmap_grid=self.heatmap_grid,
            heatmap_extent=self.heatmap_extent,
            heatmap_cmap=cmap_name,
            heatmap_alpha=0.50,
        )

        lw_trail = 1.4 if is_full else 1.1
        s_swarm = 46 if is_full else 36
        s_pbest = 60 if is_full else 45
        s_gbest = 220 if is_full else 180

        self.pso_trail_lines = [ax.plot([], [], color="#0284c7", linewidth=lw_trail, alpha=0.45, zorder=7)[0] for _ in range(40)]
        self.pso_scatter_swarm = ax.scatter([], [], c="#0284c7", s=s_swarm, edgecolors="#0f172a", linewidths=0.8, alpha=0.85, zorder=8, label="Partikel")
        self.pso_quiver = None
        self.pso_scatter_pbest = ax.scatter([], [], c="#f97316", marker=".", s=s_pbest, alpha=0.75, zorder=9, label="pbest")
        self.pso_scatter_gbest = ax.scatter([], [], c="#dc2626", marker="P", s=s_gbest, edgecolors="#ffffff", linewidths=1.8, zorder=11, label="gbest (PSO)")

        # Retikel target crosshair untuk inspeksi titik
        self.inspect_marker_pso_circle = ax.scatter([], [], marker="o", s=240 if is_full else 170, facecolors="none", edgecolors="#e11d48", linewidths=2.5, zorder=20)
        self.inspect_marker_pso_cross = ax.scatter([], [], marker="+", s=170 if is_full else 130, c="#e11d48", linewidths=2.2, zorder=21)

        if self.last_inspected_pt is not None:
            self.inspect_marker_pso_circle.set_offsets([self.last_inspected_pt])
            self.inspect_marker_pso_cross.set_offsets([self.last_inspected_pt])

        leg_y = -0.06 if is_full else -0.12
        leg_fs = 9 if is_full else 8
        ax.legend(
            [self.pso_scatter_swarm, self.pso_scatter_pbest, self.pso_scatter_gbest],
            ["Partikel Swarm", "pbest (Terbaik Pribadi)", "+ gbest (Terbaik Global)"],
            loc="upper center",
            bbox_to_anchor=(0.5, leg_y),
            ncol=3,
            fontsize=leg_fs,
            framealpha=0.95,
            edgecolor="#cbd5e1",
        )

    def _draw_conv(self, ax, is_full: bool = False):
        """Menggambar kurva konvergensi live pada aksis ax."""
        ax.clear()
        ax.set_facecolor("#ffffff")
        ax.set_title(
            "Kurva Konvergensi Live: Nilai Fitness Terbaik vs Jumlah Evaluasi Nyata",
            fontsize=11 if is_full else 9.5, weight="bold"
        )
        ax.set_xlabel("Jumlah Evaluasi Fitness (Matched Budget)", fontsize=9.5 if is_full else 8.5)
        ax.set_ylabel("Nilai Fitness", fontsize=9.5 if is_full else 8.5)
        ax.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1")
        ax.set_xlim(0, self.budget_var.get())

        ga_all_fits = [r.best_fitness for r in self.ga_history]
        pso_all_fits = [r.gbest_fitness for r in self.pso_history]
        min_f = min(min(ga_all_fits), min(pso_all_fits))
        max_f = max(max(ga_all_fits), max(pso_all_fits))
        margin = max(0.05, 0.1 * (max_f - min_f))
        ax.set_ylim(min_f - margin, max_f + margin)

        lw_line = 2.8 if is_full else 2.2
        ms_marker = 8 if is_full else 6
        self.line_ga, = ax.plot([], [], color="#2563eb", linewidth=lw_line, label=f"GA (Akhir: {self.ga_best_fit:.4f})")
        self.line_pso, = ax.plot([], [], color="#dc2626", linewidth=lw_line, label=f"PSO (Akhir: {self.pso_best_fit:.4f})")
        self.marker_ga, = ax.plot([], [], marker="o", color="#2563eb", markersize=ms_marker)
        self.marker_pso, = ax.plot([], [], marker="s", color="#dc2626", markersize=ms_marker)
        ax.legend(loc="lower right", fontsize=9.5 if is_full else 8, framealpha=0.9)

    def _render_frame(self, frame_idx: int):
        """Memperbarui posisi objek grafik pada frame tertentu."""
        idx_ga = min(frame_idx, len(self.ga_history) - 1)
        idx_pso = min(frame_idx, len(self.pso_history) - 1)

        rec_ga = self.ga_history[idx_ga]
        rec_pso = self.pso_history[idx_pso]

        # 1. Update GA jika aksis aktif
        if self.ax_ga is not None:
            self.ax_ga.set_title(
                f"GA — Gen {rec_ga.generation} | Eval: {rec_ga.evaluations}/{self.budget_var.get()} | Best Fit: {rec_ga.best_fitness:.4f}",
                fontsize=10 if self.view_mode == "ga" else 9.5, weight="bold", color="#1e293b"
            )
            self.ga_scatter_pop.set_offsets(rec_ga.population[:, :2])

            if len(rec_ga.elite_indices) > 0:
                self.ga_scatter_elite.set_offsets(rec_ga.population[rec_ga.elite_indices, :2])
            else:
                self.ga_scatter_elite.set_offsets(np.empty((0, 2)))

            if np.any(rec_ga.mutated_mask):
                self.ga_scatter_mut.set_offsets(rec_ga.population[rec_ga.mutated_mask, :2])
            else:
                self.ga_scatter_mut.set_offsets(np.empty((0, 2)))

            ga_best_pts = rec_ga.best_position.reshape((-1, 2))
            self.ga_scatter_best.set_offsets(ga_best_pts)

        # 2. Update PSO jika aksis aktif
        if self.ax_pso is not None:
            self.ax_pso.set_title(
                f"PSO — Iter {rec_pso.iteration} | Eval: {rec_pso.evaluations}/{self.budget_var.get()} | Best Fit: {rec_pso.gbest_fitness:.4f}",
                fontsize=10 if self.view_mode == "pso" else 9.5, weight="bold", color="#1e293b"
            )
            pso_coords = rec_pso.positions[:, :2]
            self.pso_scatter_swarm.set_offsets(pso_coords)

            # Quiver
            if self.pso_quiver is not None:
                self.pso_quiver.remove()
            vx = rec_pso.velocities[:, 0]
            vy = rec_pso.velocities[:, 1]
            scale_val = 600 if self.view_mode == "pso" else 800
            self.pso_quiver = self.ax_pso.quiver(
                pso_coords[:, 0], pso_coords[:, 1], vx, vy,
                color="#0369a1", scale=scale_val, width=0.0035, alpha=0.65, zorder=9
            )

            # Trails
            trail_start = max(0, idx_pso - 3)
            for p_i in range(min(40, len(pso_coords))):
                t_pts = [self.pso_history[t].positions[p_i, :2] for t in range(trail_start, idx_pso + 1)]
                if len(t_pts) >= 2:
                    arr = np.array(t_pts)
                    self.pso_trail_lines[p_i].set_data(arr[:, 0], arr[:, 1])
                else:
                    self.pso_trail_lines[p_i].set_data([], [])

            self.pso_scatter_pbest.set_offsets(rec_pso.pbest_positions[:, :2])
            pso_gbest_pts = rec_pso.gbest_position.reshape((-1, 2))
            self.pso_scatter_gbest.set_offsets(pso_gbest_pts)

        # 3. Update Konvergensi jika aksis aktif
        if self.ax_conv is not None:
            ga_ev = [self.ga_history[t].evaluations for t in range(idx_ga + 1)]
            ga_ft = [self.ga_history[t].best_fitness for t in range(idx_ga + 1)]
            pso_ev = [self.pso_history[t].evaluations for t in range(idx_pso + 1)]
            pso_ft = [self.pso_history[t].gbest_fitness for t in range(idx_pso + 1)]

            self.line_ga.set_data(ga_ev, ga_ft)
            self.line_pso.set_data(pso_ev, pso_ft)
            if len(ga_ev) > 0:
                self.marker_ga.set_data([ga_ev[-1]], [ga_ft[-1]])
            if len(pso_ev) > 0:
                self.marker_pso.set_data([pso_ev[-1]], [pso_ft[-1]])

        self.canvas.draw_idle()

        # Update Papan Skor Live
        self._update_scoreboard_metrics(rec_ga, rec_pso)

    def _update_scoreboard_metrics(self, rec_ga: GAGenerationRecord, rec_pso: PSOIterationRecord):
        """Memperbarui angka dan status pada Tab Papan Skor."""
        fit_ga = rec_ga.best_fitness
        fit_pso = rec_pso.gbest_fitness

        self.lbl_ga_fit.config(text=f"Best Fitness: {fit_ga:.4f}")
        self.lbl_ga_coord.config(text=f"Koordinat: ({rec_ga.best_position[0]:.1f}, {rec_ga.best_position[1]:.1f})")
        self.lbl_ga_eval.config(text=f"Evaluasi: {rec_ga.evaluations} / {self.budget_var.get()}")

        self.lbl_pso_fit.config(text=f"gbest Fitness: {fit_pso:.4f}")
        self.lbl_pso_coord.config(text=f"Koordinat: ({rec_pso.gbest_position[0]:.1f}, {rec_pso.gbest_position[1]:.1f})")
        self.lbl_pso_eval.config(text=f"Evaluasi: {rec_pso.evaluations} / {self.budget_var.get()}")

        # Update Banner Pemimpin
        diff = abs(fit_ga - fit_pso)
        if fit_ga > fit_pso + 1e-4:
            self.banner_leader.config(text=f"🏆 GA MEMIMPIN (+{diff:.4f})", bg="#2563eb")
        elif fit_pso > fit_ga + 1e-4:
            self.banner_leader.config(text=f"🏆 PSO MEMIMPIN (+{diff:.4f})", bg="#dc2626")
        else:
            self.banner_leader.config(text="⚖️ KEDUA ALGORITMA IMBANG", bg="#16a34a")

        self.lbl_gap_text.config(
            text=f"Selisih Nilai Fitness: {diff:.5f}\n"
                 f"Evaluasi Berjalan: GA={rec_ga.evaluations}, PSO={rec_pso.evaluations}"
        )

        # Progress bar
        curr_eval = max(rec_ga.evaluations, rec_pso.evaluations)
        total_b = self.budget_var.get()
        pct = min(100.0, (curr_eval / total_b) * 100.0)
        self.progress_eval["value"] = pct
        self.lbl_eval_counter.config(text=f"{curr_eval} / {total_b} Evaluasi ({pct:.1f}%)")

    def _toggle_simulation(self):
        """Memulai atau menjeda simulasi playback."""
        if not self.is_running:
            self.is_running = True
            self.btn_play.config(text="⏸ Jeda (Pause)", style="Pause.TButton")
            self.status_badge.config(text="STATUS: BERJALAN", background="#1e293b", foreground="#4ade80")
            self._simulation_loop()
        else:
            self.is_running = False
            self.btn_play.config(text="▶ Lanjut (Play)", style="Play.TButton")
            self.status_badge.config(text="STATUS: DIJEDA", background="#1e293b", foreground="#f97316")
            if self.timer_id is not None:
                self.root.after_cancel(self.timer_id)
                self.timer_id = None

    def _simulation_loop(self):
        """Loop iterasi berjangka waktu menggunakan root.after."""
        if not self.is_running:
            return

        if self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self._render_frame(self.current_frame)
            delay = max(5, int(self.speed_var.get()))
            self.timer_id = self.root.after(delay, self._simulation_loop)
        else:
            # Selesai
            self.is_running = False
            self.btn_play.config(text="▶ Ulangi (Play)", style="Play.TButton")
            self.status_badge.config(text="STATUS: SELESAI (KONVERGEN)", background="#1e293b", foreground="#38bdf8")

    def _step_simulation(self):
        """Maju tepat satu langkah frame."""
        if self.is_running:
            self._toggle_simulation()
        if self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self._render_frame(self.current_frame)

    def _reset_simulation(self):
        """Mereset frame ke awal (0)."""
        if self.is_running:
            self._toggle_simulation()
        self.current_frame = 0
        self._render_frame(0)
        self.status_badge.config(text="STATUS: SIAP", background="#1e293b", foreground="#38bdf8")

    def _run_to_end(self):
        """Melompat langsung ke frame terakhir."""
        if self.is_running:
            self._toggle_simulation()
        self.current_frame = self.total_frames - 1
        self._render_frame(self.current_frame)
        self.status_badge.config(text="STATUS: SELESAI (KONVERGEN)", background="#1e293b", foreground="#38bdf8")

    def _on_map_click(self, event):
        """Handler saat pengguna mengklik peta untuk inspeksi titik."""
        if event.inaxes not in (self.ax_ga, self.ax_pso):
            return
        if event.xdata is None or event.ydata is None:
            return

        x, y = float(event.xdata), float(event.ydata)
        self._inspect_coordinate(x, y)
        # Pindah tab ke Tab Inspeksi
        self.notebook.select(3)

    def _on_map_hover(self, event):
        """Menampilkan koordinat kursor di status bar header saat hover di atas peta."""
        if event.inaxes in (self.ax_ga, self.ax_pso) and event.xdata is not None and event.ydata is not None:
            self.status_badge.config(text=f"KURSOR: X={event.xdata:.1f}m, Y={event.ydata:.1f}m")

    def _inspect_coordinate(self, x: float, y: float):
        """Menghitung dan menampilkan rincian kriteria pada titik koordinat yang diklik."""
        self.last_inspected_pt = (x, y)
        pt = np.array([x, y], dtype=np.float64)
        res = self.evaluator.evaluate_features_single_point(pt)

        # Update marker retikel crosshair di peta yang aktif
        if self.ax_ga is not None and hasattr(self, "inspect_marker_ga_circle"):
            self.inspect_marker_ga_circle.set_offsets([[x, y]])
            self.inspect_marker_ga_cross.set_offsets([[x, y]])
        if self.ax_pso is not None and hasattr(self, "inspect_marker_pso_circle"):
            self.inspect_marker_pso_circle.set_offsets([[x, y]])
            self.inspect_marker_pso_cross.set_offsets([[x, y]])
        self.canvas.draw_idle()

        self.lbl_coord_click.config(text=f"Koordinat Terpilih: X = {x:.1f} m, Y = {y:.1f} m")

        # Ekstraksi informasi spasial
        d_road = res.get("d_road_geo", 0.0)
        r_name = res.get("nearest_road_name", "Jalan")
        r_class = res.get("nearest_road_class", "lokal")
        forbid_name = res.get("forbidden_zone_name", None)
        d_comp = res.get("d_comp", 9999.0)

        # 1. Update Badge Status & Total Fitness
        if res.get("is_oob", False):
            self.inspect_badge.config(text="⚠️ DI LUAR BATAS WILAYAH PETA (PENALTI)", bg="#ef4444", fg="#ffffff")
            self.lbl_total_fit.config(text=f"Estimasi Fitness: {res['raw_score']:.1f} (Ilegal - Out of Bounds)", foreground="#dc2626")
        elif res.get("is_forbidden", False):
            z_title = forbid_name if forbid_name else "Zona Terlarang"
            self.inspect_badge.config(text=f"⚠️ {z_title.upper()} (PENALTI)", bg="#ef4444", fg="#ffffff")
            self.lbl_total_fit.config(text=f"Estimasi Fitness: {res['raw_score']:.1f} (Ilegal - {z_title})", foreground="#dc2626")
        elif not res.get("is_in_road_corridor", True):
            self.inspect_badge.config(text="⚠️ DI LUAR KAWASAN JALAN (> 50m, PENALTI)", bg="#f97316", fg="#ffffff")
            self.lbl_total_fit.config(text=f"Estimasi Fitness: {res['raw_score']:.2f} (Off-road Penalti)", foreground="#ea580c")
        else:
            self.inspect_badge.config(text="✅ LOKASI VALID (LEGAL DI KORIDOR JALAN)", bg="#16a34a", fg="#ffffff")
            self.lbl_total_fit.config(text=f"Estimasi Total Fitness: {res['raw_score']:.4f}", foreground="#15803d")

        # 2. Update Informasi Jalan
        self.lbl_inspect_road_name.config(text=f"Jalan Terdekat: {r_name} ({r_class.capitalize()})")
        if d_road <= 50.0:
            self.lbl_inspect_road_dist.config(
                text=f"Jarak ke Garis Tengah: {d_road:.1f} m  (✅ Dalam Koridor <= 50m)",
                foreground="#16a34a"
            )
        else:
            self.lbl_inspect_road_dist.config(
                text=f"Jarak ke Garis Tengah: {d_road:.1f} m  (⚠️ Di Luar Koridor > 50m)",
                foreground="#dc2626"
            )

        if r_class == "hauling":
            self.lbl_inspect_road_cond.config(
                text="Kondisi: Tanah Merah / Lumpur Hauling (Kualitas Rendah 0.20 - Becek/Debu)",
                foreground="#b45309"
            )
        elif r_class == "arteri":
            self.lbl_inspect_road_cond.config(
                text="Kondisi: Aspal Prima 4 Lajur (Lintas Provinsi, Kualitas 1.00)",
                foreground="#15803d"
            )
        elif r_class == "kolektor":
            self.lbl_inspect_road_cond.config(
                text="Kondisi: Aspal 2 Arah Mulus (Jalan Utama Desa, Kualitas 0.85)",
                foreground="#15803d"
            )
        else:
            self.lbl_inspect_road_cond.config(
                text="Kondisi: Aspal Kampung / Paving (Jalan Lokal Dusun, Kualitas 0.60)",
                foreground="#475569"
            )

        # 3. Update Informasi Kompetitor
        self.lbl_inspect_comp_dist.config(text=f"Jarak ke Minimarket Kompetitor: {d_comp:.1f} m")
        if d_comp < 180.0:
            self.lbl_inspect_comp_status.config(
                text="Dampak: ⚠️ Terlalu Dekat (<180m)! Kanibalisasi langsung & perang harga sengit.",
                foreground="#dc2626"
            )
        elif 180.0 <= d_comp <= 500.0:
            self.lbl_inspect_comp_status.config(
                text="Dampak: 🎯 Zona Manis / Sweet Spot (~350m)! Memanfaatkan keramaian tanpa perang harga.",
                foreground="#16a34a"
            )
        else:
            self.lbl_inspect_comp_status.config(
                text="Dampak: ℹ️ Terlalu Jauh (>500m). Agak sepi dari arus pembeli komersial.",
                foreground="#64748b"
            )

        # 4. Update Nilai 4 Kriteria (Selalu ditampilkan dengan data riil)
        self.prog_pop["value"] = res["populasi"]
        self.lbl_pop_val.config(text=f"{res['populasi']:.2f}")

        self.prog_road["value"] = res["akses_jalan"]
        self.lbl_road_val.config(text=f"{res['akses_jalan']:.2f}")

        self.prog_fac["value"] = res["fasilitas"]
        self.lbl_fac_val.config(text=f"{res['fasilitas']:.2f}")

        self.prog_comp["value"] = res["kompetitor"]
        self.lbl_comp_val.config(text=f"{res['kompetitor']:.2f}")

        # 5. Catatan Spasial Terpadu
        notes = []
        if res.get("is_oob", False):
            notes.append("Titik berada di luar batas peta koordinat.")
        elif res.get("is_forbidden", False):
            notes.append(f"Titik berada di dalam zona terlarang: {forbid_name}. Pembangunan toko dilarang hukum lingkungan/keselamatan tambang.")
        elif not res.get("is_in_road_corridor", True):
            notes.append(f"Titik berjarak {d_road:.1f}m dari jalan (> 50m). Tidak dapat diakses kendaraan pelanggan tanpa jalan masuk.")
        else:
            notes.append("Titik berada di koridor jalan legal.")

        if res["populasi"] >= 0.5:
            notes.append("Tingkat kepadatan perumahan konsumen sangat tinggi.")
        elif res["populasi"] <= 0.15:
            notes.append("Jumlah rumah konsumen di sekitar titik ini sangat minim.")

        if r_class == "hauling":
            notes.append("Kondisi jalan hauling berupa tanah merah berlumpur yang rawan bagi pembeli retail.")

        self.lbl_inspect_notes.config(text=" ".join(notes), foreground="#334155")

    # ==========================================
    # Metode Kendali Zoom per Metode & Global
    # ==========================================
    def _zoom_axis(self, ax, factor: float, center_x: Optional[float] = None, center_y: Optional[float] = None):
        """Memperbesar atau memperkecil skala tampilan pada subplot peta."""
        if ax is None:
            return
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()

        cx = (cur_xlim[0] + cur_xlim[1]) / 2.0 if center_x is None else center_x
        cy = (cur_ylim[0] + cur_ylim[1]) / 2.0 if center_y is None else center_y

        new_w = (cur_xlim[1] - cur_xlim[0]) * factor
        new_h = (cur_ylim[1] - cur_ylim[0]) * factor

        # Batas maksimal zoom out tidak melebihi ukuran peta asli
        if new_w > self.map_model.width * 1.15:
            ax.set_xlim(0, self.map_model.width)
            ax.set_ylim(0, self.map_model.height)
        else:
            rel_x = (cx - cur_xlim[0]) / (cur_xlim[1] - cur_xlim[0]) if (cur_xlim[1] - cur_xlim[0]) > 0 else 0.5
            rel_y = (cy - cur_ylim[0]) / (cur_ylim[1] - cur_ylim[0]) if (cur_ylim[1] - cur_ylim[0]) > 0 else 0.5
            ax.set_xlim(cx - new_w * rel_x, cx + new_w * (1.0 - rel_x))
            ax.set_ylim(cy - new_h * rel_y, cy + new_h * (1.0 - rel_y))

        self.canvas.draw_idle()

    def _on_map_scroll(self, event):
        """Handler roda scroll mouse: scroll ke atas = zoom in, scroll ke bawah = zoom out."""
        if event.inaxes in (self.ax_ga, self.ax_pso) and event.xdata is not None and event.ydata is not None:
            factor = 0.82 if event.button == "up" else 1.22
            self._zoom_axis(event.inaxes, factor, center_x=event.xdata, center_y=event.ydata)

    def _zoom_in_ga(self):
        if self.ax_ga is not None:
            self._zoom_axis(self.ax_ga, 0.75)

    def _zoom_out_ga(self):
        if self.ax_ga is not None:
            self._zoom_axis(self.ax_ga, 1.33)

    def _focus_village_ga(self):
        if self.ax_ga is not None:
            self.ax_ga.set_xlim(40, 750)
            self.ax_ga.set_ylim(520, 1050)
            self.canvas.draw_idle()

    def _reset_zoom_ga(self):
        if self.ax_ga is not None:
            self.ax_ga.set_xlim(0, self.map_model.width)
            self.ax_ga.set_ylim(0, self.map_model.height)
            self.canvas.draw_idle()

    def _zoom_in_pso(self):
        if self.ax_pso is not None:
            self._zoom_axis(self.ax_pso, 0.75)

    def _zoom_out_pso(self):
        if self.ax_pso is not None:
            self._zoom_axis(self.ax_pso, 1.33)

    def _focus_village_pso(self):
        if self.ax_pso is not None:
            self.ax_pso.set_xlim(40, 750)
            self.ax_pso.set_ylim(520, 1050)
            self.canvas.draw_idle()

    def _reset_zoom_pso(self):
        if self.ax_pso is not None:
            self.ax_pso.set_xlim(0, self.map_model.width)
            self.ax_pso.set_ylim(0, self.map_model.height)
            self.canvas.draw_idle()

    def _focus_village_both(self):
        self._focus_village_ga()
        self._focus_village_pso()

    def _reset_zoom_both(self):
        self._reset_zoom_ga()
        self._reset_zoom_pso()


def main():
    root = tk.Tk()
    app = FacilityLocationTkApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
