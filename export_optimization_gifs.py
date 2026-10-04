"""
Script untuk mengekspor animasi proses optimasi GA dan PSO ke format GIF
baik untuk Mode Maksimasi (lokasi terbaik) maupun Mode Minimasi (lokasi terburuk legal).

Menghasilkan 6 berkas GIF berkualitas tinggi:
1. results/ga_maksimasi.gif
2. results/pso_maksimasi.gif
3. results/simulasi_maksimasi_split.gif
4. results/ga_minimasi.gif
5. results/pso_minimasi.gif
6. results/simulasi_minimasi_split.gif
"""

import os
import sys
import time
import tkinter as tk
import numpy as np
from PIL import Image

# Force unbuffered output so logs print immediately
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

from app_tkinter import FacilityLocationTkApp


def capture_view_frames(app, view_mode: str, step_interval: int = 1):
    """
    Mengambil frame demi frame dari kanvas Tkinter untuk view_mode tertentu.
    Menggunakan buffer Matplotlib langsung dan fastoctree quantization.
    """
    app._set_view_mode(view_mode)
    app.root.update_idletasks()
    
    total = app.total_frames
    frames = []
    
    # Ambil frame
    selected_indices = list(range(0, total, step_interval))
    if (total - 1) not in selected_indices:
        selected_indices.append(total - 1)
        
    print(f"  -> Rendering {len(selected_indices)} frames for '{view_mode}'...", flush=True)
    
    for f_idx in selected_indices:
        app._render_frame(f_idx)
        app.fig.canvas.draw()
        
        # Ekstrak buffer RGBA dari Matplotlib canvas
        w, h = app.fig.canvas.get_width_height()
        buf = app.fig.canvas.buffer_rgba()
        raw_img = Image.frombuffer("RGBA", (w, h), buf, "raw", "RGBA", 0, 1)
        
        # Konversi ke RGB lalu kuantisasi super cepat (fastoctree: ~0.01s per frame)
        rgb_img = raw_img.convert("RGB")
        quant_img = rgb_img.quantize(colors=256, method=Image.Quantize.FASTOCTREE)
        frames.append(quant_img)
        
    return frames


def save_as_gif(frames, output_path: str, fps: int = 8, final_pause_ms: int = 2500):
    """
    Menyimpan daftar frame PIL Image sebagai berkas animasi GIF dengan jeda di frame akhir.
    """
    if not frames:
        print(f"Error: Tidak ada frame untuk {output_path}", flush=True)
        return
        
    frame_duration = int(1000 / fps)
    durations = [frame_duration] * (len(frames) - 1) + [final_pause_ms]
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  [OK] Saved: {output_path} ({size_mb:.2f} MB, {len(frames)} frames)", flush=True)


def generate_all_gifs():
    t_start = time.time()
    print("=" * 70, flush=True)
    print("MEMULAI PEMBUATAN ANIMASI GIF GA & PSO (MAKSIMASI & MINIMASI)", flush=True)
    print("=" * 70, flush=True)
    
    root = tk.Tk()
    root.geometry("1100x700")
    root.withdraw()
    
    print("\n[1/3] Menginisialisasi GUI Tkinter & Menghitung Mode Maksimasi...", flush=True)
    app = FacilityLocationTkApp(root)
    root.update()
    
    # -------------------------------------------------------------
    # SIKLUS 1: MODE MAKSIMASI (Mencari Lokasi Terbaik KOPDES)
    # -------------------------------------------------------------
    print("\n[2/3] Mengekspor GIF Mode Maksimasi...", flush=True)
    
    # 1. GA Maksimasi (Layar Penuh)
    print("Generating: GA Maksimasi (Layar Penuh)...", flush=True)
    frames_ga_max = capture_view_frames(app, view_mode="ga", step_interval=1)
    save_as_gif(frames_ga_max, "d:/biocomputing-7/results/ga_maksimasi.gif", fps=10, final_pause_ms=2500)
    
    # 2. PSO Maksimasi (Layar Penuh)
    print("Generating: PSO Maksimasi (Layar Penuh)...", flush=True)
    frames_pso_max = capture_view_frames(app, view_mode="pso", step_interval=1)
    save_as_gif(frames_pso_max, "d:/biocomputing-7/results/pso_maksimasi.gif", fps=10, final_pause_ms=2500)
    
    # 3. Simulasi Split Maksimasi (GA + PSO + Konvergensi)
    print("Generating: Simulasi Maksimasi Split View...", flush=True)
    frames_split_max = capture_view_frames(app, view_mode="split", step_interval=1)
    save_as_gif(frames_split_max, "d:/biocomputing-7/results/simulasi_maksimasi_split.gif", fps=10, final_pause_ms=2500)
    
    # -------------------------------------------------------------
    # SIKLUS 2: MODE MINIMASI (Mencari Lokasi Terburuk Legal KOPDES)
    # -------------------------------------------------------------
    print("\n[3/3] Menghitung & Mengekspor GIF Mode Minimasi...", flush=True)
    app.mode_var.set("min_valid")
    app._recompute_optimization(reset_playback=True)
    root.update()
    
    # 4. GA Minimasi (Layar Penuh)
    print("Generating: GA Minimasi (Layar Penuh)...", flush=True)
    frames_ga_min = capture_view_frames(app, view_mode="ga", step_interval=1)
    save_as_gif(frames_ga_min, "d:/biocomputing-7/results/ga_minimasi.gif", fps=10, final_pause_ms=2500)
    
    # 5. PSO Minimasi (Layar Penuh)
    print("Generating: PSO Minimasi (Layar Penuh)...", flush=True)
    frames_pso_min = capture_view_frames(app, view_mode="pso", step_interval=1)
    save_as_gif(frames_pso_min, "d:/biocomputing-7/results/pso_minimasi.gif", fps=10, final_pause_ms=2500)
    
    # 6. Simulasi Split Minimasi (GA + PSO + Konvergensi)
    print("Generating: Simulasi Minimasi Split View...", flush=True)
    frames_split_min = capture_view_frames(app, view_mode="split", step_interval=1)
    save_as_gif(frames_split_min, "d:/biocomputing-7/results/simulasi_minimasi_split.gif", fps=10, final_pause_ms=2500)
    
    root.destroy()
    t_elapsed = time.time() - t_start
    print("\n" + "=" * 70, flush=True)
    print(f"SELESAI! Seluruh 6 berkas GIF berhasil dibuat dalam {t_elapsed:.1f} detik.", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    generate_all_gifs()
