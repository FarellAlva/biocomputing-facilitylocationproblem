import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))
import tkinter as tk

from app_tkinter import FacilityLocationTkApp

print("Initializing TkApp...")
root = tk.Tk()
root.geometry("1100x700")
root.withdraw()

app = FacilityLocationTkApp(root)
# Short budget for fast test
app.budget_var.set(80)
app._recompute_optimization(reset_playback=True)
root.update()

for mode in ["ga", "pso", "split"]:
    print(f"Testing view mode: {mode} (Maksimasi)...")
    app._set_view_mode(mode)
    app._render_frame(0)
    app.fig.canvas.draw()
    app._render_frame(app.total_frames - 1)
    app.fig.canvas.draw()
    app.fig.savefig(f"d:/biocomputing-7/scratch/test_final_{mode}_max.png", dpi=100)

# Switch to Minimasi
print("Switching to Minimasi...")
app.mode_var.set("min_valid")
app._recompute_optimization(reset_playback=True)
root.update()

for mode in ["ga", "pso", "split"]:
    print(f"Testing view mode: {mode} (Minimasi)...")
    app._set_view_mode(mode)
    app._render_frame(0)
    app.fig.canvas.draw()
    app._render_frame(app.total_frames - 1)
    app.fig.canvas.draw()
    app.fig.savefig(f"d:/biocomputing-7/scratch/test_final_{mode}_min.png", dpi=100)

root.destroy()
print("ALL TESTS PASSED SUCCESSFULLY!")
