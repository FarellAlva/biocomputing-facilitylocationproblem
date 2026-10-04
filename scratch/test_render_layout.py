import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))
import tkinter as tk
import matplotlib.pyplot as plt

from app_tkinter import FacilityLocationTkApp

root = tk.Tk()
root.geometry("1100x700")
root.withdraw()

app = FacilityLocationTkApp(root)
root.update()

# Render GA Full
app._set_view_mode("ga")
app._render_frame(app.total_frames - 1)
app.fig.savefig("d:/biocomputing-7/scratch/test_ga_full.png", dpi=100)

# Render PSO Full
app._set_view_mode("pso")
app._render_frame(app.total_frames - 1)
app.fig.savefig("d:/biocomputing-7/scratch/test_pso_full.png", dpi=100)

# Render Split
app._set_view_mode("split")
app._render_frame(app.total_frames - 1)
app.fig.savefig("d:/biocomputing-7/scratch/test_split.png", dpi=100)

root.destroy()
print("Saved test frames successfully!")
