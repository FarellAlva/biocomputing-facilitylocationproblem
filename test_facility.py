"""
Unit Test Ringan untuk Facility Location (test_facility.py)
Menggunakan pytest untuk menguji:
1. Counter evaluasi fitness (eval_count) dan matched budget
2. Penalti zona terlarang dan luar peta
3. Mode invert (max vs min_valid)
4. Reproducibility hasil dengan seed RNG
5. Algoritma geometri (ray casting point-in-polygon & jarak segmen)
6. Penalti kanibalisasi pada penempatan multi-toko (p=2)
"""

import os
import json
import pytest
import numpy as np

from map_model import MapModel, point_in_polygon_vectorized
from fitness import FitnessEvaluator
from ga import GeneticAlgorithm
from pso import ParticleSwarmOptimization
from stats_utils import compute_eval_to_95_percent, perform_mann_whitney_u_test


@pytest.fixture
def sample_map():
    """Fixture model peta sederhana untuk pengujian unit."""
    map_dict = {
        "name": "Peta Uji",
        "dimensions": {"width": 1000.0, "height": 800.0},
        "houses": [
            {"x": 200.0, "y": 200.0, "weight": 2.0},
            {"x": 220.0, "y": 210.0, "weight": 3.0},
            {"x": 800.0, "y": 700.0, "weight": 1.0},
        ],
        "facilities": [
            {"type": "sekolah", "x": 250.0, "y": 250.0, "weight": 1.5},
            {"type": "pasar", "x": 300.0, "y": 300.0, "weight": 2.0},
        ],
        "competitors": [
            {"x": 280.0, "y": 260.0, "name": "Kompetitor X"},
        ],
        "roads": [
            {
                "name": "Jl. Utama",
                "class": "arteri",
                "traffic": 0.8,
                "points": [[0.0, 250.0], [500.0, 250.0], [1000.0, 250.0]],
            }
        ],
        "forbidden_zones": [
            {
                "name": "Sungai Terlarang",
                "type": "sungai",
                "polygon": [[400.0, 0.0], [450.0, 0.0], [450.0, 800.0], [400.0, 800.0]],
            }
        ],
    }
    return MapModel(map_dict)


def test_geometry_point_in_polygon():
    """Menguji keakuratan ray casting point-in-polygon."""
    # Poligon persegi (100, 100) sampai (300, 300)
    polygon = np.array([[100.0, 100.0], [300.0, 100.0], [300.0, 300.0], [100.0, 300.0]])
    points = np.array([
        [200.0, 200.0],  # Di dalam
        [50.0, 50.0],    # Di luar
        [250.0, 150.0],  # Di dalam
        [350.0, 200.0],  # Di luar
    ])
    result = point_in_polygon_vectorized(points, polygon)
    assert result[0] == True
    assert result[1] == False
    assert result[2] == True
    assert result[3] == False


def test_evaluation_counter_and_heatmap(sample_map):
    """Menguji bahwa evaluasi candidate menambah counter, sedangkan prekomputasi heatmap tidak."""
    evaluator = FitnessEvaluator(sample_map, mode="max")
    assert evaluator.eval_count == 0

    # Evaluasi 1 kandidat
    evaluator.evaluate(np.array([200.0, 200.0]))
    assert evaluator.eval_count == 1

    # Evaluasi batch 5 kandidat
    batch = np.array([[100.0, 100.0] for _ in range(5)])
    evaluator.evaluate_batch(batch)
    assert evaluator.eval_count == 6

    # Prekomputasi heatmap grid tidak boleh menambah eval_count
    grid, extent = evaluator.compute_grid_heatmap(resolution_x=10, resolution_y=10)
    assert grid.shape == (10, 10)
    assert evaluator.eval_count == 6  # Tetap 6

    # Reset counter
    evaluator.reset_counter()
    assert evaluator.eval_count == 0


def test_forbidden_zone_and_oob_penalties(sample_map):
    """Menguji penalti besar untuk titik di zona terlarang dan luar peta."""
    evaluator = FitnessEvaluator(sample_map, mode="max")

    # Titik valid
    fit_valid = evaluator.evaluate(np.array([210.0, 210.0]))
    assert fit_valid > -100.0

    # Titik di dalam zona terlarang (sungai x: 400-450)
    fit_forbid = evaluator.evaluate(np.array([425.0, 400.0]))
    assert fit_forbid <= -evaluator.penalty_forbidden

    # Titik di luar peta (x < 0 atau x > 1000)
    fit_oob = evaluator.evaluate(np.array([-50.0, 300.0]))
    assert fit_oob <= -evaluator.penalty_out_of_bounds


def test_mode_min_valid(sample_map):
    """
    Menguji bahwa mode min_valid mencari lokasi valid terburuk,
    namun titik terlarang tetap terkena penalti besar.
    """
    eval_max = FitnessEvaluator(sample_map, mode="max")
    eval_min = FitnessEvaluator(sample_map, mode="min_valid")

    pt_ramai = np.array([210.0, 210.0])  # Dekat pemukiman & fasilitas di koridor jalan (skor tinggi)
    pt_sepi = np.array([900.0, 240.0])   # Jauh dari pemukiman, tetap valid di koridor jalan (skor rendah)
    pt_forbid = np.array([425.0, 400.0]) # Zona terlarang

    # Pada mode max, pt_ramai > pt_sepi
    assert eval_max.evaluate(pt_ramai) > eval_max.evaluate(pt_sepi)

    # Pada mode min_valid, pt_sepi memiliki fitness lebih tinggi daripada pt_ramai
    # (karena yang dicari adalah titik valid dengan skor terendah di koridor jalan)
    fit_min_sepi = eval_min.evaluate(pt_sepi)
    fit_min_ramai = eval_min.evaluate(pt_ramai)
    assert fit_min_sepi > fit_min_ramai

    # Titik di zona terlarang tetap harus mendapat penalti sangat buruk
    fit_min_forbid = eval_min.evaluate(pt_forbid)
    assert fit_min_forbid <= -eval_min.penalty_forbidden


def test_cannibalization_p2(sample_map):
    """Menguji bahwa 2 toko yang berimpit mendapat penalti kanibalisasi dibanding 2 toko terpisah."""
    evaluator = FitnessEvaluator(sample_map, num_stores=2, mode="max")

    # Toko 1 & 2 berimpit pada lokasi yang sama di koridor jalan
    sol_overlap = np.array([210.0, 210.0, 210.0, 210.0])
    # Toko 1 & 2 terpisah jarak aman (> 400 meter) tetap di koridor jalan
    sol_separated = np.array([210.0, 210.0, 800.0, 240.0])

    fit_overlap = evaluator.evaluate(sol_overlap)
    fit_separated = evaluator.evaluate(sol_separated)

    assert fit_separated > fit_overlap


def test_seed_reproducibility(sample_map):
    """Menguji bahwa seed yang sama menghasilkan fitness dan solusi yang persis sama."""
    evaluator1 = FitnessEvaluator(sample_map, mode="max")
    evaluator2 = FitnessEvaluator(sample_map, mode="max")

    ga1 = GeneticAlgorithm(evaluator1, pop_size=20, seed=42)
    sol1, fit1, _ = ga1.optimize(max_evaluations=100)

    ga2 = GeneticAlgorithm(evaluator2, pop_size=20, seed=42)
    sol2, fit2, _ = ga2.optimize(max_evaluations=100)

    np.testing.assert_allclose(sol1, sol2, rtol=1e-5)
    assert fit1 == pytest.approx(fit2)

    # Uji PSO reproduktif
    evaluator3 = FitnessEvaluator(sample_map, mode="max")
    evaluator4 = FitnessEvaluator(sample_map, mode="max")

    pso1 = ParticleSwarmOptimization(evaluator3, num_particles=20, seed=77)
    sol_pso1, fit_pso1, _ = pso1.optimize(max_evaluations=100)

    pso2 = ParticleSwarmOptimization(evaluator4, num_particles=20, seed=77)
    sol_pso2, fit_pso2, _ = pso2.optimize(max_evaluations=100)

    np.testing.assert_allclose(sol_pso1, sol_pso2, rtol=1e-5)
    assert fit_pso1 == pytest.approx(fit_pso2)


def test_matched_budget_limit(sample_map):
    """Menguji bahwa evaluasi berhenti tepat saat anggaran evaluasi tercapai."""
    budget = 150

    eval_ga = FitnessEvaluator(sample_map)
    ga = GeneticAlgorithm(eval_ga, pop_size=30, seed=123)
    ga.optimize(max_evaluations=budget)
    assert eval_ga.eval_count == budget

    eval_pso = FitnessEvaluator(sample_map)
    pso = ParticleSwarmOptimization(eval_pso, num_particles=30, seed=123)
    pso.optimize(max_evaluations=budget)
    assert eval_pso.eval_count == budget


def test_stats_utils():
    """Menguji kalkulasi evaluasi ke-95% dan Mann-Whitney U test."""
    evals = np.array([10, 20, 30, 40, 50, 60])
    fits = np.array([0.1, 0.2, 0.5, 0.85, 0.96, 1.0])
    # Target 95%: 0.1 + 0.95*(0.9) = 0.955 -> tercapai pada evaluasi ke-50 (fits=0.96)
    e95 = compute_eval_to_95_percent(evals, fits)
    assert e95 == 50

    # Mann-Whitney U test
    ga_fits = np.array([0.8, 0.82, 0.81, 0.79, 0.83])
    pso_fits = np.array([0.9, 0.92, 0.91, 0.89, 0.93])
    res = perform_mann_whitney_u_test(ga_fits, pso_fits)
    assert res["is_significant"] == True
    assert "PSO" in res["kesimpulan"]
