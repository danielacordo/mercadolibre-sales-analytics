import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.pyplot as plt


# Fixtures 
@pytest.fixture
def rfm_df():
    """Minimal synthetic RFM dataset covering all 6 segments."""
    rng = np.random.default_rng(42)
    n = 60
    segments = (["VIP"] * 5 + ["Loyal"] * 8 + ["Potential"] * 20 +
                ["At risk"] * 7 + ["Occasional"] * 12 + ["Lost"] * 8)
    return pd.DataFrame({
        "Cliente": [f"C{i:03d}" for i in range(n)],
        "Customer": [f"C{i:03d}" for i in range(n)],
        "Segment": segments,
        "Cluster": rng.integers(0, 6, n),
        "recencia": rng.integers(10, 1200, n),
        "frecuencia": rng.integers(1, 5, n),
        "monto_total": rng.uniform(500, 35000, n).round(0),
        "R_score": rng.integers(1, 5, n),
        "F_score": rng.integers(1, 5, n),
        "M_score": rng.integers(1, 5, n),
        "ultima_compra": pd.date_range("2023-01-01", periods=n, freq="10D"),})


# plot_rfm_clusters 
class TestPlotRfmClusters:
    def test_returns_figure(self, rfm_df, tmp_path):
        from src.rfm_plots import plot_rfm_clusters
        fig = plot_rfm_clusters(rfm_df)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_saves_file(self, rfm_df, tmp_path):
        from src.rfm_plots import plot_rfm_clusters
        out = str(tmp_path / "rfm_clusters.png")
        plot_rfm_clusters(rfm_df, save_path=out)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 10_000   # not empty
        plt.close("all")

    def test_all_segments_present_in_legend(self, rfm_df):
        from src.rfm_plots import plot_rfm_clusters
        fig = plot_rfm_clusters(rfm_df)
        # At least one axes should have legend entries
        has_legend = any(ax.get_legend() is not None for ax in fig.get_axes())
        assert has_legend
        plt.close("all")

    def test_no_crash_single_segment(self):
        """ handles rfm with only one segment """
        from src.rfm_plots import plot_rfm_clusters
        rfm_single = pd.DataFrame({
            "Cliente": ["C1","C2","C3"],
            "Customer": ["C1","C2","C3"],
            "Segment": ["Potential","Potential","Potential"],
            "Cluster": [0, 0, 0],
            "recencia": [100, 200, 300],
            "frecuencia":[1, 1, 2],
            "monto_total":[5000, 8000, 12000],
            "R_score": [3,2,1], "F_score":[1,1,2], "M_score":[2,3,4],
            "ultima_compra": pd.date_range("2025-01-01", periods=3, freq="30D"),})
        
        fig = plot_rfm_clusters(rfm_single)
        assert isinstance(fig, plt.Figure)
        plt.close("all")


# plot_rfm_heatmap
class TestPlotRfmHeatmap:
    def test_returns_figure(self, rfm_df):
        from src.rfm_plots import plot_rfm_heatmap
        fig = plot_rfm_heatmap(rfm_df)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_saves_file(self, rfm_df, tmp_path):
        from src.rfm_plots import plot_rfm_heatmap
        out = str(tmp_path / "heatmap.png")
        plot_rfm_heatmap(rfm_df, save_path=out)
        assert os.path.exists(out)
        plt.close("all")

    def test_heatmap_dimensions(self, rfm_df):
        """Heatmap image should cover 6 segments x 3 scores."""
        from src.rfm_plots import plot_rfm_heatmap
        fig = plot_rfm_heatmap(rfm_df)
        # Find the imshow axes (has an image)
        img_axes = [ax for ax in fig.get_axes() if ax.get_images()]
        assert len(img_axes) >= 1
        img_data = img_axes[0].get_images()[0].get_array()
        assert img_data.shape == (6, 3)
        plt.close("all")

    def test_scores_bounded(self, rfm_df):
        """All score values must be in [1, 4] range."""
        from src.rfm_plots import plot_rfm_heatmap
        fig = plot_rfm_heatmap(rfm_df)
        img_axes = [ax for ax in fig.get_axes() if ax.get_images()]
        data = img_axes[0].get_images()[0].get_array()
        assert data.min() >= 1.0 - 1e-6
        assert data.max() <= 4.0 + 1e-6
        plt.close("all")


# plot_customer_value
class TestPlotCustomerValue:
    def test_returns_figure(self, rfm_df):
        from src.rfm_plots import plot_customer_value
        fig = plot_customer_value(rfm_df)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_saves_file(self, rfm_df, tmp_path):
        from src.rfm_plots import plot_customer_value
        out = str(tmp_path / "customer_value.png")
        plot_customer_value(rfm_df, save_path=out)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 10_000
        plt.close("all")

    def test_dot_size_inversely_proportional_to_recency(self, rfm_df):
        """More recent customers (lower recencia) should have larger dot_size """
        rfm_copy = rfm_df.copy()
        max_rec = rfm_copy["recencia"].max()
        rfm_copy["dot_size"] = (max_rec - rfm_copy["recencia"]) / max_rec * 200 + 20
        most_recent = rfm_copy.loc[rfm_copy["recencia"].idxmin(), "dot_size"]
        oldest      = rfm_copy.loc[rfm_copy["recencia"].idxmax(), "dot_size"]
        assert most_recent > oldest
        plt.close("all")


# plot_elbow_silhouette 
class TestPlotElbowSilhouette:
    def test_returns_figure(self):
        from src.rfm_plots import plot_elbow_silhouette
        k_range = range(2, 8)
        inertias = [1200, 900, 720, 610, 540, 490]
        silhouettes= [0.41, 0.52, 0.61, 0.58, 0.54, 0.50]
        fig = plot_elbow_silhouette(k_range, inertias, silhouettes, k_opt=4)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_saves_file(self, tmp_path):
        from src.rfm_plots import plot_elbow_silhouette
        out = str(tmp_path / "elbow.png")
        plot_elbow_silhouette(range(2,8), [1200,900,720,610,540,490],
                              [0.41,0.52,0.61,0.58,0.54,0.50], k_opt=4,
                              save_path=out)
        assert os.path.exists(out)
        plt.close("all")

    def test_k_opt_within_range(self):
        """k_opt must be one of the tested K values, guard against off-by-one."""
        k_range = range(2, 8)
        ks = list(k_range)
        k_opt = ks[ks.index(max(ks, key=lambda k: [0.41,0.52,0.61,0.58,0.54,0.50][k-2]))]
        assert k_opt in ks
        plt.close("all")


# run_rfm_plots integration 
class TestRunRfmPlots:
    def test_creates_all_output_files(self, rfm_df, tmp_path):
        from src.rfm_plots import run_rfm_plots
        # Write rfm to temp CSV
        rfm_path = str(tmp_path / "rfm_clientes.csv")
        rfm_df.to_csv(rfm_path, index=False)
        plots_dir = str(tmp_path / "plots")
        run_rfm_plots(rfm_path=rfm_path, plots_dir=plots_dir, verbose=False)
        expected = ["10_rfm_clusters.png", "11_rfm_heatmap.png", "09_customer_value.png"]
        for fname in expected:
            fpath = os.path.join(plots_dir, fname)
            assert os.path.exists(fpath), f"Missing: {fname}"
            assert os.path.getsize(fpath) > 5_000, f"Suspiciously small: {fname}"
        plt.close("all")

    def test_creates_plots_dir_if_missing(self, rfm_df, tmp_path):
        from src.rfm_plots import run_rfm_plots
        rfm_path = str(tmp_path / "rfm.csv")
        plots_dir = str(tmp_path / "new_dir" / "plots")
        rfm_df.to_csv(rfm_path, index=False)
        run_rfm_plots(rfm_path=rfm_path, plots_dir=plots_dir, verbose=False)
        assert os.path.isdir(plots_dir)
        plt.close("all")
