# app.py
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="OOIP/STOIIP Monte Carlo", layout="wide")
st.title("OOIP/STOIIP Monte Carlo Estimator")
st.markdown("**Volumetric Hydrocarbon-in-Place Estimation Under Uncertainty**")

# ------------------------------------------------------------------ #
# 1. Upload & parsing
# ------------------------------------------------------------------ #
uploaded = st.file_uploader("Upload reservoir data file (CSV or Excel)", type=["csv", "xlsx", "xls"])

if uploaded:
    if uploaded.name.endswith(".csv"):
        raw = pd.read_csv(uploaded, header=None, dtype=str, na_filter=False)
    else:
        raw = pd.read_excel(uploaded, header=None, dtype=str, na_filter=False)

    header_row = None
    for i in range(len(raw)):
        if "porosity" in raw.iloc[i].astype(str).str.lower().values:
            header_row = i
            break
    if header_row is None:
        st.error("Could not find header row with 'Porosity'.")
        st.stop()

    header = raw.iloc[header_row].astype(str).str.strip()
    meta_row = raw.iloc[header_row + 1].astype(str).str.strip()
    data_rows = raw.iloc[header_row + 2:].reset_index(drop=True).astype(str).applymap(str.strip)

    col_map = {}
    header_low = header.str.lower()
    for idx, name in enumerate(header_low):
        if "porosity" in name:
            col_map["Porosity"] = idx
        elif "permeability" in name:
            col_map["Permeability_md"] = idx
        elif "net" in name and "gross" in name:
            col_map["NetToGross"] = idx
        elif "gross volume" in name or "gross vol" in name or "gross volume m3" in name:
            col_map["Gross_min"] = idx
            col_map["Gross_max"] = idx + 1
        elif "swi" in name or "water" in name:
            col_map["Swi"] = idx
        elif "formation" in name or "fvf" in name or "bo" in name or "m3/sm3" in name:
            col_map["Bo"] = idx

    required = ["Porosity", "NetToGross", "Gross_min", "Gross_max", "Swi", "Bo"]
    missing = [k for k in required if k not in col_map]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        st.stop()

    def safe_float(s):
        try:
            return float(str(s).replace(",", "").replace("\n", "").replace("\r", "").strip())
        except:
            return np.nan

    porosity = pd.to_numeric(data_rows.iloc[:, col_map["Porosity"]].apply(safe_float), errors='coerce').dropna().values
    ntg = pd.to_numeric(data_rows.iloc[:, col_map["NetToGross"]].apply(safe_float), errors='coerce').dropna().values
    gross_min = safe_float(data_rows.iloc[0, col_map["Gross_min"]])
    gross_max = safe_float(data_rows.iloc[0, col_map["Gross_max"]])
    swi = safe_float(meta_row.iloc[col_map["Swi"]])
    bo = safe_float(meta_row.iloc[col_map["Bo"]])

    if any(pd.isna(x) for x in [gross_min, gross_max, swi, bo]):
        st.error("Failed to parse **Gross Volume**, **Swi**, or **Bo**.")
        st.stop()

    if len(porosity) < 10 or len(ntg) < 10:
        st.error(f"Need ≥10 samples. Got: Porosity={len(porosity)}, N/G={len(ntg)}")
        st.stop()

    porosity = np.clip(porosity, 0, 1)
    ntg = np.clip(ntg, 0, 1)

    st.success(f"File parsed — {len(porosity)} valid samples")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Porosity", f"{porosity.mean():.3f} ± {porosity.std():.3f}")
    with col2:
        st.metric("N/G", f"{ntg.mean():.3f} ± {ntg.std():.3f}")
    with col3:
        st.metric("GRV (m³)", f"{gross_min:.2e} – {gross_max:.2e}")
        st.write(f"Swi: {swi:.3f} | Bo: {bo:.3f}")

    # ------------------------------------------------------------------ #
    # 2. Auto-detect distributions for Porosity & N/G
    # ------------------------------------------------------------------ #
    st.markdown("---")
    st.subheader("Detected Input Distributions")

    def detect_distribution(samples, name):
        if len(samples) < 15:
            return "Triangular", "Too few samples → using Triangular"
        
        tests = {}
        # Normal
        _, p_norm = stats.shapiro(samples)
        tests["Normal"] = p_norm
        # Uniform
        u = (samples - samples.min()) / (samples.max() - samples.min() + 1e-12)
        _, p_uni = stats.kstest(u, "uniform")
        tests["Uniform"] = p_uni
        # Triangular
        try:
            c, loc, scale = stats.triang.fit(samples)
            _, p_tri = stats.kstest(samples, stats.triang.cdf, args=(c, loc, scale))
            tests["Triangular"] = p_tri
        except:
            tests["Triangular"] = 0

        best = max(tests, key=tests.get)
        reason = f"Best p-value: {tests[best]:.4f}"
        if best == "Triangular" and tests[best] < 0.05:
            reason += " (fallback)"
        return best, reason

    phi_dist, phi_reason = detect_distribution(porosity, "Porosity")
    ntg_dist, ntg_reason = detect_distribution(ntg, "NetToGross")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Porosity (ϕ):** `{phi_dist}`  \n_{phi_reason}_")
    with col2:
        st.markdown(f"**Net-to-Gross (N/G):** `{ntg_dist}`  \n_{ntg_reason}_")

    # ------------------------------------------------------------------ #
    # 3. User Controls
    # ------------------------------------------------------------------ #
    st.markdown("---")
    st.subheader("Simulation Settings")

    col1, col2, col3 = st.columns(3)
    with col1:
        iterations = st.slider("Monte Carlo iterations", 1000, 100_000, 20000, 1000)
    with col2:
        output_unit = st.radio("Output units", ["Stock Tank m³", "Barrels (STB)"], horizontal=True)
    with col3:
        plot_engine = st.radio("Plotting engine", ["Plotly (interactive)", "Matplotlib (static)"], horizontal=True)

    # GRV settings
    st.markdown("**Gross Rock Volume (GRV) Settings**")
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        grv_dist = st.selectbox("GRV Distribution", ["Uniform", "Triangular", "Normal", "PERT"])
    with col_g2:
        grv_perturb = st.slider("Perturb min/max GRV", 0, 20, 5, help="±% random variation in min/max")

    # ------------------------------------------------------------------ #
    # 4. Run Simulation
    # ------------------------------------------------------------------ #
    if st.button("Run Monte Carlo Simulation", type="primary", use_container_width=True):
        with st.spinner("Simulating..."):
            rng = np.random.default_rng()

            # Perturb GRV bounds
            if grv_perturb > 0:
                factor = 1 + rng.normal(0, grv_perturb/100, size=iterations)
                gmin = gross_min * np.clip(factor, 0.8, 1.2)
                gmax = gross_max * np.clip(factor, 0.8, 1.2)
            else:
                gmin = np.full(iterations, gross_min)
                gmax = np.full(iterations, gross_max)

            # Draw ϕ and N/G
            def draw(dist, data, size):
                if dist == "Normal":
                    mu, sigma = stats.norm.fit(data)
                    return rng.normal(mu, sigma, size)
                if dist == "Uniform":
                    return rng.uniform(data.min(), data.max(), size)
                if dist == "Triangular":
                    c, loc, scale = stats.triang.fit(data)
                    return stats.triang.rvs(c, loc, scale, size=size, random_state=rng)
                return rng.choice(data, size=size, replace=True)

            phi = np.clip(draw(phi_dist, porosity, iterations), 0.001, 0.99)
            ntg = np.clip(draw(ntg_dist, ntg, iterations), 0.001, 1.0)

            # GRV
            if grv_dist == "Uniform":
                grv = rng.uniform(gmin, gmax, iterations)
            elif grv_dist == "Triangular":
                mode = (gmin + gmax) / 2
                grv = rng.triangular(gmin, mode, gmax, iterations)
            elif grv_dist == "PERT":
                grv = stats.pert.rvs(4, (gmin + gmax)/2, gmax, gmin, size=iterations, random_state=rng)
            else:  # Normal
                mean = (gmin + gmax)/2
                std = (gmax - gmin)/6
                grv = np.clip(rng.normal(mean, std, iterations), gmin, gmax)

            # Volumetric
            net_vol = grv * ntg
            hc_vol = net_vol * phi * (1 - swi)
            stoiip_m3 = hc_vol / bo
            stoiip = stoiip_m3 * (6.289811 if output_unit.startswith("Barrels") else 1)
            unit = "STB" if "Barrels" in output_unit else "m³"

            # Percentiles
            p90 = np.percentile(stoiip, 10)
            p50 = np.percentile(stoiip, 50)
            p10 = np.percentile(stoiip, 90)

        st.success("Simulation Complete!")

        # ----------------------------------------------------------------
        # 5. Results
        # ----------------------------------------------------------------
        decimals = 3
        fmt = f"{{:.{decimals}e}}"
        col1, col2, col3 = st.columns(3)
        col1.metric("P10 (High)", f"{fmt.format(p10)} {unit}")
        col2.metric("P50 (Median)", f"{fmt.format(p50)} {unit}")
        col3.metric("P90 (Low)", f"{fmt.format(p90)} {unit}")

        # ----------------------------------------------------------------
        # 6. Plotting (User Choice)
        # ----------------------------------------------------------------
        st.markdown("---")
        st.subheader("Probability Distributions")

        sorted_val = np.sort(stoiip)
        cdf_y = np.linspace(0, 1, len(sorted_val))

        if "Plotly" in plot_engine:
            fig = make_subplots(rows=2, cols=2,
                subplot_titles=("Histogram", "Ascending CDF", "Descending CDF", "Log Histogram"))
            fig.add_trace(go.Histogram(x=stoiip, nbinsx=80), row=1, col=1)
            fig.add_trace(go.Scatter(x=sorted_val, y=cdf_y, mode="lines", line=dict(color="blue")), row=1, col=2)
            fig.add_trace(go.Scatter(x=sorted_val, y=1-cdf_y, mode="lines", line=dict(color="green")), row=2, col=1)
            fig.add_trace(go.Histogram(x=stoiip, nbinsx=80), row=2, col=2)
            fig.update_xaxes(type="log", row=2, col=2)
            for val, label, colr in [(p90, "P90", "green"), (p50, "P50", "blue"), (p10, "P10", "red")]:
                fig.add_vline(x=val, line=dict(dash="dash", color=colr), annotation_text=label, row=1, col=2)
                fig.add_vline(x=val, line=dict(dash="dash", color=colr), annotation_text=label, row=2, col=1)
            fig.update_layout(height=800, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        else:  # Matplotlib
            sns.set_style("whitegrid")
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))
            axs[0,0].hist(stoiip, bins=80, color='skyblue', edgecolor='black')
            axs[0,0].set_title("Histogram")
            axs[0,1].plot(sorted_val, cdf_y, color='blue', lw=2)
            axs[0,1].set_title("Ascending CDF")
            axs[1,0].plot(sorted_val, 1-cdf_y, color='green', lw=2)
            axs[1,0].set_title("Descending CDF")
            axs[1,1].hist(stoiip, bins=80, color='lightcoral', edgecolor='black', log=True)
            axs[1,1].set_title("Log Histogram")
            for val, label, colr in [(p90, "P90", "green"), (p50, "P50", "blue"), (p10, "P10", "red")]:
                axs[0,1].axvline(val, color=colr, linestyle="--", label=label)
                axs[1,0].axvline(val, color=colr, linestyle="--")
            plt.tight_layout()
            st.pyplot(fig)

        # ----------------------------------------------------------------
        # 7. Download
        # ----------------------------------------------------------------
        results_df = pd.DataFrame({
            f"STOIIP ({unit})": np.round(stoiip, 3),
            "Porosity": np.round(phi, 4),
            "N/G": np.round(ntg, 4),
            "GRV (m³)": grv.astype(int)
        })
        st.download_button("Download Results CSV", results_df.to_csv(index=False), "monte_carlo_results.csv")
        st.download_button("Download Summary", f"""
P10: {fmt.format(p10)} {unit}
P50: {fmt.format(p50)} {unit}
P90: {fmt.format(p90)} {unit}
ϕ ~ {phi_dist} | N/G ~ {ntg_dist} | GRV ~ {grv_dist}
""", "summary.txt")

else:
    st.info("Upload your reservoir CSV/Excel file to begin.")
