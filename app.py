# app.py
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

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
    data_rows = raw.iloc[header_row + 2:].reset_index(drop=True).astype(str).apply(lambda x: x.str.strip())

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
        st.error(f"Need >=10 samples. Got: Porosity={len(porosity)}, N/G={len(ntg)}")
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
        try:
            _, p_norm = stats.shapiro(samples)
            tests["Normal"] = p_norm
        except:
            tests["Normal"] = 0
        
        try:
            u = (samples - samples.min()) / (samples.max() - samples.min() + 1e-12)
            _, p_uni = stats.kstest(u, "uniform")
            tests["Uniform"] = p_uni
        except:
            tests["Uniform"] = 0
        
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
        st.markdown(f"**Porosity (ϕ):** `{phi_dist}` \n_{phi_reason}_")
    with col2:
        st.markdown(f"**Net-to-Gross (N/G):** `{ntg_dist}` \n_{ntg_reason}_")

    # ------------------------------------------------------------------ #
    # 3. Volumetric Formula (UI Only)
    # ------------------------------------------------------------------ #
    st.markdown("---")
    st.latex(r"\text{STOIIP} = \frac{\text{GRV} \times \text{N/G} \times \phi \times (1 - S_{wi})}{B_o}")

    # ------------------------------------------------------------------ #
    # 4. User Controls
    # ------------------------------------------------------------------ #
    st.markdown("---")
    st.subheader("Simulation Settings")
    col1, col2, col3 = st.columns(3)
    with col1:
        iterations = st.slider("Monte Carlo iterations", 10000, 100000, 20000, 1000)
    with col2:
        output_unit = st.radio("Output units", ["Stock Tank m³", "Barrels (STB)"], horizontal=True)
    with col3:
        plot_engine = st.radio("Plotting engine", ["Plotly (interactive)", "Matplotlib (static)"], horizontal=True)

    # GRV distribution
    st.markdown("**Gross Rock Volume (GRV) Distribution**")
    grv_dist = st.selectbox("Select GRV distribution", ["Uniform", "Triangular", "Normal"], index=0)

    # Decimal places
    decimals = st.slider("Decimal places on charts & results", 0, 6, 3)

    # ------------------------------------------------------------------ #
    # 5. Run Simulation
    # ------------------------------------------------------------------ #
    if st.button("Run Monte Carlo Simulation", type="primary", use_container_width=True):
        with st.spinner("Simulating..."):
            rng = np.random.default_rng()

            gmin = gross_min
            gmax = gross_max

            def draw(dist, data, size, rng_instance):
                if dist == "Normal":
                    mu, sigma = stats.norm.fit(data)
                    return rng_instance.normal(mu, sigma, size)
                if dist == "Uniform":
                    return rng_instance.uniform(data.min(), data.max(), size)
                if dist == "Triangular":
                    try:
                        c, loc, scale = stats.triang.fit(data)
                        return stats.triang.rvs(c, loc, scale, size=size, random_state=rng_instance)
                    except:
                        # Fallback to simple triangular
                        mode = (data.min() + data.max()) / 2
                        return rng_instance.triangular(data.min(), mode, data.max(), size)
                return rng_instance.choice(data, size=size, replace=True)

            phi = np.clip(draw(phi_dist, porosity, iterations, rng), 0.001, 0.99)
            ntg = np.clip(draw(ntg_dist, ntg, iterations, rng), 0.001, 1.0)

            if grv_dist == "Uniform":
                grv = rng.uniform(gmin, gmax, iterations)
            elif grv_dist == "Triangular":
                mode = (gmin + gmax) / 2
                grv = rng.triangular(gmin, mode, gmax, iterations)
            else:  # Normal
                mean = (gmin + gmax) / 2
                std = (gmax - gmin) / 6
                grv = np.clip(rng.normal(mean, std, iterations), gmin, gmax)

            net_vol = grv * ntg
            hc_vol = net_vol * phi * (1 - swi)
            stoiip_m3 = hc_vol / bo
            stoiip = stoiip_m3 * (6.289811 if output_unit.startswith("Barrels") else 1)
            unit = "STB" if "Barrels" in output_unit else "m³"

            p90 = np.percentile(stoiip, 10)
            p50 = np.percentile(stoiip, 50)
            p10 = np.percentile(stoiip, 90)

        st.success("Simulation Complete!")

        # ----------------------------------------------------------------
        # 6. Results
        # ----------------------------------------------------------------
        fmt = f"{{:.{decimals}e}}"
        col1, col2, col3 = st.columns(3)
        col1.metric("P10 (High)", f"{fmt.format(p10)} {unit}")
        col2.metric("P50 (Median)", f"{fmt.format(p50)} {unit}")
        col3.metric("P90 (Low)", f"{fmt.format(p90)} {unit}")

        # --------------------------------------------------------------
        # NEW: Show ONE random Monte-Carlo realisation
        # --------------------------------------------------------------
        st.markdown("---")
        st.subheader("Example of One Random Realisation")

        # Use a fixed seed for reproducibility of the example
        example_rng = np.random.default_rng(42)
        phi_ex = np.clip(draw(phi_dist, porosity, 1, example_rng), 0.001, 0.99)[0]
        ntg_ex = np.clip(draw(ntg_dist, ntg, 1, example_rng), 0.001, 1.0)[0]

        if grv_dist == "Uniform":
            grv_ex = example_rng.uniform(gmin, gmax)
        elif grv_dist == "Triangular":
            mode = (gmin + gmax) / 2
            grv_ex = example_rng.triangular(gmin, mode, gmax)
        else:   # Normal
            mean = (gmin + gmax) / 2
            std = (gmax - gmin) / 6
            grv_ex = np.clip(example_rng.normal(mean, std), gmin, gmax)

        net_vol_ex = grv_ex * ntg_ex
        hc_vol_ex  = net_vol_ex * phi_ex * (1 - swi)
        stoiip_m3_ex = hc_vol_ex / bo
        stoiip_ex = stoiip_m3_ex * (6.289811 if output_unit.startswith("Barrels") else 1)

        # LaTeX formula with numbers plugged in
        st.latex(
            rf"\text{{STOIIP}} = \dfrac{{{grv_ex:.2e}\;\times\;{ntg_ex:.4f}\;\times\;{phi_ex:.4f}"
            rf"\;\times\;({1-swi:.4f})}}{{{bo:.4f}}}"
            rf"\;=\;{stoiip_ex:.{decimals}e}\;{unit}"
        )

        # Step-by-step written calculation
        st.markdown(
            f"**Step-by-step calculation for this random draw:**  \n"
            f"- GRV = `{grv_ex:.2e}` m³  \n"
            f"- N/G = `{ntg_ex:.4f}`  \n"
            f"- ϕ   = `{phi_ex:.4f}`  \n"
            f"- (1 - S<sub>wi</sub>) = `{1-swi:.4f}`  \n"
            f"- B<sub>o</sub> = `{bo:.4f}`  \n\n"
            f"1. Net volume = GRV × N/G = {grv_ex:.2e} × {ntg_ex:.4f} = **{net_vol_ex:.2e}** m³  \n"
            f"2. HC volume  = Net × ϕ × (1-S<sub>wi</sub>) = {net_vol_ex:.2e} × {phi_ex:.4f} × {1-swi:.4f} = **{hc_vol_ex:.2e}** m³  \n"
            f"3. STOIIP (m³) = HC / B<sub>o</sub> = {hc_vol_ex:.2e} / {bo:.4f} = **{stoiip_m3_ex:.2e}** m³  \n"
            f"4. STOIIP ({unit}) = {stoiip_m3_ex:.2e} × {6.289811 if 'Barrels' in output_unit else 1:.6f} = **{stoiip_ex:.{decimals}e}** {unit}",
            unsafe_allow_html=True
        )

        # ----------------------------------------------------------------
        # 7. Plots: Histogram + Ascending CDF + Descending CDF
        # ----------------------------------------------------------------
        st.markdown("---")
        st.subheader("STOIIP Distribution Analysis")

        sorted_val = np.sort(stoiip)
        cdf_y = np.linspace(0, 1, len(sorted_val))
        exceedance = 1 - cdf_y

        if "Plotly" in plot_engine:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    "Histogram",
                    "Ascending CDF",
                    "Descending CDF (Exceedance)",
                    "Log-Scale Histogram"
                )
            )
            # Histogram
            fig.add_trace(go.Histogram(x=stoiip, nbinsx=80, name="STOIIP", marker_color="#4e79a7"), row=1, col=1)
            # Ascending CDF
            fig.add_trace(go.Scatter(x=sorted_val, y=cdf_y, mode="lines", line=dict(color="#f28e2b", width=3)), row=1, col=2)
            # Descending CDF
            fig.add_trace(go.Scatter(x=sorted_val, y=exceedance, mode="lines", line=dict(color="#59a14f", width=3), name="Exceedance"), row=2, col=1)
            # Log Histogram
            fig.add_trace(go.Histogram(x=stoiip, nbinsx=80, name="Log", marker_color="#e15759"), row=2, col=2)

            # P10/P50/P90 lines
            for val, label, color in [(p90, "P90", "#59a14f"), (p50, "P50", "#f28e2b"), (p10, "P10", "#e15759")]:
                fig.add_vline(x=val, line=dict(dash="dash", color=color), annotation_text=f"{label}: {fmt.format(val)}", row=1, col=2)
                fig.add_vline(x=val, line=dict(dash="dash", color=color), row=2, col=1)
                # Add to legend
                fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', 
                                        line=dict(color=color, width=2, dash='dash'),
                                        name=f"{label}: {fmt.format(val)} {unit}",
                                        showlegend=True), row=2, col=1)

            fig.update_layout(height=800, showlegend=False)
            fig.update_xaxes(tickformat=f".{decimals}e")
            fig.update_xaxes(type="log", row=2, col=2)
            st.plotly_chart(fig, use_container_width=True)
        else:  # Matplotlib
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))
            # Histogram
            axs[0,0].hist(stoiip, bins=80, color="#4e79a7", edgecolor="black")
            axs[0,0].set_title("Histogram")
            axs[0,0].set_xlabel(f"STOIIP ({unit})")
            axs[0,0].set_ylabel("Frequency")
            # Ascending CDF
            axs[0,1].plot(sorted_val, cdf_y, color="#f28e2b", lw=3)
            axs[0,1].set_title("Ascending CDF")
            axs[0,1].set_xlabel(f"STOIIP ({unit})")
            axs[0,1].set_ylabel("Cumulative Probability")
            # Descending CDF
            axs[1,0].plot(sorted_val, exceedance, color="#59a14f", lw=3)
            axs[1,0].set_title("Descending CDF (Exceedance)")
            axs[1,0].set_xlabel(f"STOIIP ({unit})")
            axs[1,0].set_ylabel("Exceedance Probability")
            # Log Histogram
            axs[1,1].hist(stoiip, bins=80, color="#e15759", edgecolor="black", log=True)
            axs[1,1].set_title("Log-Scale Histogram")
            axs[1,1].set_xlabel(f"STOIIP ({unit})")
            axs[1,1].set_xscale("log")

            # P10/P50/P90 markers
            for val, label, color in [(p90, "P90", "#59a14f"), (p50, "P50", "#f28e2b"), (p10, "P10", "#e15759")]:
                val_str = f"{val:.{decimals}e}"
                axs[0,1].axvline(val, color=color, linestyle="--", linewidth=1.5)
                axs[0,1].text(val, 0.9, f"{label}: {val_str}", rotation=90, va='top', ha='right', fontsize=9, color=color)
                axs[1,0].axvline(val, color=color, linestyle="--", linewidth=1.5)

            plt.tight_layout()
            st.pyplot(fig)

        # ----------------------------------------------------------------
        # 8. Download
        # ----------------------------------------------------------------
        results_df = pd.DataFrame({
            f"STOIIP ({unit})": np.round(stoiip, decimals),
            "Porosity": np.round(phi, 4),
            "N/G": np.round(ntg, 4),
            "GRV (m³)": grv.astype(int)
        })
        st.download_button("Download Results CSV", results_df.to_csv(index=False), "monte_carlo_results.csv")
        st.download_button("Download Summary", f"""
P10 (High): {fmt.format(p10)} {unit}
P50 (Median): {fmt.format(p50)} {unit}
P90 (Low): {fmt.format(p90)} {unit}
ϕ ~ {phi_dist} | N/G ~ {ntg_dist} | GRV ~ {grv_dist}
Iterations: {iterations:,}
""", "summary.txt")

else:
    st.info("Upload your reservoir CSV/Excel file to begin.")
