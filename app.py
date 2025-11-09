import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from io import BytesIO
st.set_page_config(page_title="OOIP/STOIIP Monte Carlo", layout="wide")
st.title("OOIP/STOIIP Monte Carlo Estimator")
st.markdown("**Volumetric Hydrocarbon-in-Place Estimation Under Uncertainty**")
# ------------------------------------------------------------------ #
# NEW: Mode Selector
# ------------------------------------------------------------------ #
mode = st.selectbox("Select Mode", ["Full Monte Carlo", "Reverse CDF Only"], index=0)
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
    # Detect OOIP column
    ooip_col_idx = None
    header_low = header.str.lower()
    for idx, name in enumerate(header_low):
        if "ooip" in name:
            ooip_col_idx = idx
            break
    # ------------------------------------------------------------------ #
    # MODE: Reverse CDF Only
    # ------------------------------------------------------------------ #
    if mode == "Reverse CDF Only":
        if ooip_col_idx is None:
            st.error("No column named 'OOIP' found. This mode requires an OOIP column.")
            st.stop()
        def safe_float(s):
            try:
                return float(str(s).replace(",", "").replace("\n", "").replace("\r", "").strip())
            except:
                return np.nan
        ooip_values = pd.to_numeric(data_rows.iloc[:, ooip_col_idx].apply(safe_float), errors='coerce').dropna().values
        if len(ooip_values) == 0:
            st.error("No valid numeric OOIP values found.")
            st.stop()
        st.success(f"Loaded {len(ooip_values)} OOIP values from column '{header.iloc[ooip_col_idx]}'")
        # Sort and compute exceedance
        sorted_val = np.sort(ooip_values)
        exceedance = 1 - np.linspace(0, 1, len(sorted_val))
        # Determine unit from filename or default
        unit = "STB" if "stb" in uploaded.name.lower() else "m³"
        # ------------------------------------------------------------------ #
        # Ask for decimal places BEFORE plotting
        # ------------------------------------------------------------------ #
        st.markdown("---")
        decimals = st.slider("Decimal places on chart labels & results", 0, 10, 3, key="reverse_cdf_decimals")
        # P10, P50, P90
        p10 = np.percentile(ooip_values, 10)
        p50 = np.percentile(ooip_values, 50)
        p90 = np.percentile(ooip_values, 90)
        fmt = f"{{:.{decimals}e}}"
        # ------------------------------------------------------------------ #
        # Matplotlib Descending CDF (Exceedance) Plot - MODIFIED
        # ------------------------------------------------------------------ #
        st.subheader("Descending CDF (Exceedance) - OOIP Only")
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        ax.plot(sorted_val, exceedance, color="#59a14f", lw=3)
        ax.set_title("Descending CDF")
        ax.set_xlabel(f"OOIP ({unit})")
        ax.set_ylabel("Exceedance Probability")
        
        # Reverse X-axis: increasing from right to left
        ax.invert_xaxis()

        # Add P10, P50, P90 lines and labels
        for val, label, color in [(p10, "P10", "#59a14f"), (p50, "P50", "#f28e2b"), (p90, "P90", "#e15759")]:
            val_str = fmt.format(val)
            ax.axvline(val, color=color, linestyle="--", linewidth=1.5)
            ax.text(val, 0.9, f"{label}: {val_str}", rotation=90, va='top', ha='right', fontsize=9, color=color)

        # Move the scientific notation key (legend for 1eX) to bottom-left
        ax.xaxis.get_offset_text().set_horizontalalignment('left')
        ax.xaxis.get_offset_text().set_x(0.01)  # Position near left
        ax.xaxis.get_offset_text().set_y(-0.15)  # Below axis

        st.pyplot(fig)
        # Download PNG
        def fig_to_png(f):
            buf = BytesIO()
            f.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            return buf
        buf = fig_to_png(fig)
        st.download_button("Download Descending CDF", buf, "descending_cdf_ooip.png", "image/png")
        plt.close(fig)
        # Download OOIP values
        results_df = pd.DataFrame({"OOIP": ooip_values})
        st.download_button("Download OOIP Values", results_df.to_csv(index=False), "ooip_values.csv")
        st.stop()
    # ------------------------------------------------------------------ #
    # MODE: Full Monte Carlo (Original Logic)
    # ------------------------------------------------------------------ #
    col_map = {}
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
    # 2. Auto-detect distributions
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
    # 3. Volumetric Formula
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
        plot_engine = st.radio("Plotting engine", ["Plotly (interactive)", "Matplotlib (static)"], index=1, horizontal=True)
    st.markdown("**Gross Rock Volume (GRV) Distribution**")
    grv_dist = st.selectbox("Select GRV distribution", ["Uniform", "Triangular", "Normal"], index=0)
    decimals = st.slider("Decimal places on charts & results", 0, 10, 3) # Now 0–10
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
            else:
                mean = (gmin + gmax) / 2
                std = (gmax - gmin) / 6
                grv = np.clip(rng.normal(mean, std, iterations), gmin, gmax)
            net_vol = grv * ntg
            hc_vol = net_vol * phi * (1 - swi)
            stoiip_m3 = hc_vol / bo
            stoiip = stoiip_m3 * (6.289811 if output_unit.startswith("Barrels") else 1)
            unit = "STB" if "Barrels" in output_unit else "m³"
            p10 = np.percentile(stoiip, 10)
            p50 = np.percentile(stoiip, 50)
            p90 = np.percentile(stoiip, 90)
        st.success("Simulation Complete!")
        # ----------------------------------------------------------------
        # 6. Results
        # ----------------------------------------------------------------
        fmt = f"{{:.{decimals}e}}"
        col1, col2, col3 = st.columns(3)
        col1.metric("P10 (Low)", f"{fmt.format(p10)} {unit}")
        col2.metric("P50 (Median)", f"{fmt.format(p50)} {unit}")
        col3.metric("P90 (High)", f"{fmt.format(p90)} {unit}")
        # --------------------------------------------------------------
        # Example Realisation
        # --------------------------------------------------------------
        st.markdown("---")
        st.subheader("Example of One Random Realisation")
        example_rng = np.random.default_rng(42)
        phi_ex = np.clip(draw(phi_dist, porosity, 1, example_rng), 0.001, 0.99)[0]
        ntg_ex = np.clip(draw(ntg_dist, ntg, 1, example_rng), 0.001, 1.0)[0]
        if grv_dist == "Uniform":
            grv_ex = example_rng.uniform(gmin, gmax)
        elif grv_dist == "Triangular":
            mode = (gmin + gmax) / 2
            grv_ex = example_rng.triangular(gmin, mode, gmax)
        else:
            mean = (gmin + gmax) / 2
            std = (gmax - gmin) / 6
            grv_ex = np.clip(example_rng.normal(mean, std), gmin, gmax)
        net_vol_ex = grv_ex * ntg_ex
        hc_vol_ex = net_vol_ex * phi_ex * (1 - swi)
        stoiip_m3_ex = hc_vol_ex / bo
        stoiip_ex = stoiip_m3_ex * (6.289811 if output_unit.startswith("Barrels") else 1)
        st.latex(
            rf"\text{{STOIIP}} = \dfrac{{{grv_ex:.2e}\;\times\;{ntg_ex:.4f}\;\times\;{phi_ex:.4f}"
            rf"\;\times\;({1-swi:.4f})}}{{{bo:.4f}}}"
            rf"\;=\;{stoiip_ex:.{decimals}e}\;{unit}"
        )
        st.markdown(
            f"**Step-by-step calculation:**\n"
            f"- GRV = `{grv_ex:.2e}` m³\n"
            f"- N/G = `{ntg_ex:.4f}`\n"
            f"- ϕ = `{phi_ex:.4f}`\n"
            f"- (1 - S<sub>wi</sub>) = `{1-swi:.4f}`\n"
            f"- B<sub>o</sub> = `{bo:.4f}`\n\n"
            f{f"1. Net volume = {grv_ex:.2e} × {ntg_ex:.4f} = **{net_vol_ex:.2e}** m³\n"
            f"2. HC volume = {net_vol_ex:.2e} × {phi_ex:.4f} × {1-swi:.4f} = **{hc_vol_ex:.2e}** m³\n"
            f"3. STOIIP (m³) = {hc_vol_ex:.2e} / {bo:.4f} = **{stoiip_m3_ex:.2e}** m³\n"
            f"4. STOIIP ({unit}) = {stoiip_m3_ex:.2e} × {6.289811 if 'Barrels' in output_unit else 1:.6f} = **{stoiip_ex:.{decimals}e}** {unit}",
            unsafe_allow_html=True
        )
        # ----------------------------------------------------------------
        # 7. Plots
        # ----------------------------------------------------------------
        st.markdown("---")
        st.subheader("STOIIP Distribution Analysis")
        sorted_val = np.sort(stoiip)
        cdf_y = np.linspace(0, 1, len(sorted_val))
        exceedance = 1 - cdf_y
        def fig_to_png(fig):
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            return buf
        if "Plotly" in plot_engine:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    "Histogram",
                    "Ascending CDF",
                    "Descending CDF (Exceedance)",
                    "Log-Scale Histogram"
                ),
                vertical_spacing=0.15
            )
            fig.add_trace(go.Histogram(x=stoiip, nbinsx=80, name="STOIIP", marker_color="#4e79a7", showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=sorted_val, y=cdf_y, mode="lines", line=dict(color="#f28e2b", width=3), showlegend=False), row=1, col=2)
            fig.add_trace(go.Scatter(x=sorted_val, y=exceedance, mode="lines", line=dict(color="#59a14f", width=3), name="Exceedance", showlegend=False), row=2, col=1)
            fig.add_trace(go.Histogram(x=stoiip, nbinsx=80, name="Log", marker_color="#e15759", showlegend=False), row=2, col=2)
            for val, label, color in [(p10, "P10", "#59a14f"), (p50, "P50", "#f28e2b"), (p90, "P90", "#e15759")]:
                val_str = fmt.format(val)
                fig.add_vline(x=val, line=dict(dash="dash", color=color),
                              annotation_text=f"{label}: {val_str}", row=1, col=2)
                fig.add_vline(x=val, line=dict(dash="dash", color=color),
                              annotation_text=f"{label}: {val_str}",
                              annotation_position="top", row=2, col=1)
                fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                                        line=dict(dash='dash', color=color),
                                        name=f"{label}: {val_str} {unit}", showlegend=True))
            fig.update_layout(height=900,
                              legend=dict(orientation="h", yanchor="top", y=-0.05, xanchor="left", x=0.0,
                                          bgcolor="rgba(255,255,255,0.8)", bordercolor="black", borderwidth=1))
            fig.update_xaxes(tickformat=f".{decimals}e")
            fig.update_xaxes(type="log", row=2, col=2)
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Matplotlib plots
            fig1, ax1 = plt.subplots(figsize=(8, 6), dpi=300)
            ax1.hist(stoiip, bins=80, color="#4e79a7", edgecolor="black")
            ax1.set_title("Histogram")
            ax1.set_xlabel(f"STOIIP ({unit})")
            ax1.set_ylabel("Frequency")
            st.pyplot(fig1)
            buf1 = fig_to_png(fig1)
            st.download_button("Download Histogram", buf1, "histogram.png", "image/png")
            fig2, ax2 = plt.subplots(figsize=(8, 6), dpi=300)
            ax2.plot(sorted_val, cdf_y, color="#f28e2b", lw=3)
            ax2.set_title("Ascending CDF")
            ax2.set_xlabel(f"STOIIP ({unit})")
            ax2.set_ylabel("Cumulative Probability")
            for val, label, color in [(p10, "P10", "#59a14f"), (p50, "P50", "#f28e2b"), (p90, "P90", "#e15759")]:
                val_str = f"{val:.{decimals}e}"
                ax2.axvline(val, color=color, linestyle="--", linewidth=1.5)
                ax2.text(val, 0.9, f"{label}: {val_str}", rotation=90, va='top', ha='right', fontsize=9, color=color)
            st.pyplot(fig2)
            buf2 = fig_to_png(fig2)
            st.download_button("Download Ascending CDF", buf2, "ascending_cdf.png", "image/png")
            fig3, ax3 = plt.subplots(figsize=(8, 6), dpi=300)
            ax3.plot(sorted_val, exceedance, color="#59a14f", lw=3)
            ax3.set_title("Descending CDF")
            ax3.set_xlabel(f"STOIIP ({unit})")
            ax3.set_ylabel("Exceedance Probability")
            for val, label, color in [(p10, "P10", "#59a14f"), (p50, "P50", "#f28e2b"), (p90, "P90", "#e15759")]:
                val_str = f"{val:.{decimals}e}"
                ax3.axvline(val, color=color, linestyle="--", linewidth=1.5)
                ax3.text(val, 0.9, f"{label}: {val_str}", rotation=90, va='top', ha='right', fontsize=9, color=color)
            st.pyplot(fig3)
            buf3 = fig_to_png(fig3)
            st.download_button("Download Descending CDF", buf3, "descending_cdf.png", "image/png")
            fig4, ax4 = plt.subplots(figsize=(8, 6), dpi=300)
            ax4.hist(stoiip, bins=80, color="#e15759", edgecolor="black", log=True)
            ax4.set_title("Log-Scale Histogram")
            ax4.set_xlabel(f"STOIIP ({unit})")
            ax4.set_xscale("log")
            st.pyplot(fig4)
            buf4 = fig_to_png(fig4)
            st.download_button("Download Log Histogram", buf4, "log_histogram.png", "image/png")
            plt.close('all')
        # ----------------------------------------------------------------
        # 8. Download Results
        # ----------------------------------------------------------------
        results_df = pd.DataFrame({
            f"STOIIP ({unit})": np.round(stoiip, decimals),
            "Porosity": np.round(phi, 4),
            "N/G": np.round(ntg, 4),
            "GRV (m³)": grv.astype(int)
        })
        st.download_button("Download Results CSV", results_df.to_csv(index=False), "monte_carlo_results.csv")
        st.download_button("Download Summary", f"""
P10 (Low): {fmt.format(p10)} {unit}
P50 (Median): {fmt.format(p50)} {unit}
P90 (High): {fmt.format(p90)} {unit}
ϕ ~ {phi_dist} | N/G ~ {ntg_dist} | GRV ~ {grv_dist}
Iterations: {iterations:,}
""", "summary.txt")
else:
    st.info("Upload your reservoir CSV/Excel file to begin.")
