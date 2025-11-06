# app.py
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="OOIP/STOIIP Monte Carlo", layout="wide")
st.title("🛢️ OOIP/STOIIP Monte Carlo Estimator")
st.markdown("**Volumetric Hydrocarbon-in-Place Estimation Under Uncertainty**")

# ------------------------------------------------------------------ #
# 1. Upload & parsing
# ------------------------------------------------------------------ #
uploaded = st.file_uploader("Upload reservoir data file (CSV or Excel)", type=["csv", "xlsx", "xls"])

if uploaded:
    # --- Read as strings ---
    if uploaded.name.endswith(".csv"):
        raw = pd.read_csv(uploaded, header=None, dtype=str, na_filter=False)
    else:
        raw = pd.read_excel(uploaded, header=None, dtype=str, na_filter=False)

    # --- Find header row ---
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

    # --- Map columns ---
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

    # --- Safe float conversion ---
    def safe_float(s):
        try:
            return float(str(s).replace(",", "").replace("\n", "").replace("\r", "").strip())
        except:
            return np.nan

    # --- Extract samples ---
    porosity = pd.to_numeric(data_rows.iloc[:, col_map["Porosity"]].apply(safe_float), errors='coerce').dropna().values
    ntg = pd.to_numeric(data_rows.iloc[:, col_map["NetToGross"]].apply(safe_float), errors='coerce').dropna().values

    # --- Extract parameters ---
    first_data = data_rows.iloc[0]
    gross_min = safe_float(first_data.iloc[col_map["Gross_min"]])
    gross_max = safe_float(first_data.iloc[col_map["Gross_max"]])
    swi = safe_float(meta_row.iloc[col_map["Swi"]])
    bo = safe_float(meta_row.iloc[col_map["Bo"]])

    if any(pd.isna(x) for x in [gross_min, gross_max, swi, bo]):
        st.error("Failed to parse **Gross Volume**, **Swi**, or **Bo**. Please check file format.")
        st.stop()

    # Validate data
    if len(porosity) < 10 or len(ntg) < 10:
        st.error(f"Insufficient samples: Porosity={len(porosity)}, NTG={len(ntg)}. Need at least 10 samples each.")
        st.stop()

    # Clip values to valid ranges
    porosity = np.clip(porosity, 0, 1)
    ntg = np.clip(ntg, 0, 1)

    # Display parsed data
    st.success(f"✅ File parsed successfully — **{len(porosity)} samples** detected!")
   
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Porosity samples", len(porosity))
        st.write(f"Range: {porosity.min():.3f} - {porosity.max():.3f}")
    with col2:
        st.metric("N/G samples", len(ntg))
        st.write(f"Range: {ntg.min():.3f} - {ntg.max():.3f}")
    with col3:
        st.metric("Gross Volume (m³)", f"{gross_min:.2e} - {gross_max:.2e}")
        st.write(f"Swi: {swi:.3f} | Bo: {bo:.3f}")

    # ------------------------------------------------------------------ #
    # 2. Distribution detection (for suggestion only)
    # ------------------------------------------------------------------ #
    st.markdown("---")
    st.subheader("📊 Suggested Distributions (based on data fit)")

    def best_distribution(samples, name):
        if len(samples) < 20:
            return "Triangular", {"reason": "Too few samples"}
        tests = {}
        # Shapiro-Wilk for normality
        try:
            _, p = stats.shapiro(samples)
            tests["Normal"] = p
        except:
            pass
        # KS for uniform
        try:
            u = (samples - samples.min()) / (samples.max() - samples.min() + 1e-12)
            _, p = stats.kstest(u, "uniform")
            tests["Uniform"] = p
        except:
            pass
        # Triangular
        try:
            a, b = samples.min(), samples.max()
            c_loc = mode_est = stats.mode(samples, keepdims=False)[0] or np.median(samples)
            c = (c_loc - a) / (b - a + 1e-12)
            _, p = stats.kstest(samples, stats.triang.cdf, args=(c, a, b-a))
            tests["Triangular"] = p
        except:
            pass
        if not tests:
            return "Triangular", {"reason": "Fallback"}
        best = max(tests, key=tests.get)
        return best, tests

    phi_suggested, phi_info = best_distribution(porosity, "Porosity")
    ntg_suggested, ntg_info = best_distribution(ntg, "NetToGross")

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Porosity (ϕ):**")
        st.write(f"Suggested: **{phi_suggested}**")
    with col2:
        st.write("**Net-to-Gross (N/G):**")
        st.write(f"Suggested: **{ntg_suggested}**")

    # ------------------------------------------------------------------ #
    # 3. User Controls - Distribution selection
    # ------------------------------------------------------------------ #
    st.markdown("---")
    st.subheader("⚙️ Parameter Distributions")

    dist_options = ["Uniform", "Triangular", "Normal", "Bootstrap (empirical)"]

    col1, col2, col3 = st.columns(3)
    with col1:
        phi_dist = st.selectbox("Porosity Distribution", dist_options, index=dist_options.index(phi_suggested) if phi_suggested in dist_options else 0)
    with col2:
        ntg_dist = st.selectbox("N/G Distribution", dist_options, index=dist_options.index(ntg_suggested) if ntg_suggested in dist_options else 0)
    with col3:
        grv_dist = st.selectbox("GRV Distribution", ["Uniform", "Triangular", "Normal", "Pert (3-point)"], index=0)

    # GRV extra params
    st.markdown("**GRV Advanced Settings**")
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        grv_most_likely = st.number_input("Most likely GRV (mode)", value=(gross_min + gross_max)/2, format="%.2e")
    with col_g2:
        grv_random_points = st.slider("Number of random points for min/max perturbation", 3, 50, 10,
                                      help="Generates random min/max around original values for realism")

    # ------------------------------------------------------------------ #
    # 4. Simulation Settings
    # ------------------------------------------------------------------ #
    st.markdown("---")
    st.subheader("🚀 Simulation Settings")
    col1, col2, col3 = st.columns(3)
   
    with col1:
        use_slider = st.radio("Iteration input method:", ["Slider", "Direct Input"], horizontal=True)
        if use_slider == "Slider":
            iterations = st.slider("Monte Carlo iterations", 1000, 100_000, 20000, 1000)
        else:
            iterations = st.number_input("Monte Carlo iterations", min_value=1000, max_value=500_000, value=20000, step=5000)
   
    with col2:
        output_unit = st.radio("Output units:", ["Stock Tank m³", "Barrels (STB)"], horizontal=True)
        decimals = st.slider("Decimal places", 0, 6, 3)
   
    with col3:
        show_markers = st.checkbox("Mark P10/P50/P90 on plots", True)

    st.info("💡 Different distributions for ϕ and N/G are **perfectly fine and recommended** when data suggests it!")

    # ------------------------------------------------------------------ #
    # 5. Run simulation
    # ------------------------------------------------------------------ #
    if st.button("🚀 Run Monte Carlo Simulation", type="primary", use_container_width=True):
        with st.spinner("Running Monte Carlo simulation..."):
            prog = st.progress(0)
            rng = np.random.default_rng()

            # Perturb GRV min/max
            prog.progress(10)
            perturb = rng.normal(1.0, 0.05, size=(iterations, grv_random_points))  # 5% std
            factors_min = perturb.min(axis=1)
            factors_max = perturb.max(axis=1)
            grv_min_rand = gross_min * factors_min
            grv_max_rand = gross_max * factors_max
            grv_min_rand = np.clip(grv_min_rand, gross_min*0.8, gross_min*1.2)
            grv_max_rand = np.clip(grv_max_rand, gross_max*0.8, gross_max*1.2)

            def draw(dist, samples, size):
                if dist == "Bootstrap (empirical)":
                    return rng.choice(samples, size=size, replace=True)
                if dist == "Normal":
                    mu, sigma = stats.norm.fit(samples)
                    return rng.normal(mu, sigma, size)
                if dist == "Uniform":
                    return rng.uniform(samples.min(), samples.max(), size)
                if dist == "Triangular":
                    c, loc, scale = stats.triang.fit(samples)
                    return stats.triang.rvs(c, loc, scale, size=size, random_state=rng)
                return rng.choice(samples, size=size, replace=True)

            prog.progress(30)
            phi = np.clip(draw(phi_dist, porosity, iterations), 0.001, 0.99)
            ntg = np.clip(draw(ntg_dist, ntg, iterations), 0.001, 1.0)

            prog.progress(50)
            if grv_dist == "Uniform":
                grv = rng.uniform(grv_min_rand, grv_max_rand, iterations)
            elif grv_dist == "Pert (3-point)":
                grv = stats.pert.rvs(4, grv_most_likely, grv_max_rand, grv_min_rand, size=iterations, random_state=rng)
            elif grv_dist == "Triangular":
                grv = rng.triangular(grv_min_rand, grv_most_likely, grv_max_rand, iterations)
            else:  # Normal
                mean = (grv_min_rand + grv_max_rand) / 2
                std = (grv_max_rand - grv_min_rand) / 6
                grv = np.clip(rng.normal(mean, std, iterations), grv_min_rand, grv_max_rand)

            prog.progress(70)
            net_rock_volume = grv * ntg
            hc_pore_volume = net_rock_volume * phi * (1 - swi)
            stoiip_m3 = hc_pore_volume / bo

            if output_unit == "Barrels (STB)":
                stoiip = stoiip_m3 * 6.289811  # more precise
                unit_label = "STB"
            else:
                stoiip = stoiip_m3
                unit_label = "m³"

            prog.progress(100)

        st.success("✅ Simulation complete!")

        # ----------------------------------------------------------------
        # 6. Percentiles (corrected order)
        # ----------------------------------------------------------------
        p90 = np.percentile(stoiip, 10)   # Low case (90% exceed)
        p50 = np.percentile(stoiip, 50)   # Median
        p10 = np.percentile(stoiip, 90)   # High case (10% exceed)

        fmt = f"{{:.{decimals}e}}"

        # ----------------------------------------------------------------
        # 7. Results
        # ----------------------------------------------------------------
        st.markdown("---")
        st.subheader("📈 Results")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mean STOIIP", f"{fmt.format(stoiip.mean())} {unit_label}")
        col2.metric("P50 (Median)", f"{fmt.format(p50)} {unit_label}")
        col3.metric("Std Dev", f"{fmt.format(stoiip.std())} {unit_label}")
        col4.metric("CoV", f"{stoiip.std()/stoiip.mean():.2%}")

        st.markdown("### 🎯 Proven Reserves (PRMS Style)")
        pcol1, pcol2, pcol3 = st.columns(3)
        pcol1.metric("P10 (High Estimate)", f"{fmt.format(p10)} {unit_label}",
                     help="10% probability this value will be exceeded")
        pcol2.metric("P50 (Best Estimate)", f"{fmt.format(p50)} {unit_label}")
        pcol3.metric("P90 (Low Estimate)", f"{fmt.format(p90)} {unit_label}",
                     help="90% probability this value will be exceeded")

        # ----------------------------------------------------------------
        # 8. Plots (FIXED descending CDF)
        # ----------------------------------------------------------------
        st.markdown("---")
        st.subheader("📊 Probability Distributions")
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f"Histogram of STOIIP ({unit_label})",
                "Ascending CDF",
                "Descending CDF (Exceedance Probability)",
                "Histogram (Log Scale)"
            ),
            specs=[[{"type": "histogram"}, {"type": "xy"}],
                   [{"type": "xy"}, {"type": "histogram"}]]
        )

        # Histogram
        fig.add_trace(go.Histogram(x=stoiip, nbinsx=100, name="STOIIP", marker_color='skyblue'), row=1, col=1)

        # Ascending CDF
        sorted_stoiip = np.sort(stoiip)
        prob = np.linspace(0, 1, len(sorted_stoiip))
        fig.add_trace(go.Scatter(x=sorted_stoiip, y=prob, mode="lines", name="CDF", line=dict(color='blue', width=3)), row=1, col=2)

        # Descending CDF - FIXED
        fig.add_trace(go.Scatter(x=sorted_stoiip, y=1 - prob, mode="lines", name="Exceedance", line=dict(color='green', width=3)), row=2, col=1)

        # Log histogram
        fig.add_trace(go.Histogram(x=stoiip, nbinsx=100, name="Log", marker_color='lightcoral'), row=2, col=2)

        if show_markers:
            for val, label, color in [(p90, "P90 (Low)", "green"), (p50, "P50", "blue"), (p10, "P10 (High)", "red")]:
                fig.add_vline(x=val, line=dict(dash="dash", color=color, width=2),
                              annotation_text=f"{label}: {fmt.format(val)}", row=1, col=2)
                fig.add_vline(x=val, line=dict(dash="dash", color=color, width=2),
                              annotation_text=f"{label}: {fmt.format(val)}", row=2, col=1)

        fig.update_layout(height=1000, showlegend=False, hovermode="closest")
        fig.update_xaxes(title_text=f"STOIIP ({unit_label})", tickformat=f".{decimals}e")
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_yaxes(title_text="Cumulative Probability", row=1, col=2)
        fig.update_yaxes(title_text="Exceedance Probability", row=2, col=1)
        fig.update_xaxes(type="log", row=2, col=2)

        st.plotly_chart(fig, use_container_width=True)

        # ----------------------------------------------------------------
        # 9. Summary & Download
        # ----------------------------------------------------------------
        st.markdown("---")
        st.download_button("📥 Download Full Results CSV", 
                           data=pd.DataFrame({f'STOIIP ({unit_label})': np.round(stoiip, decimals),
                                              'Porosity': np.round(phi, 4),
                                              'N/G': np.round(ntg, 4),
                                              'GRV (m³)': np.round(grv, 0)}).to_csv(index=False),
                           file_name=f"OOIP_results_{iterations}.csv")

        st.download_button("📄 Download Summary Report",
                           data=f"""OOIP Monte Carlo Report - Reservoir 3
Iterations: {iterations:,}
P10 (High): {fmt.format(p10)} {unit_label}
P50 (Median): {fmt.format(p50)} {unit_label}
P90 (Low): {fmt.format(p90)} {unit_label}
Porosity Dist: {phi_dist} | N/G Dist: {ntg_dist} | GRV Dist: {grv_dist}
""",
                           file_name="OOIP_summary.txt")

else:
    st.info("👆 Upload your Reservoir CSV/Excel file to start!")
    st.markdown("**Expected format:** Porosity, NetToGross, Gross Volume min/max, Swi, Bo")
