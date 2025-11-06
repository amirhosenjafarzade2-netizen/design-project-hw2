# app.py
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

st.set_page_config(page_title="OOIP Monte Carlo", layout="wide")
st.title("OOIP Monte Carlo Estimator")

# ------------------------------------------------------------------ #
# 1. Upload & robust parsing
# ------------------------------------------------------------------ #
uploaded = st.file_uploader("Upload reference file (CSV or Excel)", type=["csv", "xlsx", "xls"])

if uploaded:
    # ---- read file ------------------------------------------------
    if uploaded.name.endswith(".csv"):
        raw = pd.read_csv(uploaded, header=None)      # no header assumed
    else:
        raw = pd.read_excel(uploaded, header=None)

    # ---- locate header row (first row that contains "Porosity") ----
    header_row = None
    for i, row in raw.iterrows():
        if "Porosity" in row.values:
            header_row = i
            break
    if header_row is None:
        st.error("Could not find a row containing the word **Porosity**.")
        st.stop()

    # split data
    header = raw.iloc[header_row]
    data   = raw.iloc[header_row+1:].reset_index(drop=True)

    # ---- map columns ------------------------------------------------
    col_map = {}
    for idx, name in enumerate(header):
        name_str = str(name).strip()
        if "Porosity" in name_str:
            col_map["Porosity"] = idx
        elif "Permeability" in name_str:
            col_map["Permeability_md"] = idx
        elif "NetToGross" in name_str or "Net To Gross" in name_str:
            col_map["NetToGross"] = idx
        elif "Gross Volume" in name_str:
            col_map["Gross_min"] = idx
            col_map["Gross_max"] = idx + 1          # min & max are side-by-side
        elif "Swi" in name_str:
            col_map["Swi"] = idx
        elif "Formation volume Factor" in name_str or "Bo" in name_str:
            col_map["Bo"] = idx

    missing = [k for k in ["Porosity","Permeability_md","NetToGross",
                           "Gross_min","Gross_max","Swi","Bo"] if k not in col_map]
    if missing:
        st.error(f"Missing columns: {', '.join(missing)}")
        st.stop()

    # ---- extract samples -------------------------------------------
    porosity = pd.to_numeric(data.iloc[:, col_map["Porosity"]], errors="coerce").dropna().values
    perm     = pd.to_numeric(data.iloc[:, col_map["Permeability_md"]], errors="coerce").dropna().values
    ntg      = pd.to_numeric(data.iloc[:, col_map["NetToGross"]], errors="coerce").dropna().values

    # min / max Gross Volume, Swi, Bo are in the **first data row** (row 0 after header)
    gross_min = pd.to_numeric(data.iloc[0, col_map["Gross_min"]], errors="coerce")
    gross_max = pd.to_numeric(data.iloc[0, col_map["Gross_max"]], errors="coerce")
    swi       = pd.to_numeric(data.iloc[0, col_map["Swi"]], errors="coerce")
    bo        = pd.to_numeric(data.iloc[0, col_map["Bo"]], errors="coerce")

    if any(v is None for v in [gross_min, gross_max, swi, bo]):
        st.error("Could not parse Gross Volume, Swi or Bo from the first data row.")
        st.stop()

    st.success("File parsed successfully!")
    st.write(f"**Samples:** Porosity={len(porosity)}, Permeability={len(perm)}, NetToGross={len(ntg)}")
    st.write(f"**Gross Volume:** {gross_min:.2e} – {gross_max:.2e} m³  |  **Swi:** {swi:.3f}  |  **Bo:** {bo:.3f}")

    # ------------------------------------------------------------------ #
    # 2. Distribution detection (Porosity & NetToGross only)
    # ------------------------------------------------------------------ #
    def best_distribution(samples, name):
        if len(samples) < 20:
            return "bootstrap", None

        tests = {}
        # Normal
        _, p = stats.shapiro(samples)
        tests["Normal"] = p

        # Uniform (KS against fitted uniform)
        u = (samples - samples.min()) / (samples.max() - samples.min() + 1e-12)
        _, p = stats.kstest(u, "uniform")
        tests["Uniform"] = p

        # Lognormal
        if (samples > 0).all():
            _, p = stats.shapiro(np.log(samples))
            tests["Lognormal"] = p

        # Triangular (fit min, mode≈median, max)
        a, b = samples.min(), samples.max()
        c = np.median(samples)
        cdf_tri = stats.triang.cdf
        cdf_tri.stats = lambda *args, **kwds: None
        try:
            _, p = stats.kstest(samples, lambda x: cdf_tri((x-a)/(b-a), c=(c-a)/(b-a)))
            tests["Triangular"] = p
        except:
            pass

        best = max(tests, key=tests.get)
        st.write(f"**{name}:** best fit → **{best}** (p={tests[best]:.3f})")
        return best, tests[best]

    phi_dist, _ = best_distribution(porosity, "Porosity")
    ntg_dist, _ = best_distribution(ntg,      "NetToGross")

    # ------------------------------------------------------------------ #
    # 3. Monte-Carlo settings
    # ------------------------------------------------------------------ #
    st.markdown("---")
    iterations = st.slider("Monte Carlo iterations", 100, 20_000, 5_000, 500)
    show_p = st.checkbox("Mark P10 / P50 / P90 on plots", True)

    # ------------------------------------------------------------------ #
    # 4. Run simulation
    # ------------------------------------------------------------------ #
    if st.button("Run Monte Carlo"):
        prog = st.progress(0)
        status = st.empty()

        # ---- sample from detected distributions ------------------------
        rng = np.random.default_rng()

        def draw(dist, samples, size):
            if dist == "Normal":
                return rng.normal(np.mean(samples), np.std(samples), size)
            if dist == "Lognormal" and (samples > 0).all():
                mu = np.mean(np.log(samples))
                sigma = np.std(np.log(samples))
                return rng.lognormal(mu, sigma, size)
            if dist == "Uniform":
                return rng.uniform(samples.min(), samples.max(), size)
            if dist == "Triangular":
                return rng.triangular(samples.min(), np.median(samples), samples.max(), size)
            # fallback bootstrap
            return rng.choice(samples, size=size, replace=True)

        phi = np.clip(draw(phi_dist, porosity, iterations), 0, 1)
        ntg = np.clip(draw(ntg_dist, ntg,      iterations), 0, 1)
        gross = rng.uniform(gross_min, gross_max, iterations)

        # ---- Net volume = Gross × NTG × Porosity ----------------------
        net_vol = gross * ntg * phi

        # ---- OOIP (metric-adapted) ------------------------------------
        # 7758 converts acre-ft → bbl; here we keep it for consistency with standard formula
        ooip = (7758 * net_vol * (1 - swi)) / bo

        prog.progress(100)
        status.success("Simulation finished!")

        # ----------------------------------------------------------------
        # 5. Results
        # ----------------------------------------------------------------
        p10 = np.percentile(ooip, 90)   # high case
        p50 = np.percentile(ooip, 50)
        p90 = np.percentile(ooip, 10)   # low case

        col1, col2, col3 = st.columns(3)
        col1.metric("Mean OOIP", f"{ooip.mean():.2e}")
        col2.metric("P50 (Median)", f"{p50:.2e}")
        col3.metric("Std", f"{ooip.std():.2e}")

        st.markdown("**P10 / P50 / P90**")
        st.write(f"P10 (high) = **{p10:.2e}**  |  P50 = **{p50:.2e}**  |  P90 (low) = **{p90:.2e}**")

        # ----------------------------------------------------------------
        # 6. Plots
        # ----------------------------------------------------------------
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Histogram", "Ascending CDF", "Descending CDF", "Histogram (log)"),
            specs=[[{"type":"histogram"},{"type":"xy"}],
                   [{"type":"xy"},      {"type":"histogram"}]]
        )

        # Histogram
        fig.add_trace(go.Histogram(x=ooip, nbinsx=60, name="OOIP"), row=1, col=1)

        # CDFs
        sorted_o = np.sort(ooip)
        prob = np.linspace(0, 1, len(sorted_o))

        # Ascending
        fig.add_trace(go.Scatter(x=sorted_o, y=prob, mode="lines", name="CDF"), row=1, col=2)
        # Descending (exceedance)
        fig.add_trace(go.Scatter(x=sorted_o[::-1], y=1-prob[::-1], mode="lines", name="Exceedance"), row=2, col=1)

        # Log histogram
        fig.add_trace(go.Histogram(x=ooip, nbinsx=60, name="OOIP log", xaxis="x4"), row=2, col=2)

        if show_p:
            for val, label, color in [(p90, "P90 (low)", "green"),
                                      (p50, "P50",      "blue"),
                                      (p10, "P10 (high)","red")]:
                fig.add_vline(x=val, line=dict(dash="dash", color=color),
                              annotation_text=label, row=1, col=2)
                fig.add_vline(x=val, line=dict(dash="dash", color=color),
                              annotation_text=label, row=2, col=1)

        fig.update_layout(height=800, showlegend=False)
        fig.update_xaxes(type="log", row=2, col=2)
        st.plotly_chart(fig, use_container_width=True)

        # ----------------------------------------------------------------
        # 7. Download
        # ----------------------------------------------------------------
        csv = pd.DataFrame({"OOIP": ooip}).to_csv(index=False)
        st.download_button("Download OOIP samples (CSV)", csv, "ooip_samples.csv", "text/csv")
