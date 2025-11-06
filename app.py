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
# 1. Upload & FINAL ROBUST PARSING – 200 samples guaranteed
# ------------------------------------------------------------------ #
uploaded = st.file_uploader("Upload reference file (CSV or Excel)", type=["csv", "xlsx", "xls"])

if uploaded:
    # --- Read as strings to preserve scientific notation ---
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
    meta_row = raw.iloc[header_row + 1].astype(str).str.strip()  # min/max labels
    data_rows = raw.iloc[header_row + 2 :].reset_index(drop=True).astype(str).applymap(str.strip)

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
        elif "gross volume" in name or "gross vol" in name:
            col_map["Gross_min"] = idx
            col_map["Gross_max"] = idx + 1
        elif "swi" in name:
            col_map["Swi"] = idx
        elif "formation" in name or "bo" in name:
            col_map["Bo"] = idx

    required = ["Porosity","Permeability_md","NetToGross","Gross_min","Gross_max","Swi","Bo"]
    missing = [k for k in required if k not in col_map]
    if missing:
        st.error(f"Missing columns: {', '.join(missing)}")
        st.stop()

    # --- Safe float conversion ---
    def safe_float(s):
        try:
            return float(str(s).replace(",", "").replace("\n","").replace("\r","").strip())
        except:
            return np.nan

    # --- FORCE 200 SAMPLES FROM COLUMNS 0, 1, 2 (Porosity, Perm, NTG) ---
    porosity_raw = pd.to_numeric(data_rows.iloc[:, 0].apply(safe_float), errors='coerce')
    ntg_raw      = pd.to_numeric(data_rows.iloc[:, 2].apply(safe_float), errors='coerce')

    porosity = porosity_raw.dropna().values
    ntg      = ntg_raw.dropna().values

    # --- STRICT 200 SAMPLE CHECK ---
    if len(porosity) != 200 or len(ntg) != 200:
        st.error(f"Expected 200 samples. Got Porosity={len(porosity)}, NTG={len(ntg)}. Check for invalid entries.")
        st.stop()

    # --- Extract Gross min/max from FIRST DATA ROW ---
    first_data = data_rows.iloc[0]
    gross_min = safe_float(first_data.iloc[col_map["Gross_min"]])
    gross_max = safe_float(first_data.iloc[col_map["Gross_max"]])
    swi       = safe_float(meta_row.iloc[col_map["Swi"]])
    bo        = safe_float(meta_row.iloc[col_map["Bo"]])

    if any(pd.isna(x) for x in [gross_min, gross_max, swi, bo]):
        st.error("Failed to parse Gross Volume, Swi, or Bo.")
        st.stop()

    st.success("200/200 samples loaded successfully!")
    st.write(f"**Gross Vol:** {gross_min:,.2e} – {gross_max:,.2e} m³ | **Swi:** {swi:.3f} | **Bo:** {bo:.3f}")

    # ------------------------------------------------------------------ #
    # 2. Distribution detection
    # ------------------------------------------------------------------ #
    def best_distribution(samples, name):
        if len(samples) < 20:
            st.write(f"**{name}:** too few samples → using **bootstrap**")
            return "bootstrap"

        tests = {}
        _, p = stats.shapiro(samples)
        tests["Normal"] = p

        u = (samples - samples.min()) / (samples.max() - samples.min() + 1e-12)
        _, p = stats.kstest(u, "uniform")
        tests["Uniform"] = p

        if (samples > 0).all():
            _, p = stats.shapiro(np.log(samples))
            tests["Lognormal"] = p

        a, b = samples.min(), samples.max()
        c = np.median(samples)
        try:
            _, p = stats.kstest(samples, lambda x: stats.triang.cdf((x-a)/(b-a), c=(c-a)/(b-a)))
            tests["Triangular"] = p
        except:
            pass

        best = max(tests, key=tests.get)
        st.write(f"**{name}:** best fit → **{best}** (p={tests[best]:.3f})")
        return best

    phi_dist = best_distribution(porosity, "Porosity")
    ntg_dist = best_distribution(ntg,      "NetToGross")

    # ------------------------------------------------------------------ #
    # 3. UI Controls
    # ------------------------------------------------------------------ #
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        iterations = st.slider("Monte Carlo iterations", 100, 20_000, 5_000, 500)
    with col2:
        decimals = st.slider("Decimal places on plots & results", 0, 4, 2)

    show_p = st.checkbox("Mark P10 / P50 / P90 on plots", True)

    # ------------------------------------------------------------------ #
    # 4. Run simulation
    # ------------------------------------------------------------------ #
    if st.button("Run Monte Carlo", type="primary"):
        prog = st.progress(0)
        status = st.empty()

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
            return rng.choice(samples, size=size, replace=True)

        phi = np.clip(draw(phi_dist, porosity, iterations), 0, 1)
        ntg = np.clip(draw(ntg_dist, ntg,      iterations), 0, 1)
        gross = rng.uniform(gross_min, gross_max, iterations)

        net_vol = gross * ntg * phi
        ooip = (7758 * net_vol * (1 - swi)) / bo

        prog.progress(100)
        status.success("Simulation complete!")

        # ----------------------------------------------------------------
        # 5. Format numbers
        # ----------------------------------------------------------------
        fmt = f"{{:.{decimals}e}}"

        p10 = np.percentile(ooip, 90)
        p50 = np.percentile(ooip, 50)
        p90 = np.percentile(ooip, 10)

        # ----------------------------------------------------------------
        # 6. Results
        # ----------------------------------------------------------------
        col1, col2, col3 = st.columns(3)
        col1.metric("Mean OOIP", fmt.format(ooip.mean()))
        col2.metric("P50 (Median)", fmt.format(p50))
        col3.metric("Std Dev", fmt.format(ooip.std()))

        st.markdown(f"**P10 / P50 / P90** (rounded to {decimals} decimal(s))")
        st.write(f"**P10 (high):** {fmt.format(p10)} | **P50:** {fmt.format(p50)} | **P90 (low):** {fmt.format(p90)}")

        # ----------------------------------------------------------------
        # 7. Plots
        # ----------------------------------------------------------------
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Histogram", "Ascending CDF", "Descending CDF", "Histogram (log)"),
            specs=[[{"type":"histogram"},{"type":"xy"}],
                   [{"type":"xy"},      {"type":"histogram"}]]
        )

        fig.add_trace(go.Histogram(x=ooip, nbinsx=60, name="OOIP",
                                 hovertemplate=f"OOIP: %{{x:{fmt}}} <br>Count: %{{y}}<extra></extra>"),
                      row=1, col=1)

        sorted_o = np.sort(ooip)
        prob = np.linspace(0, 1, len(sorted_o))

        fig.add_trace(go.Scatter(x=sorted_o, y=prob, mode="lines", name="CDF",
                                 hovertemplate=f"OOIP: %{{x:{fmt}}} <br>Prob: %{{y:.1%}}<extra></extra>"),
                      row=1, col=2)

        fig.add_trace(go.Scatter(x=sorted_o[::-1], y=1-prob[::-1], mode="lines", name="Exceedance",
                                 hovertemplate=f"OOIP: %{{x:{fmt}}} <br>Prob >: %{{y:.1%}}<extra></extra>"),
                      row=2, col=1)

        fig.add_trace(go.Histogram(x=ooip, nbinsx=60, name="OOIP (log)",
                                 hovertemplate=f"OOIP: %{{x:{fmt}}} <br>Count: %{{y}}<extra></extra>"),
                      row=2, col=2)

        if show_p:
            for val, label, color in [(p90, "P90 (low)", "green"),
                                      (p50, "P50", "blue"),
                                      (p10, "P10 (high)", "red")]:
                val_fmt = fmt.format(val)
                fig.add_vline(x=val, line=dict(dash="dash", color=color),
                              annotation_text=f"{label}: {val_fmt}", row=1, col=2)
                fig.add_vline(x=val, line=dict(dash="dash", color=color),
                              annotation_text=f"{label}: {val_fmt}", row=2, col=1)

        fig.update_layout(height=800, showlegend=False, hovermode="x unified")
        fig.update_xaxes(tickformat=f".{decimals}e", row=1, col=1)
        fig.update_xaxes(tickformat=f".{decimals}e", row=1, col=2)
        fig.update_xaxes(tickformat=f".{decimals}e", row=2, col=1)
        fig.update_xaxes(type="log", tickformat=f".{decimals}e", row=2, col=2)

        st.plotly_chart(fig, use_container_width=True)

        # ----------------------------------------------------------------
        # 8. Download
        # ----------------------------------------------------------------
        ooip_rounded = np.round(ooip, decimals)
        csv = pd.DataFrame({"OOIP": ooip_rounded}).to_csv(index=False)
        st.download_button("Download OOIP samples (CSV)", csv, "ooip_samples.csv", "text/csv")
