import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="OOIP Descending CDF", layout="wide")
st.title("OOIP Descending CDF Estimator")
st.markdown("**Descending CDF (Exceedance) Plot for OOIP**")

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
    data_rows = raw.iloc[header_row + 2:].reset_index(drop=True).astype(str).apply(lambda x: x.str.strip())

    # Detect OOIP column
    ooip_col_idx = None
    header_low = header.str.lower()
    for idx, name in enumerate(header_low):
        if "ooip" in name:
            ooip_col_idx = idx
            break

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

    # Sort descending for exceedance plot
    sorted_val_desc = np.sort(ooip_values)[::-1]
    exceedance = np.linspace(1, 0, len(sorted_val_desc))
    unit = "STB" if "stb" in uploaded.name.lower() else "m³"

    st.markdown("---")
    decimals = st.slider("Decimal places on chart labels & results", 0, 10, 3, key="reverse_cdf_decimals")

    p10 = np.percentile(ooip_values, 10)
    p50 = np.percentile(ooip_values, 50)
    p90 = np.percentile(ooip_values, 90)
    fmt = f"{{:.{decimals}e}}"

    # ------------------- Matplotlib Descending CDF ------------------- #
    st.subheader("Descending CDF (Exceedance) - OOIP Only")
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.plot(sorted_val_desc, exceedance, color="#59a14f", lw=3)  # descending order
    ax.set_title("Descending CDF")
    ax.set_xlabel(f"OOIP ({unit})")
    ax.set_ylabel("Exceedance Probability")
    # Reverse X-axis (right to left)
    ax.invert_xaxis()
    # P10 / P50 / P90
    for val, label, color in [(p10, "P10", "#59a14f"),
                              (p50, "P50", "#f28e2b"),
                              (p90, "P90", "#e15759")]:
        val_str = fmt.format(val)
        ax.axvline(val, color=color, linestyle="--", linewidth=1.5)
        ax.text(val, 0.9, f"{label}: {val_str}", rotation=90,
                va='top', ha='right', fontsize=9, color=color)
    # Scientific-notation offset to bottom-left
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    offset_text = ax.xaxis.get_offset_text()
    offset_text.set_position((0.02, 0.02))
    offset_text.set_horizontalalignment('left')
    offset_text.set_fontsize(10)
    st.pyplot(fig)

    def fig_to_png(f):
        buf = BytesIO()
        f.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        return buf

    buf = fig_to_png(fig)
    st.download_button("Download Descending CDF", buf, "descending_cdf_ooip.png", "image/png")
    plt.close(fig)

    results_df = pd.DataFrame({"OOIP": ooip_values})
    st.download_button("Download OOIP Values", results_df.to_csv(index=False), "ooip_values.csv")
else:
    st.info("Upload your reservoir CSV/Excel file to begin.")
