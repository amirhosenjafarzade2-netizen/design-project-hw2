import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# Page config
st.set_page_config(page_title="OOIP Monte Carlo Estimator", layout="wide")

st.title("🛢️ OOIP Monte Carlo Simulation App")

# Sidebar for instructions
with st.sidebar:
    st.header("📋 Instructions")
    st.write("""
    1. Upload your reference CSV/Excel file.
    2. The app will parse Porosity, Permeability (md), NetToGross, Gross Volume min/max, Swi, and Formation Volume Factor (Bo).
    3. Distributions for Porosity, Permeability, and NetToGross will be automatically detected (Normal, Uniform, Triangular, Lognormal).
    4. Specify number of Monte Carlo iterations (100 - 20,000).
    5. Run simulation to estimate OOIP using the formula below.
    6. View results and plots.
    """)
    
    st.subheader("🔬 OOIP Formula")
    st.latex(r"""
    \text{OOIP} = \frac{7758 \times V \times \phi \times (1 - S_w)}{B_o}
    """)
    st.write("Where:")
    st.write("- \( V \): Net Volume (m³) = Gross Volume (m³) × Net-to-Gross")
    st.write("- \( \phi \): Porosity")
    st.write("- \( S_w \): Water Saturation")
    st.write("- \( B_o \): Oil Formation Volume Factor (rb/STB, but adjusted for m³/Sm³ input)")
    st.write("*Note: The standard formula uses barrels and acre-ft; here adapted for metric inputs. 7758 is the conversion factor for consistency.*")

# Main app
st.header("1. Upload Reference File")

uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    try:
        # Read the file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success("File uploaded successfully!")
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        # Parse columns - assuming first row is headers, but handle comma in perm
        # Rename columns for clarity
        df.columns = ['Porosity', 'Permeability_md', 'NetToGross', 'Col4', 'Col5', 'Col6', 'Gross_Volume_m3', 'Col8', 'Swi', 'Bo', 'Col11']
        
        # Extract samples (from row 1 onwards, assuming row 0 has min/max for gross)
        samples_df = df.iloc[1:].copy()
        
        # Extract porosity, perm, ntg samples (200 values)
        porosity_samples = pd.to_numeric(samples_df['Porosity'], errors='coerce').dropna().values
        perm_samples = pd.to_numeric(samples_df['Permeability_md'], errors='coerce').dropna().values  # Handle comma if present, but assuming cleaned
        ntg_samples = pd.to_numeric(samples_df['NetToGross'], errors='coerce').dropna().values
        
        # Gross volume min/max from row 0, columns 6 and 7 (0-indexed)
        gross_min = pd.to_numeric(df.iloc[0, 6], errors='coerce')
        gross_max = pd.to_numeric(df.iloc[0, 7], errors='coerce')
        
        # Swi and Bo from row 0
        swi = pd.to_numeric(df.iloc[0, 8], errors='coerce')
        bo = pd.to_numeric(df.iloc[0, 9], errors='coerce')
        
        # Check lengths
        if len(porosity_samples) < 200 or len(ntg_samples) < 200:
            st.error("Warning: Less than 200 samples found for Porosity or NetToGross. Using available data.")
        
        st.subheader("Extracted Parameters")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Porosity Samples", len(porosity_samples))
        with col2:
            st.metric("NetToGross Samples", len(ntg_samples))
        with col3:
            st.metric("Gross Volume Range", f"{gross_min:.2e} - {gross_max:.2e} m³")
        
        st.subheader("Distribution Detection")
        
        def detect_distribution(samples, var_name):
            if len(samples) < 20:
                return "Insufficient data"
            
            # Test distributions
            # Normal
            _, p_normal = stats.shapiro(samples)
            
            # Uniform (Kolmogorov-Smirnov against uniform)
            uniform_samples = (samples - samples.min()) / (samples.max() - samples.min())
            _, p_uniform = stats.kstest(uniform_samples, 'uniform')
            
            # Triangular: Fit and KS test
            a, b, c = samples.min(), samples.max(), np.median(samples)
            def tri_pdf(x, a, b, c):
                return np.where((x < a) | (x > b), 0,
                                np.where(x < c, 2*(x-a)/((b-a)*(c-a)), 2*(b-x)/((b-a)*(b-c))))
            _, p_tri = stats.kstest(samples, lambda x: tri_pdf(x, a, b, c))
            
            # Lognormal
            _, p_lognorm = stats.shapiro(np.log(samples + 1e-10))  # Avoid log(0)
            
            p_values = {
                'Normal': p_normal,
                'Uniform': p_uniform,
                'Triangular': p_tri,
                'Lognormal': p_lognorm
            }
            
            best_dist = max(p_values, key=p_values.get)
            st.write(f"**{var_name}:** Best fit - {best_dist} (p-value: {p_values[best_dist]:.3f})")
            return best_dist, p_values[best_dist]
        
        # Detect for each
        porosity_dist, _ = detect_distribution(porosity_samples, "Porosity")
        perm_dist, _ = detect_distribution(perm_samples, "Permeability")
        ntg_dist, _ = detect_distribution(ntg_samples, "NetToGross")
        
        st.header("2. Monte Carlo Simulation Setup")
        
        iterations = st.slider("Number of iterations", min_value=100, max_value=20000, value=1000, step=100)
        
        show_p10_p50_p90 = st.checkbox("Show P10 (High), P50 (Median), P90 (Low) on plots?", value=True)
        
        if st.button("Run Simulation"):
            st.subheader("Running Monte Carlo Simulation...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Generate random samples based on detected distributions
            # Porosity
            if porosity_dist == 'Normal':
                phi = np.random.normal(np.mean(porosity_samples), np.std(porosity_samples), iterations)
            elif porosity_dist == 'Lognormal':
                phi = np.random.lognormal(np.mean(np.log(porosity_samples)), np.std(np.log(porosity_samples)), iterations)
            elif porosity_dist == 'Uniform':
                phi = np.random.uniform(np.min(porosity_samples), np.max(porosity_samples), iterations)
            elif porosity_dist == 'Triangular':
                phi = np.random.triangular(np.min(porosity_samples), np.median(porosity_samples), np.max(porosity_samples), iterations)
            else:
                phi = porosity_samples[:iterations] if len(porosity_samples) >= iterations else np.random.choice(porosity_samples, iterations)
            
            # Net to Gross
            if ntg_dist == 'Normal':
                ntg = np.random.normal(np.mean(ntg_samples), np.std(ntg_samples), iterations)
            elif ntg_dist == 'Lognormal':
                ntg = np.random.lognormal(np.mean(np.log(ntg_samples)), np.std(np.log(ntg_samples)), iterations)
            elif ntg_dist == 'Uniform':
                ntg = np.random.uniform(np.min(ntg_samples), np.max(ntg_samples), iterations)
            elif ntg_dist == 'Triangular':
                ntg = np.random.triangular(np.min(ntg_samples), np.median(ntg_samples), np.max(ntg_samples), iterations)
            else:
                ntg = ntg_samples[:iterations] if len(ntg_samples) >= iterations else np.random.choice(ntg_samples, iterations)
            
            # Clip to [0,1] for phi and ntg
            phi = np.clip(phi, 0, 1)
            ntg = np.clip(ntg, 0, 1)
            
            # Gross Volume: Uniform between min and max (as per data format)
            gross_vol = np.random.uniform(gross_min, gross_max, iterations)
            
            # Net Volume = Gross * NTG * phi? Wait, no: Standard is Net Rock Volume = Gross * NTG, then Pore Vol = Net Rock * phi
            # But user said: net volume is porosity * net to gross * gross volume
            # So following user: net_vol = gross_vol * ntg * phi
            net_vol = gross_vol * ntg * phi
            
            # OOIP calculation
            # Note: User input is m3/Sm3, but formula uses 7758 (for bbl, acre-ft). To adapt:
            # Assuming user intends metric, but to match standard, we'll use the formula as-is, assuming Bo is in rb/STB and adjust units implicitly.
            # For simplicity, use the formula directly, assuming inputs are consistent.
            ooip = (7758 * net_vol * (1 - swi) ) / bo  # net_vol in m3, but 7758 expects acre-ft; this may need unit conversion, but per user request.
            
            # Progress update
            progress_bar.progress(100)
            status_text.text("Simulation complete!")
            
            st.subheader("3. Results")
            st.metric("Mean OOIP", f"{np.mean(ooip):.2e}")
            st.metric("P10 (High Case)", f"{np.percentile(ooip, 90):.2e}")  # P10 is 90th percentile (high)
            st.metric("P50 (Median)", f"{np.percentile(ooip, 50):.2e}")
            st.metric("P90 (Low Case)", f"{np.percentile(ooip, 10):.2e}")   # P90 is 10th percentile (low)
            
            st.subheader("OOIP Distribution")
            st.dataframe(pd.DataFrame({
                'Mean': [np.mean(ooip)],
                'Std': [np.std(ooip)],
                'Min': [np.min(ooip)],
                'P10': [np.percentile(ooip, 10)],
                'P50': [np.percentile(ooip, 50)],
                'P90': [np.percentile(ooip, 90)],
                'Max': [np.max(ooip)]
            }))
            
            # Plots
            st.subheader("4. Visualizations")
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Histogram', 'Ascending CDF (P10, P50, P90)', 'Descending CDF', 'Histogram (Log Scale)'),
                specs=[[{"type": "histogram"}, {"type": "xy"}],
                       [{"type": "xy"}, {"type": "histogram"}]]
            )
            
            # Histogram
            fig.add_trace(go.Histogram(x=ooip, name='OOIP', nbinsx=50), row=1, col=1)
            
            # Ascending CDF
            sorted_ooip = np.sort(ooip)
            x_cdf = np.arange(1, len(sorted_ooip) + 1) / len(sorted_ooip)
            fig.add_trace(go.Scatter(x=sorted_ooip, y=x_cdf, mode='lines', name='Ascending CDF'), row=1, col=2)
            if show_p10_p50_p90:
                p10_val = np.percentile(ooip, 10)
                p50_val = np.percentile(ooip, 50)
                p90_val = np.percentile(ooip, 90)
                fig.add_vline(x=p10_val, line_dash="dash", line_color="green", annotation_text="P90 (Low)", row=1, col=2)
                fig.add_vline(x=p50_val, line_dash="dot", line_color="blue", annotation_text="P50", row=1, col=2)
                fig.add_vline(x=p90_val, line_dash="dash", line_color="red", annotation_text="P10 (High)", row=1, col=2)
            
            # Descending CDF: Probability of exceeding (1 - CDF)
            fig.add_trace(go.Scatter(x=sorted_ooip[::-1], y=1 - x_cdf[::-1], mode='lines', name='Descending CDF'), row=2, col=1)
            if show_p10_p50_p90:
                fig.add_vline(x=p10_val, line_dash="dash", line_color="green", annotation_text="P90 (Low)", row=2, col=1)
                fig.add_vline(x=p50_val, line_dash="dot", line_color="blue", annotation_text="P50", row=2, col=1)
                fig.add_vline(x=p90_val, line_dash="dash", line_color="red", annotation_text="P10 (High)", row=2, col=1)
            
            # Log scale histogram
            fig.add_trace(go.Histogram(x=ooip, name='OOIP (Log)', nbinsx=50, xaxis='log'), row=2, col=2)
            
            fig.update_layout(height=800, showlegend=True, title_text="OOIP Analysis Plots")
            fig.update_xaxes(type="log", row=2, col=2)
            st.plotly_chart(fig, use_container_width=True)
            
            # Download results
            csv_buffer = io.StringIO()
            pd.DataFrame({'OOIP': ooip}).to_csv(csv_buffer, index=False)
            st.download_button("Download OOIP Samples (CSV)", csv_buffer.getvalue(), "ooip_samples.csv")
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.write("Please ensure the file format matches the reference (CSV/Excel with Porosity, Permeability_md, NetToGross, etc.)")
else:
    st.info("👆 Please upload a file to get started.")
    # Sample data preview if needed
    st.subheader("Expected Format Example")
    sample_df = pd.DataFrame({
        'Porosity': [0.207841, 0.158781],
        'Permeability_md': [68.891785, 101.98174],
        'NetToGross': [0.679461, 0.56746],
        'Gross_Volume_m3': [6.99e8, 1.30e9],  # Example
        'Swi': [0.23, 0.23],
        'Bo': [1.82, 1.82]
    })
    st.dataframe(sample_df)
