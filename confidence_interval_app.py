import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import t
import streamlit as st

# Streamlit title
st.title("Confidence Interval Calculator (99% Confidence Level)")

# Sample Data
data = [1.13, 1.55, 1.43, 0.92, 1.25, 1.36, 1.32, 0.85, 1.07, 1.48, 1.20, 1.33, 1.18, 1.22, 1.29]
n = len(data)
confidence_level = 0.99
population_std_dev = 0.2  # Known population standard deviation (for Z)

# Calculate sample mean and sample std deviation
sample_mean = np.mean(data)
sample_std_dev = np.std(data, ddof=1)

# T-distribution calculations
degrees_of_freedom = n - 1
t_critical = t.ppf(1 - (1 - confidence_level) / 2, degrees_of_freedom)
margin_of_error_t = t_critical * (sample_std_dev / np.sqrt(n))
ci_t = (sample_mean - margin_of_error_t, sample_mean + margin_of_error_t)

# Z-distribution calculations
z_critical = 2.576  # for 99% confidence
margin_of_error_z = z_critical * (population_std_dev / np.sqrt(n))
ci_z = (sample_mean - margin_of_error_z, sample_mean + margin_of_error_z)

# Display the results
st.subheader("Sample Statistics")
st.write(f"Sample Mean: {sample_mean:.4f}")
st.write(f"Sample Standard Deviation: {sample_std_dev:.4f}")
st.write(f"Sample Size (n): {n}")

st.subheader("T-Distribution (Sample Std Dev)")
st.write(f"T Critical Value: {t_critical:.4f}")
st.write(f"Margin of Error (T): {margin_of_error_t:.4f}")
st.write(f"Confidence Interval (T): {ci_t[0]:.4f} to {ci_t[1]:.4f}")

st.subheader("Z-Distribution (Population Std Dev = 0.2)")
st.write(f"Z Critical Value: {z_critical}")
st.write(f"Margin of Error (Z): {margin_of_error_z:.4f}")
st.write(f"Confidence Interval (Z): {ci_z[0]:.4f} to {ci_z[1]:.4f}")

# Optional: Visualize the data
st.subheader("Data Visualization")
fig, ax = plt.subplots()
sns.histplot(data, kde=True, bins=10, ax=ax)
ax.axvline(sample_mean, color='red', linestyle='--', label='Sample Mean')
ax.axvline(ci_t[0], color='green', linestyle='--', label='Lower CI (T)')
ax.axvline(ci_t[1], color='green', linestyle='--', label='Upper CI (T)')
ax.legend()
st.pyplot(fig)
