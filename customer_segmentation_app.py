import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px

# -------------------- PAGE SETUP WITH BACKGROUND --------------------
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1507525428034-b723cf961d3e"); /* Stylish ecommerce image */
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: #ffffff;
    }
    .block-container {
        background-color: rgba(0, 0, 0, 0.6);  /* Adds semi-transparent overlay for readability */
        padding: 2rem;
        border-radius: 12px;
    }
    h1, h2, h3, h4, h5, h6, p {
        color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\Saurav Kumar\python files\UPWORK PROJECTS\Customer segmentation\archive\customer_segmentsfull.csv", parse_dates=["InvoiceDate"])
    df.dropna(subset=["CustomerID"], inplace=True)
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    return df

df = load_data()

# -------------------- RFM CALCULATION --------------------
snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
rfm = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
    "InvoiceNo": "count",
    "TotalPrice": "sum"
})
rfm.columns = ["Recency", "Frequency", "Monetary"]

# Handling duplicate bin edges
r_labels = [4, 3, 2, 1]
f_labels = [1, 2, 3, 4]
m_labels = [1, 2, 3, 4]

rfm["R_Score"] = pd.qcut(rfm["Recency"], q=4, labels=r_labels)
rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), q=4, labels=f_labels)
rfm["M_Score"] = pd.qcut(rfm["Monetary"].rank(method="first"), q=4, labels=m_labels)

rfm["RFM_Score"] = (
    rfm["R_Score"].astype(str) +
    rfm["F_Score"].astype(str) +
    rfm["M_Score"].astype(str)
)

# -------------------- SEGMENTATION --------------------
def segment(row):
    score = row["RFM_Score"]
    if score >= "444":
        return "Champions"
    elif score >= "344":
        return "Loyal"
    elif score >= "244":
        return "Potential"
    else:
        return "Others"

rfm["Segment"] = rfm.apply(segment, axis=1)
# --- Display ---
st.title("🧠 Customer Segmentation Dashboard")
st.markdown("### Based on RFM (Recency, Frequency, Monetary) Analysis")

col1, col2, col3 = st.columns(3)
col1.metric("Total Customers", rfm.shape[0])
col2.metric("Best RFM Score", rfm["RFM_Score"].min())
col3.metric("Worst RFM Score", rfm["RFM_Score"].max())


# Display Metrics
st.subheader("📊 RFM Summary Metrics")
col1.metric("Total Customers", rfm.shape[0])
col2.metric("Average Recency", f"{rfm['Recency'].mean():.1f} days")
col3.metric("Avg. Monetary Value", f"${rfm['Monetary'].mean():,.2f}")

# --- Visualization ---
st.plotly_chart(px.histogram(rfm, x="RFM_Score", title="Customer RFM Score Distribution", color="RFM_Score"))

# -------------------- VISUALIZATIONS --------------------
st.subheader("📊 RFM Segment Distribution")
fig, ax = plt.subplots()
segment_counts = rfm["Segment"].value_counts()
sns.barplot(x=segment_counts.index, y=segment_counts.values, palette="viridis", ax=ax)
ax.set_ylabel("Number of Customers")
st.pyplot(fig)

st.subheader("📈 Recency vs Frequency (Colored by Segment)")
fig2, ax2 = plt.subplots()
sns.scatterplot(data=rfm, x="Recency", y="Frequency", hue="Segment", palette="Set2", ax=ax2)
st.pyplot(fig2)

# -------------------- RAW DATA --------------------
with st.expander("📄 View Raw RFM Data"):
    st.dataframe(rfm.reset_index())
    st.download_button(
        label="Download RFM Data",
        data=rfm.to_csv().encode('utf-8'),
        file_name='rfm_data.csv',
        mime='text/csv'
    )
# -------------------- FOOTER --------------------
st.markdown(
    """
    <style>
    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Footer hidden for a cleaner look
# -------------------- END OF APP --------------------