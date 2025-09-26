import streamlit as st
import plotly.express as px #Charts
import pandas as pd
import os #file management
import warnings #for any warnings
warnings.filterwarnings('ignore') #ignore the warnings

st.set_page_config(page_title="Sales Dashboard", page_icon=":bar_chart",layout="wide")
st.title(" :bar_chart: Sales Dashboard")
# Sample sales data
data = {
    "Region": ["North", "South", "East", "West"],
    "Sales": [15000, 12000, 18000, 10000]
}

df = pd.DataFrame(data)

# Show dataframe in the app
st.dataframe(df)

# Create a Plotly bar chart
fig = px.bar(df, x="Region", y="Sales", title="Sales by Region", color="Sales")
st.plotly_chart(fig, use_container_width=True)
