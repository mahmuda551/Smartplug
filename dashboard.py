tumi ei solution amr deya code er moddhe replace kore deo

import streamlit as st
from pymongo import MongoClient
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import plotly.express as px
from datetime import datetime

# --- MongoDB Setup ---
@st.cache_resource
def get_collection():
    """Connects to MongoDB using credentials from st.secrets."""
    client = MongoClient(st.secrets["mongo"]["uri"])
    db = client[st.secrets["mongo"]["db_name"]]
    return db[st.secrets["mongo"]["collection_name"]]

collection = get_collection()

# --- Streamlit UI ---
st.set_page_config(page_title="Energy Dashboard", layout="wide")
st.title("🔌 EnergyFlow Tracker")

# ✅ Auto-refresh every 2 minutes
st_autorefresh(interval=2 * 60 * 1000, limit=None, key="2min_autorefresh")

# --- Fetch and Clean Data ---
def fetch_data():
    cursor = collection.find().sort("timestamp", 1)
    raw_docs = list(cursor)

    if not raw_docs:
        return pd.DataFrame()

    flattened_data = []
    for doc in raw_docs:
        if not isinstance(doc, dict):
            continue

        new_doc = {'timestamp': doc.get('timestamp')}
        status_value = doc.get('status')

        if isinstance(status_value, list):
            for item in status_value:
                if isinstance(item, dict) and 'code' in item and 'value' in item:
                    new_doc[item['code']] = item['value']

        flattened_data.append(new_doc)

    df = pd.DataFrame(flattened_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    numeric_cols = ['cur_current', 'cur_power', 'cur_voltage']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df["date"] = df["timestamp"].dt.date
    df["month"] = df["timestamp"].dt.to_period("M").astype(str)
    df["year"] = df["timestamp"].dt.year
    df["week"] = df["timestamp"].dt.isocalendar().week

    return df

data = fetch_data()

# --- Sidebar Filters ---
if not data.empty:
    st.sidebar.header("📊 Filter Data")

    min_date = min(data["timestamp"]).date()
    max_date = max(data["timestamp"]).date()

    start_date = st.sidebar.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

    if start_date > end_date:
        st.sidebar.error("Start date must be before or equal to end date.")

    summary_period = st.sidebar.radio("Summary Period", options=["None", "Day", "Week", "Month"], index=0)

    mask = (data["date"] >= start_date) & (data["date"] <= end_date)
    filtered_data = data[mask]
else:
    filtered_data = pd.DataFrame()

# --- Aggregated Summary Function ---
def get_aggregated_summary(df, period):
    if df.empty:
        return None

    df = df.sort_values("timestamp")
    df['time_diff_hr'] = df['timestamp'].diff().dt.total_seconds() / 3600
    df['power_wh'] = df['cur_power'] * df['time_diff_hr']

    if period == "Day":
        group_col = 'date'
    elif period == "Week":
        group_col = 'week'
    elif period == "Month":
        group_col = 'month'
    else:
        return None

    grouped = df.groupby(group_col).agg(
        total_energy_kwh=pd.NamedAgg(column='power_wh', aggfunc=lambda x: x.sum(skipna=True) / 1000),
        max_power=pd.NamedAgg(column='cur_power', aggfunc='max'),
        min_power=pd.NamedAgg(column='cur_power', aggfunc='min'),
        avg_current=pd.NamedAgg(column='cur_current', aggfunc='mean')
    ).reset_index()

    return grouped

# --- Display Section ---
if filtered_data.empty:
    st.warning("No data found for the selected date range.")
else:
    # 📍 Latest Readings
    st.subheader("📊 Latest Metrics")
    latest = filtered_data.iloc[-1]
    col1, col2, col3 = st.columns(3)
    col1.metric("Power (W)", f"{latest.get('cur_power', 'N/A') / 10}" if pd.notna(latest.get('cur_power')) else "N/A")
    col2.metric("Voltage (V)", f"{latest.get('cur_voltage', 'N/A') / 10:.1f}" if pd.notna(latest.get('cur_voltage')) else "N/A")
    col3.metric("Current (mA)", f"{latest.get('cur_current', 'N/A')}" if pd.notna(latest.get('cur_current')) else "N/A")

    # ⚠️ Alerts
    POWER_THRESHOLD_W = 100.0
    CURRENT_THRESHOLD_MA = 500
    if latest.get("cur_power") and latest["cur_power"] > POWER_THRESHOLD_W:
        st.error(f"⚠️ High Power Alert! Power is {latest['cur_power'] / 10:.1f} W")
    if latest.get("cur_current") and latest["cur_current"] > CURRENT_THRESHOLD_MA:
        st.error(f"⚠️ High Current Alert! Current is {latest['cur_current']} mA")

    # 📅 Aggregated Summary
    if summary_period != "None":
        summary_df = get_aggregated_summary(filtered_data, summary_period)
        if summary_df is not None and not summary_df.empty:
            st.subheader(f"📅 {summary_period}ly Aggregated Summary")
            st.dataframe(summary_df.round(3))

            x_col = {'Day': 'date', 'Week': 'week', 'Month': 'month'}[summary_period]
            fig = px.bar(summary_df, x=x_col, y='total_energy_kwh',
                         labels={x_col: summary_period, 'total_energy_kwh': 'Total Energy (kWh)'},
                         title=f"Total Energy Used Per {summary_period}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No summary data available.")
    else:
        st.subheader("📈 Real-time Metrics Over Time")
        fig_power = px.line(filtered_data, x='timestamp', y='cur_power', title="Power (W)")
        st.plotly_chart(fig_power.update_yaxes(title_text='Power (W)'), use_container_width=True)
        
        fig_voltage = px.line(filtered_data, x='timestamp', y='cur_voltage', title="Voltage (V)")
        st.plotly_chart(fig_voltage.update_yaxes(title_text='Voltage (V)'), use_container_width=True)

        fig_current = px.line(filtered_data, x='timestamp', y='cur_current', title="Current (mA)")
        st.plotly_chart(fig_current.update_yaxes(title_text='Current (mA)'), use_container_width=True)

        # ✅ Total energy usage calculation
        st.subheader("⚡ Total Energy Usage (kWh)")
        df_energy = filtered_data.sort_values("timestamp").copy()
        df_energy["time_diff_hr"] = df_energy["timestamp"].diff().dt.total_seconds() / 3600
        df_energy.dropna(subset=['time_diff_hr', 'cur_power'], inplace=True)
        df_energy = df_energy[df_energy["time_diff_hr"] > 0]
        df_energy["power_wh"] = df_energy["cur_power"] * df_energy["time_diff_hr"]
        total_energy_kwh = df_energy["power_wh"].sum() / 1000
        
        st.success(f"Total Energy Used: **{total_energy_kwh:.4f} kWh** between {start_date} and {end_date}")

# --- Raw Data ---
with st.expander("📄 Raw Data (Filtered)"):
    st.dataframe(filtered_data)
