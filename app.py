import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_excel(uploaded_file)

st.set_page_config(page_title="Benford Analysis Dashboard", layout="wide")

st.title("Benford's Law Analysis Dashboard")

st.write("Upload a dataset to perform Benford's Law analysis and statistical testing.")
uploaded_file = st.file_uploader("Upload a CSV or Excel file",type=["csv", "xlsx"])

if uploaded_file is not None:
    df = load_data(uploaded_file)

    st.success("File uploaded successfully!")

    st.subheader("Data Preview")
    st.dataframe(df.head())

    st.write("Number of rows:", len(df))
    st.write("Number of columns:", len(df.columns))
    st.sidebar.header("Data Filters")

    pending_filters = []

    for i in range(1, 4):
        filter_column = st.sidebar.selectbox(f"Filter {i} Column",["None"] + list(df.columns),key=f"column_{i}")

        if filter_column != "None":
            selected_values = st.sidebar.multiselect(f"Filter {i} Values",df[filter_column].dropna().unique(),key=f"values_{i}_{filter_column}")

            if selected_values:
                pending_filters.append((filter_column, selected_values))

    if "applied_filters" not in st.session_state or st.session_state.get("file_id") != uploaded_file.file_id:
        st.session_state.applied_filters = []
        st.session_state.file_id = uploaded_file.file_id

    if st.sidebar.button("Apply Filters"):
        st.session_state.applied_filters = pending_filters

    filtered_df = df.copy()

    for column, values in st.session_state.applied_filters:
        filtered_df = filtered_df[filtered_df[column].isin(values)]

    st.subheader("Filtered Data")
    st.dataframe(filtered_df.head())
    st.write("Remaining rows:", len(filtered_df))
    st.subheader("Benford's Law Analysis")

    numeric_columns = filtered_df.select_dtypes(include=np.number).columns.tolist()

    if not numeric_columns:
        st.warning("No numerical columns available for analysis.")
        st.stop()

    selected_column = st.selectbox("Select a numerical column to analyze",numeric_columns)

    st.write("Selected column:", selected_column)
    values = filtered_df[selected_column].dropna()

    values = values[np.isfinite(values)]

    values = values.abs()

    values = values[values > 0]

    st.write("Number of values used for Benford analysis:", len(values))

    if len(values) == 0:
        st.warning("No valid values available for Benford analysis.")
        st.stop()
        st.subheader("First-Digit Analysis")

    first_digits = values.astype(str).str.replace(".", "", regex=False).str.lstrip("0").str[0].astype(int)

    digits = np.arange(1, 10)

    actual_first = first_digits.value_counts(normalize=True).reindex(digits, fill_value=0)

    expected_first = np.log10(1 + 1 / digits)

    first_digit_results = pd.DataFrame({"Digit": digits,"Actual Percentage": actual_first.values * 100,"Expected Percentage": expected_first * 100})

    st.dataframe(first_digit_results)
    st.subheader("First-Two-Digit Analysis")

    first_two_digits = values.map(lambda x: int(f"{x:.15e}".replace(".", "")[:2]))

    two_digits = np.arange(10, 100)

    actual_two = first_two_digits.value_counts(normalize=True).reindex(two_digits, fill_value=0)

    expected_two = np.log10(1 + 1 / two_digits)

    two_digit_results = pd.DataFrame({"Digits": two_digits,"Actual Percentage": actual_two.values * 100,"Expected Percentage": expected_two * 100})

    st.dataframe(two_digit_results)
    st.subheader("First-Digit Visualization")

    fig1, ax1 = plt.subplots(figsize=(10, 5))

    ax1.bar(digits, actual_first.values * 100, label="Actual")
    ax1.plot(digits, expected_first * 100, color="red", marker="o", label="Expected")

    ax1.set_title("First-Digit Benford Analysis")
    ax1.set_xlabel("First Digit")
    ax1.set_ylabel("Percentage (%)")
    ax1.set_xticks(digits)
    ax1.legend()

    st.pyplot(fig1)
    st.subheader("First-Two-Digit Visualization")

    fig2, ax2 = plt.subplots(figsize=(14, 5))

    ax2.bar(two_digits, actual_two.values * 100, label="Actual")
    ax2.plot(two_digits, expected_two * 100, color="red", label="Expected")

    ax2.set_title("First-Two-Digit Benford Analysis")
    ax2.set_xlabel("First Two Digits")
    ax2.set_ylabel("Percentage (%)")
    ax2.set_xticks(np.arange(10, 100, 5))
    ax2.legend()

    st.pyplot(fig2)
    st.subheader("KS Statistical Analysis")

    critical_value = 1.36 / np.sqrt(len(values))

    cum_diff_first = np.cumsum(actual_first.values) - np.cumsum(expected_first)

    max_diff_first = np.max(np.abs(cum_diff_first))

    cum_diff_two = np.cumsum(actual_two.values) - np.cumsum(expected_two)

    max_diff_two = np.max(np.abs(cum_diff_two))

    st.write("First-Digit KS Statistic:", max_diff_first)

    st.write("First-Two-Digit KS Statistic:", max_diff_two)

    st.write("Critical Value (alpha = 0.05):", critical_value)
    st.subheader("Statistical Conclusions")

    if max_diff_first > critical_value:
        st.warning("First-Digit Analysis: Reject the null hypothesis. The actual distribution is statistically different from Benford's Law.")
    else:
        st.success("First-Digit Analysis: Fail to reject the null hypothesis. There is insufficient evidence of a statistically significant difference.")

    if max_diff_two > critical_value:
        st.warning("First-Two-Digit Analysis: Reject the null hypothesis. The actual distribution is statistically different from Benford's Law.")
    else:
        st.success("First-Two-Digit Analysis: Fail to reject the null hypothesis. There is insufficient evidence of a statistically significant difference.")

    st.info("A statistically significant difference from Benford's Law does not necessarily indicate fraud.")
    st.subheader("Summary Statistics")

    summary_stats = filtered_df[selected_column].describe().to_frame(name="Value")

    summary_stats.loc["median"] = filtered_df[selected_column].median()

    summary_stats.loc["total"] = filtered_df[selected_column].sum()

    st.dataframe(summary_stats)
    st.download_button(label="Download Summary Statistics",data=summary_stats.to_csv(),file_name="summary_statistics.csv",mime="text/csv")
    st.download_button(label="Download First-Digit Analysis",data=first_digit_results.to_csv(index=False),file_name="first_digit_analysis.csv",mime="text/csv")

    st.download_button(label="Download First-Two-Digit Analysis",data=two_digit_results.to_csv(index=False),file_name="first_two_digit_analysis.csv",mime="text/csv")
    import io

    chart1 = io.BytesIO()
    fig1.savefig(chart1, format="png", bbox_inches="tight")

    st.download_button(label="Download First-Digit Chart",data=chart1.getvalue(),file_name="first_digit_chart.png",mime="image/png")

    chart2 = io.BytesIO()
    fig2.savefig(chart2, format="png", bbox_inches="tight")

    st.download_button(label="Download First-Two-Digit Chart",data=chart2.getvalue(),file_name="first_two_digit_chart.png",mime="image/png")