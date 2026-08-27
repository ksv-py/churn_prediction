from pathlib import Path

import plotly.express as px
import streamlit as st

from analysis import SOURCE_NAME, SOURCE_URL, load_churn_data, portfolio_metrics, train_churn_model


st.set_page_config(page_title="Retention Desk", page_icon="RD", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: 0; }
    .hero { padding: 1.2rem 0 1rem; border-bottom: 1px solid #d8dce1; }
    .eyebrow { color: #bf4f3f; font-weight: 700; text-transform: uppercase; letter-spacing: .12em; font-size: .72rem; }
    .hero h1 { color: #202b3c; font-size: 2.7rem; margin: .25rem 0; }
    .hero p { color: #586474; max-width: 700px; font-size: 1.04rem; }
    [data-testid="stMetricValue"] { color: #202b3c; }
    .note { border-left: 4px solid #bf4f3f; background: #fff4f0; padding: .75rem 1rem; color: #5e4039; }
    </style>
    """, unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Retention Desk / churn intelligence</div>
      <h1>Know who needs a reason to stay.</h1>
      <p>A focused view of customer attrition, contract pressure, tenure, and revenue exposure for the next retention conversation.</p>
    </div>
    """, unsafe_allow_html=True,
)

source_path = Path(__file__).parent / "data" / SOURCE_NAME
if not source_path.exists():
    st.error(f"The original reference file {SOURCE_NAME} is not in the data folder.")
    st.markdown('<div class="note">The reference repository documents this Excel export but does not commit the binary file. Add the original file to data/ to activate the dashboard.</div>', unsafe_allow_html=True)
    st.link_button("Open Kaggle dataset", SOURCE_URL)
    st.stop()

try:
    data = load_churn_data(source_path)
except Exception as error:
    st.error(str(error))
    st.stop()

with st.sidebar:
    st.header("Focus the book")
    states = st.multiselect("State", sorted(data["State"].dropna().unique()), default=sorted(data["State"].dropna().unique()))
    contracts = st.multiselect("Plan", sorted(data["Contract"].dropna().unique()), default=sorted(data["Contract"].dropna().unique()))
    show_churned = st.checkbox("Show churned customers", value=True)

view = data[data["State"].isin(states) & data["Contract"].isin(contracts)].copy()
if not show_churned:
    view = view[view["churned"].eq(0)]
metrics = portfolio_metrics(view)

cards = st.columns(4)
cards[0].metric("Customers", f"{int(metrics['customers']):,}")
cards[1].metric("Churn rate", f"{metrics['churn_rate']:.1%}")
cards[2].metric("Revenue at stake", f"${metrics['at_risk_revenue']:,.0f}")
cards[3].metric("Avg. tenure", f"{metrics['avg_tenure']:.1f} mo")
st.caption("Source: original reference Excel export. Revenue at stake is observed total revenue for customers labelled Churned, not predicted loss.")

left, right = st.columns([1.1, 1])
with left:
    st.subheader("Where churn concentrates")
    by_state = view.groupby("State", as_index=False).agg(customers=("churned", "size"), churn_rate=("churned", "mean"), revenue_at_stake=("Total_Revenue", lambda values: values[view.loc[values.index, "churned"].eq(1)].sum())).sort_values("churn_rate", ascending=False)
    fig = px.bar(by_state.head(12), x="churn_rate", y="State", orientation="h", text_auto=".1%", color="churn_rate", color_continuous_scale=["#dce8f3", "#bf4f3f"])
    fig.update_layout(xaxis_tickformat=".0%", xaxis_title="Churn rate", yaxis_title=None, coloraxis_showscale=False, height=390, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Contract pressure")
    contract_view = view.groupby("Contract", as_index=False).agg(churn_rate=("churned", "mean"), customers=("churned", "size"))
    fig = px.bar(contract_view, x="Contract", y="churn_rate", text_auto=".1%", color="Contract", color_discrete_sequence=["#315f78", "#bf4f3f", "#d9984a"])
    fig.update_layout(yaxis_tickformat=".0%", yaxis_title="Churn rate", xaxis_title=None, showlegend=False, height=390, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Tenure and revenue exposure")
tenure_view = view.groupby("tenure_band", as_index=False, observed=True).agg(customers=("churned", "size"), churn_rate=("churned", "mean"), revenue=("Total_Revenue", "sum"))
fig = px.bar(tenure_view, x="tenure_band", y="churn_rate", color="revenue", text_auto=".1%", color_continuous_scale=["#e3eef4", "#315f78"])
fig.update_layout(yaxis_tickformat=".0%", yaxis_title="Churn rate", xaxis_title=None, height=320, coloraxis_colorbar_title="Revenue", margin=dict(t=10, b=10))
st.plotly_chart(fig, use_container_width=True)

with st.expander("Model check"):
    if view["churned"].nunique() == 2:
        _, roc_auc = train_churn_model(data)
        st.write(f"Random forest baseline ROC AUC: **{roc_auc:.2f}** on a stratified held-out sample.")
        st.caption("This is a prioritisation baseline, not an automated decision. Validate with time-based evaluation, calibration, and retention experiments before production use.")
    else:
        st.info("Select both churned and non-churned customers to evaluate the model.")

st.subheader("Retention queue")
queue_columns = [column for column in ["Customer_ID", "State", "Contract", "Tenure_in_Months", "Monthly_Charge", "Total_Revenue", "Churn_Category", "Churn_Reason"] if column in view.columns]
queue = view[view["churned"].eq(1)][queue_columns].sort_values("Total_Revenue", ascending=False).head(25)
st.dataframe(queue, use_container_width=True, hide_index=True)
