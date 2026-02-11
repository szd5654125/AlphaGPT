import streamlit as st
import time
from data_service import DashboardService
from visualizer import plot_pnl_distribution, plot_market_scatter

st.set_page_config(
    page_title="MemeAlpha Commander",
    page_icon="🐕",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
    }
    .stDataFrame { border: none; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_service():
    return DashboardService()

svc = get_service()

st.sidebar.title("MemeAlpha Bot")
st.sidebar.markdown("---")

with st.sidebar:
    st.subheader("Wallet Status")
    bal = svc.get_wallet_balance()
    st.metric("SOL Balance", f"{bal:.4f} SOL")
    
    st.markdown("---")
    st.subheader("Control Panel")
    if st.button("Refresh Data"):
        st.rerun()
        
    if st.button("EMERGENCY STOP", type="primary"):
        with open("STOP_SIGNAL", "w") as f:
            f.write("STOP")
        st.error("STOP SIGNAL SENT, Process will terminate on next cycle.")

col1, col2, col3, col4 = st.columns(4)
portfolio_df = svc.load_portfolio()
market_df = svc.get_market_overview()
strategy_data = svc.load_strategy_info()

open_positions = len(portfolio_df)
total_invested = portfolio_df['initial_cost_sol'].sum() if not portfolio_df.empty else 0.0

with col1:
    st.metric("Open Positions", f"{open_positions} / 5")
with col2:
    st.metric("Total Invested", f"{total_invested:.2f} SOL")
with col3:
    if not portfolio_df.empty:
        current_val = (portfolio_df['amount_held'] * portfolio_df['highest_price']).sum()
        pnl_sol = current_val - total_invested
        st.metric("Unrealized PnL (Est)", f"{pnl_sol:+.3f} SOL", delta_color="normal")
    else:
        st.metric("Unrealized PnL", "0.00 SOL")
with col4:
    st.metric("Active Strategy", "AlphaGPT-v1", help=str(strategy_data))

tab1, tab2, tab3 = st.tabs(["Portfolio", "Market Scanner", "Logs"])

with tab1:
    st.subheader("Active Holdings")
    if not portfolio_df.empty:
        # Display Table
        display_cols = ['symbol', 'entry_price', 'highest_price', 'amount_held', 'pnl_pct', 'is_moonbag']
        
        # Format for display
        show_df = portfolio_df[display_cols].copy()
        show_df['pnl_pct'] = show_df['pnl_pct'].apply(lambda x: f"{x:.2%}")
        show_df['entry_price'] = show_df['entry_price'].apply(lambda x: f"{x:.6f}")
        
        st.dataframe(show_df, use_container_width=True, hide_index=True)
        
        # Display Chart
        st.plotly_chart(plot_pnl_distribution(portfolio_df), use_container_width=True)
    else:
        st.info("No active positions. The bot is scanning...")

with tab2:
    st.subheader("Top Opportunities (DB Snapshot)")
    if not market_df.empty:
        st.plotly_chart(plot_market_scatter(market_df), use_container_width=True)
        st.dataframe(market_df, use_container_width=True)
    else:
        st.warning("No market data found in DB. Is the Data Pipeline running?")

with tab3:
    st.subheader("System Logs (Tail 20)")
    logs = svc.get_recent_logs(20)
    if logs:
        st.code("".join(logs), language="text")
    else:
        st.caption("No logs found or log file path incorrect.")

time.sleep(1) 
if st.checkbox("Auto-Refresh (30s)", value=True):
    time.sleep(30)
    st.rerun()