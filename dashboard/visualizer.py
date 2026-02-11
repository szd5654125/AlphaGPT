import plotly.express as px
import plotly.graph_objects as go

def plot_pnl_distribution(portfolio_df):
    if portfolio_df.empty:
        return go.Figure()
    
    colors = ['#00FF00' if x > 0 else '#FF0000' for x in portfolio_df['pnl_pct']]
    
    fig = go.Figure(data=[go.Bar(
        x=portfolio_df['symbol'],
        y=portfolio_df['pnl_pct'],
        marker_color=colors
    )])
    
    fig.update_layout(
        title="Current Positions PnL %",
        yaxis_tickformat='.2%',
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def plot_market_scatter(market_df):
    if market_df.empty: return go.Figure()
    
    fig = px.scatter(
        market_df,
        x="liquidity",
        y="volume",
        size="fdv",
        color="symbol",
        hover_name="symbol",
        log_x=True,
        log_y=True,
        title="Market Liquidity vs Volume (Bubble Size = FDV)",
        template="plotly_dark"
    )
    return fig