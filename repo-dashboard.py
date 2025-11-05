# repo_market_dashboard.py
"""

Implemented with the help of Claude (AI)

Comprehensive Repo Market Liquidity Dashboard
Multi-spread monitoring system for repo market stress indicators
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
from typing import Optional, Dict, List
import time

# Page configuration
st.set_page_config(
    page_title="Repo Market Liquidity Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-critical {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .alert-warning {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .alert-normal {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .info-box {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .knowledge-box {
        background-color: #fff9c4;
        border-left: 4px solid #fbc02d;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)


class RepoMarketDataFetcher:
    """Fetches repo market data from FRED API"""

    def __init__(self):
        self.fred_base_url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
        self.series_config = {
            'SOFR': 'Secured Overnight Financing Rate',
            'EFFR': 'Effective Federal Funds Rate',
            'OBFR': 'Overnight Bank Funding Rate',
            'IORB': 'Interest on Reserve Balances',
            'RRPONTSYAWARD': 'Overnight Reverse Repo Rate',
        }

    @st.cache_data(ttl=3600)
    def fetch_series(_self, series_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch a single series from FRED"""
        try:
            params = {
                'id': series_id,
                'cosd': start_date,
                'coed': end_date
            }
            response = requests.get(_self.fred_base_url, params=params, timeout=10)
            response.raise_for_status()

            df = pd.read_csv(pd.io.common.StringIO(response.text))
            df.columns = ['DATE', series_id]
            df['DATE'] = pd.to_datetime(df['DATE'])
            df[series_id] = pd.to_numeric(df[series_id], errors='coerce')
            return df
        except Exception as e:
            st.error(f"Error fetching {series_id}: {str(e)}")
            return pd.DataFrame()

    def fetch_all_data(self, days_back: int = 365) -> pd.DataFrame:
        """Fetch all required series and merge them"""
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

        df = self.fetch_series('SOFR', start_date, end_date)

        for series_id in ['EFFR', 'OBFR', 'IORB', 'RRPONTSYAWARD']:
            series_df = self.fetch_series(series_id, start_date, end_date)
            if not series_df.empty:
                df = pd.merge(df, series_df, on='DATE', how='outer')

        df = df.sort_values('DATE').reset_index(drop=True)
        df = df.ffill().dropna()

        return df


class SpreadCalculator:
    """Calculate various repo market spreads"""

    @staticmethod
    def calculate_spreads(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all relevant spreads"""
        spreads = df.copy()

        if 'SOFR' in df.columns and 'EFFR' in df.columns:
            spreads['SOFR_EFFR'] = (df['SOFR'] - df['EFFR']) * 100

        if 'SOFR' in df.columns and 'IORB' in df.columns:
            spreads['SOFR_IORB'] = (df['SOFR'] - df['IORB']) * 100

        if 'EFFR' in df.columns and 'IORB' in df.columns:
            spreads['EFFR_IORB'] = (df['EFFR'] - df['IORB']) * 100

        if 'SOFR' in df.columns and 'RRPONTSYAWARD' in df.columns:
            spreads['SOFR_RRP'] = (df['SOFR'] - df['RRPONTSYAWARD']) * 100

        if 'OBFR' in df.columns and 'EFFR' in df.columns:
            spreads['OBFR_EFFR'] = (df['OBFR'] - df['EFFR']) * 100

        if 'OBFR' in df.columns and 'IORB' in df.columns:
            spreads['OBFR_IORB'] = (df['OBFR'] - df['IORB']) * 100

        return spreads

    @staticmethod
    def get_spread_stats(df: pd.DataFrame, spread_name: str) -> Dict:
        """Calculate statistics for a specific spread"""
        if spread_name not in df.columns:
            return {}

        spread_data = df[spread_name].dropna()
        if len(spread_data) == 0:
            return {}

        current = spread_data.iloc[-1]
        mean = spread_data.mean()
        std = spread_data.std()

        stats = {
            'current': current,
            'mean': mean,
            'std': std,
            'min': spread_data.min(),
            'max': spread_data.max(),
            'median': spread_data.median(),
            'z_score': (current - mean) / std if std > 0 else 0,
            'percentile': (spread_data <= current).sum() / len(spread_data) * 100
        }

        return stats


class KnowledgeBase:
    """Repository of educational content about repo market stress"""

    @staticmethod
    def get_spread_explanation(spread_name: str) -> Dict[str, str]:
        """Get explanation for specific spread"""
        explanations = {
            'SOFR_EFFR': {
                'title': '📖 SOFR-EFFR Spread',
                'what': 'The difference between secured (repo) and unsecured (fed funds) overnight rates.',
                'why_widens': """
                **Spread widens when:**
                - Banks hoard cash and won't lend unsecured
                - Counterparty concerns increase
                - Quarter-end/month-end balance sheet constraints
                - Regulatory pressure reduces willingness to lend
                - Treasury settlement needs spike
                """,
                'stress_signals': """
                **Stress Indicators:**
                - Positive spread > 5-10 bps indicates tightness
                - Sustained widening suggests structural issues
                - Volatile swings signal market dysfunction
                """,
                'historical': """
                **Historical Events:**
                - Sept 2019: Spread spiked to 30+ bps during repo crisis
                - March 2020: COVID panic drove spreads to extremes
                - Oct 2025: Recent QT pressures pushing spreads higher
                """
            },
            'SOFR_IORB': {
                'title': '📖 SOFR-IORB Spread',
                'what': 'Measures if repo rates are trading above the Fed\'s floor rate (what banks earn on reserves).',
                'why_widens': """
                **Spread widens when:**
                - Reserve scarcity forces banks to pay up for cash
                - System-wide liquidity tightens
                - Quantitative Tightening (QT) drains reserves
                - Banks need collateral for regulatory reasons
                - Money funds prefer repo over reverse repo
                """,
                'stress_signals': """
                **Stress Indicators:**
                - Positive spread suggests reserves becoming scarce
                - Persistent elevation indicates QT approaching limit
                - Volatility signals uneven reserve distribution
                """,
                'historical': """
                **Historical Events:**
                - Pre-2019: Reserves ample, spread mostly negative
                - Sept 2019: Reserve scarcity caused repo crisis
                - 2025: QT pushing reserves lower, spread widening
                """
            },
            'EFFR_IORB': {
                'title': '📖 EFFR-IORB Spread',
                'what': 'Shows where fed funds trade relative to what the Fed pays on reserves.',
                'why_widens': """
                **Spread widens when:**
                - Banks desperate for unsecured cash
                - System reserves very tight
                - GSEs (Fannie/Freddie) willing to lend at higher rates
                - Bank reluctance to lend pushes rates up
                """,
                'stress_signals': """
                **Stress Indicators:**
                - Should trade slightly above IORB (5-10 bps normal)
                - Large positive spread = severe reserve shortage
                - Negative spread rarely seen, would signal glut
                """,
                'historical': """
                **Historical Events:**
                - 2019: Stayed close to IORB until Sept crisis
                - Sept 2019: Spiked dramatically during crisis
                - QT periods: Tends to drift higher
                """
            },
            'SOFR_RRP': {
                'title': '📖 SOFR-RRP Spread',
                'what': 'Difference between market repo rate and Fed\'s reverse repo facility rate.',
                'why_widens': """
                **Spread widens when:**
                - Strong demand for Treasury collateral
                - Money funds prefer lending in market vs Fed
                - Collateral scarcity drives up repo rates
                - Quarter-end balance sheet effects
                """,
                'stress_signals': """
                **Stress Indicators:**
                - Large positive spread = collateral scarcity
                - Negative spread would mean Fed facility more attractive
                - Widening indicates market stress
                """,
                'historical': """
                **Historical Events:**
                - RRP created post-2008 as Fed policy tool
                - Usage peaked during QE as liquidity flooded system
                - QT period: Usage declining as reserves drain
                """
            },
            'OBFR_EFFR': {
                'title': '📖 OBFR-EFFR Spread',
                'what': 'Overnight Bank Funding Rate vs Fed Funds - captures broader funding including Eurodollars.',
                'why_widens': """
                **Spread widens when:**
                - Offshore dollar funding becomes expensive
                - Eurodollar market stress
                - Foreign banks face dollar shortage
                - Cross-border funding concerns
                """,
                'stress_signals': """
                **Stress Indicators:**
                - Usually very tight (< 2 bps)
                - Widening suggests offshore stress
                - Can signal global dollar shortage
                """,
                'historical': """
                **Historical Events:**
                - March 2020: Widened during COVID dollar scramble
                - Generally stable except during crises
                - Tracks EFFR closely in normal times
                """
            },
            'OBFR_IORB': {
                'title': '📖 OBFR-IORB Spread',
                'what': 'Broader overnight bank funding relative to Fed\'s reserve rate.',
                'why_widens': """
                **Spread widens when:**
                - Similar to EFFR-IORB but includes Eurodollar effects
                - Global dollar funding tightness
                - Combined domestic and offshore stress
                """,
                'stress_signals': """
                **Stress Indicators:**
                - Reflects both US and offshore conditions
                - Wider than EFFR-IORB suggests offshore premium
                """,
                'historical': """
                **Historical Events:**
                - Introduced as broader funding measure
                - Useful for tracking global dollar conditions
                """
            }
        }
        return explanations.get(spread_name, {})

    @staticmethod
    def get_general_stress_factors() -> str:
        """Get general information about repo stress factors"""
        return """
        ### 🎓 Common Causes of Repo Market Stress

        **1. Reserve Scarcity**
        - QT (Quantitative Tightening) drains reserves from system
        - Banks become cautious about lending excess reserves
        - Uneven distribution causes some banks to hoard cash

        **2. Regulatory Constraints**
        - Basel III leverage ratios limit balance sheet expansion
        - Banks reduce intermediation at quarter/year-ends
        - Supplementary Leverage Ratio (SLR) makes repo expensive

        **3. Treasury Issuance**
        - Large Treasury auctions absorb dealer balance sheets
        - Primary dealer inventories create funding needs
        - Settlement dates concentrate demand for cash

        **4. Tax Dates & Government Cash Flows**
        - Treasury General Account (TGA) buildups drain reserves
        - Corporate tax dates shift money from banks to government
        - Creates temporary but significant liquidity drains

        **5. Technical Factors**
        - Month-end and quarter-end window dressing
        - GSIB (Global Systemically Important Bank) scoring dates
        - Money fund rebalancing

        **6. Crisis Events**
        - Flight to quality increases demand for Treasuries
        - Counterparty concerns reduce unsecured lending
        - Margin calls and collateral demands spike
        """

    @staticmethod
    def get_historical_crises() -> str:
        """Get information about historical repo crises"""
        return """
        ### 📚 Historical Repo Market Crises

        **September 2019 Crisis**
        - SOFR spiked to 5.25% (from ~2.5%)
        - EFFR-IORB spread jumped to 20+ bps
        - **Causes**: Corporate tax payments + Treasury settlement + reserve scarcity from QT
        - **Fed Response**: Emergency repo operations ($75B+/day) and restart of balance sheet expansion
        - **Lesson**: "Ample reserves" threshold was higher than expected

        **March 2020 COVID Crisis**
        - Extreme volatility across all funding markets
        - Global dollar shortage as investors fled to cash
        - OBFR-EFFR spread widened significantly
        - **Causes**: Pandemic panic + Treasury market dysfunction
        - **Fed Response**: Massive QE, repo facilities, dollar swap lines
        - **Lesson**: Need robust standing facilities for crises

        **October-November 2025 Stress**
        - Recent elevated SOFR levels and SRF usage
        - Fed injecting liquidity via Standing Repo Facility
        - **Causes**: QT reserves approaching scarcity, government shutdown effects, fiscal pressures
        - **Status**: Ongoing monitoring, QT likely to end soon
        - **Lesson**: Same vulnerabilities as 2019 re-emerging

        **Key Takeaway**: Repo stress often emerges from combination of structural factors (reserves, regulation) 
        plus temporary shocks (tax dates, issuance, crises). Fed must balance QT with maintaining adequate reserves.
        """


class DashboardVisualizer:
    """Create visualizations for the dashboard"""

    @staticmethod
    def create_spread_timeseries(df: pd.DataFrame, spread_name: str,
                                 title: str, highlight_days: int = 60) -> go.Figure:
        """Create interactive time series plot for a spread"""
        if spread_name not in df.columns:
            return go.Figure()

        stats = SpreadCalculator.get_spread_stats(df, spread_name)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df['DATE'],
            y=df[spread_name],
            mode='lines',
            name='Spread',
            line=dict(color='#2E86AB', width=1.5),
            opacity=0.7,
            hovertemplate='<b>Date</b>: %{x}<br><b>Spread</b>: %{y:.2f} bps<extra></extra>'
        ))

        recent_df = df[df['DATE'] >= df['DATE'].max() - timedelta(days=highlight_days)]
        fig.add_trace(go.Scatter(
            x=recent_df['DATE'],
            y=recent_df[spread_name],
            mode='lines',
            name=f'Last {highlight_days} days',
            line=dict(color='#A23B72', width=3),
            hovertemplate='<b>Date</b>: %{x}<br><b>Spread</b>: %{y:.2f} bps<extra></extra>'
        ))

        if stats:
            fig.add_hline(y=stats['mean'], line_dash="dash",
                          line_color="green", opacity=0.7,
                          annotation_text=f"Mean: {stats['mean']:.2f} bps",
                          annotation_position="right")

            fig.add_hrect(
                y0=stats['mean'] + stats['std'],
                y1=stats['mean'] + 2 * stats['std'],
                fillcolor="orange", opacity=0.1,
                annotation_text="Elevated", annotation_position="right"
            )
            fig.add_hrect(
                y0=stats['mean'] + 2 * stats['std'],
                y1=df[spread_name].max() * 1.1,
                fillcolor="red", opacity=0.1,
                annotation_text="Critical", annotation_position="right"
            )

        fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Spread (basis points)",
            hovermode='x unified',
            height=450,
            template='plotly_white',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        return fig

    @staticmethod
    def create_multi_spread_comparison(df: pd.DataFrame, spread_names: List[str]) -> go.Figure:
        """Create comparison chart for multiple spreads"""
        fig = go.Figure()

        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4B51']

        for idx, spread in enumerate(spread_names):
            if spread in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['DATE'],
                    y=df[spread],
                    mode='lines',
                    name=spread.replace('_', '-'),
                    line=dict(color=colors[idx % len(colors)], width=2),
                    hovertemplate='<b>%{fullData.name}</b><br>%{y:.2f} bps<extra></extra>'
                ))

        fig.update_layout(
            title="Multi-Spread Comparison",
            xaxis_title="Date",
            yaxis_title="Spread (basis points)",
            hovermode='x unified',
            height=500,
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        return fig

    @staticmethod
    def create_distribution_plot(df: pd.DataFrame, spread_name: str) -> go.Figure:
        """Create distribution histogram for a spread"""
        if spread_name not in df.columns:
            return go.Figure()

        stats = SpreadCalculator.get_spread_stats(df, spread_name)

        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=df[spread_name].dropna(),
            nbinsx=50,
            name='Distribution',
            marker_color='#2E86AB',
            opacity=0.7,
            hovertemplate='<b>Range</b>: %{x}<br><b>Count</b>: %{y}<extra></extra>'
        ))

        if stats:
            fig.add_vline(x=stats['current'], line_dash="solid",
                          line_color="red", line_width=2,
                          annotation_text=f"Current: {stats['current']:.2f}",
                          annotation_position="top")

            fig.add_vline(x=stats['mean'], line_dash="dash",
                          line_color="green", line_width=2,
                          annotation_text=f"Mean: {stats['mean']:.2f}",
                          annotation_position="bottom")

        fig.update_layout(
            title=f"{spread_name.replace('_', '-')} Distribution",
            xaxis_title="Spread (basis points)",
            yaxis_title="Frequency",
            height=350,
            template='plotly_white',
            showlegend=False
        )

        return fig

    @staticmethod
    def create_correlation_heatmap(df: pd.DataFrame, spread_names: List[str]) -> go.Figure:
        """Create correlation heatmap for multiple spreads"""
        available_spreads = [s for s in spread_names if s in df.columns]

        if len(available_spreads) < 2:
            return go.Figure()

        corr_matrix = df[available_spreads].corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=[s.replace('_', '-') for s in corr_matrix.columns],
            y=[s.replace('_', '-') for s in corr_matrix.index],
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation"),
            hovertemplate='<b>%{y} vs %{x}</b><br>Correlation: %{z:.2f}<extra></extra>'
        ))

        fig.update_layout(
            title="Spread Correlation Matrix",
            height=500,
            template='plotly_white'
        )

        return fig


def get_stress_level(z_score: float) -> tuple:
    """Determine stress level based on z-score"""
    if abs(z_score) > 2:
        return ("CRITICAL", "🔴", "alert-critical")
    elif abs(z_score) > 1:
        return ("ELEVATED", "🟡", "alert-warning")
    else:
        return ("NORMAL", "🟢", "alert-normal")


def main():
    """Main dashboard application"""

    st.markdown('<h1 class="main-header">📊 Repo Market Liquidity Monitor</h1>',
                unsafe_allow_html=True)
    st.markdown("### Real-time tracking of repo market spreads and liquidity conditions")

    st.sidebar.title("⚙️ Dashboard Controls")

    days_back = st.sidebar.slider(
        "Historical Data Range (days)",
        min_value=30,
        max_value=730,
        value=365,
        step=30
    )

    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    auto_refresh = st.sidebar.checkbox("Auto-refresh (5 min)", value=False)
    if auto_refresh:
        time.sleep(300)
        st.rerun()

    st.sidebar.markdown("---")

    with st.sidebar.expander("📚 Understanding Repo Stress", expanded=False):
        st.markdown("""
        **What are we monitoring?**

        Spreads between different overnight rates that indicate 
        liquidity conditions in the financial system.

        **Why does it matter?**

        - Wide spreads = funding stress
        - Can predict financial instability
        - Guides Fed policy decisions

        **Key Events:**
        - Sept 2019: Repo crisis
        - March 2020: COVID stress
        - Nov 2025: Current monitoring
        """)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This dashboard monitors key repo market spreads to assess liquidity conditions. "
        "Data is sourced from FRED (Federal Reserve Economic Data)."
    )

    with st.spinner("Fetching latest repo market data..."):
        fetcher = RepoMarketDataFetcher()
        df = fetcher.fetch_all_data(days_back=days_back)

        if df.empty:
            st.error("Unable to fetch data. Please try again later.")
            return

        df = SpreadCalculator.calculate_spreads(df)

    st.sidebar.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.sidebar.markdown(f"**Latest Data:** {df['DATE'].max().strftime('%Y-%m-%d')}")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview",
        "📊 Individual Spreads",
        "🔍 Detailed Analysis",
        "📉 Rate Levels",
        "🎓 Knowledge Base"
    ])

    with tab1:
        st.header("Market Overview")

        spread_columns = [col for col in df.columns if '_' in col and col not in ['DATE']]

        if spread_columns:
            cols = st.columns(min(len(spread_columns), 6))

            for idx, spread in enumerate(spread_columns):
                stats = SpreadCalculator.get_spread_stats(df, spread)
                if stats:
                    stress_level, emoji, _ = get_stress_level(stats['z_score'])

                    with cols[idx % 6]:
                        st.metric(
                            label=f"{emoji} {spread.replace('_', '-')}",
                            value=f"{stats['current']:.2f} bps",
                            delta=f"{stats['current'] - stats['mean']:.2f} vs mean",
                            help=f"Z-score: {stats['z_score']:.2f} | Status: {stress_level}"
                        )

        st.markdown("---")

        st.subheader("Spread Comparison")
        if spread_columns:
            fig_comparison = DashboardVisualizer.create_multi_spread_comparison(
                df, spread_columns
            )
            st.plotly_chart(fig_comparison, use_container_width=True)

        with st.expander("💡 What Causes These Spreads to Widen?", expanded=False):
            st.markdown(KnowledgeBase.get_general_stress_factors())

        st.markdown("---")

        st.subheader("🚨 Liquidity Alerts")
        alerts = []
        for spread in spread_columns:
            stats = SpreadCalculator.get_spread_stats(df, spread)
            if stats and abs(stats['z_score']) > 1.5:
                stress_level, emoji, css_class = get_stress_level(stats['z_score'])
                alerts.append({
                    'spread': spread,
                    'level': stress_level,
                    'emoji': emoji,
                    'z_score': stats['z_score'],
                    'current': stats['current'],
                    'css': css_class
                })

        if alerts:
            for alert in sorted(alerts, key=lambda x: abs(x['z_score']), reverse=True):
                st.markdown(
                    f'<div class="{alert["css"]}">'
                    f'<strong>{alert["emoji"]} {alert["spread"].replace("_", "-")}</strong>: '
                    f'{alert["level"]} - Current spread: {alert["current"]:.2f} bps '
                    f'(Z-score: {alert["z_score"]:.2f})'
                    f'</div>',
                    unsafe_allow_html=True
                )
        else:
            st.success("✅ All spreads are within normal ranges")

    with tab2:
        st.header("Individual Spread Analysis")

        if spread_columns:
            selected_spread = st.selectbox(
                "Select Spread to Analyze",
                spread_columns,
                format_func=lambda x: x.replace('_', '-')
            )

            knowledge = KnowledgeBase.get_spread_explanation(selected_spread)

            if knowledge:
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown(f'<div class="knowledge-box">', unsafe_allow_html=True)
                    st.markdown(f"## {knowledge['title']}")
                    st.markdown(f"**What it measures:** {knowledge['what']}")
                    st.markdown(knowledge['why_widens'])
                    st.markdown('</div>', unsafe_allow_html=True)

                with col2:
                    st.markdown(f'<div class="info-box">', unsafe_allow_html=True)
                    st.markdown(knowledge['stress_signals'])
                    st.markdown('</div>', unsafe_allow_html=True)

                with st.expander("📚 Historical Context", expanded=False):
                    st.markdown(knowledge['historical'])

            st.markdown("---")

            stats = SpreadCalculator.get_spread_stats(df, selected_spread)

            if stats:
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Current", f"{stats['current']:.2f} bps")
                with col2:
                    st.metric("Mean", f"{stats['mean']:.2f} bps")
                with col3:
                    st.metric("Std Dev", f"{stats['std']:.2f} bps")
                with col4:
                    stress_level, emoji, _ = get_stress_level(stats['z_score'])
                    st.metric(f"{emoji} Status", stress_level)

                st.subheader("Time Series")
                fig_ts = DashboardVisualizer.create_spread_timeseries(
                    df, selected_spread,
                    f"{selected_spread.replace('_', '-')} Spread Over Time"
                )
                st.plotly_chart(fig_ts, use_container_width=True)

                col1, col2 = st.columns(2)

                with col1:
                    fig_dist = DashboardVisualizer.create_distribution_plot(df, selected_spread)
                    st.plotly_chart(fig_dist, use_container_width=True)

                with col2:
                    st.subheader("Statistics Summary")

                    stats_df = pd.DataFrame({
                        'Metric': ['Current', 'Mean', 'Median', 'Std Dev',
                                   'Min', 'Max', 'Z-Score', 'Percentile'],
                        'Value': [
                            f"{stats['current']:.2f} bps",
                            f"{stats['mean']:.2f} bps",
                            f"{stats['median']:.2f} bps",
                            f"{stats['std']:.2f} bps",
                            f"{stats['min']:.2f} bps",
                            f"{stats['max']:.2f} bps",
                            f"{stats['z_score']:.2f}",
                            f"{stats['percentile']:.1f}%"
                        ]
                    })

                    st.dataframe(stats_df, use_container_width=True, hide_index=True)

                    st.markdown('<div class="info-box">', unsafe_allow_html=True)
                    st.markdown("**💡 Interpretation:**")

                    if stats['z_score'] > 2:
                        st.markdown("""
                                - 🔴 **Critical stress level**
                                - Spread is unusually high
                                - Similar to crisis conditions
                                - Fed intervention may be needed
                                """)
                    elif stats['z_score'] > 1:
                        st.markdown("""
                                - 🟡 **Elevated conditions**
                                - Above historical average
                                - Monitor for deterioration
                                - Some liquidity tightness
                                """)
                    elif stats['z_score'] < -1:
                        st.markdown("""
                                - 🔵 **Below normal**
                                - Unusually tight spread
                                - Ample liquidity in system
                                - May indicate QE effects
                                """)
                    else:
                        st.markdown("""
                                - 🟢 **Normal conditions**
                                - Within historical range
                                - Stable liquidity
                                - No immediate concerns
                                """)

                    st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.header("Detailed Analysis")

        st.markdown("""
            <div class="knowledge-box">
            <strong>🎓 About Cross-Spread Analysis:</strong><br>
            Understanding how different spreads correlate helps identify whether stress is isolated or systemic.
            High correlation means spreads move together (systemic issue), low correlation suggests specific market segments affected.
            </div>
            """, unsafe_allow_html=True)

        if len(spread_columns) >= 2:
            st.subheader("Spread Correlation Analysis")
            fig_corr = DashboardVisualizer.create_correlation_heatmap(df, spread_columns)
            st.plotly_chart(fig_corr, use_container_width=True)

            with st.expander("💡 Understanding Correlations", expanded=False):
                st.markdown("""
                    **High Positive Correlation (>0.7):**
                    - Spreads move together
                    - Suggests common underlying factor (e.g., reserve scarcity)
                    - Indicates systemic stress

                    **Low Correlation (<0.3):**
                    - Spreads move independently
                    - Different market segments affected
                    - May indicate specific issues (e.g., Eurodollar vs domestic)

                    **Negative Correlation:**
                    - Rare but important
                    - One market easing while another tightens
                    - May indicate flight-to-quality or segmentation
                    """)

        st.markdown("---")

        st.subheader("Rolling Statistics & Volatility")

        with st.expander("💡 Why Rolling Statistics Matter", expanded=False):
            st.markdown("""
                **Rolling averages smooth out daily noise to reveal trends:**
                - Rising trend = increasing stress
                - Falling trend = improving conditions
                - Widening bands = increasing volatility/uncertainty

                **Volatility (standard deviation) signals:**
                - High volatility = market stress and uncertainty
                - Low volatility = stable conditions
                - Spiking volatility = crisis conditions
                """)

        if spread_columns:
            selected_for_rolling = st.selectbox(
                "Select spread for rolling analysis",
                spread_columns,
                format_func=lambda x: x.replace('_', '-'),
                key='rolling_select'
            )

            window = st.slider("Rolling window (days)", 5, 90, 30)

            rolling_mean = df[selected_for_rolling].rolling(window=window).mean()
            rolling_std = df[selected_for_rolling].rolling(window=window).std()

            fig_rolling = go.Figure()

            fig_rolling.add_trace(go.Scatter(
                x=df['DATE'], y=df[selected_for_rolling],
                mode='lines', name='Actual Spread',
                line=dict(color='lightgray', width=1),
                opacity=0.5,
                hovertemplate='<b>Actual</b>: %{y:.2f} bps<extra></extra>'
            ))

            fig_rolling.add_trace(go.Scatter(
                x=df['DATE'], y=rolling_mean,
                mode='lines', name=f'{window}-day MA',
                line=dict(color='#2E86AB', width=3),
                hovertemplate='<b>Rolling Mean</b>: %{y:.2f} bps<extra></extra>'
            ))

            fig_rolling.add_trace(go.Scatter(
                x=df['DATE'],
                y=rolling_mean + rolling_std,
                mode='lines', name='+1 Std Dev',
                line=dict(width=0),
                showlegend=False,
                hovertemplate='<b>+1σ</b>: %{y:.2f} bps<extra></extra>'
            ))

            fig_rolling.add_trace(go.Scatter(
                x=df['DATE'],
                y=rolling_mean - rolling_std,
                mode='lines', name='±1 Std Dev',
                line=dict(width=0),
                fillcolor='rgba(46, 134, 171, 0.2)',
                fill='tonexty',
                hovertemplate='<b>-1σ</b>: %{y:.2f} bps<extra></extra>'
            ))

            fig_rolling.update_layout(
                title=f"{selected_for_rolling.replace('_', '-')} - {window}-Day Rolling Statistics",
                xaxis_title="Date",
                yaxis_title="Spread (basis points)",
                height=450,
                template='plotly_white',
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            st.plotly_chart(fig_rolling, use_container_width=True)

            st.subheader("Volatility Analysis")

            col1, col2 = st.columns(2)

            with col1:
                fig_vol = go.Figure()
                fig_vol.add_trace(go.Scatter(
                    x=df['DATE'],
                    y=rolling_std,
                    mode='lines',
                    name='Rolling Std Dev',
                    line=dict(color='#F18F01', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(241, 143, 1, 0.2)'
                ))

                fig_vol.update_layout(
                    title=f"{window}-Day Rolling Volatility",
                    xaxis_title="Date",
                    yaxis_title="Standard Deviation (bps)",
                    height=350,
                    template='plotly_white'
                )

                st.plotly_chart(fig_vol, use_container_width=True)

            with col2:
                recent_vol = rolling_std.tail(30).mean()
                historical_vol = rolling_std.mean()

                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.markdown("### 📊 Volatility Summary")
                st.metric(
                    "Recent Volatility (30d avg)",
                    f"{recent_vol:.2f} bps",
                    delta=f"{((recent_vol / historical_vol - 1) * 100):.1f}% vs historical"
                )
                st.metric(
                    "Historical Average",
                    f"{historical_vol:.2f} bps"
                )

                if recent_vol > historical_vol * 1.5:
                    st.warning("⚠️ Volatility significantly elevated")
                elif recent_vol > historical_vol * 1.2:
                    st.info("ℹ️ Volatility moderately elevated")
                else:
                    st.success("✅ Volatility within normal range")

                st.markdown('</div>', unsafe_allow_html=True)

    with tab4:
        st.header("Underlying Rate Levels")

        st.markdown("""
            <div class="knowledge-box">
            <strong>🎓 Understanding Rate Levels:</strong><br>
            These are the absolute levels of different overnight rates. Spreads are derived from differences between these rates.
            Watching absolute levels helps understand Fed policy stance and overall rate environment.
            </div>
            """, unsafe_allow_html=True)

        rate_columns = ['SOFR', 'EFFR', 'OBFR', 'IORB', 'RRPONTSYAWARD']
        available_rates = [r for r in rate_columns if r in df.columns]

        if available_rates:
            st.subheader("Current Rate Levels")
            cols = st.columns(len(available_rates))

            rate_descriptions = {
                'SOFR': 'Secured repo rate',
                'EFFR': 'Fed funds rate',
                'OBFR': 'Bank funding rate',
                'IORB': 'Fed floor rate',
                'RRPONTSYAWARD': 'Reverse repo rate'
            }

            for idx, rate in enumerate(available_rates):
                with cols[idx]:
                    current_rate = df[rate].iloc[-1]
                    prev_rate = df[rate].iloc[-2] if len(df) > 1 else current_rate
                    st.metric(
                        label=rate,
                        value=f"{current_rate:.3f}%",
                        delta=f"{(current_rate - prev_rate):.3f}%",
                        help=rate_descriptions.get(rate, '')
                    )

            st.markdown("---")

            st.subheader("Rate Comparison Over Time")

            selected_rates = st.multiselect(
                "Select rates to display",
                available_rates,
                default=available_rates[:4]
            )

            if selected_rates:
                fig_rates = go.Figure()

                colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D',
                          '#6A994E', '#BC4B51', '#8D5B4C']

                for idx, rate in enumerate(selected_rates):
                    fig_rates.add_trace(go.Scatter(
                        x=df['DATE'],
                        y=df[rate],
                        mode='lines',
                        name=rate,
                        line=dict(color=colors[idx % len(colors)], width=2),
                        hovertemplate=f'<b>{rate}</b>: %{{y:.3f}}%<extra></extra>'
                    ))

                fig_rates.update_layout(
                    title="Rate Levels Comparison",
                    xaxis_title="Date",
                    yaxis_title="Rate (%)",
                    height=500,
                    template='plotly_white',
                    hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )

                st.plotly_chart(fig_rates, use_container_width=True)

                with st.expander("💡 Interpreting Rate Relationships", expanded=False):
                    st.markdown("""
                        **Normal Hierarchy (from highest to lowest):**
                        1. **SOFR** - Highest (secured lending has term premium)
                        2. **EFFR** - Middle (unsecured but among banks)
                        3. **IORB** - Floor (what Fed pays on reserves)
                        4. **RRP** - Lowest (Fed's reverse repo facility)

                        **What disruptions mean:**
                        - **SOFR > EFFR significantly**: Collateral/repo stress
                        - **EFFR < IORB**: Extreme liquidity abundance (rare)
                        - **EFFR > IORB significantly**: Reserve scarcity
                        - **Rates converging**: System stress or policy transition

                        **Fed Policy:**
                        - Fed sets IORB and RRP to create a "corridor"
                        - Market rates should trade within this corridor
                        - Breaking above = stress, breaking below = excess liquidity
                        """)

            st.markdown("---")

            st.subheader("Recent Rate History")
            recent_data = df[['DATE'] + available_rates].tail(10).sort_values('DATE', ascending=False)
            recent_data['DATE'] = recent_data['DATE'].dt.strftime('%Y-%m-%d')

            st.dataframe(
                recent_data.style.format({rate: '{:.3f}%' for rate in available_rates}),
                use_container_width=True,
                hide_index=True
            )

    with tab5:
        st.header("🎓 Knowledge Base")

        st.markdown("""
            <div class="main-header" style="font-size: 1.8rem;">
            Understanding Repo Market Stress
            </div>
            """, unsafe_allow_html=True)

        with st.expander("📚 Historical Repo Market Crises", expanded=True):
            st.markdown(KnowledgeBase.get_historical_crises())

        with st.expander("🔍 Common Stress Factors", expanded=False):
            st.markdown(KnowledgeBase.get_general_stress_factors())

        with st.expander("🏛️ Fed Policy Response Tools", expanded=False):
            st.markdown("""
                ### Federal Reserve Liquidity Tools

                **1. Standing Repo Facility (SRF)**
                - Allows banks to borrow overnight (rate adjusts with policy)
                - Automatic backstop - no Fed discretion needed
                - Usage indicates stress when significantly elevated
                - Created after 2019 crisis to prevent repeat

                **2. Discount Window**
                - Emergency lending to banks
                - Carries stigma - banks reluctant to use
                - Higher rate than SRF

                **3. Reverse Repo Facility (RRP)**
                - Allows money funds to park cash at Fed
                - Sets floor for money market rates
                - High usage indicates excess liquidity
                - Declining usage suggests QT working

                **4. Open Market Operations**
                - Emergency repo operations (like Sept 2019)
                - Fed can inject massive liquidity quickly
                - Temporary measure for acute stress

                **5. Quantitative Easing (QE)**
                - Large-scale asset purchases
                - Permanently expands reserves
                - Used in major crises (2008, 2020)

                **6. Interest on Reserve Balances (IORB)**
                - Sets floor for overnight rates
                - Can be adjusted to influence spreads
                - Rate set by Fed policy (typically at top of fed funds range)

                ### When Does Fed Intervene?

                - **SOFR consistently > 4.50%**: Triggers automatic SRF usage
                - **Sustained spread widening**: Suggests structural issues
                - **Market dysfunction**: Unable to trade, extreme volatility
                - **Financial stability risk**: Threatens broader system
                """)

        with st.expander("📖 Glossary of Terms", expanded=False):
            st.markdown("""
                ### Key Terms Explained

                **SOFR (Secured Overnight Financing Rate)**
                - Broadest measure of overnight Treasury repo rates
                - Based on actual transactions (~$1 trillion daily)
                - Replaced LIBOR as benchmark rate

                **EFFR (Effective Federal Funds Rate)**
                - Volume-weighted median of unsecured overnight lending
                - Between banks and GSEs
                - Fed's primary policy rate

                **OBFR (Overnight Bank Funding Rate)**
                - Broader than EFFR, includes Eurodollar deposits
                - Captures offshore dollar funding

                **IORB (Interest on Reserve Balances)**
                - Rate Fed pays banks on reserves
                - Sets floor for overnight rates
                - Key monetary policy tool

                **Reverse Repo (RRP)**
                - Fed facility where money funds park cash
                - Alternative to bank deposits
                - Provides policy rate floor

                **Basis Point (bp)**
                - 1/100th of 1% (0.01%)
                - Standard unit for rate differences
                - Example: 4.50% - 4.40% = 10 bps

                **Z-Score**
                - Statistical measure of how unusual current level is
                - Number of standard deviations from mean
                - |Z| > 2 indicates unusual/stressed conditions

                **QT (Quantitative Tightening)**
                - Fed shrinking balance sheet
                - Drains reserves from system
                - Can cause repo stress if too aggressive

                **QE (Quantitative Easing)**
                - Fed expanding balance sheet
                - Adds reserves to system
                - Used to ease financial conditions

                **Repo (Repurchase Agreement)**
                - Short-term collateralized loan
                - Borrower sells securities, agrees to repurchase
                - Core funding mechanism for financial system

                **Reserves**
                - Bank deposits held at Federal Reserve
                - Required for settlement and regulation
                - Scarcity causes repo market stress

                **Treasury General Account (TGA)**
                - Government's checking account at Fed
                - Buildups drain reserves from banks
                - Can cause temporary stress

                **Leverage Ratio**
                - Bank capital ÷ total assets
                - Regulation limits balance sheet growth
                - Makes repo intermediation expensive at quarter-ends
                """)

        with st.expander("👁️ What to Watch Now", expanded=False):
            st.markdown("""
                ### Current Market Monitoring (November 2025)

                **Key Indicators:**

                1. **SRF Usage**
                   - Watch for sustained elevation above $10B
                   - Daily spikes > $20B indicate acute stress
                   - Check New York Fed data releases

                2. **SOFR-IORB Spread**
                   - Normal: -5 to +10 bps
                   - Elevated: +10 to +20 bps
                   - Critical: > +20 bps sustained

                3. **SOFR-EFFR Spread**
                   - Normal: 0 to +5 bps
                   - Elevated: +5 to +15 bps
                   - Critical: > +15 bps (like Sept 2019)

                4. **Reserve Levels**
                   - Monitor Fed H.4.1 weekly report
                   - Watch for decline below $3 trillion
                   - Compare to RRP usage

                5. **QT Pace**
                   - Currently $60B/month Treasury runoff
                   - Fed may slow or stop if stress increases
                   - FOMC statements for guidance

                **Calendar Dates to Watch:**

                - **Month-ends**: Balance sheet pressures
                - **Quarter-ends**: Regulatory reporting dates
                - **Mid-March, June, Sept, Dec**: Tax payment dates
                - **Major Treasury auctions**: Settlement needs
                - **FOMC meetings**: Policy change announcements

                **Resources for Monitoring:**

                - New York Fed Markets webpage
                - Fed H.4.1 weekly balance sheet report
                - Treasury TGA balance
                - Primary dealer statistics
                - Money market fund flows
                """)

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Data Source:** Federal Reserve Economic Data (FRED)")
    with col2:
        st.markdown(f"**Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    with col3:
        st.markdown("**© 2025 Repo Market Monitor**")

if __name__ == "__main__":
    main()