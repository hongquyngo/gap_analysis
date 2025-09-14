# utils/gap/charts.py

"""
Visualization module for GAP Analysis System
Provides chart components using Plotly and Altair
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, List, Optional, Any
import numpy as np

class GAPCharts:
    """Creates visualization components for GAP analysis"""
    
    def __init__(self, formatter):
        """
        Initialize charts with formatter
        
        Args:
            formatter: Instance of GAPFormatter for consistent formatting
        """
        self.formatter = formatter
        self.color_scheme = {
            'severe_shortage': '#FF4444',
            'high_shortage': '#FF8800',
            'low_shortage': '#FFAA00',
            'balanced': '#00AA00',
            'surplus': '#0088FF',
            'high_surplus': '#8800FF'
        }
    
    def create_status_pie_chart(self, gap_df: pd.DataFrame) -> go.Figure:
        """
        Create pie chart showing distribution of items by GAP status
        
        Args:
            gap_df: DataFrame with GAP calculations
            
        Returns:
            Plotly figure object
        """
        # Count items by status
        status_counts = gap_df['gap_status'].value_counts()
        
        # Map to display labels and colors
        labels = []
        values = []
        colors = []
        
        status_labels = {
            'severe_shortage': '🔴 Severe Shortage',
            'high_shortage': '🟠 High Shortage',
            'low_shortage': '🟡 Low Shortage',
            'balanced': '✅ Balanced',
            'surplus': '🔵 Surplus',
            'high_surplus': '🟣 High Surplus'
        }
        
        for status, count in status_counts.items():
            labels.append(status_labels.get(status, status))
            values.append(count)
            colors.append(self.color_scheme.get(status, '#888888'))
        
        fig = go.Figure(data=[
            go.Pie(
                labels=labels,
                values=values,
                marker=dict(colors=colors),
                hole=0.3,  # Donut chart
                textinfo='label+percent+value',
                textposition='auto',
                hovertemplate='<b>%{label}</b><br>' +
                             'Count: %{value}<br>' +
                             'Percentage: %{percent}<br>' +
                             '<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title={
                'text': 'Distribution by GAP Status',
                'x': 0.5,
                'xanchor': 'center'
            },
            height=400,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.05
            )
        )
        
        return fig
    
    def create_top_shortage_bar_chart(self, gap_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
        """
        Create bar chart showing top shortage items
        
        Args:
            gap_df: DataFrame with GAP calculations
            top_n: Number of top items to show
            
        Returns:
            Plotly figure object
        """
        # Get top shortage items
        shortage_df = gap_df[gap_df['net_gap'] < 0].copy()
        
        if shortage_df.empty:
            # No shortage items - create empty chart with message
            fig = go.Figure()
            fig.add_annotation(
                text="No shortage items found",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(
                title="Top Shortage Items",
                height=400,
                xaxis=dict(visible=False),
                yaxis=dict(visible=False)
            )
            return fig
        
        # Sort by absolute shortage and get top N
        shortage_df['abs_gap'] = abs(shortage_df['net_gap'])
        top_items = shortage_df.nlargest(top_n, 'abs_gap')
        
        # Prepare display names
        if 'product_name' in top_items.columns:
            display_names = top_items.apply(
                lambda x: f"{x.get('pt_code', '')} - {x['product_name'][:25]}..." 
                          if len(x.get('product_name', '')) > 25 
                          else f"{x.get('pt_code', '')} - {x.get('product_name', '')}",
                axis=1
            )
        elif 'brand' in top_items.columns:
            display_names = top_items['brand']
        else:
            display_names = top_items.index.astype(str)
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=top_items['abs_gap'],
                y=display_names,
                orientation='h',
                marker=dict(
                    color=top_items['gap_status'].map(self.color_scheme),
                    line=dict(width=1, color='rgba(0,0,0,0.3)')
                ),
                text=top_items['abs_gap'].apply(lambda x: self.formatter.format_number(x)),
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>' +
                             'Shortage: %{x:,.0f} units<br>' +
                             '<extra></extra>',
                customdata=top_items[['gap_percentage', 'total_demand']],
                hovertext=[
                    f"GAP: {row['gap_percentage']:.1f}%<br>" +
                    f"Demand: {self.formatter.format_number(row['total_demand'])}"
                    for _, row in top_items.iterrows()
                ]
            )
        ])
        
        fig.update_layout(
            title={
                'text': f'Top {top_n} Shortage Items',
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title="Shortage Quantity (units)",
            yaxis_title="",
            height=max(400, top_n * 40),  # Dynamic height based on items
            showlegend=False,
            yaxis=dict(autorange="reversed"),  # Worst at top
            margin=dict(l=200)  # More space for product names
        )
        
        return fig
    
    def create_gap_heatmap(self, gap_df: pd.DataFrame, group_by: str = 'brand') -> go.Figure:
        """
        Create heatmap showing GAP by category/brand
        
        Args:
            gap_df: DataFrame with GAP calculations
            group_by: Grouping field (brand, category, etc.)
            
        Returns:
            Plotly figure object
        """
        if group_by not in gap_df.columns:
            # Fallback if grouping column doesn't exist
            fig = go.Figure()
            fig.add_annotation(
                text=f"'{group_by}' grouping not available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(
                title="GAP Heatmap",
                height=400,
                xaxis=dict(visible=False),
                yaxis=dict(visible=False)
            )
            return fig
        
        # Prepare data for heatmap
        # Create a pivot with gap percentage
        heatmap_data = gap_df.pivot_table(
            values='gap_percentage',
            index=group_by,
            aggfunc='mean'
        ).reset_index()
        
        # Sort by gap percentage
        heatmap_data = heatmap_data.sort_values('gap_percentage')
        
        # Create color scale (red for shortage, green for balanced, blue for surplus)
        colorscale = [
            [0.0, '#FF0000'],    # Deep red for severe shortage
            [0.4, '#FFAA00'],    # Orange for shortage
            [0.5, '#00AA00'],    # Green for balanced
            [0.6, '#00AAFF'],    # Light blue for small surplus
            [1.0, '#0000FF']     # Blue for high surplus
        ]
        
        # Normalize values for color scale (-100% to +100%)
        z_values = heatmap_data['gap_percentage'].values.reshape(-1, 1)
        z_normalized = np.clip(z_values, -100, 100)
        
        fig = go.Figure(data=go.Heatmap(
            z=z_normalized,
            y=heatmap_data[group_by],
            x=['GAP %'],
            text=z_values,
            texttemplate='%{text:.1f}%',
            textfont={"size": 12},
            colorscale=colorscale,
            zmid=0,
            colorbar=dict(
                title="GAP %",
                tickmode='linear',
                tick0=-100,
                dtick=25,
                ticksuffix='%'
            ),
            hovertemplate='<b>%{y}</b><br>' +
                         'GAP: %{z:.1f}%<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': f'GAP Percentage by {group_by.title()}',
                'x': 0.5,
                'xanchor': 'center'
            },
            height=max(400, len(heatmap_data) * 30),  # Dynamic height
            xaxis=dict(visible=False),
            yaxis_title="",
            margin=dict(l=150)  # Space for labels
        )
        
        return fig
    
    def create_supply_demand_comparison(self, gap_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
        """
        Create grouped bar chart comparing supply vs demand
        
        Args:
            gap_df: DataFrame with GAP calculations
            top_n: Number of items to show
            
        Returns:
            Plotly figure object
        """
        # Get items with largest absolute gaps
        gap_df_sorted = gap_df.copy()
        gap_df_sorted['abs_gap'] = abs(gap_df_sorted['net_gap'])
        top_items = gap_df_sorted.nlargest(top_n, 'abs_gap')
        
        # Prepare display names
        if 'product_name' in top_items.columns:
            display_names = top_items.apply(
                lambda x: f"{x.get('pt_code', '')}",  # Just PT code for space
                axis=1
            )
        elif 'brand' in top_items.columns:
            display_names = top_items['brand']
        else:
            display_names = top_items.index.astype(str)
        
        # Create grouped bar chart
        fig = go.Figure()
        
        # Add supply bars
        fig.add_trace(go.Bar(
            name='Supply',
            x=display_names,
            y=top_items['total_supply'],
            marker_color='#0088FF',
            text=top_items['total_supply'].apply(lambda x: self.formatter.format_number(x)),
            textposition='outside',
            hovertemplate='Supply: %{y:,.0f}<extra></extra>'
        ))
        
        # Add demand bars
        fig.add_trace(go.Bar(
            name='Demand',
            x=display_names,
            y=top_items['total_demand'],
            marker_color='#FF8800',
            text=top_items['total_demand'].apply(lambda x: self.formatter.format_number(x)),
            textposition='outside',
            hovertemplate='Demand: %{y:,.0f}<extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': 'Supply vs Demand Comparison',
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title="Product",
            yaxis_title="Quantity (units)",
            barmode='group',
            height=500,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis_tickangle=-45
        )
        
        return fig
    
    def create_trend_sparkline(self, values: List[float], dates: List = None) -> go.Figure:
        """
        Create small sparkline chart for trends
        
        Args:
            values: List of values to plot
            dates: Optional list of dates
            
        Returns:
            Plotly figure object
        """
        if not values:
            return None
        
        x_values = dates if dates else list(range(len(values)))
        
        fig = go.Figure(data=go.Scatter(
            x=x_values,
            y=values,
            mode='lines',
            line=dict(
                color='#0088FF' if values[-1] >= values[0] else '#FF4444',
                width=2
            ),
            fill='tozeroy',
            fillcolor='rgba(0,136,255,0.1)' if values[-1] >= values[0] else 'rgba(255,68,68,0.1)',
            showlegend=False
        ))
        
        fig.update_layout(
            height=60,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                visible=False,
                fixedrange=True
            ),
            yaxis=dict(
                visible=False,
                fixedrange=True
            ),
            hovermode=False
        )
        
        return fig
    
    def create_kpi_cards(self, metrics: Dict[str, Any]) -> None:
        """
        Create KPI cards using Streamlit columns
        
        Args:
            metrics: Dictionary of metrics from calculator
        """
        # First row of KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📦 Total Products",
                value=self.formatter.format_number(metrics['total_products']),
                delta=None
            )
        
        with col2:
            shortage_pct = (metrics['shortage_items'] / metrics['total_products'] * 100) if metrics['total_products'] > 0 else 0
            st.metric(
                label="⚠️ Shortage Items",
                value=self.formatter.format_number(metrics['shortage_items']),
                delta=f"{shortage_pct:.1f}% of total",
                delta_color="inverse"
            )
        
        with col3:
            st.metric(
                label="🚨 Critical Items",
                value=self.formatter.format_number(metrics['critical_items']),
                delta="Immediate action" if metrics['critical_items'] > 0 else "All good",
                delta_color="inverse" if metrics['critical_items'] > 0 else "normal"
            )
        
        with col4:
            st.metric(
                label="📊 Coverage Rate",
                value=f"{metrics['overall_coverage']:.1f}%",
                delta="Target: 95%" if metrics['overall_coverage'] < 95 else "On target",
                delta_color="normal" if metrics['overall_coverage'] >= 95 else "inverse"
            )
        
        # Second row of KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📉 Total Shortage",
                value=self.formatter.format_number(metrics['total_shortage']),
                delta="units",
                help="Total quantity short across all products"
            )
        
        with col2:
            st.metric(
                label="📈 Total Surplus",
                value=self.formatter.format_number(metrics['total_surplus']),
                delta="units",
                help="Total excess quantity across all products"
            )
        
        with col3:
            st.metric(
                label="💰 At Risk Value",
                value=self.formatter.format_currency(metrics['at_risk_value_usd']),
                delta=None,
                help="Potential revenue at risk due to shortages"
            )
        
        with col4:
            st.metric(
                label="👥 Affected Customers",
                value=self.formatter.format_number(metrics['affected_customers']),
                delta=None,
                help="Number of customers impacted by shortages"
            )