# utils/net_gap/charts.py

"""
Visualization module for GAP Analysis System - Clean Version
Provides chart components using Plotly
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from typing import Dict, List, Optional, Any

# Chart configuration constants
CHART_HEIGHT_DEFAULT = 400
CHART_HEIGHT_PER_ITEM = 40
MIN_CHART_HEIGHT = 300
MAX_CHART_HEIGHT = 800

# Color scheme for GAP statuses
STATUS_COLORS = {
    'SEVERE_SHORTAGE': '#FF4444',
    'HIGH_SHORTAGE': '#FF8800',
    'MODERATE_SHORTAGE': '#FFAA00',
    'BALANCED': '#00AA00',
    'LIGHT_SURPLUS': '#0088FF',
    'MODERATE_SURPLUS': '#8800FF',
    'HIGH_SURPLUS': '#FF8800',
    'SEVERE_SURPLUS': '#FF4444',
    'NO_DEMAND': '#CCCCCC',
    'NO_DEMAND_INCOMING': '#999999',
    'UNKNOWN': '#888888'
}

STATUS_LABELS = {
    'SEVERE_SHORTAGE': '🔴 Severe Shortage',
    'HIGH_SHORTAGE': '🟠 High Shortage',
    'MODERATE_SHORTAGE': '🟡 Moderate Shortage',
    'BALANCED': '✅ Balanced',
    'LIGHT_SURPLUS': '🔵 Light Surplus',
    'MODERATE_SURPLUS': '🟣 Moderate Surplus',
    'HIGH_SURPLUS': '🟠 High Surplus',
    'SEVERE_SURPLUS': '🔴 Severe Surplus',
    'NO_DEMAND': '⚪ No Demand',
    'NO_DEMAND_INCOMING': '🟤 No Demand (PO Incoming)',
    'UNKNOWN': '❓ Unknown'
}

# Chart theme configuration
CHART_THEME = {
    'font_family': 'Arial, sans-serif',
    'font_size': 12,
    'title_font_size': 16,
    'background_color': 'rgba(0,0,0,0)',
    'grid_color': 'rgba(128,128,128,0.2)'
}


class GAPCharts:
    """Creates visualization components for GAP analysis"""
    
    def __init__(self, formatter):
        """
        Initialize charts with formatter
        
        Args:
            formatter: Instance of GAPFormatter for consistent formatting
        """
        self.formatter = formatter
    
    def create_kpi_cards(self, metrics: Dict[str, Any]) -> None:
        """
        Create KPI cards using Streamlit columns
        
        Args:
            metrics: Dictionary of metrics from calculator
        """
        # First row - Main metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📦 Total Products",
                value=self.formatter.format_number(metrics['total_products']),
                help="Total number of products analyzed"
            )
        
        with col2:
            shortage_pct = self._calculate_percentage(
                metrics['shortage_items'], 
                metrics['total_products']
            )
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
            coverage = metrics['overall_coverage']
            st.metric(
                label="📊 Coverage Rate",
                value=f"{coverage:.1f}%",
                delta=self._get_coverage_delta(coverage),
                delta_color="normal" if coverage >= 95 else "inverse"
            )
        
        # Second row - Volume and value metrics
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
                help="Potential revenue at risk due to shortages"
            )
        
        with col4:
            st.metric(
                label="👥 Affected Customers",
                value=self.formatter.format_number(metrics['affected_customers']),
                help="Number of customers impacted by shortages"
            )
    
    def create_status_pie_chart(self, gap_df: pd.DataFrame) -> go.Figure:
        """
        Create pie chart showing distribution of items by GAP status
        
        Args:
            gap_df: DataFrame with GAP calculations
            
        Returns:
            Plotly figure object
        """
        if gap_df.empty:
            return self._create_empty_chart("No data available for status distribution")
        
        # Count items by status
        status_counts = gap_df['gap_status'].value_counts()
        
        # Prepare data for pie chart
        labels = []
        values = []
        colors = []
        
        for status, count in status_counts.items():
            labels.append(STATUS_LABELS.get(status, status))
            values.append(count)
            colors.append(STATUS_COLORS.get(status, '#888888'))
        
        # Create pie chart
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
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Distribution by GAP Status',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': CHART_THEME['title_font_size']}
            },
            height=CHART_HEIGHT_DEFAULT,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.05
            ),
            font=dict(
                family=CHART_THEME['font_family'],
                size=CHART_THEME['font_size']
            ),
            paper_bgcolor=CHART_THEME['background_color']
        )
        
        return fig
    
    def create_top_shortage_bar_chart(self, gap_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
        """
        Create bar chart showing top shortage items
        
        Args:
            gap_df: DataFrame with GAP calculations (should be filtered for shortages)
            top_n: Number of top items to show
            
        Returns:
            Plotly figure object
        """
        if gap_df.empty:
            return self._create_empty_chart("No shortage items found")
        
        # Get top shortage items
        shortage_df = gap_df.copy()
        shortage_df['abs_gap'] = abs(shortage_df['net_gap'])
        top_items = shortage_df.nlargest(min(top_n, len(shortage_df)), 'abs_gap')
        
        # Prepare display names
        display_names = self._prepare_display_names(top_items)
        
        # Get colors based on status
        colors = [STATUS_COLORS.get(status, '#888888') for status in top_items['gap_status']]
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=top_items['abs_gap'],
                y=display_names,
                orientation='h',
                marker=dict(
                    color=colors,
                    line=dict(width=1, color='rgba(0,0,0,0.3)')
                ),
                text=top_items['abs_gap'].apply(lambda x: self.formatter.format_number(x)),
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>' +
                             'Shortage: %{x:,.0f} units<br>' +
                             'GAP: %{customdata[0]:.1f}%<br>' +
                             'Demand: %{customdata[1]:,.0f}<br>' +
                             '<extra></extra>',
                customdata=np.column_stack((
                    top_items['gap_percentage'],
                    top_items['total_demand']
                ))
            )
        ])
        
        # Calculate dynamic height
        chart_height = self._calculate_dynamic_height(len(top_items))
        
        # Update layout
        fig.update_layout(
            title={
                'text': f'Top {len(top_items)} Shortage Items',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': CHART_THEME['title_font_size']}
            },
            xaxis_title="Shortage Quantity (units)",
            yaxis_title="",
            height=chart_height,
            showlegend=False,
            yaxis=dict(autorange="reversed"),  # Worst at top
            margin=dict(l=200),  # More space for product names
            font=dict(
                family=CHART_THEME['font_family'],
                size=CHART_THEME['font_size']
            ),
            paper_bgcolor=CHART_THEME['background_color']
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
        if gap_df.empty:
            return self._create_empty_chart("No data to compare")
        
        # Get items with largest absolute gaps
        gap_df_sorted = gap_df.copy()
        gap_df_sorted['abs_gap'] = abs(gap_df_sorted['net_gap'])
        top_items = gap_df_sorted.nlargest(min(top_n, len(gap_df_sorted)), 'abs_gap')
        
        # Prepare display names (shorter for x-axis)
        display_names = self._prepare_short_names(top_items)
        
        # Create figure
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
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Supply vs Demand Comparison',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': CHART_THEME['title_font_size']}
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
            xaxis_tickangle=-45,
            font=dict(
                family=CHART_THEME['font_family'],
                size=CHART_THEME['font_size']
            ),
            paper_bgcolor=CHART_THEME['background_color']
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
        if gap_df.empty or group_by not in gap_df.columns:
            return self._create_empty_chart(f"Cannot create heatmap - '{group_by}' not available")
        
        # Prepare data for heatmap
        heatmap_data = gap_df.pivot_table(
            values='gap_percentage',
            index=group_by,
            aggfunc='mean'
        ).reset_index()
        
        if heatmap_data.empty:
            return self._create_empty_chart("No data for heatmap")
        
        # Sort by gap percentage
        heatmap_data = heatmap_data.sort_values('gap_percentage')
        
        # Prepare z values and normalize
        z_values = heatmap_data['gap_percentage'].values.reshape(-1, 1)
        z_normalized = np.clip(z_values, -100, 100)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=z_normalized,
            y=heatmap_data[group_by],
            x=['GAP %'],
            text=z_values,
            texttemplate='%{text:.1f}%',
            textfont={"size": 12},
            colorscale=self._get_gap_colorscale(),
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
        
        # Calculate dynamic height
        chart_height = self._calculate_dynamic_height(len(heatmap_data), min_height=400)
        
        # Update layout
        fig.update_layout(
            title={
                'text': f'GAP Percentage by {group_by.title()}',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': CHART_THEME['title_font_size']}
            },
            height=chart_height,
            xaxis=dict(visible=False),
            yaxis_title="",
            margin=dict(l=150),
            font=dict(
                family=CHART_THEME['font_family'],
                size=CHART_THEME['font_size']
            ),
            paper_bgcolor=CHART_THEME['background_color']
        )
        
        return fig
    
    # Helper methods
    def _create_empty_chart(self, message: str) -> go.Figure:
        """Create an empty chart with a message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            height=CHART_HEIGHT_DEFAULT,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            paper_bgcolor=CHART_THEME['background_color']
        )
        return fig
    
    def _calculate_percentage(self, part: float, whole: float) -> float:
        """Calculate percentage safely"""
        return (part / whole * 100) if whole > 0 else 0
    
    def _get_coverage_delta(self, coverage: float) -> str:
        """Get coverage delta message"""
        if coverage >= 100:
            return "Excellent"
        elif coverage >= 95:
            return "On target"
        elif coverage >= 90:
            return "Below target"
        else:
            return f"Target: 95%"
    
    def _calculate_dynamic_height(self, n_items: int, min_height: int = MIN_CHART_HEIGHT) -> int:
        """Calculate dynamic chart height based on items"""
        calculated_height = max(min_height, n_items * CHART_HEIGHT_PER_ITEM)
        return min(calculated_height, MAX_CHART_HEIGHT)
    
    def _prepare_display_names(self, df: pd.DataFrame) -> List[str]:
        """Prepare display names for items"""
        if 'product_name' in df.columns and 'pt_code' in df.columns:
            return df.apply(
                lambda x: f"{x['pt_code']} - {x['product_name'][:25]}{'...' if len(str(x['product_name'])) > 25 else ''}",
                axis=1
            ).tolist()
        elif 'brand' in df.columns:
            return df['brand'].tolist()
        elif 'category' in df.columns:
            return df['category'].tolist()
        else:
            return df.index.astype(str).tolist()
    
    def _prepare_short_names(self, df: pd.DataFrame) -> List[str]:
        """Prepare short names for x-axis"""
        if 'pt_code' in df.columns:
            return df['pt_code'].tolist()
        elif 'brand' in df.columns:
            return df['brand'].tolist()
        elif 'category' in df.columns:
            return df['category'].tolist()
        else:
            return df.index.astype(str).tolist()
    
    def _get_gap_colorscale(self) -> List[List]:
        """Get color scale for GAP heatmap"""
        return [
            [0.0, '#FF0000'],    # Deep red for severe shortage
            [0.4, '#FFAA00'],    # Orange for shortage
            [0.5, '#00AA00'],    # Green for balanced
            [0.6, '#00AAFF'],    # Light blue for small surplus
            [1.0, '#0000FF']     # Blue for high surplus
        ]