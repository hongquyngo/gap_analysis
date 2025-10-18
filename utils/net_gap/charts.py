# utils/net_gap/charts.py

"""
Visualization Module - Version 3.4 CLEANED
- Removed unused charts (coverage distribution, supply vs demand)
- Added top surplus chart
- Simplified chart functions
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

# Chart configuration
CHART_HEIGHT_DEFAULT = 400
CHART_HEIGHT_PER_ITEM = 35
MIN_CHART_HEIGHT = 300
MAX_CHART_HEIGHT = 700

# Color scheme
STATUS_COLORS = {
    'CRITICAL_BREACH': '#8B0000',
    'BELOW_SAFETY': '#FF4444',
    'AT_REORDER': '#FFA500',
    'HAS_EXPIRED': '#8B4513',
    'EXPIRY_RISK': '#FF8C00',
    'SEVERE_SHORTAGE': '#FF0000',
    'HIGH_SHORTAGE': '#FF8800',
    'MODERATE_SHORTAGE': '#FFAA00',
    'BALANCED': '#00AA00',
    'LIGHT_SURPLUS': '#0088FF',
    'MODERATE_SURPLUS': '#0066CC',
    'HIGH_SURPLUS': '#FF8800',
    'SEVERE_SURPLUS': '#FF4444',
    'NO_DEMAND': '#CCCCCC',
    'NO_DEMAND_INCOMING': '#999999',
    'UNKNOWN': '#888888'
}

STATUS_LABELS = {
    'CRITICAL_BREACH': '🚨 Critical Safety Breach',
    'BELOW_SAFETY': '⚠️ Below Safety Stock',
    'AT_REORDER': '📦 At Reorder Point',
    'HAS_EXPIRED': '❌ Has Expired Stock',
    'EXPIRY_RISK': '⏰ Expiry Risk',
    'SEVERE_SHORTAGE': '🔴 Severe Shortage',
    'HIGH_SHORTAGE': '🟠 High Shortage',
    'MODERATE_SHORTAGE': '🟡 Moderate Shortage',
    'BALANCED': '✅ Balanced',
    'LIGHT_SURPLUS': '🔵 Light Surplus',
    'MODERATE_SURPLUS': '🟣 Moderate Surplus',
    'HIGH_SURPLUS': '🟠 High Surplus',
    'SEVERE_SURPLUS': '🔴 Severe Surplus',
    'NO_DEMAND': '⚪ No Demand',
    'NO_DEMAND_INCOMING': '🟤 No Demand (PO Coming)',
    'UNKNOWN': '❓ Unknown'
}

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
        Initialize charts
        
        Args:
            formatter: GAPFormatter instance
        """
        self.formatter = formatter
        self._include_safety = False
    
    def create_kpi_cards(
        self, 
        metrics: Dict[str, Any], 
        include_safety: bool = False,
        enable_customer_dialog: bool = True
    ) -> None:
        """
        Create KPI cards using Streamlit columns
        
        Args:
            metrics: Metrics dictionary
            include_safety: Safety stock included
            enable_customer_dialog: Show customer dialog button
        """
        self._include_safety = include_safety
        
        # First row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📦 Total Products",
                value=self.formatter.format_number(metrics['total_products']),
                help="Total number of products analyzed"
            )
        
        with col2:
            shortage_label = "⚠️ Below Requirements" if include_safety else "⚠️ Shortage Items"
            shortage_pct = self._calculate_percentage(
                metrics['shortage_items'], 
                metrics['total_products']
            )
            st.metric(
                label=shortage_label,
                value=self.formatter.format_number(metrics['shortage_items']),
                delta=f"{shortage_pct:.1f}% of total",
                delta_color="inverse"
            )
        
        with col3:
            critical_label = "🚨 Safety Breaches" if include_safety else "🚨 Critical Items"
            st.metric(
                label=critical_label,
                value=self.formatter.format_number(metrics['critical_items']),
                delta="Immediate action" if metrics['critical_items'] > 0 else "All good",
                delta_color="inverse" if metrics['critical_items'] > 0 else "normal"
            )
        
        with col4:
            coverage = metrics['overall_coverage']
            st.metric(
                label="📊 Coverage Rate",
                value=f"{coverage:.1f}%",
                delta=self._get_coverage_delta(coverage, include_safety),
                delta_color="normal" if coverage >= 95 else "inverse"
            )
        
        # Second row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📉 Total Shortage",
                value=self.formatter.format_number(metrics['total_shortage']),
                delta="units"
            )
        
        with col2:
            st.metric(
                label="📈 Total Surplus",
                value=self.formatter.format_number(metrics['total_surplus']),
                delta="units"
            )
        
        with col3:
            st.metric(
                label="💰 At Risk Value",
                value=self.formatter.format_currency(metrics['at_risk_value_usd'])
            )
        
        with col4:
            # Affected Customers with dialog button
            affected_count = metrics['affected_customers']
            
            metric_container = st.container()
            
            with metric_container:
                st.metric(
                    label="👥 Affected Customers",
                    value=self.formatter.format_number(affected_count)
                )
                
                if enable_customer_dialog and affected_count > 0:
                    if st.button(
                        f"📋 View Details",
                        key="view_customer_details",
                        type="primary",
                        use_container_width=True
                    ):
                        from .session_manager import get_session_manager
                        session_mgr = get_session_manager()
                        session_mgr.open_customer_dialog()
                        st.rerun()
        
        # Third row (safety metrics)
        if include_safety and 'below_safety_count' in metrics:
            st.divider()
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="🔒 Below Safety",
                    value=self.formatter.format_number(metrics.get('below_safety_count', 0))
                )
            
            with col2:
                st.metric(
                    label="📦 At Reorder",
                    value=self.formatter.format_number(metrics.get('at_reorder_count', 0))
                )
            
            with col3:
                st.metric(
                    label="💵 Safety Value",
                    value=self.formatter.format_currency(
                        metrics.get('safety_stock_value', 0),
                        abbreviate=True
                    )
                )
            
            with col4:
                expired_count = metrics.get('has_expired_count', 0)
                expiry_risk = metrics.get('expiry_risk_count', 0)
                
                if expired_count > 0:
                    st.metric(
                        label="❌ Expired",
                        value=expired_count,
                        delta=f"+{expiry_risk} at risk",
                        delta_color="inverse"
                    )
                else:
                    st.metric(
                        label="📅 Expiry Status",
                        value="Clear",
                        delta=f"{expiry_risk} watch" if expiry_risk > 0 else "All good"
                    )
    
    def create_status_pie_chart(self, gap_df: pd.DataFrame) -> go.Figure:
        """Create pie chart showing status distribution"""
        if gap_df.empty:
            return self._create_empty_chart("No data available")
        
        status_counts = gap_df['gap_status'].value_counts()
        
        labels = []
        values = []
        colors = []
        
        for status, count in status_counts.items():
            labels.append(STATUS_LABELS.get(status, status))
            values.append(count)
            colors.append(STATUS_COLORS.get(status, '#888888'))
        
        fig = go.Figure(data=[
            go.Pie(
                labels=labels,
                values=values,
                marker=dict(colors=colors),
                hole=0.3,
                textinfo='label+percent+value',
                textposition='auto',
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<br><extra></extra>'
            )
        ])
        
        title = 'GAP Status Distribution'
        if self._include_safety:
            title += ' (Including Safety Stock)'
        
        fig.update_layout(
            title={'text': title, 'x': 0.5, 'xanchor': 'center'},
            height=CHART_HEIGHT_DEFAULT,
            showlegend=True,
            font=dict(family=CHART_THEME['font_family'], size=CHART_THEME['font_size']),
            paper_bgcolor=CHART_THEME['background_color']
        )
        
        return fig
    
    def create_top_shortage_bar_chart(
        self, 
        gap_df: pd.DataFrame, 
        top_n: int = 10
    ) -> go.Figure:
        """Create bar chart showing top shortage items"""
        if gap_df.empty:
            return self._create_empty_chart("No shortage items")
        
        shortage_df = gap_df.copy()
        
        # Convert net_gap to numeric first to avoid dtype object error
        shortage_df['net_gap'] = pd.to_numeric(shortage_df['net_gap'], errors='coerce').fillna(0)
        
        # Filter shortage items (negative gap)
        shortage_df = shortage_df[shortage_df['net_gap'] < 0].copy()
        shortage_df['abs_gap'] = shortage_df['net_gap'].abs()
        
        if shortage_df.empty:
            return self._create_empty_chart("No shortage items found")
        
        top_items = shortage_df.nlargest(min(top_n, len(shortage_df)), 'abs_gap')
        
        display_names = self._prepare_display_names(top_items)
        colors = [STATUS_COLORS.get(status, '#888888') for status in top_items['gap_status']]
        
        fig = go.Figure(data=[
            go.Bar(
                x=top_items['abs_gap'],
                y=display_names,
                orientation='h',
                marker=dict(color=colors, line=dict(width=1, color='rgba(0,0,0,0.3)')),
                text=top_items['abs_gap'].apply(lambda x: self.formatter.format_number(x)),
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Shortage: %{x:,.0f} units<br><extra></extra>'
            )
        ])
        
        chart_height = self._calculate_dynamic_height(len(top_items))
        
        fig.update_layout(
            title={'text': f'Top {len(top_items)} Shortage Items', 'x': 0.5, 'xanchor': 'center'},
            xaxis_title="Shortage Quantity (units)",
            height=chart_height,
            yaxis=dict(autorange="reversed"),
            margin=dict(l=200),
            font=dict(family=CHART_THEME['font_family'], size=CHART_THEME['font_size']),
            paper_bgcolor=CHART_THEME['background_color']
        )
        
        return fig
    
    def create_top_surplus_bar_chart(
        self, 
        gap_df: pd.DataFrame, 
        top_n: int = 10
    ) -> go.Figure:
        """Create bar chart showing top surplus items"""
        if gap_df.empty:
            return self._create_empty_chart("No surplus items")
        
        surplus_df = gap_df.copy()
        
        # Convert net_gap to numeric first
        surplus_df['net_gap'] = pd.to_numeric(surplus_df['net_gap'], errors='coerce').fillna(0)
        
        # Filter surplus items (positive gap)
        surplus_df = surplus_df[surplus_df['net_gap'] > 0].copy()
        
        if surplus_df.empty:
            return self._create_empty_chart("No surplus items found")
        
        top_items = surplus_df.nlargest(min(top_n, len(surplus_df)), 'net_gap')
        
        display_names = self._prepare_display_names(top_items)
        colors = [STATUS_COLORS.get(status, '#0088FF') for status in top_items['gap_status']]
        
        fig = go.Figure(data=[
            go.Bar(
                x=top_items['net_gap'],
                y=display_names,
                orientation='h',
                marker=dict(color=colors, line=dict(width=1, color='rgba(0,0,0,0.3)')),
                text=top_items['net_gap'].apply(lambda x: self.formatter.format_number(x)),
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Surplus: %{x:,.0f} units<br><extra></extra>'
            )
        ])
        
        chart_height = self._calculate_dynamic_height(len(top_items))
        
        fig.update_layout(
            title={'text': f'Top {len(top_items)} Surplus Items', 'x': 0.5, 'xanchor': 'center'},
            xaxis_title="Surplus Quantity (units)",
            height=chart_height,
            yaxis=dict(autorange="reversed"),
            margin=dict(l=200),
            font=dict(family=CHART_THEME['font_family'], size=CHART_THEME['font_size']),
            paper_bgcolor=CHART_THEME['background_color']
        )
        
        return fig
    
    # Helper methods
    def _create_empty_chart(self, message: str) -> go.Figure:
        """Create empty chart with message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
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
    
    def _get_coverage_delta(self, coverage: float, include_safety: bool = False) -> str:
        """Get coverage delta message"""
        if include_safety:
            if coverage >= 110:
                return "Excellent"
            elif coverage >= 100:
                return "Good"
            elif coverage >= 90:
                return "Below target"
            else:
                return "Critical"
        else:
            if coverage >= 100:
                return "Excellent"
            elif coverage >= 95:
                return "On target"
            elif coverage >= 90:
                return "Below target"
            else:
                return "Target: 95%"
    
    def _calculate_dynamic_height(self, n_items: int) -> int:
        """Calculate dynamic chart height"""
        calculated = max(MIN_CHART_HEIGHT, n_items * CHART_HEIGHT_PER_ITEM)
        return min(calculated, MAX_CHART_HEIGHT)
    
    def _prepare_display_names(self, df: pd.DataFrame) -> List[str]:
        """Prepare display names for items"""
        if 'product_name' in df.columns and 'pt_code' in df.columns:
            return df.apply(
                lambda x: f"{x['pt_code']} - {x['product_name'][:25]}{'...' if len(str(x['product_name'])) > 25 else ''}",
                axis=1
            ).tolist()
        elif 'brand' in df.columns:
            return df['brand'].tolist()
        else:
            return df.index.astype(str).tolist()