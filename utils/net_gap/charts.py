# utils/net_gap/charts.py

"""
Visualization module for GAP Analysis System - Version 2.1 FIXED
- Fixed dtype issues with abs_gap column
- Enhanced numeric validation
- Context-aware visualizations
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

# Chart configuration constants
CHART_HEIGHT_DEFAULT = 400
CHART_HEIGHT_PER_ITEM = 35
MIN_CHART_HEIGHT = 300
MAX_CHART_HEIGHT = 700

# Enhanced color scheme with safety stock statuses
STATUS_COLORS = {
    # Safety stock specific statuses
    'CRITICAL_BREACH': '#8B0000',  # Dark red
    'BELOW_SAFETY': '#FF4444',     # Red
    'AT_REORDER': '#FFA500',       # Orange
    'HAS_EXPIRED': '#8B4513',      # Saddle brown
    'EXPIRY_RISK': '#FF8C00',      # Dark orange
    
    # Traditional statuses
    'SEVERE_SHORTAGE': '#FF0000',   # Red
    'HIGH_SHORTAGE': '#FF8800',     # Orange-red
    'MODERATE_SHORTAGE': '#FFAA00', # Orange
    'BALANCED': '#00AA00',          # Green
    'LIGHT_SURPLUS': '#0088FF',     # Light blue
    'MODERATE_SURPLUS': '#0066CC',  # Medium blue
    'HIGH_SURPLUS': '#FF8800',      # Orange (concern)
    'SEVERE_SURPLUS': '#FF4444',    # Red (critical)
    'NO_DEMAND': '#CCCCCC',         # Gray
    'NO_DEMAND_INCOMING': '#999999', # Dark gray
    'UNKNOWN': '#888888'             # Medium gray
}

STATUS_LABELS = {
    # Safety stock specific statuses
    'CRITICAL_BREACH': '🚨 Critical Safety Breach',
    'BELOW_SAFETY': '⚠️ Below Safety Stock',
    'AT_REORDER': '📦 At Reorder Point',
    'HAS_EXPIRED': '❌ Has Expired Stock',
    'EXPIRY_RISK': '⏰ Expiry Risk',
    
    # Traditional statuses
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

# Chart theme configuration
CHART_THEME = {
    'font_family': 'Arial, sans-serif',
    'font_size': 12,
    'title_font_size': 16,
    'background_color': 'rgba(0,0,0,0)',
    'grid_color': 'rgba(128,128,128,0.2)'
}


class GAPCharts:
    """Creates visualization components for GAP analysis with safety stock support"""
    
    def __init__(self, formatter):
        """
        Initialize charts with formatter
        
        Args:
            formatter: Instance of GAPFormatter for consistent formatting
        """
        self.formatter = formatter
        self._include_safety = False
    
    def _ensure_numeric_column(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Ensure a column is numeric type
        
        Args:
            df: DataFrame
            column: Column name to convert
            
        Returns:
            DataFrame with numeric column
        """
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors='coerce').fillna(0)
        return df
    
    def create_kpi_cards(self, metrics: Dict[str, Any], include_safety: bool = False,
                        enable_customer_dialog: bool = True) -> None:
        """
        Create KPI cards using Streamlit columns
        Enhanced with safety stock metrics when enabled
        
        Args:
            metrics: Dictionary of metrics from calculator
            include_safety: Whether safety stock is included in analysis
            enable_customer_dialog: Whether to show customer dialog button
        """
        self._include_safety = include_safety
        
        # First row - Main metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📦 Total Products",
                value=self.formatter.format_number(metrics['total_products']),
                help="Total number of products analyzed"
            )
        
        with col2:
            # Context-aware shortage label
            if include_safety:
                shortage_label = "⚠️ Below Requirements"
                help_text = "Items below demand or safety stock requirements"
            else:
                shortage_label = "⚠️ Shortage Items"
                help_text = "Items with insufficient supply to meet demand"
            
            shortage_pct = self._calculate_percentage(
                metrics['shortage_items'], 
                metrics['total_products']
            )
            st.metric(
                label=shortage_label,
                value=self.formatter.format_number(metrics['shortage_items']),
                delta=f"{shortage_pct:.1f}% of total",
                delta_color="inverse",
                help=help_text
            )
        
        with col3:
            # Critical items (adapts based on safety)
            if include_safety:
                critical_label = "🚨 Safety Breaches"
                critical_help = "Items critically below safety stock or with expired inventory"
            else:
                critical_label = "🚨 Critical Items"
                critical_help = "Items requiring immediate action"
            
            st.metric(
                label=critical_label,
                value=self.formatter.format_number(metrics['critical_items']),
                delta="Immediate action" if metrics['critical_items'] > 0 else "All good",
                delta_color="inverse" if metrics['critical_items'] > 0 else "normal",
                help=critical_help
            )
        
        with col4:
            coverage = metrics['overall_coverage']
            st.metric(
                label="📊 Coverage Rate",
                value=f"{coverage:.1f}%",
                delta=self._get_coverage_delta(coverage, include_safety),
                delta_color="normal" if coverage >= 95 else "inverse",
                help="Overall supply coverage considering demand" + 
                     (" and safety requirements" if include_safety else "")
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
            # Affected Customers with dialog button
            affected_count = metrics['affected_customers']
            
            metric_container = st.container()
            
            with metric_container:
                st.metric(
                    label="👥 Affected Customers",
                    value=self.formatter.format_number(affected_count),
                    help="Number of unique customers impacted by shortages"
                )
                
                if enable_customer_dialog and affected_count > 0:
                    if st.button(
                        f"📋 View Details",
                        key="view_customer_details",
                        type="primary",
                        use_container_width=True
                    ):
                        st.session_state.show_customer_dialog = True
                        st.rerun()
        
        # Third row - Safety stock specific metrics (only if enabled and available)
        if include_safety and 'below_safety_count' in metrics:
            st.divider()
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="🔒 Below Safety",
                    value=self.formatter.format_number(metrics.get('below_safety_count', 0)),
                    help="Items with inventory below safety stock requirement"
                )
            
            with col2:
                st.metric(
                    label="📦 At Reorder",
                    value=self.formatter.format_number(metrics.get('at_reorder_count', 0)),
                    help="Items at or below reorder point"
                )
            
            with col3:
                st.metric(
                    label="💵 Safety Value",
                    value=self.formatter.format_currency(
                        metrics.get('safety_stock_value', 0),
                        abbreviate=True
                    ),
                    help="Total value of safety stock requirements"
                )
            
            with col4:
                expired_count = metrics.get('has_expired_count', 0)
                expiry_risk = metrics.get('expiry_risk_count', 0)
                
                if expired_count > 0:
                    st.metric(
                        label="❌ Expired",
                        value=expired_count,
                        delta=f"+{expiry_risk} at risk",
                        delta_color="inverse",
                        help="Products with expired or expiring inventory"
                    )
                else:
                    st.metric(
                        label="📅 Expiry Status",
                        value="Clear",
                        delta=f"{expiry_risk} watch" if expiry_risk > 0 else "All good",
                        help="No expired inventory detected"
                    )
    
    def create_status_pie_chart(self, gap_df: pd.DataFrame) -> go.Figure:
        """
        Create pie chart showing distribution of items by GAP status
        Adapts to show safety-specific statuses when enabled
        
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
        title = 'Distribution by GAP Status'
        if self._include_safety:
            title += ' (Including Safety Stock)'
        
        fig.update_layout(
            title={
                'text': title,
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
        FIXED: Ensure proper numeric type handling for abs_gap
        
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
        
        # FIX: Ensure net_gap is numeric before calculating abs_gap
        shortage_df = self._ensure_numeric_column(shortage_df, 'net_gap')
        
        # Create abs_gap column with proper numeric type
        shortage_df['abs_gap'] = shortage_df['net_gap'].abs()
        
        # Double-check abs_gap is numeric
        if shortage_df['abs_gap'].dtype == 'object':
            logger.warning("abs_gap still object dtype after conversion, forcing to numeric")
            shortage_df['abs_gap'] = pd.to_numeric(shortage_df['abs_gap'], errors='coerce').fillna(0)
        
        # Log dtype for debugging
        logger.debug(f"abs_gap dtype: {shortage_df['abs_gap'].dtype}")
        
        # Now safely use nlargest
        try:
            top_items = shortage_df.nlargest(min(top_n, len(shortage_df)), 'abs_gap')
        except Exception as e:
            logger.error(f"Error in nlargest operation: {e}")
            # Fallback: sort manually
            shortage_df = shortage_df.sort_values('abs_gap', ascending=False)
            top_items = shortage_df.head(min(top_n, len(shortage_df)))
        
        # Prepare display names
        display_names = self._prepare_display_names(top_items)
        
        # Get colors based on status
        colors = [STATUS_COLORS.get(status, '#888888') for status in top_items['gap_status']]
        
        # Ensure numeric columns for hover data
        top_items = self._ensure_numeric_column(top_items, 'gap_percentage')
        top_items = self._ensure_numeric_column(top_items, 'total_demand')
        
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
        title = f'Top {len(top_items)} Shortage Items'
        if self._include_safety:
            title += ' (Considering Safety Stock)'
        
        fig.update_layout(
            title={
                'text': title,
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
        Shows available supply when safety stock is considered
        
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
        
        # Ensure net_gap is numeric before calculating abs_gap
        gap_df_sorted = self._ensure_numeric_column(gap_df_sorted, 'net_gap')
        gap_df_sorted['abs_gap'] = gap_df_sorted['net_gap'].abs()
        
        try:
            top_items = gap_df_sorted.nlargest(min(top_n, len(gap_df_sorted)), 'abs_gap')
        except Exception as e:
            logger.error(f"Error in nlargest operation: {e}")
            gap_df_sorted = gap_df_sorted.sort_values('abs_gap', ascending=False)
            top_items = gap_df_sorted.head(min(top_n, len(gap_df_sorted)))
        
        # Prepare display names (shorter for x-axis)
        display_names = self._prepare_short_names(top_items)
        
        # Create figure
        fig = go.Figure()
        
        # Determine what to show as supply
        if self._include_safety and 'available_supply' in top_items.columns:
            top_items = self._ensure_numeric_column(top_items, 'available_supply')
            supply_values = top_items['available_supply']
            supply_label = 'Available Supply'
            supply_hover = 'Available (after safety): %{y:,.0f}'
        else:
            top_items = self._ensure_numeric_column(top_items, 'total_supply')
            supply_values = top_items['total_supply']
            supply_label = 'Total Supply'
            supply_hover = 'Supply: %{y:,.0f}'
        
        # Ensure demand is numeric
        top_items = self._ensure_numeric_column(top_items, 'total_demand')
        
        # Add supply bars
        fig.add_trace(go.Bar(
            name=supply_label,
            x=display_names,
            y=supply_values,
            marker_color='#0088FF',
            text=supply_values.apply(lambda x: self.formatter.format_number(x)),
            textposition='outside',
            hovertemplate=supply_hover + '<extra></extra>'
        ))
        
        # Add demand bars
        fig.add_trace(go.Bar(
            name='Total Demand',
            x=display_names,
            y=top_items['total_demand'],
            marker_color='#FF8800',
            text=top_items['total_demand'].apply(lambda x: self.formatter.format_number(x)),
            textposition='outside',
            hovertemplate='Demand: %{y:,.0f}<extra></extra>'
        ))
        
        # Add safety stock line if included
        if self._include_safety and 'safety_stock_qty' in top_items.columns:
            top_items = self._ensure_numeric_column(top_items, 'safety_stock_qty')
            fig.add_trace(go.Scatter(
                name='Safety Stock',
                x=display_names,
                y=top_items['safety_stock_qty'],
                mode='lines+markers',
                line=dict(color='red', dash='dash'),
                marker=dict(size=8),
                hovertemplate='Safety: %{y:,.0f}<extra></extra>'
            ))
        
        # Update layout
        title = 'Supply vs Demand Comparison'
        if self._include_safety:
            title += ' (With Safety Requirements)'
        
        fig.update_layout(
            title={
                'text': title,
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
    
    def create_coverage_distribution(self, gap_df: pd.DataFrame) -> go.Figure:
        """
        Create histogram showing distribution of coverage ratios
        More useful than heatmap for understanding overall inventory health
        
        Args:
            gap_df: DataFrame with GAP calculations
            
        Returns:
            Plotly figure object
        """
        if gap_df.empty:
            return self._create_empty_chart("No data for coverage distribution")
        
        # Ensure coverage_ratio is numeric
        gap_df = self._ensure_numeric_column(gap_df.copy(), 'coverage_ratio')
        
        # Filter out extreme values for better visualization
        coverage_data = gap_df[gap_df['coverage_ratio'] < 10]['coverage_ratio'] * 100
        
        # Define bins and colors
        bins = [0, 50, 70, 90, 110, 150, 200, 300, 1000]
        bin_labels = ['<50%', '50-70%', '70-90%', '90-110%', '110-150%', '150-200%', '200-300%', '>300%']
        bin_colors = ['#FF0000', '#FF8800', '#FFAA00', '#00AA00', '#0088FF', '#0066CC', '#FF8800', '#FF4444']
        
        # Create histogram
        fig = go.Figure(data=[
            go.Histogram(
                x=coverage_data,
                nbinsx=30,
                marker_color='#0088FF',
                hovertemplate='Coverage: %{x:.0f}%<br>Count: %{y}<extra></extra>'
            )
        ])
        
        # Add reference lines
        fig.add_vline(x=100, line_dash="dash", line_color="green", 
                     annotation_text="Target (100%)")
        
        if self._include_safety:
            fig.add_vline(x=90, line_dash="dot", line_color="orange", 
                         annotation_text="Min Safe (90%)")
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Coverage Ratio Distribution',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': CHART_THEME['title_font_size']}
            },
            xaxis_title="Coverage (%)",
            yaxis_title="Number of Products",
            height=CHART_HEIGHT_DEFAULT,
            showlegend=False,
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
    
    def _get_coverage_delta(self, coverage: float, include_safety: bool = False) -> str:
        """Get coverage delta message"""
        if include_safety:
            # Stricter targets with safety stock
            if coverage >= 110:
                return "Excellent"
            elif coverage >= 100:
                return "Good"
            elif coverage >= 90:
                return "Below target"
            else:
                return "Critical"
        else:
            # Standard targets
            if coverage >= 100:
                return "Excellent"
            elif coverage >= 95:
                return "On target"
            elif coverage >= 90:
                return "Below target"
            else:
                return "Target: 95%"
    
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
        else:
            return df.index.astype(str).tolist()
    
    def _prepare_short_names(self, df: pd.DataFrame) -> List[str]:
        """Prepare short names for x-axis"""
        if 'pt_code' in df.columns:
            return df['pt_code'].tolist()
        elif 'brand' in df.columns:
            return df['brand'].tolist()
        else:
            return df.index.astype(str).tolist()