# utils/period_gap/gap_display.py
"""
Display Functions for Period GAP Analysis
Handles all visualization and presentation logic
Updated with improved shortage categorization display
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def show_gap_summary(
    gap_df: pd.DataFrame, 
    display_options: Dict[str, Any],
    df_demand_filtered: Optional[pd.DataFrame] = None,
    df_supply_filtered: Optional[pd.DataFrame] = None
):
    """
    Show GAP analysis summary with improved shortage categorization
    
    Args:
        gap_df: GAP analysis results
        display_options: Display configuration options
        df_demand_filtered: Filtered demand data (for additional context)
        df_supply_filtered: Filtered supply data (for additional context)
    """
    from .formatters import format_number, format_currency
    from .period_helpers import is_past_period
    from .shortage_analyzer import categorize_shortage_type, get_shortage_summary
    
    st.markdown("### 📊 GAP Analysis Summary")
    
    if gap_df.empty:
        st.warning("No GAP data available for summary.")
        return
    
    # Verify required columns exist
    required_columns = ['pt_code', 'gap_quantity', 'period', 'total_demand_qty', 
                       'total_available', 'supply_in_period', 'fulfillment_rate_percent']
    missing_columns = [col for col in required_columns if col not in gap_df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns in GAP data: {missing_columns}")
        return
    
    # Categorize shortage types
    shortage_categorization = categorize_shortage_type(gap_df)
    net_shortage_products = shortage_categorization['net_shortage']
    timing_gap_products = shortage_categorization['timing_gap']
    
    # Calculate essential metrics
    total_products = gap_df['pt_code'].nunique()
    total_periods = gap_df['period'].nunique()
    
    # Metrics for different shortage types
    products_with_net_shortage = len(net_shortage_products)
    products_with_timing_gap = len(timing_gap_products)
    products_no_issue = total_products - products_with_net_shortage - products_with_timing_gap
    
    # Calculate shortage quantities for each type
    net_shortage_qty = gap_df[gap_df['pt_code'].isin(net_shortage_products)]['gap_quantity'].clip(upper=0).abs().sum()
    timing_gap_qty = gap_df[gap_df['pt_code'].isin(timing_gap_products)]['gap_quantity'].clip(upper=0).abs().sum()
    
    # Calculate backlog metrics if tracking
    track_backlog = display_options.get('track_backlog', True)
    if track_backlog and 'backlog_to_next' in gap_df.columns:
        final_backlog_by_product = gap_df.groupby('pt_code')['backlog_to_next'].last()
        total_backlog = final_backlog_by_product.sum()
        products_with_backlog = (final_backlog_by_product > 0).sum()
    else:
        total_backlog = 0
        products_with_backlog = 0
    
    # Determine overall status with improved categorization
    if products_with_net_shortage > 0:
        status_color = "#dc3545"
        status_bg_color = "#f8d7da"
        status_icon = "🚨"
        status_text = "Net Shortage Detected"
        status_detail = f"{products_with_net_shortage} products need new orders | {products_with_timing_gap} products need expedite/reschedule"
    elif products_with_timing_gap > 0:
        status_color = "#ffc107"
        status_bg_color = "#fff3cd"
        status_icon = "⚠️"
        status_text = "Timing Gaps Detected"
        status_detail = f"{products_with_timing_gap} products have sufficient supply but need schedule adjustments"
    elif products_with_backlog > 0:
        status_color = "#fd7e14"
        status_bg_color = "#fff3cd"
        status_icon = "⚠️"
        status_text = "Backlog Detected"
        status_detail = f"{products_with_backlog} products have unfulfilled demand carried forward"
    else:
        status_color = "#28a745"
        status_bg_color = "#d4edda"
        status_icon = "✅"
        status_text = "Supply Meets Demand"
        status_detail = "All products have sufficient supply with proper timing"
    
    # Main status card
    st.markdown(f"""
    <div style="background-color: {status_bg_color}; padding: 20px; border-radius: 10px; border-left: 5px solid {status_color};">
        <h2 style="margin: 0; color: {status_color};">{status_icon} {status_text}</h2>
        <p style="margin: 10px 0 0 0; font-size: 18px; color: #333;">
            {status_detail}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("")
    
    # Show tracking mode info
    if track_backlog:
        st.info("📊 **Backlog Tracking: ON** - Unfulfilled demand accumulates to next periods")
    else:
        st.info("📊 **Backlog Tracking: OFF** - Each period calculated independently")
    
    # Show categorization breakdown
    st.markdown("#### 🎯 Shortage Categorization")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📦 Net Shortage",
            f"{products_with_net_shortage} products",
            delta=f"{format_number(net_shortage_qty)} units" if net_shortage_qty > 0 else "No shortage",
            delta_color="inverse" if products_with_net_shortage > 0 else "off",
            help="Products where total supply < total demand - Need new orders"
        )
    
    with col2:
        st.metric(
            "⏱️ Timing Gaps",
            f"{products_with_timing_gap} products",
            delta=f"{format_number(timing_gap_qty)} units" if timing_gap_qty > 0 else "No gaps",
            delta_color="inverse" if products_with_timing_gap > 0 else "off",
            help="Products with sufficient supply but timing mismatches - Need expedite/reschedule"
        )
    
    with col3:
        st.metric(
            "✅ No Issues",
            f"{products_no_issue} products",
            delta=f"{(products_no_issue/total_products*100):.0f}% of total" if total_products > 0 else "0%",
            delta_color="normal" if products_no_issue > 0 else "off"
        )
    
    with col4:
        coverage_rate = ((products_no_issue) / total_products * 100) if total_products > 0 else 100
        st.metric(
            "Coverage Rate",
            f"{coverage_rate:.0f}%",
            delta=f"{total_products - products_with_net_shortage - products_with_timing_gap} fully covered",
            delta_color="normal" if coverage_rate >= 80 else "inverse"
        )
    
    # Expandable action items
    with st.expander("📋 View Action Items", expanded=(products_with_net_shortage > 0 or products_with_timing_gap > 0)):
        
        if products_with_net_shortage > 0 or products_with_timing_gap > 0:
            
            action_col1, action_col2 = st.columns(2)
            
            with action_col1:
                st.markdown("##### 📦 Products Needing New Orders")
                if products_with_net_shortage > 0:
                    # Get top products with net shortage
                    net_shortage_df = gap_df[gap_df['pt_code'].isin(net_shortage_products)]
                    product_shortage = net_shortage_df.groupby('pt_code').agg({
                        'gap_quantity': lambda x: x[x < 0].sum() if any(x < 0) else 0,
                        'total_demand_qty': 'sum',
                        'supply_in_period': 'sum'
                    })
                    product_shortage['net_shortage'] = product_shortage['total_demand_qty'] - product_shortage['supply_in_period']
                    product_shortage = product_shortage[product_shortage['net_shortage'] > 0]
                    product_shortage = product_shortage.sort_values('net_shortage', ascending=False).head(5)
                    
                    for pt_code, row in product_shortage.iterrows():
                        st.caption(f"• **{pt_code}**: Order {format_number(row['net_shortage'])} units")
                else:
                    st.caption("No products need new orders")
            
            with action_col2:
                st.markdown("##### ⏱️ Products Needing Expedite/Reschedule")
                if products_with_timing_gap > 0:
                    # Get top products with timing gaps
                    timing_gap_df = gap_df[gap_df['pt_code'].isin(timing_gap_products)]
                    product_timing = timing_gap_df.groupby('pt_code').agg({
                        'gap_quantity': lambda x: x[x < 0].sum() if any(x < 0) else 0,
                        'period': lambda x: x[timing_gap_df.loc[x.index, 'gap_quantity'] < 0].iloc[0] if any(timing_gap_df.loc[x.index, 'gap_quantity'] < 0) else None
                    })
                    product_timing['gap_quantity'] = product_timing['gap_quantity'].abs()
                    product_timing = product_timing[product_timing['gap_quantity'] > 0]
                    product_timing = product_timing.sort_values('gap_quantity', ascending=False).head(5)
                    
                    for pt_code, row in product_timing.iterrows():
                        period_str = row['period'] if pd.notna(row['period']) else "Unknown"
                        st.caption(f"• **{pt_code}**: Expedite for {period_str}")
                else:
                    st.caption("No products need schedule adjustments")
        
        else:
            st.success("✅ No action items - All products are properly supplied")
        
        # Summary statistics
        st.markdown("##### 📊 Supply vs Demand Balance")
        
        if track_backlog and 'effective_demand' in gap_df.columns:
            total_demand = gap_df.groupby(['pt_code', 'period'])['effective_demand'].first().sum()
            display_demand_label = "Total Effective Demand"
        else:
            total_demand = gap_df['total_demand_qty'].sum()
            display_demand_label = "Total Demand"
        
        total_supply = gap_df['supply_in_period'].sum()
        net_position = total_supply - total_demand
        
        balance_col1, balance_col2, balance_col3 = st.columns([2, 1, 2])
        
        with balance_col1:
            st.metric(display_demand_label, format_number(total_demand))
        
        with balance_col2:
            if net_position >= 0:
                st.markdown("<h2 style='text-align: center; color: green;'>→</h2>", unsafe_allow_html=True)
            else:
                st.markdown("<h2 style='text-align: center; color: red;'>→</h2>", unsafe_allow_html=True)
        
        with balance_col3:
            st.metric(
                "Total Supply", 
                format_number(total_supply),
                delta=format_number(net_position),
                delta_color="normal" if net_position >= 0 else "inverse"
            )
        
        if total_demand > 0:
            supply_rate = min(total_supply / total_demand * 100, 100)
            st.progress(supply_rate / 100)
            st.caption(f"Supply covers {supply_rate:.1f}% of total {display_demand_label.lower()}")


def show_gap_detail_table(
    gap_df: pd.DataFrame,
    display_filters: Dict[str, Any],
    df_demand_filtered: Optional[pd.DataFrame] = None,
    df_supply_filtered: Optional[pd.DataFrame] = None
):
    """Show detailed GAP analysis table"""
    from .period_helpers import prepare_gap_detail_display, format_gap_display_df
    from .shortage_analyzer import categorize_shortage_type
    
    st.markdown("### 📋 GAP Details by Product & Period")
    
    if gap_df.empty:
        st.info("No data matches the selected filters.")
        return
    
    # Add shortage categorization info
    shortage_categorization = categorize_shortage_type(gap_df)
    
    # Show filter status
    filter_status = display_filters.get('period_filter', 'All')
    if filter_status == "Net Shortage":
        st.info(f"📦 Showing {len(shortage_categorization['net_shortage'])} products with net shortage (need new orders)")
    elif filter_status == "Timing Gap":
        st.info(f"⏱️ Showing {len(shortage_categorization['timing_gap'])} products with timing gaps (need expedite/reschedule)")
    else:
        st.caption(f"Showing {len(gap_df):,} records")
    
    # Prepare display dataframe
    display_df = prepare_gap_detail_display(
        gap_df, 
        display_filters, 
        df_demand_filtered, 
        df_supply_filtered
    )
    
    # Add shortage type column
    def get_shortage_type(pt_code):
        if pt_code in shortage_categorization['net_shortage']:
            return "Net Shortage"
        elif pt_code in shortage_categorization['timing_gap']:
            return "Timing Gap"
        else:
            return "No Issue"
    
    display_df['shortage_type'] = display_df['pt_code'].apply(get_shortage_type)
    
    # Format the dataframe
    formatted_df = format_gap_display_df(display_df, display_filters)
    
    # Apply row highlighting if enabled
    if display_filters.get("enable_row_highlighting", False):
        from .period_helpers import highlight_gap_rows_enhanced
        styled_df = formatted_df.style.apply(highlight_gap_rows_enhanced, axis=1)
        st.dataframe(styled_df, use_container_width=True, height=600)
    else:
        st.dataframe(formatted_df, use_container_width=True, height=600)


def show_gap_pivot_view(gap_df: pd.DataFrame, display_options: Dict[str, Any]):
    """Show GAP pivot view with past period indicators and shortage type info"""
    from .helpers import create_period_pivot
    from .formatters import format_number
    from .period_helpers import is_past_period
    from .shortage_analyzer import categorize_shortage_type
    
    st.markdown("### 📊 Pivot View - GAP by Period")
    
    if gap_df.empty:
        st.info("No data to display in pivot view.")
        return
    
    # Get shortage categorization
    shortage_categorization = categorize_shortage_type(gap_df)
    
    # Create pivot
    pivot_df = create_period_pivot(
        df=gap_df,
        group_cols=["product_name", "pt_code"],
        period_col="period",
        value_col="gap_quantity",
        agg_func="sum",
        period_type=display_options["period_type"],
        show_only_nonzero=False,
        fill_value=0
    )
    
    if pivot_df.empty:
        st.info("No data to display after pivoting.")
        return
    
    # Add shortage type column
    def get_shortage_type(pt_code):
        if pt_code in shortage_categorization['net_shortage']:
            return "📦"  # Net shortage icon
        elif pt_code in shortage_categorization['timing_gap']:
            return "⏱️"  # Timing gap icon
        else:
            return "✅"  # No issue icon
    
    pivot_df.insert(2, 'Type', pivot_df['pt_code'].apply(get_shortage_type))
    
    # Add past period indicators to column names
    renamed_columns = {}
    for col in pivot_df.columns:
        if col not in ["product_name", "pt_code", "Type"]:
            if is_past_period(str(col), display_options["period_type"]):
                renamed_columns[col] = f"🔴 {col}"
    
    if renamed_columns:
        pivot_df = pivot_df.rename(columns=renamed_columns)
    
    # Show legend
    st.info("**Type:** 📦 = Net Shortage (need orders) | ⏱️ = Timing Gap (need reschedule) | ✅ = No Issue | **Period:** 🔴 = Past period")
    
    # Format numbers
    for col in pivot_df.columns[3:]:  # Skip product_name, pt_code, and Type columns
        pivot_df[col] = pivot_df[col].apply(lambda x: format_number(x))
    
    st.dataframe(pivot_df, use_container_width=True, height=400)