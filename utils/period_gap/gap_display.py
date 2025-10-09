# utils/period_gap/gap_display.py
"""
Display Functions for Period GAP Analysis
Handles all visualization and presentation logic
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
    Show GAP analysis summary with backlog tracking support
    
    Args:
        gap_df: GAP analysis results
        display_options: Display configuration options
        df_demand_filtered: Filtered demand data (for additional context)
        df_supply_filtered: Filtered supply data (for additional context)
    """
    from .formatters import format_number, format_currency
    from .period_helpers import is_past_period
    
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
    
    # === KEY INSIGHTS ===
    st.markdown("#### 🎯 Key Insights")
    
    # Calculate essential metrics
    total_shortage = gap_df[gap_df['gap_quantity'] < 0]['gap_quantity'].abs().sum()
    shortage_products = gap_df[gap_df['gap_quantity'] < 0]['pt_code'].nunique()
    total_products = gap_df['pt_code'].nunique()
    total_periods = gap_df['period'].nunique()
    periods_with_shortage = gap_df[gap_df['gap_quantity'] < 0]['period'].nunique()
    
    # Calculate backlog metrics if tracking
    track_backlog = display_options.get('track_backlog', True)
    if track_backlog and 'backlog_to_next' in gap_df.columns:
        final_backlog_by_product = gap_df.groupby('pt_code')['backlog_to_next'].last()
        total_backlog = final_backlog_by_product.sum()
        products_with_backlog = (final_backlog_by_product > 0).sum()
        
        max_backlog_by_product = gap_df.groupby('pt_code')['backlog_qty'].max()
        peak_total_backlog = max_backlog_by_product.sum()
        products_ever_had_backlog = (max_backlog_by_product > 0).sum()
    else:
        total_backlog = 0
        products_with_backlog = 0
        peak_total_backlog = 0
        products_ever_had_backlog = 0
    
    # Determine overall status
    if total_shortage == 0 and total_backlog == 0:
        status_color = "#28a745"
        status_bg_color = "#d4edda"
        status_icon = "✅"
        status_text = "No Shortage Detected"
        status_detail = "Supply meets demand for all products across all periods"
    elif products_with_backlog > 0:
        status_color = "#fd7e14"
        status_bg_color = "#fff3cd"
        status_icon = "⚠️"
        status_text = "Backlog Detected"
        status_detail = f"{products_with_backlog} of {total_products} products have unfulfilled demand carried forward"
    elif shortage_products / total_products > 0.5 or periods_with_shortage / total_periods > 0.5:
        status_color = "#dc3545"
        status_bg_color = "#f8d7da"
        status_icon = "🚨"
        status_text = "Critical Shortage"
        status_detail = f"{shortage_products} of {total_products} products need immediate attention"
    else:
        status_color = "#ffc107"
        status_bg_color = "#fff3cd"
        status_icon = "⚠️"
        status_text = "Partial Shortage"
        status_detail = f"{shortage_products} of {total_products} products have shortage in some periods"
    
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
    
    # Essential metrics
    if track_backlog and 'backlog_to_next' in gap_df.columns:
        col1, col2, col3, col4 = st.columns(4)
    else:
        col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Shortage Quantity",
            format_number(total_shortage),
            delta=f"{shortage_products} products" if shortage_products > 0 else "No products",
            delta_color="inverse" if shortage_products > 0 else "off"
        )
    
    with col2:
        coverage_rate = ((total_products - shortage_products) / total_products * 100) if total_products > 0 else 100
        st.metric(
            "Product Coverage Rate",
            f"{coverage_rate:.0f}%",
            delta=f"{total_products - shortage_products} of {total_products} covered",
            delta_color="normal" if coverage_rate >= 80 else "inverse"
        )
    
    with col3:
        if track_backlog and 'backlog_to_next' in gap_df.columns:
            st.metric(
                "Current Backlog",
                format_number(total_backlog),
                delta=f"{products_with_backlog} products" if products_with_backlog > 0 else "All clear",
                delta_color="inverse" if total_backlog > 0 else "off"
            )
        else:
            period_type = display_options.get('period_type', 'Weekly')
            past_periods = gap_df[
                gap_df['period'].apply(lambda x: is_past_period(str(x), period_type))
            ]['period'].nunique()
            
            future_periods = total_periods - past_periods
            st.metric(
                "Planning Horizon",
                f"{future_periods} {period_type.lower()} periods",
                delta=f"{past_periods} periods passed" if past_periods > 0 else "All future",
                delta_color="inverse" if past_periods > 0 else "off"
            )
    
    if track_backlog and 'backlog_to_next' in gap_df.columns:
        with col4:
            st.metric(
                "Peak Backlog",
                format_number(peak_total_backlog),
                delta=f"{products_ever_had_backlog} products affected",
                delta_color="inverse" if peak_total_backlog > total_backlog else "normal"
            )
    
    # === EXPANDABLE DETAILS ===
    with st.expander("📈 View Detailed Analysis", expanded=bool(shortage_products > 0 or products_with_backlog > 0)):
        
        if shortage_products > 0 or products_with_backlog > 0:
            st.markdown("##### 🎯 Action Required")
            
            action_col1, action_col2 = st.columns(2)
            
            with action_col1:
                st.markdown("**🔴 Products with Issues:**")
                
                if track_backlog and 'backlog_to_next' in gap_df.columns:
                    product_issues = gap_df.groupby('pt_code').agg({
                        'gap_quantity': lambda x: x[x < 0].sum() if any(x < 0) else 0,
                        'backlog_to_next': 'last',
                        'period': 'count'
                    }).rename(columns={'period': 'affected_periods'})
                    
                    product_issues['total_issue'] = product_issues['gap_quantity'].abs() + product_issues['backlog_to_next']
                    product_issues = product_issues[product_issues['total_issue'] > 0]
                    product_issues = product_issues.sort_values('total_issue', ascending=False).head(5)
                    
                    for pt_code, row in product_issues.iterrows():
                        issue_text = []
                        if row['gap_quantity'] < 0:
                            issue_text.append(f"Shortage: {format_number(abs(row['gap_quantity']))}")
                        if row['backlog_to_next'] > 0:
                            issue_text.append(f"Backlog: {format_number(row['backlog_to_next'])}")
                        st.caption(f"• **{pt_code}**: {' | '.join(issue_text)} ({row['affected_periods']} periods)")
                else:
                    product_shortages = gap_df[gap_df['gap_quantity'] < 0].groupby('pt_code').agg({
                        'gap_quantity': 'sum',
                        'period': 'count'
                    }).rename(columns={'period': 'affected_periods'})
                    product_shortages['gap_quantity'] = product_shortages['gap_quantity'].abs()
                    product_shortages = product_shortages.sort_values('gap_quantity', ascending=False).head(5)
                    
                    for pt_code, row in product_shortages.iterrows():
                        st.caption(f"• **{pt_code}**: {format_number(row['gap_quantity'])} units ({row['affected_periods']} periods)")
            
            with action_col2:
                st.markdown("**📅 Periods with Issues:**")
                
                period_type = display_options.get('period_type', 'Weekly')
                
                if track_backlog and 'backlog_qty' in gap_df.columns:
                    period_issues = gap_df.groupby('period').agg({
                        'gap_quantity': lambda x: x[x < 0].sum() if any(x < 0) else 0,
                        'backlog_qty': 'sum',
                        'pt_code': 'nunique'
                    }).rename(columns={'pt_code': 'products_affected'})
                    
                    period_issues['has_issue'] = (period_issues['gap_quantity'] < 0) | (period_issues['backlog_qty'] > 0)
                    period_issues = period_issues[period_issues['has_issue']]
                    period_issues['total_issue'] = period_issues['gap_quantity'].abs() + period_issues['backlog_qty']
                    period_issues = period_issues.sort_values('total_issue', ascending=False).head(5)
                    
                    for period, row in period_issues.iterrows():
                        is_past = is_past_period(str(period), period_type)
                        indicator = "🔴" if is_past else "🟡"
                        issue_parts = []
                        if row['gap_quantity'] < 0:
                            issue_parts.append(f"Gap: {format_number(abs(row['gap_quantity']))}")
                        if row['backlog_qty'] > 0:
                            issue_parts.append(f"Backlog: {format_number(row['backlog_qty'])}")
                        st.caption(f"{indicator} **{period}**: {' | '.join(issue_parts)} ({row['products_affected']} products)")
                else:
                    period_shortages = gap_df[gap_df['gap_quantity'] < 0].groupby('period').agg({
                        'gap_quantity': 'sum',
                        'pt_code': 'nunique'
                    }).rename(columns={'pt_code': 'products_affected'})
                    period_shortages['gap_quantity'] = period_shortages['gap_quantity'].abs()
                    period_shortages = period_shortages.sort_values('gap_quantity', ascending=False).head(5)
                    
                    for period, row in period_shortages.iterrows():
                        is_past = is_past_period(str(period), period_type)
                        indicator = "🔴" if is_past else "🟡"
                        st.caption(f"{indicator} **{period}**: {format_number(row['gap_quantity'])} units ({row['products_affected']} products)")
        
        # Supply vs Demand Overview
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
    
    st.markdown("### 📄 GAP Details by Product & Period")
    
    if gap_df.empty:
        st.info("No data matches the selected filters.")
        return
    
    st.caption(f"Showing {len(gap_df):,} records")
    
    # Prepare display dataframe
    display_df = prepare_gap_detail_display(
        gap_df, 
        display_filters, 
        df_demand_filtered, 
        df_supply_filtered
    )
    
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
    """Show GAP pivot view with past period indicators"""
    from .helpers import create_period_pivot, apply_period_indicators
    from .formatters import format_number
    
    st.markdown("### 📊 Pivot View - GAP by Period")
    
    if gap_df.empty:
        st.info("No data to display in pivot view.")
        return
    
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
    
    # Add past period indicators
    display_pivot = apply_period_indicators(
        df=pivot_df,
        period_type=display_options["period_type"],
        exclude_cols=["product_name", "pt_code"],
        indicator="🔴"
    )
    
    # Show legend
    st.info("🔴 = Past period (already occurred)")
    
    # Format numbers
    for col in display_pivot.columns[2:]:
        display_pivot[col] = display_pivot[col].apply(lambda x: format_number(x))
    
    st.dataframe(display_pivot, use_container_width=True)