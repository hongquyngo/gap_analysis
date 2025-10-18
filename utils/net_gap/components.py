# utils/net_gap/components.py

"""
UI Components for GAP Analysis - COMPLETE VERSION
Preserves 100% of original detailed functionality
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging

from .constants import TABLE_PRESETS, STATUS_ICONS, FIELD_TOOLTIPS, UI_CONFIG
from .formatters import GAPFormatter

logger = logging.getLogger(__name__)


def render_kpi_cards(metrics: Dict[str, Any], include_safety: bool = False):
    """Render KPI metric cards - COMPLETE"""
    
    # Row 1: Core metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📦 Total Products",
            f"{metrics['total_products']:,}",
            help="Total number of products analyzed"
        )
    
    with col2:
        shortage_pct = (metrics['shortage_items'] / max(metrics['total_products'], 1)) * 100
        st.metric(
            "⚠️ Shortage Items",
            f"{metrics['shortage_items']:,}",
            f"{shortage_pct:.1f}% of total",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            "🚨 Critical Items",
            f"{metrics['critical_items']:,}",
            "Immediate action" if metrics['critical_items'] > 0 else "All good",
            delta_color="inverse" if metrics['critical_items'] > 0 else "normal"
        )
    
    with col4:
        coverage = metrics['overall_coverage']
        st.metric(
            "📊 Coverage Rate",
            f"{coverage:.1f}%",
            "Target: 95%" if coverage < 95 else "On target",
            delta_color="normal" if coverage >= 95 else "inverse"
        )
    
    # Row 2: Supply/Demand metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📉 Total Shortage",
            f"{metrics['total_shortage']:,.0f}",
            "units"
        )
    
    with col2:
        st.metric(
            "📈 Total Surplus",
            f"{metrics['total_surplus']:,.0f}",
            "units"
        )
    
    with col3:
        st.metric(
            "💰 At Risk Value",
            f"${metrics['at_risk_value_usd']:,.0f}",
            help="Revenue at risk from shortages"
        )
    
    with col4:
        st.metric(
            "👥 Affected Customers",
            f"{metrics.get('affected_customers', 0):,}",
            help="Unique customers affected by shortages"
        )
    
    # Row 3: Safety metrics (if applicable)
    if include_safety and 'below_safety_count' in metrics:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🔒 Below Safety",
                f"{metrics.get('below_safety_count', 0):,}",
                help="Items below safety stock level"
            )
        
        with col2:
            st.metric(
                "📦 At Reorder",
                f"{metrics.get('at_reorder_count', 0):,}",
                help="Items at or below reorder point"
            )
        
        with col3:
            st.metric(
                "💵 Safety Value",
                f"${metrics.get('safety_stock_value', 0):,.0f}",
                help="Total value of safety stock"
            )
        
        with col4:
            expired_count = metrics.get('has_expired_count', 0)
            expiry_risk = metrics.get('expiry_risk_count', 0)
            
            if expired_count > 0:
                st.metric(
                    "⌛ Expired",
                    f"{expired_count:,}",
                    f"+{expiry_risk} at risk",
                    delta_color="inverse"
                )
            else:
                st.metric(
                    "✅ Expiry Status",
                    "Clear",
                    f"{expiry_risk} watch" if expiry_risk > 0 else "All good"
                )


def prepare_detailed_display(
    df: pd.DataFrame,
    formatter: GAPFormatter,
    preset: str = 'standard',
    include_safety: bool = False,
    show_all_columns: bool = False
) -> pd.DataFrame:
    """
    Prepare COMPLETE display dataframe with ALL formatting
    Preserves 100% of original detailed columns
    """
    
    display_df = df.copy()
    
    # Status display mapping
    status_display = {
        'CRITICAL_BREACH': '🚨 Critical Breach',
        'BELOW_SAFETY': '⚠️ Below Safety',
        'AT_REORDER': '📦 At Reorder',
        'HAS_EXPIRED': '❌ Has Expired',
        'EXPIRY_RISK': '⏰ Expiry Risk',
        'NO_DEMAND': '⚪ No Demand',
        'SEVERE_SHORTAGE': '🔴 Severe Shortage',
        'HIGH_SHORTAGE': '🟠 High Shortage',
        'MODERATE_SHORTAGE': '🟡 Moderate Shortage',
        'BALANCED': '✅ Balanced',
        'LIGHT_SURPLUS': '🔵 Light Surplus',
        'MODERATE_SURPLUS': '🟣 Moderate Surplus',
        'HIGH_SURPLUS': '🟠 High Surplus',
        'SEVERE_SURPLUS': '🔴 Severe Surplus'
    }
    
    # Format Status column
    if 'gap_status' in display_df.columns:
        display_df['Status'] = display_df['gap_status'].map(status_display).fillna('❓ Unknown')
    
    # Format Supply columns - COMPLETE
    if 'total_supply' in display_df.columns:
        display_df['Supply'] = display_df['total_supply'].apply(
            lambda x: formatter.format_number(x, field_name='supply')
        )
    
    # Format individual supply sources
    supply_source_cols = ['supply_inventory', 'supply_can_pending', 
                          'supply_warehouse_transfer', 'supply_purchase_order']
    for col in supply_source_cols:
        if col in display_df.columns:
            display_name = col.replace('supply_', '').replace('_', ' ').title()
            display_df[display_name] = display_df[col].apply(
                lambda x: formatter.format_number(x, field_name=col)
            )
    
    # Format Demand columns - COMPLETE
    if 'total_demand' in display_df.columns:
        display_df['Demand'] = display_df['total_demand'].apply(formatter.format_number)
    
    # Format individual demand sources
    demand_source_cols = ['demand_oc_pending', 'demand_forecast']
    for col in demand_source_cols:
        if col in display_df.columns:
            display_name = col.replace('demand_', '').replace('_', ' ').title()
            display_df[display_name] = display_df[col].apply(formatter.format_number)
    
    # Format GAP columns
    if 'net_gap' in display_df.columns:
        display_df['Net GAP'] = display_df['net_gap'].apply(
            lambda x: formatter.format_number(x, show_sign=True)
        )
    
    # Format Coverage
    if 'coverage_ratio' in display_df.columns:
        display_df['Coverage'] = display_df['coverage_ratio'].apply(formatter.format_coverage)
    
    if 'gap_percentage' in display_df.columns:
        display_df['GAP %'] = display_df['gap_percentage'].apply(
            lambda x: formatter.format_percentage(x, show_sign=True)
        )
    
    # Safety Stock columns - COMPLETE
    if include_safety:
        if 'safety_stock_qty' in display_df.columns:
            display_df['Safety Stock'] = display_df['safety_stock_qty'].apply(
                lambda x: formatter.format_number(x, field_name='safety_stock_qty')
            )
        
        if 'available_supply' in display_df.columns:
            display_df['Available'] = display_df['available_supply'].apply(
                lambda x: formatter.format_number(x, field_name='available_supply')
            )
            
            # Calculate True GAP (ignoring safety)
            if 'total_supply' in display_df.columns and 'total_demand' in display_df.columns:
                true_gap = display_df['total_supply'] - display_df['total_demand']
                display_df['True GAP'] = true_gap.apply(
                    lambda x: formatter.format_number(x, show_sign=True)
                )
        
        if 'safety_coverage' in display_df.columns:
            display_df['Safety Cov'] = display_df['safety_coverage'].apply(
                lambda x: f"{x:.1f}x" if pd.notna(x) and x < 999 else "N/A"
            )
        
        if 'days_of_supply' in display_df.columns:
            display_df['Days Supply'] = display_df['days_of_supply'].apply(formatter.format_days)
        
        if 'below_reorder' in display_df.columns:
            display_df['Reorder'] = display_df['below_reorder'].apply(
                lambda x: '⚠️ Yes' if x else '✅ No'
            )
    
    # Financial columns - COMPLETE
    if 'avg_unit_cost_usd' in display_df.columns:
        display_df['Unit Cost'] = display_df['avg_unit_cost_usd'].apply(
            lambda x: formatter.format_currency(x, decimals=2)
        )
    
    if 'avg_selling_price_usd' in display_df.columns:
        display_df['Sell Price'] = display_df['avg_selling_price_usd'].apply(
            lambda x: formatter.format_currency(x, decimals=2)
        )
    
    if 'at_risk_value_usd' in display_df.columns:
        display_df['At Risk Value'] = display_df['at_risk_value_usd'].apply(
            lambda x: formatter.format_currency(x, abbreviate=True)
        )
    
    if 'gap_value_usd' in display_df.columns:
        display_df['GAP Value'] = display_df['gap_value_usd'].apply(
            lambda x: formatter.format_currency(x, abbreviate=True)
        )
    
    if 'supply_value_usd' in display_df.columns:
        display_df['Supply Value'] = display_df['supply_value_usd'].apply(
            lambda x: formatter.format_currency(x, abbreviate=True)
        )
    
    if 'demand_value_usd' in display_df.columns:
        display_df['Demand Value'] = display_df['demand_value_usd'].apply(
            lambda x: formatter.format_currency(x, abbreviate=True)
        )
    
    # Customer metrics
    if 'customer_count' in display_df.columns:
        display_df['Customers'] = display_df['customer_count'].apply(
            lambda x: f"{int(x):,}" if pd.notna(x) else "0"
        )
    
    # Urgency metrics
    if 'count_overdue' in display_df.columns:
        display_df['Overdue'] = display_df['count_overdue'].apply(
            lambda x: f"{int(x):,}" if x > 0 else "-"
        )
    
    if 'count_urgent' in display_df.columns:
        display_df['Urgent'] = display_df['count_urgent'].apply(
            lambda x: f"{int(x):,}" if x > 0 else "-"
        )
    
    # Priority
    if 'priority' in display_df.columns:
        priority_display = {1: 'P1-Critical', 2: 'P2-High', 3: 'P3-Medium', 4: 'P4-Low', 99: 'P99-OK'}
        display_df['Priority'] = display_df['priority'].map(priority_display).fillna('Unknown')
    
    # Action
    if 'suggested_action' in display_df.columns:
        display_df['Action'] = display_df['suggested_action']
    
    # Expiry information
    if 'expired_qty' in display_df.columns:
        display_df['Expired Qty'] = display_df['expired_qty'].apply(
            lambda x: formatter.format_number(x) if x > 0 else "-"
        )
    
    if 'near_expiry_qty' in display_df.columns:
        display_df['Near Expiry'] = display_df['near_expiry_qty'].apply(
            lambda x: formatter.format_number(x) if x > 0 else "-"
        )
    
    # Select columns based on preset or show all
    if show_all_columns:
        # Return ALL formatted columns
        return display_df
    else:
        # Select columns based on preset
        columns = get_display_columns(preset, display_df.columns, include_safety)
        return display_df[columns] if columns else display_df


def get_display_columns(preset: str, available_columns: list, include_safety: bool) -> list:
    """
    Get display columns based on preset
    Returns COMPLETE column list preserving all details
    """
    
    # Base columns (always shown)
    base_cols = ['pt_code', 'product_name', 'brand']
    
    # Supply/Demand columns
    supply_demand_cols = ['Supply', 'Demand', 'Net GAP']
    
    # Detailed supply columns
    supply_detail_cols = ['Inventory', 'Can Pending', 'Warehouse Transfer', 'Purchase Order']
    
    # Detailed demand columns  
    demand_detail_cols = ['Oc Pending', 'Forecast']
    
    # Analysis columns
    analysis_cols = ['Coverage', 'GAP %', 'Status', 'Priority', 'Action']
    
    # Safety columns
    safety_cols = ['Safety Stock', 'Available', 'True GAP', 'Safety Cov', 'Days Supply', 'Reorder']
    
    # Financial columns
    financial_cols = ['Unit Cost', 'Sell Price', 'At Risk Value', 'GAP Value', 
                     'Supply Value', 'Demand Value']
    
    # Customer columns
    customer_cols = ['Customers', 'Overdue', 'Urgent']
    
    # Expiry columns
    expiry_cols = ['Expired Qty', 'Near Expiry']
    
    # Build column list based on preset
    if preset == 'standard':
        columns = base_cols + supply_demand_cols + analysis_cols[:3]
    elif preset == 'safety' and include_safety:
        columns = base_cols + safety_cols + analysis_cols[:2]
    elif preset == 'financial':
        columns = base_cols + supply_demand_cols + financial_cols
    elif preset == 'detailed':
        columns = base_cols + supply_demand_cols + supply_detail_cols + demand_detail_cols + \
                 analysis_cols + financial_cols[:4]
        if include_safety:
            columns.extend(safety_cols[:3])
    else:  # 'all' or fallback
        columns = base_cols + supply_demand_cols + supply_detail_cols + demand_detail_cols + \
                 analysis_cols + financial_cols + customer_cols
        if include_safety:
            columns.extend(safety_cols)
        columns.extend(expiry_cols)
    
    # Filter to available columns only
    return [col for col in columns if col in available_columns]


def render_data_table(
    df: pd.DataFrame,
    items_per_page: int = 25,
    current_page: int = 1,
    formatter: Optional[GAPFormatter] = None,
    include_safety: bool = False
):
    """
    Simplified data table - load ALL columns, show default subset
    User controls visibility via Streamlit's built-in column selector
    """
    
    if formatter is None:
        formatter = GAPFormatter()
    
    if df.empty:
        st.info("No data matches current filters")
        return None
    
    # Prepare ALL columns with complete formatting
    display_df = prepare_detailed_display(
        df, 
        formatter, 
        preset='all',  # Always load ALL columns
        include_safety=include_safety,
        show_all_columns=True
    )
    
    # Default visible columns (like current "Detailed" preset)
    default_visible = [
        'pt_code', 'product_name', 'brand',
        'Supply', 'Inventory', 'Can Pending', 'Warehouse Transfer', 'Purchase Order',
        'Demand', 'Oc Pending', 'Forecast',
        'Net GAP', 'Coverage', 'GAP %', 'Status', 'Priority', 'Action'
    ]
    
    # Filter to available columns
    visible_columns = [col for col in default_visible if col in display_df.columns]
    
    # Add remaining columns (hidden by default)
    all_other_columns = [col for col in display_df.columns if col not in visible_columns]
    column_order = visible_columns + all_other_columns
    
    # Reorder dataframe
    display_df = display_df[column_order]
    
    # Pagination
    total_items = len(display_df)
    total_pages = max(1, (total_items + items_per_page - 1) // items_per_page)
    page = min(current_page, total_pages)
    
    start_idx = (page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, total_items)
    
    # Display info
    st.caption(f"Showing {start_idx+1}-{end_idx} of {total_items} items | Columns: {len(display_df.columns)}")
    
    # Display table with ALL columns (user can toggle visibility)
    st.dataframe(
        display_df.iloc[start_idx:end_idx],
        use_container_width=True,
        hide_index=True,
        column_config={
            col: st.column_config.Column(col, help=FIELD_TOOLTIPS.get(col))
            for col in display_df.columns if col in FIELD_TOOLTIPS
        },
        column_order=column_order,  # Default visible columns first
        height=min(600, (end_idx - start_idx) * 35 + 100)
    )
    
    return {
        'page': page,
        'total_pages': total_pages,
        'total_items': total_items,
        'columns': len(display_df.columns)
    }


def render_table_presets(on_preset_change=None, include_safety: bool = False):
    """Render table preset buttons"""
    
    # Define available presets
    presets = {
        'standard': '📊 Standard',
        'safety': '🔒 Safety',
        'financial': '💰 Financial', 
        'detailed': '📋 Detailed',
        'all': '🔍 All Columns'
    }
    
    # Remove safety preset if not applicable
    if not include_safety:
        presets.pop('safety', None)
    
    cols = st.columns(len(presets))
    
    for idx, (key, label) in enumerate(presets.items()):
        with cols[idx]:
            if st.button(label, use_container_width=True, key=f"preset_{key}"):
                if on_preset_change:
                    on_preset_change(key)
                return key
    
    return None


def render_table_configuration(current_config: Dict[str, bool]) -> Dict[str, bool]:
    """
    Render advanced table configuration options
    Preserves original column selection functionality
    """
    
    with st.expander("⚙️ Advanced Table Configuration", expanded=False):
        st.markdown("**Select column groups to display:**")
        
        new_config = {}
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Basic Information**")
            new_config['basic'] = st.checkbox("Product Info", value=current_config.get('basic', True))
            new_config['supply'] = st.checkbox("Supply & Demand", value=current_config.get('supply', True))
            new_config['supply_details'] = st.checkbox("Supply Sources", value=current_config.get('supply_details', False))
        
        with col2:
            st.markdown("**Analysis**")
            new_config['analysis'] = st.checkbox("GAP Analysis", value=current_config.get('analysis', True))
            new_config['safety'] = st.checkbox("Safety Stock", value=current_config.get('safety', False))
            new_config['demand_details'] = st.checkbox("Demand Sources", value=current_config.get('demand_details', False))
        
        with col3:
            st.markdown("**Additional**")
            new_config['financial'] = st.checkbox("Financial", value=current_config.get('financial', False))
            new_config['customer'] = st.checkbox("Customer Info", value=current_config.get('customer', False))
            new_config['expiry'] = st.checkbox("Expiry Info", value=current_config.get('expiry', False))
        
        # Quick selection buttons
        st.markdown("**Quick Selection:**")
        button_cols = st.columns(4)
        
        if button_cols[0].button("Essential", use_container_width=True):
            return {'basic': True, 'supply': True, 'analysis': True}
        
        if button_cols[1].button("Complete", use_container_width=True):
            return {k: True for k in new_config.keys()}
        
        if button_cols[2].button("Financial Focus", use_container_width=True):
            return {'basic': True, 'supply': True, 'financial': True, 'analysis': True}
        
        if button_cols[3].button("Reset", use_container_width=True):
            return {'basic': True, 'supply': True, 'analysis': True}
    
    return new_config


def render_pagination(current_page: int, total_pages: int, key_prefix: str = "page"):
    """Render pagination controls - COMPLETE"""
    
    if total_pages <= 1:
        return current_page
    
    col1, col2, col3, col4, col5 = st.columns([1, 1, 3, 1, 1])
    
    new_page = current_page
    
    with col1:
        if st.button("⏮", disabled=(current_page == 1), key=f"{key_prefix}_first"):
            new_page = 1
    
    with col2:
        if st.button("◀", disabled=(current_page == 1), key=f"{key_prefix}_prev"):
            new_page = current_page - 1
    
    with col3:
        st.markdown(
            f"<div style='text-align: center; padding: 8px;'>"
            f"Page <b>{current_page}</b> of <b>{total_pages}</b>"
            f"</div>",
            unsafe_allow_html=True
        )
    
    with col4:
        if st.button("▶", disabled=(current_page == total_pages), key=f"{key_prefix}_next"):
            new_page = current_page + 1
    
    with col5:
        if st.button("⏭", disabled=(current_page == total_pages), key=f"{key_prefix}_last"):
            new_page = total_pages
    
    return new_page


def render_status_summary(gap_df: pd.DataFrame):
    """Render detailed status summary"""
    from .constants import GAP_CATEGORIES
    
    if gap_df.empty:
        return
    
    # Count by category
    counts = {}
    for category, config in GAP_CATEGORIES.items():
        mask = gap_df['gap_status'].isin(config['statuses'])
        count = len(gap_df[mask])
        if count > 0:
            counts[category] = {
                'count': count,
                'pct': (count / len(gap_df)) * 100,
                'icon': config['icon'],
                'label': config['label']
            }
    
    # Display as metrics
    cols = st.columns(len(counts))
    for idx, (category, data) in enumerate(counts.items()):
        with cols[idx]:
            st.metric(
                f"{data['icon']} {data['label']}",
                f"{data['count']:,}",
                f"{data['pct']:.1f}%"
            )


def render_quick_filter():
    """Render quick filter for results"""
    filter_options = {
        'all': '📊 All Items',
        'shortage': '🔴 Shortage Only',
        'optimal': '✅ Optimal Only',
        'surplus': '📦 Surplus Only',
        'inactive': '⭕ No Demand',
        'critical': '🚨 Critical Only'
    }
    
    selected = st.radio(
        "Quick Filter",
        options=list(filter_options.keys()),
        format_func=lambda x: filter_options[x],
        horizontal=True,
        label_visibility="collapsed",
        key="quick_filter",
        help="Filter displayed results by status"
    )
    
    return selected


def apply_quick_filter(df: pd.DataFrame, filter_type: str) -> pd.DataFrame:
    """Apply quick filter to dataframe - ENHANCED"""
    from .constants import GAP_CATEGORIES
    
    if filter_type == 'all' or df.empty:
        return df
    
    if filter_type == 'critical':
        # Show only critical priority items
        return df[df['priority'] == 1]
    
    # Map filter to category
    category_map = {
        'shortage': 'SHORTAGE',
        'optimal': 'OPTIMAL',
        'surplus': 'SURPLUS',
        'inactive': 'INACTIVE'
    }
    
    category = category_map.get(filter_type)
    if category and category in GAP_CATEGORIES:
        statuses = GAP_CATEGORIES[category]['statuses']
        return df[df['gap_status'].isin(statuses)]
    
    return df