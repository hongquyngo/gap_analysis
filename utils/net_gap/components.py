# utils/net_gap/components.py

"""
UI Components for GAP Analysis - Enhanced Version
Added formula guide and improved dataframe display
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging

from .constants import STATUS_ICONS, FIELD_TOOLTIPS, UI_CONFIG
from .formatters import GAPFormatter

logger = logging.getLogger(__name__)


def render_formula_guide():
    """Render expandable formula explanation guide"""
    
    with st.expander("📊 **GAP Calculation Guide** - Click to understand the formulas", expanded=False):
        
        # Main formulas
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            ### Core Formulas
            
            **Net GAP** = Available Supply - Total Demand
            - When safety enabled: Available = Supply - Safety Stock
            - When safety disabled: Available = Supply
            
            **True GAP** = Total Supply - Total Demand
            - Always ignores safety stock
            - Shows actual supply vs demand difference
            
            **Coverage Ratio** = (Supply ÷ Demand) × 100%
            - Shows supply as percentage of demand
            - >100% means surplus, <100% means shortage
            """)
        
        with col2:
            st.markdown("""
            ### Financial Calculations
            
            **At Risk Value** = |Shortage Qty| × Selling Price
            - Revenue that could be lost due to shortage
            
            **GAP Value** = Net GAP × Unit Cost
            - Inventory value of the gap
            
            **Safety Stock Impact** = Net GAP - True GAP
            - Shows how safety stock affects the gap
            - Negative means safety stock creates shortage
            """)
        
        # Example scenarios
        st.markdown("---")
        st.markdown("### Example Scenarios")
        
        example_data = pd.DataFrame([
            {
                'Scenario': '✅ Safe',
                'Supply': 100,
                'Safety': 20,
                'Demand': 70,
                'Net GAP': '+10',
                'True GAP': '+30',
                'Interpretation': 'OK with safety, 30 units actual surplus'
            },
            {
                'Scenario': '⚠️ Risk',
                'Supply': 100,
                'Safety': 20,
                'Demand': 90,
                'Net GAP': '-10',
                'True GAP': '+10',
                'Interpretation': 'Shortage with safety, but stock exists'
            },
            {
                'Scenario': '🔴 Critical',
                'Supply': 50,
                'Safety': 20,
                'Demand': 80,
                'Net GAP': '-50',
                'True GAP': '-30',
                'Interpretation': 'Real shortage even without safety'
            }
        ])
        
        st.dataframe(
            example_data,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Supply': st.column_config.NumberColumn(format="%d"),
                'Safety': st.column_config.NumberColumn(format="%d"),
                'Demand': st.column_config.NumberColumn(format="%d"),
            }
        )
        
        # Quick tips
        st.info("""
        💡 **Quick Tips:**
        - **Net GAP < 0**: You have a shortage considering safety requirements
        - **True GAP < 0**: You have a real shortage (not enough stock even without safety)
        - **Net GAP < 0 but True GAP > 0**: Safety stock is causing the shortage
        """)


def render_kpi_cards(metrics: Dict[str, Any], include_safety: bool = False):
    """Render KPI metric cards"""
    
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
    include_safety: bool = False
) -> pd.DataFrame:
    """
    Prepare display dataframe with selected meaningful columns
    Removed product_id as pt_code serves the same purpose
    """
    
    if df.empty:
        return df
    
    # Create a copy for display
    display_df = pd.DataFrame()
    
    # Product identification (no product_id, using pt_code instead)
    if 'pt_code' in df.columns:
        display_df['PT Code'] = df['pt_code']
    if 'product_name' in df.columns:
        display_df['Product Name'] = df['product_name']
    if 'brand' in df.columns:
        display_df['Brand'] = df['brand']
    if 'standard_uom' in df.columns:
        display_df['UOM'] = df['standard_uom']
    
    # Supply columns (raw values)
    if 'total_supply' in df.columns:
        display_df['Total Supply'] = df['total_supply'].apply(
            lambda x: formatter.format_number(x, field_name='total_supply')
        )
    
    # Supply breakdown
    if 'supply_inventory' in df.columns:
        display_df['Inventory'] = df['supply_inventory'].apply(
            lambda x: formatter.format_number(x, field_name='supply_inventory')
        )
    if 'supply_can_pending' in df.columns:
        display_df['CAN Pending'] = df['supply_can_pending'].apply(
            lambda x: formatter.format_number(x, field_name='supply_can_pending')
        )
    if 'supply_warehouse_transfer' in df.columns:
        display_df['Transfer'] = df['supply_warehouse_transfer'].apply(
            lambda x: formatter.format_number(x, field_name='supply_warehouse_transfer')
        )
    if 'supply_purchase_order' in df.columns:
        display_df['PO'] = df['supply_purchase_order'].apply(
            lambda x: formatter.format_number(x, field_name='supply_purchase_order')
        )
    
    # Demand columns
    if 'total_demand' in df.columns:
        display_df['Total Demand'] = df['total_demand'].apply(formatter.format_number)
    
    if 'demand_oc_pending' in df.columns:
        display_df['OC Pending'] = df['demand_oc_pending'].apply(formatter.format_number)
    if 'demand_forecast' in df.columns:
        display_df['Forecast'] = df['demand_forecast'].apply(formatter.format_number)
    
    # GAP Analysis - Keep both Net GAP and True GAP
    if 'net_gap' in df.columns:
        display_df['Net GAP'] = df['net_gap'].apply(
            lambda x: formatter.format_number(x, show_sign=True)
        )
    
    # True GAP - Important to keep for transparency
    if include_safety and 'total_supply' in df.columns and 'total_demand' in df.columns:
        true_gap = df['total_supply'] - df['total_demand']
        display_df['True GAP'] = true_gap.apply(
            lambda x: formatter.format_number(x, show_sign=True)
        )
    
    # Coverage metrics
    if 'coverage_ratio' in df.columns:
        display_df['Coverage %'] = df['coverage_ratio'].apply(formatter.format_coverage)
    
    if 'gap_percentage' in df.columns:
        display_df['GAP %'] = df['gap_percentage'].apply(
            lambda x: formatter.format_percentage(x, show_sign=True)
        )
    
    # Safety Stock columns (when enabled)
    if include_safety:
        if 'safety_stock_qty' in df.columns:
            display_df['Safety Stock'] = df['safety_stock_qty'].apply(
                lambda x: formatter.format_number(x, field_name='safety_stock_qty')
            )
        
        if 'available_supply' in df.columns:
            display_df['Available Supply'] = df['available_supply'].apply(
                lambda x: formatter.format_number(x, field_name='available_supply')
            )
        
        if 'reorder_point' in df.columns:
            display_df['Reorder Point'] = df['reorder_point'].apply(
                lambda x: formatter.format_number(x, field_name='reorder_point')
            )
        
        if 'below_reorder' in df.columns:
            display_df['Below Reorder'] = df['below_reorder'].apply(
                lambda x: '⚠️ Yes' if x else '✅ No'
            )
        
        if 'safety_coverage' in df.columns:
            display_df['Safety Coverage'] = df['safety_coverage'].apply(
                lambda x: f"{x:.1f}x" if pd.notna(x) and x < 999 else "N/A"
            )
    
    # Financial columns
    if 'avg_unit_cost_usd' in df.columns:
        display_df['Unit Cost'] = df['avg_unit_cost_usd'].apply(
            lambda x: formatter.format_currency(x, decimals=2)
        )
    
    if 'avg_selling_price_usd' in df.columns:
        display_df['Sell Price'] = df['avg_selling_price_usd'].apply(
            lambda x: formatter.format_currency(x, decimals=2)
        )
    
    if 'at_risk_value_usd' in df.columns:
        display_df['At Risk Value'] = df['at_risk_value_usd'].apply(
            lambda x: formatter.format_currency(x, abbreviate=True)
        )
    
    if 'gap_value_usd' in df.columns:
        display_df['GAP Value'] = df['gap_value_usd'].apply(
            lambda x: formatter.format_currency(x, abbreviate=True)
        )
    
    # Status columns
    if 'gap_status' in df.columns:
        status_icons = {
            'CRITICAL_BREACH': '🚨',
            'SEVERE_SHORTAGE': '🔴',
            'HIGH_SHORTAGE': '🟠',
            'MODERATE_SHORTAGE': '🟡',
            'BALANCED': '✅',
            'LIGHT_SURPLUS': '🔵',
            'MODERATE_SURPLUS': '🟣',
            'HIGH_SURPLUS': '🟠',
            'SEVERE_SURPLUS': '🔴',
            'BELOW_SAFETY': '⚠️',
            'NO_DEMAND': '⚪'
        }
        display_df['Status'] = df['gap_status'].map(
            lambda x: f"{status_icons.get(x, '❓')} {x.replace('_', ' ').title()}"
        )
    
    if 'priority' in df.columns:
        priority_map = {1: 'P1-Critical', 2: 'P2-High', 3: 'P3-Medium', 4: 'P4-Low', 99: 'P99-OK'}
        display_df['Priority'] = df['priority'].map(priority_map).fillna('Unknown')
    
    if 'suggested_action' in df.columns:
        display_df['Action'] = df['suggested_action']
    
    # Customer impact
    if 'customer_count' in df.columns:
        display_df['Customers'] = df['customer_count'].apply(
            lambda x: f"{int(x):,}" if pd.notna(x) and x > 0 else "-"
        )
    
    return display_df


def render_data_table(
    df: pd.DataFrame,
    items_per_page: int = 25,
    current_page: int = 1,
    formatter: Optional[GAPFormatter] = None,
    include_safety: bool = False
):
    """Enhanced data table with column visibility control"""
    
    if formatter is None:
        formatter = GAPFormatter()
    
    if df.empty:
        st.info("No data matches current filters")
        return None
    
    # Prepare display dataframe
    display_df = prepare_detailed_display(
        df, 
        formatter, 
        include_safety=include_safety
    )
    
    # Define default visible columns (essential only)
    default_visible = [
        'PT Code', 'Product Name', 'Brand',
        'Total Supply', 'Total Demand', 
        'Net GAP', 'Coverage %',
        'Status', 'Priority'
    ]
    
    # Add True GAP if safety is enabled
    if include_safety and 'True GAP' in display_df.columns:
        default_visible.insert(6, 'True GAP')  # Insert after Net GAP
    
    # Configure column visibility
    column_config = {}
    
    # Set help text for important columns
    help_texts = {
        'Net GAP': 'Supply minus Demand (considering safety if enabled)',
        'True GAP': 'Supply minus Demand (ignoring safety stock)',
        'Coverage %': 'Supply as percentage of Demand',
        'Available Supply': 'Supply after safety stock reservation',
        'Safety Coverage': 'How many times safety stock is covered',
        'At Risk Value': 'Revenue at risk due to shortage'
    }
    
    for col, help_text in help_texts.items():
        if col in display_df.columns:
            column_config[col] = st.column_config.Column(
                col,
                help=help_text
            )
    
    # Pagination
    total_items = len(display_df)
    total_pages = max(1, (total_items + items_per_page - 1) // items_per_page)
    page = min(current_page, total_pages)
    
    start_idx = (page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, total_items)
    
    # Display info
    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption(f"Showing {start_idx+1}-{end_idx} of {total_items} items | {len(display_df.columns)} columns available")
    with col2:
        st.caption("👁️ Click column headers to show/hide")
    
    # Display table with configurable columns
    st.dataframe(
        display_df.iloc[start_idx:end_idx],
        use_container_width=True,
        hide_index=True,
        column_config=column_config,
        height=min(600, (end_idx - start_idx) * 35 + 50),
        # Enable column visibility toggle
        key=f"gap_table_{page}"
    )
    
    return {
        'page': page,
        'total_pages': total_pages,
        'total_items': total_items,
        'columns': len(display_df.columns)
    }


def render_pagination(current_page: int, total_pages: int, key_prefix: str = "page"):
    """Render pagination controls"""
    
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
    """Apply quick filter to dataframe"""
    from .constants import GAP_CATEGORIES
    
    if filter_type == 'all' or df.empty:
        return df
    
    if filter_type == 'critical':
        return df[df['priority'] == 1]
    
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