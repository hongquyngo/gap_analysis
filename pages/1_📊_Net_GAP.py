# pages/1_📊_Net_GAP.py

"""
Net GAP Analysis Page - Version 2.2
- Fixed Coverage display (percentage format)
- Added comprehensive tooltips and help information
- Improved user guidance for calculated fields
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import logging
from typing import Dict, Any, Optional, List, Tuple
import io
import time
import os
from pathlib import Path

# Configure page
st.set_page_config(
    page_title="Net GAP Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import utilities
project_root = os.environ.get('PROJECT_ROOT', Path(__file__).parent.parent)
if str(project_root) not in os.sys.path:
    os.sys.path.insert(0, str(project_root))

from utils.auth import AuthManager
from utils.net_gap.session_manager import get_session_manager
from utils.net_gap.data_loader import GAPDataLoader, DataLoadError, ValidationError
from utils.net_gap.calculator import GAPCalculator
from utils.net_gap.formatters import GAPFormatter
from utils.net_gap.filters import GAPFilters, QUICK_FILTER_BASE, QUICK_FILTER_SAFETY
from utils.net_gap.charts import GAPCharts
from utils.net_gap.customer_dialog import CustomerImpactDialog, show_customer_popup
from utils.net_gap.field_explanations import show_field_explanations, get_field_tooltip

# Constants
MAX_EXPORT_ROWS = 10000
DATA_LOAD_WARNING_SECONDS = 5
VERSION = "2.2"


def initialize_components():
    """Initialize all GAP analysis components"""
    session_manager = get_session_manager()
    data_loader = GAPDataLoader()
    calculator = GAPCalculator()
    formatter = GAPFormatter()
    filters = GAPFilters(data_loader)
    charts = GAPCharts(formatter)
    customer_dialog = CustomerImpactDialog(calculator, formatter)
    
    return session_manager, data_loader, calculator, formatter, filters, charts, customer_dialog


def validate_data(df: pd.DataFrame, data_type: str) -> bool:
    """
    Validate dataframe has required columns
    
    Args:
        df: DataFrame to validate
        data_type: 'supply' or 'demand'
    
    Returns:
        True if valid, False otherwise
    """
    if df.empty:
        return True  # Empty is valid, just no data
    
    if data_type == 'supply':
        required = ['supply_source', 'available_quantity', 'product_id']
    else:  # demand
        required = ['demand_source', 'required_quantity', 'product_id']
    
    missing = [col for col in required if col not in df.columns]
    if missing:
        st.error(f"Missing required columns in {data_type} data: {missing}")
        return False
    return True


def handle_error(e: Exception) -> None:
    """Handle errors with appropriate user messages"""
    error_type = type(e).__name__
    error_msg = str(e).lower()
    
    logger.error(f"Error in Net GAP analysis: {e}", exc_info=True)
    
    # Handle custom exceptions
    if isinstance(e, ValidationError):
        st.error(f"⚠️ Validation Error: {str(e)}")
        st.info("Please check your filter selections and try again.")
        return
    
    if isinstance(e, DataLoadError):
        st.error(f"❌ Data Loading Error: {str(e)}")
        st.info("Please try refreshing the page or contact support if the issue persists.")
        return
    
    # Handle general errors
    if "connection" in error_msg or "connect" in error_msg:
        st.error("🔌 Database connection issue. Please refresh the page and try again.")
    elif "permission" in error_msg or "denied" in error_msg:
        st.error("🔒 Access denied. Please check your permissions.")
    elif "timeout" in error_msg:
        st.error("⏱️ Request timed out. Try using more specific filters.")
    else:
        st.error(f"❌ An error occurred: {error_type}")
    
    with st.expander("Error Details", expanded=False):
        st.code(str(e))


def load_data_with_timing(
    data_loader: GAPDataLoader,
    filter_values: Dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load supply, demand, and safety stock data with timing and validation
    
    Args:
        data_loader: GAPDataLoader instance
        filter_values: Filter configuration
        
    Returns:
        Tuple of (supply_df, demand_df, safety_stock_df)
    """
    start_time = time.time()
    include_safety = filter_values.get('include_safety_stock', False)
    
    with st.spinner("Loading data..."):
        # Load supply data (convert lists to tuples for cache)
        supply_df = data_loader.load_supply_data(
            entity_name=filter_values.get('entity'),
            date_from=filter_values.get('date_range')[0],
            date_to=filter_values.get('date_range')[1],
            product_ids=filter_values.get('products_tuple'),
            brands=filter_values.get('brands_tuple')
        )
        
        # Load demand data
        demand_df = data_loader.load_demand_data(
            entity_name=filter_values.get('entity'),
            date_from=filter_values.get('date_range')[0],
            date_to=filter_values.get('date_range')[1],
            product_ids=filter_values.get('products_tuple'),
            brands=filter_values.get('brands_tuple'),
            customers=filter_values.get('customers_tuple')
        )
        
        # Load safety stock if included
        safety_stock_df = None
        if include_safety:
            safety_stock_df = data_loader.load_safety_stock_data(
                entity_name=filter_values.get('entity'),
                product_ids=filter_values.get('products_tuple'),
                include_customer_specific=True
            )
    
    load_time = time.time() - start_time
    if load_time > DATA_LOAD_WARNING_SECONDS:
        st.info(f"⏱️ Data loading took {load_time:.1f} seconds")
    
    logger.info(
        f"Data loaded: {len(supply_df)} supply, {len(demand_df)} demand, "
        f"{len(safety_stock_df) if safety_stock_df is not None else 0} safety stock records"
    )
    
    return supply_df, demand_df, safety_stock_df


def calculate_gap_with_progress(
    calculator: GAPCalculator,
    supply_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    safety_stock_df: Optional[pd.DataFrame],
    filter_values: Dict[str, Any]
) -> pd.DataFrame:
    """
    Calculate GAP with progress indicator
    
    Args:
        calculator: GAPCalculator instance
        supply_df: Supply data
        demand_df: Demand data
        safety_stock_df: Safety stock data (optional)
        filter_values: Filter configuration
        
    Returns:
        GAP analysis DataFrame
    """
    include_safety = filter_values.get('include_safety_stock', False)
    
    with st.spinner("Calculating GAP analysis..."):
        gap_df = calculator.calculate_net_gap(
            supply_df=supply_df,
            demand_df=demand_df,
            safety_stock_df=safety_stock_df,
            group_by=filter_values.get('group_by', 'product'),
            selected_supply_sources=filter_values.get('supply_sources'),
            selected_demand_sources=filter_values.get('demand_sources'),
            include_safety_stock=include_safety
        )
    
    logger.info(f"GAP calculated for {len(gap_df)} items")
    return gap_df


def export_to_excel(
    gap_df: pd.DataFrame, 
    metrics: Dict, 
    filters: Dict, 
    include_safety: bool = False
) -> bytes:
    """
    Export GAP analysis to Excel with safety stock information
    
    Args:
        gap_df: GAP analysis dataframe
        metrics: Summary metrics
        filters: Applied filters
        include_safety: Whether safety stock is included
    
    Returns:
        Excel file as bytes
    """
    output = io.BytesIO()
    
    # Check data size
    if len(gap_df) > MAX_EXPORT_ROWS:
        st.warning(
            f"⚠️ Large dataset ({len(gap_df)} rows). "
            f"Export limited to {MAX_EXPORT_ROWS} rows."
        )
        gap_df = gap_df.head(MAX_EXPORT_ROWS)
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Summary sheet
        summary_rows = [
            ('Total Products', metrics['total_products']),
            ('Shortage Items', metrics['shortage_items']),
            ('Critical Items', metrics['critical_items']),
            ('Coverage Rate (%)', f"{metrics['overall_coverage']:.1f}%"),
            ('Total Shortage', metrics['total_shortage']),
            ('Total Surplus', metrics['total_surplus']),
            ('At Risk Value (USD)', f"${metrics['at_risk_value_usd']:,.2f}"),
            ('Affected Customers', metrics['affected_customers']),
            ('Supply Sources', ', '.join(filters.get('supply_sources', []))),
            ('Demand Sources', ', '.join(filters.get('demand_sources', [])))
        ]
        
        # Add safety stock metrics if included
        if include_safety:
            summary_rows.extend([
                ('Safety Stock Included', 'Yes'),
                ('Below Safety Count', metrics.get('below_safety_count', 0)),
                ('At Reorder Count', metrics.get('at_reorder_count', 0)),
                ('Safety Stock Value', f"${metrics.get('safety_stock_value', 0):,.2f}")
            ])
        
        summary_data = pd.DataFrame(summary_rows, columns=['Metric', 'Value'])
        summary_data.to_excel(writer, sheet_name='Summary', index=False)
        
        # Detail sheet
        gap_df.to_excel(writer, sheet_name='GAP Details', index=False)
        
        # Filters sheet
        filters_data = pd.DataFrame({
            'Filter': [
                'Entity', 'Date Range', 'Supply Sources', 'Demand Sources', 
                'Products', 'Brands', 'Customers', 'Group By', 'Safety Stock'
            ],
            'Value': [
                filters.get('entity', 'All'),
                f"{filters.get('date_range', ['N/A'])[0]} to {filters.get('date_range', ['N/A', 'N/A'])[1]}",
                ', '.join(filters.get('supply_sources', [])),
                ', '.join(filters.get('demand_sources', [])),
                f"{len(filters.get('products', []))} selected" if filters.get('products') else 'All',
                ', '.join(filters.get('brands', [])) or 'All',
                f"{len(filters.get('customers', []))} selected" if filters.get('customers') else 'All',
                filters.get('group_by', 'product'),
                'Yes' if filters.get('include_safety_stock', False) else 'No'
            ]
        })
        filters_data.to_excel(writer, sheet_name='Applied Filters', index=False)
    
    output.seek(0)
    logger.info("Excel export generated successfully")
    return output.getvalue()


def display_visualizations(
    gap_df: pd.DataFrame, 
    filter_values: Dict, 
    charts: GAPCharts, 
    include_safety: bool
) -> None:
    """Display visualization tabs"""
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Status Distribution", 
        "📉 Top Shortages", 
        "📈 Supply vs Demand", 
        "🔍 Coverage Analysis"
    ])
    
    with tab1:
        if not gap_df.empty:
            fig_pie = charts.create_status_pie_chart(gap_df)
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns([3, 1])
        with col2:
            top_n = st.number_input("Top N items", min_value=5, max_value=20, value=10)
        
        # Filter for shortage items based on context
        if include_safety:
            shortage_statuses = [
                'SEVERE_SHORTAGE', 'HIGH_SHORTAGE', 'MODERATE_SHORTAGE',
                'BELOW_SAFETY', 'CRITICAL_BREACH'
            ]
        else:
            shortage_statuses = ['SEVERE_SHORTAGE', 'HIGH_SHORTAGE', 'MODERATE_SHORTAGE']
        
        shortage_df = gap_df[gap_df['gap_status'].isin(shortage_statuses)]
        
        if not shortage_df.empty:
            fig_bar = charts.create_top_shortage_bar_chart(shortage_df, top_n=top_n)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No shortage items to display")
    
    with tab3:
        if not gap_df.empty:
            fig_comparison = charts.create_supply_demand_comparison(gap_df, top_n=15)
            st.plotly_chart(fig_comparison, use_container_width=True)
    
    with tab4:
        if not gap_df.empty:
            fig_coverage = charts.create_coverage_distribution(gap_df)
            st.plotly_chart(fig_coverage, use_container_width=True)


def format_coverage_value(row: pd.Series) -> str:
    """
    Format coverage ratio for better readability
    
    Args:
        row: DataFrame row containing coverage_ratio, total_demand, total_supply
        
    Returns:
        Formatted string representation
    """
    ratio = row.get('coverage_ratio', 0)
    demand = row.get('total_demand', 0)
    supply = row.get('total_supply', 0)
    
    # Special cases
    if pd.isna(ratio):
        return "—"
    
    if demand == 0:
        if supply > 0:
            return "No Demand"
        else:
            return "—"
    
    if supply == 0:
        return "0%"
    
    # Convert to percentage
    percentage = ratio * 100
    
    # Format based on value range
    if percentage > 999:
        return ">999%"
    elif percentage < 1:
        return f"{percentage:.1f}%"
    else:
        return f"{int(percentage)}%"


def prepare_display_dataframe(
    gap_df: pd.DataFrame, 
    filter_values: Dict, 
    formatter: GAPFormatter, 
    include_safety: bool,
    session_manager
) -> pd.DataFrame:
    """Prepare dataframe for display with formatting and column selection"""
    display_df = gap_df.copy()
    
    # Enhanced status mapping with safety stock statuses
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
    
    display_df['Status'] = display_df['gap_status'].map(status_display).fillna('❓ Unknown')
    
    # Format numeric columns
    if 'total_supply' in display_df.columns:
        display_df['Supply'] = display_df['total_supply'].apply(
            lambda x: formatter.format_number(x)
        )
    
    if 'total_demand' in display_df.columns:
        display_df['Demand'] = display_df['total_demand'].apply(
            lambda x: formatter.format_number(x)
        )
    
    if 'net_gap' in display_df.columns:
        display_df['Net GAP'] = display_df['net_gap'].apply(
            lambda x: formatter.format_number(x, show_sign=True)
        )
    
    # IMPROVED Coverage formatting
    if 'coverage_ratio' in display_df.columns:
        display_df['Coverage'] = display_df.apply(format_coverage_value, axis=1)
    
    if 'gap_percentage' in display_df.columns:
        display_df['GAP %'] = display_df['gap_percentage'].apply(
            lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A"
        )
    
    # Safety stock columns (when included)
    if include_safety:
        if 'safety_stock_qty' in display_df.columns:
            display_df['Safety Stock'] = display_df['safety_stock_qty'].apply(
                lambda x: formatter.format_number(x)
            )
        
        if 'available_supply' in display_df.columns:
            display_df['Available'] = display_df['available_supply'].apply(
                lambda x: formatter.format_number(x)
            )
            true_gap = display_df['available_supply'] - display_df.get('total_demand', 0)
            display_df['True GAP'] = true_gap.apply(
                lambda x: formatter.format_number(x, show_sign=True)
            )
        
        if 'safety_coverage' in display_df.columns:
            display_df['Safety Cov'] = display_df['safety_coverage'].apply(
                lambda x: f"{x:.1f}x" if pd.notna(x) and x < 999 else "N/A"
            )
        
        if 'below_reorder' in display_df.columns:
            display_df['Reorder'] = display_df['below_reorder'].apply(
                lambda x: '⚠️ Yes' if x else '✅ No'
            )
    
    # Rename suggested_action for better display
    if 'suggested_action' in display_df.columns:
        display_df['Action'] = display_df['suggested_action']
    
    # Format financial columns with currency
    if 'at_risk_value_usd' in display_df.columns:
        display_df['At Risk Value'] = display_df['at_risk_value_usd'].apply(
            lambda x: formatter.format_currency(x, abbreviate=True)
        )
    
    if 'gap_value_usd' in display_df.columns:
        display_df['GAP Value'] = display_df['gap_value_usd'].apply(
            lambda x: formatter.format_currency(x, abbreviate=True)
        )
    
    # Get selected columns from session manager
    col_config = session_manager.get_table_columns_config()
    columns = []
    
    # Basic Info
    if col_config['basic']:
        if filter_values.get('group_by') == 'product':
            columns.extend(['pt_code', 'product_name', 'brand'])
        else:
            columns.extend(['brand'])
    
    # Supply & Demand
    if col_config['supply']:
        columns.extend(['Supply', 'Demand', 'Net GAP'])
    
    # Safety Stock columns
    if col_config['safety'] and include_safety:
        if 'Safety Stock' in display_df.columns:
            columns.extend(['Safety Stock', 'Available', 'True GAP'])
    
    # Analysis
    if col_config['analysis']:
        columns.extend(['Coverage', 'GAP %', 'Status', 'Action'])
    
    # Financial
    if col_config['financial']:
        financial_cols = ['At Risk Value', 'GAP Value']
        columns.extend([c for c in financial_cols if c in display_df.columns])
    
    # Details
    if col_config['details']:
        detail_cols = [
            'supply_inventory', 'supply_can_pending', 
            'supply_warehouse_transfer', 'supply_purchase_order',
            'demand_oc_pending', 'demand_forecast'
        ]
        columns.extend([c for c in detail_cols if c in display_df.columns])
    
    # Filter to available columns
    columns = [col for col in columns if col in display_df.columns]
    
    return display_df[columns] if columns else display_df


def display_paginated_table(
    df: pd.DataFrame, 
    items_per_page: int,
    session_manager,
    key_prefix: str = "main"
) -> None:
    """
    Display paginated dataframe with SessionStateManager
    
    Args:
        df: DataFrame to display
        items_per_page: Number of items per page
        session_manager: SessionStateManager instance
        key_prefix: Unique prefix for widget keys to avoid conflicts
    """
    if df.empty:
        st.info("No data matches the current filters")
        return
    
    total_items = len(df)
    total_pages = max(1, (total_items + items_per_page - 1) // items_per_page)
    
    # Get and validate page from session manager
    page = session_manager.get_current_page()
    session_manager.set_current_page(page, total_pages)
    page = session_manager.get_current_page()
    
    # Calculate indices
    start_idx = (page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, total_items)
    
    # Display the data table with tooltips
    page_df = df.iloc[start_idx:end_idx]
    
    # Add column configuration with tooltips
    column_config = {}
    for col in page_df.columns:
        tooltip = get_field_tooltip(col)
        if tooltip:
            column_config[col] = st.column_config.Column(
                col,
                help=tooltip,
                width="medium"
            )
    
    st.dataframe(
        page_df, 
        use_container_width=True, 
        hide_index=True,
        column_config=column_config
    )
    
    # Pagination controls
    if total_pages > 1:
        col1, col2, col3, col4, col5 = st.columns([1, 1, 3, 1, 1])
        
        with col1:
            if st.button("⏮️", disabled=(page == 1), use_container_width=True, key=f"{key_prefix}_page_first"):
                session_manager.set_current_page(1, total_pages)
                st.rerun()
        
        with col2:
            if st.button("⬅️", disabled=(page == 1), use_container_width=True, key=f"{key_prefix}_page_prev"):
                session_manager.set_current_page(page - 1, total_pages)
                st.rerun()
        
        with col3:
            st.markdown(
                f"<div style='text-align: center; padding: 8px;'>"
                f"Page <b>{page}</b> of <b>{total_pages}</b> "
                f"({start_idx + 1:,}-{end_idx:,} of {total_items:,})"
                f"</div>",
                unsafe_allow_html=True
            )
        
        with col4:
            if st.button("➡️", disabled=(page == total_pages), use_container_width=True, key=f"{key_prefix}_page_next"):
                session_manager.set_current_page(page + 1, total_pages)
                st.rerun()
        
        with col5:
            if st.button("⏭️", disabled=(page == total_pages), use_container_width=True, key=f"{key_prefix}_page_last"):
                session_manager.set_current_page(total_pages, total_pages)
                st.rerun()


def display_data_table(
    gap_df: pd.DataFrame, 
    filter_values: Dict, 
    formatter: GAPFormatter, 
    metrics: Dict, 
    include_safety: bool,
    session_manager,
    filters: GAPFilters
) -> None:
    """Display the main data table with quick filter, search, column selection, and export"""
    
    # Quick Filter Section
    st.markdown("### 🔍 Quick Filter")
    st.caption("Filter results without recalculating GAP")
    
    # Get appropriate filter options based on safety stock
    if include_safety:
        filter_options = QUICK_FILTER_SAFETY
    else:
        filter_options = QUICK_FILTER_BASE
    
    # Create radio buttons with format_func for labels
    quick_filter = st.radio(
        "Select filter view",
        options=list(filter_options.keys()),
        format_func=lambda x: filter_options[x]['label'],
        horizontal=True,
        label_visibility="collapsed",
        key="quick_filter_radio_post_calc",
        help="Filter displayed results by status. Does not require recalculation."
    )
    
    # Display tooltip for selected filter
    selected_help = filter_options[quick_filter]['help']
    st.caption(f"ℹ️ {selected_help}")
    
    # Reset pagination when filter changes
    if st.session_state.get('_last_quick_filter') != quick_filter:
        session_manager.reset_pagination()
        st.session_state['_last_quick_filter'] = quick_filter
    
    # Apply quick filter immediately
    original_count = len(gap_df)
    if quick_filter != 'all':
        gap_df = filters.apply_quick_filter(gap_df, quick_filter, include_safety)
        filtered_count = len(gap_df)
        if filtered_count < original_count:
            filter_label = filter_options[quick_filter]['label']
            st.info(f"📊 Showing {filtered_count:,} of {original_count:,} items ({filter_label})")
    
    st.divider()
    
    # Help Section for Field Explanations - Create as a separate button/dialog
    help_col1, help_col2, help_col3 = st.columns([1, 3, 1])
    with help_col1:
        if st.button("📐 View Formulas", use_container_width=True, key="show_formulas_btn"):
            st.session_state['show_formulas'] = not st.session_state.get('show_formulas', False)
    
    # Show formulas section if button clicked
    if st.session_state.get('show_formulas', False):
        with st.container():
            st.divider()
            show_field_explanations(include_safety)
            st.divider()
    
    # Column configuration section
    with st.expander("⚙️ Table Configuration", expanded=False):
        st.markdown("**Select columns to display:**")
        
        col_config = session_manager.get_table_columns_config()
        
        # Column selectors
        col1, col2, col3 = st.columns(3)
        
        new_config = {}
        
        with col1:
            new_config['basic'] = st.checkbox(
                "Basic Info", 
                value=col_config['basic'],
                key="col_basic_check",
                help="PT Code, Product Name, Brand"
            )
            new_config['supply'] = st.checkbox(
                "Supply & Demand", 
                value=col_config['supply'],
                key="col_supply_check",
                help="Total Supply, Total Demand, Net GAP"
            )
        
        with col2:
            if include_safety:
                new_config['safety'] = st.checkbox(
                    "Safety Stock", 
                    value=col_config['safety'],
                    key="col_safety_check",
                    help="Safety Stock, Available Supply, True GAP"
                )
            else:
                new_config['safety'] = False
            
            new_config['analysis'] = st.checkbox(
                "Analysis", 
                value=col_config['analysis'],
                key="col_analysis_check",
                help="Coverage, GAP %, Status, Action"
            )
        
        with col3:
            new_config['financial'] = st.checkbox(
                "Financial", 
                value=col_config['financial'],
                key="col_financial_check",
                help="At Risk Value, GAP Value in USD"
            )
            new_config['details'] = st.checkbox(
                "Details", 
                value=col_config['details'],
                key="col_details_check",
                help="Individual supply/demand source breakdowns"
            )
        
        # Update config in session manager
        session_manager.set_table_columns_config(new_config)
        
        # Preset buttons
        st.markdown("**Quick presets:**")
        preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)
        
        with preset_col1:
            if st.button("📊 Standard View", use_container_width=True):
                session_manager.apply_table_preset('standard')
                st.rerun()
        
        with preset_col2:
            if st.button("🔒 Safety View", use_container_width=True):
                session_manager.apply_table_preset('safety')
                st.rerun()
        
        with preset_col3:
            if st.button("💰 Financial View", use_container_width=True):
                session_manager.apply_table_preset('financial')
                st.rerun()
        
        with preset_col4:
            if st.button("📋 All Columns", use_container_width=True):
                session_manager.apply_table_preset('all')
                st.rerun()
    
    # Prepare display dataframe
    display_df = prepare_display_dataframe(
        gap_df, filter_values, formatter, include_safety, session_manager
    )
    
    # Controls row
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_term = st.text_input(
            "🔎 Search in results", 
            placeholder="Type to filter...",
            key="table_search"
        )
        if search_term:
            mask = display_df.astype(str).apply(
                lambda x: x.str.contains(search_term, case=False, na=False)
            ).any(axis=1)
            display_df = display_df[mask]
    
    with col2:
        items_per_page = st.selectbox(
            "Items per page", 
            [10, 25, 50, 100], 
            index=1,
            key="items_per_page"
        )
    
    with col3:
        if st.button("📥 Export to Excel", type="primary", use_container_width=True):
            excel_data = export_to_excel(gap_df, metrics, filter_values, include_safety)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            st.download_button(
                label="📥 Download Excel",
                data=excel_data,
                file_name=f"gap_analysis_{timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            st.success("✅ Export ready!")
    
    # Display info
    st.caption(f"Showing {len(display_df.columns)} columns | {len(display_df)} items found")
    
    # Reset pagination when search changes
    if search_term:
        session_manager.reset_pagination()
    
    # Display paginated table with tooltips
    display_paginated_table(display_df, items_per_page, session_manager, key_prefix="main_table")


def render_action_buttons(session_manager, filters) -> Tuple[bool, bool]:
    """
    Render action buttons
    
    Returns:
        Tuple of (calculate_clicked, force_recalculation)
    """
    col1, col2, col3 = st.columns([1, 1, 2])
    
    force_recalc = False
    
    with col1:
        if st.button("🔄 Reset Filters", use_container_width=True):
            session_manager.reset_filters()
            session_manager.reset_pagination()
            session_manager.clear_gap_calculation()
            st.rerun()
    
    with col2:
        calculate_clicked = st.button(
            "📊 Calculate GAP", 
            type="primary", 
            use_container_width=True
        )
        if calculate_clicked:
            # Force recalculation when button explicitly clicked
            force_recalc = True
    
    with col3:
        active_count = filters.count_active_filters()
        if active_count > 0:
            st.info(f"✓ {active_count} filters active")
    
    return calculate_clicked, force_recalc


def main():
    """Main application logic with refactored components"""
    # Initialize authentication
    auth_manager = AuthManager()
    
    # Check authentication
    if not auth_manager.check_session():
        st.warning("⚠️ Please login to access this page")
        st.stop()
    
    # Initialize components
    (session_manager, data_loader, calculator, formatter, 
     filters, charts, customer_dialog) = initialize_components()
    
    # Page header
    st.title("📊 Net GAP Analysis")
    st.markdown("Comprehensive supply-demand analysis with safety stock management")
    
    # User info in sidebar
    st.sidebar.markdown(f"👤 **User:** {auth_manager.get_user_display_name()}")
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        auth_manager.logout()
        st.rerun()
    
    # Show customer dialog if needed
    if session_manager.show_customer_dialog():
        # Prepare dialog data if not already prepared
        shortage_ids, _ = session_manager.get_dialog_data()
        
        if not shortage_ids:
            # Dialog was triggered but data not prepared
            gap_df_filtered, _ = session_manager.get_gap_results()
            if gap_df_filtered is not None and not gap_df_filtered.empty:
                customer_dialog.show_dialog(
                    gap_df_filtered,
                    st.session_state.get('_temp_demand_df')
                )
        
        # Show the popup if data is available
        if '_temp_calculator' in st.session_state:
            show_customer_popup()
    
    # Render filters
    try:
        filter_values = filters.render_filters()
    except Exception as e:
        handle_error(e)
        st.stop()
    
    # Action buttons
    calculate_clicked, force_recalc = render_action_buttons(session_manager, filters)
    
    # Check if we should show results
    should_recalculate = force_recalc or session_manager.should_recalculate(filter_values)
    has_results = session_manager.is_gap_calculated()
    
    # If no results and calculate button not clicked, show initial message
    if not has_results and not calculate_clicked:
        st.info("👆 Configure filters and click 'Calculate GAP' to begin analysis")
        st.stop()
    
    # If results exist but filters changed, show warning
    if has_results and should_recalculate and not calculate_clicked:
        st.warning(
            "⚠️ Filters have changed since last calculation. "
            "Click 'Calculate GAP' to update results."
        )
        st.info("Showing previous results below...")
    
    try:
        # Determine if we need to calculate or use cached results
        if should_recalculate or not has_results:
            # Perform calculation
            with st.spinner("🔄 Calculating GAP analysis..."):
                # Load data
                supply_df, demand_df, safety_stock_df = load_data_with_timing(
                    data_loader, filter_values
                )
                
                # Validate data
                if not validate_data(supply_df, 'supply') or not validate_data(demand_df, 'demand'):
                    st.stop()
                
                # Check if data is available
                if supply_df.empty and demand_df.empty:
                    st.warning(
                        "No data available for the selected filters. "
                        "Please adjust your filters and try again."
                    )
                    st.stop()
                
                # Calculate GAP (full dataset, no quick filter applied yet)
                gap_df = calculate_gap_with_progress(
                    calculator, supply_df, demand_df, safety_stock_df, filter_values
                )
                
                # Calculate metrics on full dataset
                metrics = calculator.get_summary_metrics(gap_df)
                
                # Store results in session manager (full dataset)
                session_manager.set_gap_calculated(gap_df, metrics, filter_values)
                
                # Store raw data for dialog
                st.session_state['_temp_demand_df'] = demand_df
                
                logger.info(f"GAP calculation completed and cached: {len(gap_df)} items")
        
        else:
            # Use cached results
            gap_df, metrics = session_manager.get_gap_results()
            include_safety = filter_values.get('include_safety_stock', False)
            
            # Always reload demand_df for dialog functionality
            if '_temp_demand_df' not in st.session_state or st.session_state.get('_temp_demand_df') is None:
                with st.spinner("Loading customer data..."):
                    _, demand_df, _ = load_data_with_timing(data_loader, filter_values)
                    st.session_state['_temp_demand_df'] = demand_df
                    logger.info("Demand data reloaded for dialog support")
            
            logger.info(f"Using cached GAP results: {len(gap_df)} items")
        
        # Always ensure temporary data is available for dialogs and interactions
        st.session_state['_temp_calculator'] = calculator
        st.session_state['_temp_formatter'] = formatter
        st.session_state['_temp_gap_df'] = gap_df
        
        # Ensure demand_df is in temp state
        if '_temp_demand_df' not in st.session_state or st.session_state.get('_temp_demand_df') is None:
            with st.spinner("Loading customer data..."):
                _, demand_df, _ = load_data_with_timing(data_loader, filter_values)
                st.session_state['_temp_demand_df'] = demand_df
        
        # Display configuration summary
        include_safety = filter_values.get('include_safety_stock', False)
        config_text = f"""
        **Analysis Configuration:**
        - Supply: {', '.join(filter_values.get('supply_sources', []))}
        - Demand: {', '.join(filter_values.get('demand_sources', []))}
        """
        if include_safety:
            config_text += "\n- ✅ **Safety stock requirements included**"
        
        config_text += f"\n- {filters.get_filter_summary(filter_values)}"
        
        st.info(config_text)
        
        # Display KPI cards
        st.subheader("📈 Key Metrics")
        
        charts.create_kpi_cards(
            metrics, 
            include_safety=include_safety,
            enable_customer_dialog=(filter_values.get('group_by') == 'product')
        )
        
        st.divider()
        
        # Visualizations
        st.subheader("📊 Visual Analysis")
        display_visualizations(gap_df, filter_values, charts, include_safety)
        
        st.divider()
        
        # Detailed data table (with quick filter inside)
        st.subheader("📋 Detailed GAP Analysis")
        display_data_table(
            gap_df, filter_values, formatter, 
            metrics, include_safety, session_manager, filters
        )
        
    except (ValidationError, DataLoadError) as e:
        handle_error(e)
    except Exception as e:
        handle_error(e)
    
    # Footer
    st.divider()
    
    # Show calculation timestamp if available
    calc_time = st.session_state.get(session_manager.KEY_CALCULATION_TIMESTAMP)
    if calc_time:
        time_str = calc_time.strftime('%Y-%m-%d %H:%M:%S')
        st.caption(f"Last calculated: {time_str} | Net GAP Analysis v{VERSION}")
    else:
        st.caption(f"Net GAP Analysis v{VERSION}")


if __name__ == "__main__":
    main()