# pages/1_📊_Net_GAP.py

"""
Net GAP Analysis Page - Version 3.2 ENHANCED
- Added exclusion support for products, brands, and expired inventory
- Fixed customer dialog auto-popup bug
- Enhanced product and entity display formatting
- Improved state management and error handling
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, Any, Tuple
import io
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
from utils.net_gap.customer_dialog import show_customer_popup
from utils.net_gap.field_explanations import show_field_explanations, get_field_tooltip

# Constants
MAX_EXPORT_ROWS = 10000
VERSION = "3.2"


def initialize_components():
    """Initialize all GAP analysis components"""
    session_manager = get_session_manager()
    data_loader = GAPDataLoader()
    calculator = GAPCalculator()
    formatter = GAPFormatter()
    filters = GAPFilters(data_loader)
    charts = GAPCharts(formatter)
    
    return session_manager, data_loader, calculator, formatter, filters, charts


def handle_error(e: Exception) -> None:
    """Handle errors with appropriate user messages"""
    error_type = type(e).__name__
    error_msg = str(e).lower()
    
    logger.error(f"Error in Net GAP analysis: {e}", exc_info=True)
    
    if isinstance(e, ValidationError):
        st.error(f"⚠️ Validation Error: {str(e)}")
        st.info("Please check your filter selections and try again.")
        return
    
    if isinstance(e, DataLoadError):
        st.error(f"❌ Data Loading Error: {str(e)}")
        st.info("Please try refreshing the page or contact support if the issue persists.")
        return
    
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


def load_and_calculate_gap(
    data_loader: GAPDataLoader,
    calculator: GAPCalculator,
    filter_values: Dict[str, Any]
):
    """
    Load data and calculate GAP with exclusion support
    
    Returns:
        GAPCalculationResult object
    """
    with st.spinner("🔄 Loading data and calculating GAP..."):
        # Load supply data with exclusions
        supply_df = data_loader.load_supply_data(
            entity_name=filter_values.get('entity'),
            product_ids=filter_values.get('products_tuple'),
            brands=filter_values.get('brands_tuple'),
            exclude_products=filter_values.get('exclude_products', False),
            exclude_brands=filter_values.get('exclude_brands', False),
            exclude_expired=filter_values.get('exclude_expired_inventory', True)
        )
        
        # Load demand data with exclusions
        demand_df = data_loader.load_demand_data(
            entity_name=filter_values.get('entity'),
            product_ids=filter_values.get('products_tuple'),
            brands=filter_values.get('brands_tuple'),
            exclude_products=filter_values.get('exclude_products', False),
            exclude_brands=filter_values.get('exclude_brands', False)
        )
        
        # Load safety stock if included
        safety_stock_df = None
        if filter_values.get('include_safety_stock', False):
            safety_stock_df = data_loader.load_safety_stock_data(
                entity_name=filter_values.get('entity'),
                product_ids=filter_values.get('products_tuple'),
                include_customer_specific=True
            )
        
        # Validate data
        if supply_df.empty and demand_df.empty:
            st.warning(
                "No data available for the selected filters. "
                "Please adjust your filters and try again."
            )
            st.stop()
        
        logger.info(f"Data loaded: {len(supply_df)} supply, {len(demand_df)} demand records")
        
        # Calculate GAP
        result = calculator.calculate_net_gap(
            supply_df=supply_df,
            demand_df=demand_df,
            safety_stock_df=safety_stock_df,
            group_by=filter_values.get('group_by', 'product'),
            selected_supply_sources=filter_values.get('supply_sources'),
            selected_demand_sources=filter_values.get('demand_sources'),
            include_safety_stock=filter_values.get('include_safety_stock', False)
        )
        
        logger.info(f"GAP calculated: {len(result.gap_df)} items, "
                   f"{result.metrics.get('affected_customers', 0)} affected customers")
        
        return result


def format_value_for_export(value: Any, field_name: str) -> Any:
    """Format values for Excel export (fix 999 issue)"""
    if pd.isna(value) or value is None:
        return None
    
    if field_name == 'safety_coverage':
        if value >= 999:
            return None
        return round(value, 2)
    
    if field_name == 'days_of_supply':
        if value >= 999:
            return None
        return round(value, 1)
    
    if field_name == 'coverage_ratio':
        if value > 10:
            return None
        return round(value, 2)
    
    if isinstance(value, (int, float)):
        return round(value, 2) if isinstance(value, float) else value
    
    return value


def export_to_excel(
    gap_df: pd.DataFrame, 
    metrics: Dict, 
    filters: Dict, 
    include_safety: bool = False
) -> bytes:
    """Export GAP analysis to Excel with proper formatting"""
    output = io.BytesIO()
    
    export_df = gap_df.copy()
    if len(export_df) > MAX_EXPORT_ROWS:
        st.warning(f"⚠️ Large dataset. Export limited to {MAX_EXPORT_ROWS} rows.")
        export_df = export_df.head(MAX_EXPORT_ROWS)
    
    export_columns = [
        'product_id', 'product_name', 'pt_code', 'brand',
        'total_supply', 'total_demand', 'net_gap',
        'coverage_ratio', 'gap_percentage', 'gap_status',
        'priority', 'suggested_action'
    ]
    
    if include_safety:
        export_columns.extend([
            'safety_stock_qty', 'available_supply', 
            'safety_coverage', 'days_of_supply', 'below_reorder'
        ])
    
    if 'at_risk_value_usd' in export_df.columns:
        export_columns.extend(['at_risk_value_usd', 'gap_value_usd'])
    
    export_columns = [col for col in export_columns if col in export_df.columns]
    export_df = export_df[export_columns].copy()
    
    for col in export_df.columns:
        if col in ['safety_coverage', 'days_of_supply', 'coverage_ratio']:
            export_df[col] = export_df[col].apply(
                lambda x: format_value_for_export(x, col)
            )
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Summary sheet with exclusion info
        summary_rows = [
            ('Report Generated', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
            ('', ''),
            ('FILTERS APPLIED', ''),
            ('Entity', filters.get('entity', 'All')),
            ('Products', f"{len(filters.get('products', []))} selected" if filters.get('products') else 'All'),
            ('Products Mode', 'EXCLUDED' if filters.get('exclude_products') else 'INCLUDED'),
            ('Brands', ', '.join(filters.get('brands', [])) or 'All'),
            ('Brands Mode', 'EXCLUDED' if filters.get('exclude_brands') else 'INCLUDED'),
            ('Expired Inventory', 'EXCLUDED' if filters.get('exclude_expired_inventory') else 'INCLUDED'),
            ('', ''),
            ('METRICS', ''),
            ('Total Products', metrics['total_products']),
            ('Shortage Items', metrics['shortage_items']),
            ('Critical Items', metrics['critical_items']),
            ('Coverage Rate (%)', f"{metrics['overall_coverage']:.1f}"),
            ('', ''),
            ('Total Supply', metrics['total_supply']),
            ('Total Demand', metrics['total_demand']),
            ('Net GAP', metrics['net_gap']),
            ('', ''),
            ('Total Shortage', metrics['total_shortage']),
            ('Total Surplus', metrics['total_surplus']),
            ('At Risk Value (USD)', f"${metrics['at_risk_value_usd']:,.2f}"),
            ('Affected Customers', metrics['affected_customers']),
        ]
        
        if include_safety:
            summary_rows.extend([
                ('', ''),
                ('SAFETY STOCK ANALYSIS', ''),
                ('Below Safety Count', metrics.get('below_safety_count', 0)),
                ('At Reorder Count', metrics.get('at_reorder_count', 0)),
                ('Safety Stock Value', f"${metrics.get('safety_stock_value', 0):,.2f}"),
                ('Expired Items', metrics.get('has_expired_count', 0)),
                ('Expiry Risk Items', metrics.get('expiry_risk_count', 0))
            ])
        
        summary_data = pd.DataFrame(summary_rows, columns=['Metric', 'Value'])
        summary_data.to_excel(writer, sheet_name='Summary', index=False)
        
        # Detail sheet
        export_df.to_excel(writer, sheet_name='GAP Details', index=False)
        
        # Notes sheet
        notes_data = pd.DataFrame({
            'Note': [
                'EXCLUSION FILTERS',
                f"- Products: {filters.get('products', []) or 'None'} ({'EXCLUDED' if filters.get('exclude_products') else 'INCLUDED'})",
                f"- Brands: {filters.get('brands', []) or 'None'} ({'EXCLUDED' if filters.get('exclude_brands') else 'INCLUDED'})",
                f"- Expired Inventory: {'EXCLUDED' if filters.get('exclude_expired_inventory') else 'INCLUDED'}",
                '',
                'SPECIAL VALUES',
                '- Blank cells in Safety Coverage or Days of Supply indicate values > 999',
                '- Blank cells in Coverage Ratio indicate excessive surplus (>10x demand)',
                '',
                'STATUS CODES',
                '- CRITICAL_BREACH: Inventory below 50% of safety stock',
                '- BELOW_SAFETY: Inventory below safety stock level',
                '- SEVERE_SHORTAGE: Coverage < 50%',
                '- HIGH_SHORTAGE: Coverage 50-70%',
                '- MODERATE_SHORTAGE: Coverage 70-90%',
                '- BALANCED: Coverage 90-110%',
                '- SURPLUS: Coverage > 110%'
            ]
        })
        notes_data.to_excel(writer, sheet_name='Notes', index=False)
    
    output.seek(0)
    logger.info("Excel export generated successfully with exclusion info")
    return output.getvalue()


def format_coverage_value(row: pd.Series) -> str:
    """Format coverage ratio for display"""
    ratio = row.get('coverage_ratio', 0)
    demand = row.get('total_demand', 0)
    supply = row.get('total_supply', 0)
    
    if pd.isna(ratio):
        return "–"
    
    if demand == 0:
        return "No Demand" if supply > 0 else "–"
    
    if supply == 0:
        return "0%"
    
    percentage = ratio * 100
    
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
    """Prepare dataframe for display with formatting"""
    display_df = gap_df.copy()
    
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
        display_df['Supply'] = display_df['total_supply'].apply(formatter.format_number)
    
    if 'total_demand' in display_df.columns:
        display_df['Demand'] = display_df['total_demand'].apply(formatter.format_number)
    
    if 'net_gap' in display_df.columns:
        display_df['Net GAP'] = display_df['net_gap'].apply(
            lambda x: formatter.format_number(x, show_sign=True)
        )
    
    if 'coverage_ratio' in display_df.columns:
        display_df['Coverage'] = display_df.apply(format_coverage_value, axis=1)
    
    if 'gap_percentage' in display_df.columns:
        display_df['GAP %'] = display_df['gap_percentage'].apply(
            lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A"
        )
    
    # Safety stock columns
    if include_safety:
        if 'safety_stock_qty' in display_df.columns:
            display_df['Safety Stock'] = display_df['safety_stock_qty'].apply(
                formatter.format_number
            )
        
        if 'available_supply' in display_df.columns:
            display_df['Available'] = display_df['available_supply'].apply(
                formatter.format_number
            )
            true_gap = display_df['total_supply'] - display_df.get('total_demand', 0)
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
    
    if 'suggested_action' in display_df.columns:
        display_df['Action'] = display_df['suggested_action']
    
    # Financial columns
    if 'at_risk_value_usd' in display_df.columns:
        display_df['At Risk Value'] = display_df['at_risk_value_usd'].apply(
            lambda x: formatter.format_currency(x, abbreviate=True)
        )
    
    if 'gap_value_usd' in display_df.columns:
        display_df['GAP Value'] = display_df['gap_value_usd'].apply(
            lambda x: formatter.format_currency(x, abbreviate=True)
        )
    
    # Get selected columns
    col_config = session_manager.get_table_columns_config()
    columns = []
    
    if col_config['basic']:
        if filter_values.get('group_by') == 'product':
            columns.extend(['pt_code', 'product_name', 'brand'])
        else:
            columns.extend(['brand'])
    
    if col_config['supply']:
        columns.extend(['Supply', 'Demand', 'Net GAP'])
    
    if col_config['safety'] and include_safety:
        if 'Safety Stock' in display_df.columns:
            columns.extend(['Safety Stock', 'Available', 'True GAP'])
    
    if col_config['analysis']:
        columns.extend(['Coverage', 'GAP %', 'Status', 'Action'])
    
    if col_config['financial']:
        financial_cols = ['At Risk Value', 'GAP Value']
        columns.extend([c for c in financial_cols if c in display_df.columns])
    
    if col_config['details']:
        detail_cols = [
            'supply_inventory', 'supply_can_pending', 
            'supply_warehouse_transfer', 'supply_purchase_order',
            'demand_oc_pending', 'demand_forecast'
        ]
        columns.extend([c for c in detail_cols if c in display_df.columns])
    
    columns = [col for col in columns if col in display_df.columns]
    
    return display_df[columns] if columns else display_df


def display_paginated_table(
    df: pd.DataFrame, 
    items_per_page: int,
    session_manager,
    key_prefix: str = "main"
) -> None:
    """Display paginated dataframe"""
    if df.empty:
        st.info("No data matches the current filters")
        return
    
    total_items = len(df)
    total_pages = max(1, (total_items + items_per_page - 1) // items_per_page)
    
    page = session_manager.get_current_page()
    session_manager.set_current_page(page, total_pages)
    page = session_manager.get_current_page()
    
    start_idx = (page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, total_items)
    
    page_df = df.iloc[start_idx:end_idx]
    
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
    
    if total_pages > 1:
        col1, col2, col3, col4, col5 = st.columns([1, 1, 3, 1, 1])
        
        with col1:
            if st.button("⮈", disabled=(page == 1), use_container_width=True, key=f"{key_prefix}_first"):
                session_manager.set_current_page(1, total_pages)
                st.rerun()
        
        with col2:
            if st.button("⬅️", disabled=(page == 1), use_container_width=True, key=f"{key_prefix}_prev"):
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
            if st.button("➡️", disabled=(page == total_pages), use_container_width=True, key=f"{key_prefix}_next"):
                session_manager.set_current_page(page + 1, total_pages)
                st.rerun()
        
        with col5:
            if st.button("⭆", disabled=(page == total_pages), use_container_width=True, key=f"{key_prefix}_last"):
                session_manager.set_current_page(total_pages, total_pages)
                st.rerun()


def display_data_table(
    result,
    filter_values: Dict, 
    formatter: GAPFormatter, 
    include_safety: bool,
    session_manager,
    filters: GAPFilters
) -> None:
    """Display main data table with quick filter and controls"""
    
    gap_df = result.gap_df.copy()
    metrics = result.metrics
    
    # Quick Filter Section
    st.markdown("### 🔍 Quick Filter")
    st.caption("Filter results without recalculating GAP")
    
    filter_options = QUICK_FILTER_SAFETY if include_safety else QUICK_FILTER_BASE
    
    quick_filter = st.radio(
        "Select filter view",
        options=list(filter_options.keys()),
        format_func=lambda x: filter_options[x]['label'],
        horizontal=True,
        label_visibility="collapsed",
        key="quick_filter_radio",
        help="Filter displayed results by status"
    )
    
    st.caption(f"ℹ️ {filter_options[quick_filter]['help']}")
    
    if st.session_state.get('_last_quick_filter') != quick_filter:
        session_manager.reset_pagination()
        st.session_state['_last_quick_filter'] = quick_filter
    
    original_count = len(gap_df)
    if quick_filter != 'all':
        gap_df = filters.apply_quick_filter(gap_df, quick_filter, include_safety)
        filtered_count = len(gap_df)
        if filtered_count < original_count:
            st.info(f"📊 Showing {filtered_count:,} of {original_count:,} items "
                   f"({filter_options[quick_filter]['label']})")
    
    st.divider()
    
    # Help Section
    help_col1, help_col2, help_col3 = st.columns([1, 3, 1])
    with help_col1:
        if st.button("📖 View Formulas", use_container_width=True, key="show_formulas"):
            st.session_state['show_formulas'] = not st.session_state.get('show_formulas', False)
    
    if st.session_state.get('show_formulas', False):
        with st.container():
            st.divider()
            show_field_explanations(include_safety)
            st.divider()
    
    # Column configuration
    with st.expander("⚙️ Table Configuration", expanded=False):
        st.markdown("**Select columns to display:**")
        
        col_config = session_manager.get_table_columns_config()
        
        col1, col2, col3 = st.columns(3)
        
        new_config = {}
        
        with col1:
            new_config['basic'] = st.checkbox("Basic Info", value=col_config['basic'], key="col_basic")
            new_config['supply'] = st.checkbox("Supply & Demand", value=col_config['supply'], key="col_supply")
        
        with col2:
            if include_safety:
                new_config['safety'] = st.checkbox("Safety Stock", value=col_config['safety'], key="col_safety")
            else:
                new_config['safety'] = False
            new_config['analysis'] = st.checkbox("Analysis", value=col_config['analysis'], key="col_analysis")
        
        with col3:
            new_config['financial'] = st.checkbox("Financial", value=col_config['financial'], key="col_financial")
            new_config['details'] = st.checkbox("Details", value=col_config['details'], key="col_details")
        
        session_manager.set_table_columns_config(new_config)
        
        st.markdown("**Quick presets:**")
        preset_cols = st.columns(4)
        
        if preset_cols[0].button("📊 Standard", use_container_width=True):
            session_manager.apply_table_preset('standard')
            st.rerun()
        if preset_cols[1].button("🔒 Safety", use_container_width=True):
            session_manager.apply_table_preset('safety')
            st.rerun()
        if preset_cols[2].button("💰 Financial", use_container_width=True):
            session_manager.apply_table_preset('financial')
            st.rerun()
        if preset_cols[3].button("📋 All", use_container_width=True):
            session_manager.apply_table_preset('all')
            st.rerun()
    
    # Prepare display
    display_df = prepare_display_dataframe(
        gap_df, filter_values, formatter, include_safety, session_manager
    )
    
    # Controls
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
            session_manager.reset_pagination()
    
    with col2:
        items_per_page = st.selectbox(
            "Items per page", 
            [10, 25, 50, 100], 
            index=1,
            key="items_per_page"
        )
    
    with col3:
        if st.button("📥 Export to Excel", type="primary", use_container_width=True):
            excel_data = export_to_excel(
                result.gap_df,
                metrics, 
                filter_values, 
                include_safety
            )
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            st.download_button(
                label="📥 Download Excel",
                data=excel_data,
                file_name=f"gap_analysis_{timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            st.success("✅ Export ready!")
    
    st.caption(f"Showing {len(display_df.columns)} columns | {len(display_df)} items found")
    
    display_paginated_table(display_df, items_per_page, session_manager, key_prefix="main")


def render_action_buttons(session_manager, filters) -> Tuple[bool, bool]:
    """Render action buttons"""
    col1, col2, col3 = st.columns([1, 1, 2])
    
    force_recalc = False
    
    with col1:
        if st.button("🔄 Reset Filters", use_container_width=True):
            session_manager.reset_filters()
            session_manager.clear_gap_calculation()
            st.rerun()
    
    with col2:
        calculate_clicked = st.button(
            "📊 Calculate GAP", 
            type="primary", 
            use_container_width=True
        )
        if calculate_clicked:
            force_recalc = True
    
    with col3:
        active_count = filters.count_active_filters()
        if active_count > 0:
            st.info(f"✓ {active_count} filters active")
    
    return calculate_clicked, force_recalc


def main():
    """Main application logic"""
    # Authentication
    auth_manager = AuthManager()
    
    if not auth_manager.check_session():
        st.warning("⚠️ Please login to access this page")
        st.stop()
    
    # Initialize components
    session_manager, data_loader, calculator, formatter, filters, charts = initialize_components()
    
    # Page header
    st.title("📊 Net GAP Analysis")
    st.markdown("Comprehensive supply-demand analysis with exclusion filters and safety stock management")
    
    # Sidebar
    st.sidebar.markdown(f"👤 **User:** {auth_manager.get_user_display_name()}")
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        auth_manager.logout()
        st.rerun()
    
    # FIXED: Check dialog state properly
    if session_manager.show_customer_dialog():
        result = session_manager.get_gap_result()
        if result and result.customer_impact:
            show_customer_popup()
        else:
            logger.warning("Customer dialog requested but no customer impact data")
            session_manager.close_customer_dialog()
    
    # Render filters
    try:
        filter_values = filters.render_filters()
    except Exception as e:
        handle_error(e)
        st.stop()
    
    # Action buttons
    calculate_clicked, force_recalc = render_action_buttons(session_manager, filters)
    
    # Check if calculation needed
    should_recalculate = force_recalc or session_manager.should_recalculate(filter_values)
    has_results = session_manager.is_gap_calculated()
    
    # Show initial message
    if not has_results and not calculate_clicked:
        st.info("👆 Configure filters and click 'Calculate GAP' to begin analysis")
        st.stop()
    
    # Show warning if filters changed
    if has_results and should_recalculate and not calculate_clicked:
        st.warning(
            "⚠️ Filters have changed since last calculation. "
            "Click 'Calculate GAP' to update results."
        )
        st.info("Showing previous results below...")
    
    try:
        # Calculate or use cached
        if should_recalculate or not has_results:
            result = load_and_calculate_gap(data_loader, calculator, filter_values)
            session_manager.set_gap_calculated(result)
        else:
            result = session_manager.get_gap_result()
            logger.info(f"Using cached result: {len(result.gap_df)} items")
        
        # Display configuration with exclusion info
        include_safety = filter_values.get('include_safety_stock', False)
        
        config_parts = [
            "**Analysis Configuration:**",
            f"- Supply: {', '.join(filter_values.get('supply_sources', []))}",
            f"- Demand: {', '.join(filter_values.get('demand_sources', []))}"
        ]
        
        if include_safety:
            config_parts.append("- ✅ **Safety stock requirements included**")
        
        # Add exclusion info
        exclusions = []
        if filter_values.get('exclude_products') and filter_values.get('products'):
            exclusions.append(f"{len(filter_values['products'])} products excluded")
        if filter_values.get('exclude_brands') and filter_values.get('brands'):
            exclusions.append(f"{len(filter_values['brands'])} brands excluded")
        if filter_values.get('exclude_expired_inventory'):
            exclusions.append("expired inventory excluded")
        
        if exclusions:
            config_parts.append(f"- 🚫 **Exclusions:** {', '.join(exclusions)}")
        
        config_parts.append(f"- {filters.get_filter_summary(filter_values)}")
        
        st.info("\n".join(config_parts))
        
        # KPI cards
        st.subheader("📈 Key Metrics")
        charts.create_kpi_cards(
            result.metrics, 
            include_safety=include_safety,
            enable_customer_dialog=(filter_values.get('group_by') == 'product')
        )
        
        st.divider()
        
        # Visualizations
        st.subheader("📊 Visual Analysis")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Status Distribution", 
            "📉 Top Shortages", 
            "📈 Supply vs Demand", 
            "🔍 Coverage Analysis"
        ])
        
        with tab1:
            if not result.gap_df.empty:
                fig_pie = charts.create_status_pie_chart(result.gap_df)
                st.plotly_chart(fig_pie, use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns([3, 1])
            with col2:
                top_n = st.number_input("Top N items", min_value=5, max_value=20, value=10)
            
            if include_safety:
                shortage_statuses = [
                    'SEVERE_SHORTAGE', 'HIGH_SHORTAGE', 'MODERATE_SHORTAGE',
                    'BELOW_SAFETY', 'CRITICAL_BREACH'
                ]
            else:
                shortage_statuses = ['SEVERE_SHORTAGE', 'HIGH_SHORTAGE', 'MODERATE_SHORTAGE']
            
            shortage_df = result.gap_df[result.gap_df['gap_status'].isin(shortage_statuses)]
            
            if not shortage_df.empty:
                fig_bar = charts.create_top_shortage_bar_chart(shortage_df, top_n=top_n)
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("No shortage items to display")
        
        with tab3:
            if not result.gap_df.empty:
                fig_comparison = charts.create_supply_demand_comparison(result.gap_df, top_n=15)
                st.plotly_chart(fig_comparison, use_container_width=True)
        
        with tab4:
            if not result.gap_df.empty:
                fig_coverage = charts.create_coverage_distribution(result.gap_df)
                st.plotly_chart(fig_coverage, use_container_width=True)
        
        st.divider()
        
        # Detailed table
        st.subheader("📋 Detailed GAP Analysis")
        display_data_table(
            result, filter_values, formatter, 
            include_safety, session_manager, filters
        )
        
    except (ValidationError, DataLoadError) as e:
        handle_error(e)
    except Exception as e:
        handle_error(e)
    
    # Footer
    st.divider()
    if result:
        calc_time = result.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        st.caption(f"Last calculated: {calc_time} | Net GAP Analysis v{VERSION}")
    else:
        st.caption(f"Net GAP Analysis v{VERSION}")


if __name__ == "__main__":
    main()