# pages/1_📊_Net_GAP.py

"""
Net GAP Analysis Page - Version 2.0
- Integrated safety stock analysis
- Simplified single-phase filtering
- Context-aware calculations and visualizations
- Unified GAP metric that adapts to safety stock
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import logging
from typing import Dict, Any, Optional
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
from utils.net_gap.data_loader import GAPDataLoader
from utils.net_gap.calculator import GAPCalculator
from utils.net_gap.formatters import GAPFormatter
from utils.net_gap.filters import GAPFilters
from utils.net_gap.charts import GAPCharts
from utils.net_gap.customer_dialog import CustomerImpactDialog

# Constants
MAX_EXPORT_ROWS = 10000
DATA_LOAD_WARNING_SECONDS = 5
VERSION = "2.0"

def initialize_components():
    """Initialize all GAP analysis components"""
    data_loader = GAPDataLoader()
    calculator = GAPCalculator()
    formatter = GAPFormatter()
    filters = GAPFilters(data_loader)
    charts = GAPCharts(formatter)
    customer_dialog = CustomerImpactDialog(calculator, formatter)
    return data_loader, calculator, formatter, filters, charts, customer_dialog

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

def export_to_excel(gap_df: pd.DataFrame, metrics: Dict, filters: Dict, 
                   include_safety: bool = False) -> bytes:
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
        st.warning(f"⚠️ Large dataset ({len(gap_df)} rows). Export limited to {MAX_EXPORT_ROWS} rows.")
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
            'Filter': ['Entity', 'Date Range', 'Supply Sources', 'Demand Sources', 
                      'Products', 'Brands', 'Customers', 'Group By', 'Safety Stock'],
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
    return output.getvalue()

def display_visualizations(gap_df: pd.DataFrame, filter_values: Dict, 
                          charts: Any, include_safety: bool) -> None:
    """Display visualization tabs"""
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Status Distribution", 
        "📉 Top Shortages", 
        "📈 Supply vs Demand", 
        "📐 Coverage Analysis"
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
            shortage_statuses = ['SEVERE_SHORTAGE', 'HIGH_SHORTAGE', 'MODERATE_SHORTAGE',
                               'BELOW_SAFETY', 'CRITICAL_BREACH']
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

def prepare_display_dataframe(gap_df: pd.DataFrame, filter_values: Dict, 
                             formatter: Any, include_safety: bool) -> pd.DataFrame:
    """Prepare dataframe for display with formatting"""
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
    format_columns = {
        'total_supply': lambda x: formatter.format_number(x),
        'total_demand': lambda x: formatter.format_number(x),
        'net_gap': lambda x: formatter.format_number(x, show_sign=True),
        'coverage_ratio': lambda x: f"{x:.2f}x" if x < 10 else "999x+",
        'gap_percentage': lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A"
    }
    
    # Add safety stock columns if included
    if include_safety and 'safety_stock_qty' in display_df.columns:
        format_columns['safety_stock_qty'] = lambda x: formatter.format_number(x)
        format_columns['safety_coverage'] = lambda x: f"{x:.1f}x" if pd.notna(x) and x < 999 else "N/A"
    
    for col, fmt_func in format_columns.items():
        if col in display_df.columns:
            display_df[f"{col}_display"] = display_df[col].apply(fmt_func)
    
    # Select columns based on grouping and safety stock
    if filter_values.get('group_by') == 'product':
        columns = ['pt_code', 'product_name', 'brand']
    else:  # brand
        columns = ['brand']
    
    columns.extend(['total_supply_display', 'total_demand_display'])
    
    # Add safety stock column if included
    if include_safety and 'safety_stock_qty_display' in display_df.columns:
        columns.append('safety_stock_qty_display')
    
    columns.extend(['net_gap_display', 'Status', 'suggested_action'])
    
    # Filter to available columns
    columns = [col for col in columns if col in display_df.columns]
    
    return display_df[columns]

def display_paginated_table(df: pd.DataFrame, items_per_page: int) -> None:
    """Display paginated dataframe with improved pagination UI"""
    if df.empty:
        st.info("No data matches the current filters")
        return
    
    total_items = len(df)
    total_pages = (total_items + items_per_page - 1) // items_per_page
    
    # Initialize page in session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 1
    
    # Ensure current page is valid
    if st.session_state.current_page > total_pages:
        st.session_state.current_page = total_pages
    if st.session_state.current_page < 1:
        st.session_state.current_page = 1
    
    page = st.session_state.current_page
    
    # Calculate indices
    start_idx = (page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, total_items)
    
    # Display the data table
    page_df = df.iloc[start_idx:end_idx]
    st.dataframe(page_df, use_container_width=True, hide_index=True)
    
    # Pagination controls
    if total_pages > 1:
        col1, col2, col3, col4, col5 = st.columns([1, 1, 3, 1, 1])
        
        with col1:
            if st.button("⏮️", disabled=(page == 1), use_container_width=True):
                st.session_state.current_page = 1
                st.rerun()
        
        with col2:
            if st.button("⬅️", disabled=(page == 1), use_container_width=True):
                st.session_state.current_page = page - 1
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
            if st.button("➡️", disabled=(page == total_pages), use_container_width=True):
                st.session_state.current_page = page + 1
                st.rerun()
        
        with col5:
            if st.button("⏭️", disabled=(page == total_pages), use_container_width=True):
                st.session_state.current_page = total_pages
                st.rerun()

def display_data_table(gap_df: pd.DataFrame, filter_values: Dict, 
                      formatter: Any, metrics: Dict, include_safety: bool) -> None:
    """Display the main data table with search and export"""
    
    # Prepare display dataframe
    display_df = prepare_display_dataframe(gap_df, filter_values, formatter, include_safety)
    
    # Controls row
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_term = st.text_input("🔍 Search in results", placeholder="Type to filter...")
        if search_term:
            mask = display_df.astype(str).apply(
                lambda x: x.str.contains(search_term, case=False, na=False)
            ).any(axis=1)
            display_df = display_df[mask]
    
    with col2:
        items_per_page = st.selectbox("Items per page", [10, 25, 50, 100], index=1)
    
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
    
    # Display paginated table
    display_paginated_table(display_df, items_per_page)

def main():
    """Main application logic with integrated safety stock support"""
    # Initialize authentication
    auth_manager = AuthManager()
    
    # Check authentication
    if not auth_manager.check_session():
        st.warning("⚠️ Please login to access this page")
        st.stop()
    
    # Initialize session state for dialog
    if 'show_customer_dialog' not in st.session_state:
        st.session_state.show_customer_dialog = False
    
    # Initialize components
    data_loader, calculator, formatter, filters, charts, customer_dialog = initialize_components()
    
    # Page header
    st.title("📊 Net GAP Analysis")
    st.markdown("Comprehensive supply-demand analysis with safety stock management")
    
    # User info in sidebar
    st.sidebar.markdown(f"👤 **User:** {auth_manager.get_user_display_name()}")
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        auth_manager.logout()
        st.rerun()
    
    # Render unified filters
    filter_values = filters.render_filters()
    
    # Action buttons
    col1, col2, col3 = st.columns([1, 1, 2])
    
    calculate_clicked = False
    with col1:
        if st.button("🔄 Reset Filters", use_container_width=True):
            st.session_state.gap_filters = filters._get_default_filters()
            st.rerun()
    
    with col2:
        if st.button("📊 Calculate GAP", type="primary", use_container_width=True):
            calculate_clicked = True
    
    with col3:
        # Show active filter count
        active_count = sum([
            1 for k, v in filter_values.items() 
            if k not in ['group_by', 'quick_filter'] and v
        ])
        if active_count > 0:
            st.info(f"✓ {active_count} filters active")
    
    if not calculate_clicked:
        st.info("👆 Configure filters and click 'Calculate GAP' to begin analysis")
        st.stop()
    
    try:
        # Check if safety stock is included
        include_safety = filter_values.get('include_safety_stock', False)
        
        # Load data with timing
        start_time = time.time()
        
        with st.spinner("Loading data..."):
            # Load supply and demand data
            supply_df = data_loader.load_supply_data(
                entity_name=filter_values.get('entity'),
                date_from=filter_values.get('date_range')[0],
                date_to=filter_values.get('date_range')[1],
                product_ids=filter_values.get('products'),
                brands=filter_values.get('brands')
            )
            
            demand_df = data_loader.load_demand_data(
                entity_name=filter_values.get('entity'),
                date_from=filter_values.get('date_range')[0],
                date_to=filter_values.get('date_range')[1],
                product_ids=filter_values.get('products'),
                brands=filter_values.get('brands'),
                customers=filter_values.get('customers')
            )
            
            # Load safety stock if included
            safety_stock_df = None
            if include_safety:
                # Convert entity name to ID if needed (simplified for now)
                safety_stock_df = data_loader.load_safety_stock_data(
                    entity_id=None,  # Would need entity ID mapping
                    product_ids=filter_values.get('products'),
                    include_customer_specific=True
                )
        
        load_time = time.time() - start_time
        if load_time > DATA_LOAD_WARNING_SECONDS:
            st.info(f"⏱️ Data loading took {load_time:.1f} seconds")
        
        # Validate data
        if not validate_data(supply_df, 'supply') or not validate_data(demand_df, 'demand'):
            st.stop()
        
        # Check if data is available
        if supply_df.empty and demand_df.empty:
            st.warning("No data available for the selected filters. Please adjust your filters.")
            st.stop()
        
        # Calculate GAP with safety stock support
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
        
        # Apply quick filter (context-aware)
        gap_df_filtered = filters.apply_quick_filter(
            gap_df, 
            filter_values.get('quick_filter', 'all'),
            include_safety=include_safety
        )
        
        if gap_df_filtered.empty:
            st.info("No items match the selected quick filter.")
            st.stop()
        
        # Calculate metrics
        metrics = calculator.get_summary_metrics(gap_df_filtered)
        
        # Customer dialog handling
        if st.session_state.show_customer_dialog:
            customer_dialog.show_dialog(gap_df_filtered, calculator._filtered_demand_df)
        
        # Display configuration summary
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
        display_visualizations(gap_df_filtered, filter_values, charts, include_safety)
        
        st.divider()
        
        # Detailed data table
        st.subheader("📋 Detailed GAP Analysis")
        display_data_table(gap_df_filtered, filter_values, formatter, metrics, include_safety)
        
    except Exception as e:
        handle_error(e)
    
    # Footer
    st.divider()
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Net GAP Analysis v{VERSION}")

if __name__ == "__main__":
    main()