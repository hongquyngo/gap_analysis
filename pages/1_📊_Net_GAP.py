# pages/1_📊_Net_GAP.py

"""
Net GAP Analysis Page - Clean Version
Simple supply-demand GAP analysis without time dimension
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

# Import utilities - Improved path handling
project_root = os.environ.get('PROJECT_ROOT', Path(__file__).parent.parent)
if str(project_root) not in os.sys.path:
    os.sys.path.insert(0, str(project_root))

from utils.auth import AuthManager
from utils.net_gap.data_loader import GAPDataLoader
from utils.net_gap.calculator import GAPCalculator
from utils.net_gap.formatters import GAPFormatter
from utils.net_gap.filters import GAPFilters
from utils.net_gap.charts import GAPCharts

# Constants
MAX_EXPORT_ROWS = 10000
DATA_LOAD_WARNING_SECONDS = 5

def initialize_components():
    """Initialize all GAP analysis components (non-cached for DB connections)"""
    data_loader = GAPDataLoader()
    calculator = GAPCalculator()
    formatter = GAPFormatter()
    filters = GAPFilters(data_loader)
    charts = GAPCharts(formatter)
    return data_loader, calculator, formatter, filters, charts

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

def export_to_excel(gap_df: pd.DataFrame, metrics: Dict, filters: Dict) -> bytes:
    """
    Export GAP analysis to Excel
    
    Args:
        gap_df: GAP analysis dataframe
        metrics: Summary metrics
        filters: Applied filters
    
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
        summary_data = pd.DataFrame({
            'Metric': [
                'Total Products',
                'Shortage Items',
                'Critical Items',
                'Coverage Rate (%)',
                'Total Shortage',
                'Total Surplus',
                'At Risk Value (USD)',
                'Supply Sources',
                'Demand Sources'
            ],
            'Value': [
                metrics['total_products'],
                metrics['shortage_items'],
                metrics['critical_items'],
                f"{metrics['overall_coverage']:.1f}%",
                metrics['total_shortage'],
                metrics['total_surplus'],
                f"${metrics['at_risk_value_usd']:,.2f}",
                ', '.join(filters.get('supply_sources', [])),
                ', '.join(filters.get('demand_sources', []))
            ]
        })
        summary_data.to_excel(writer, sheet_name='Summary', index=False)
        
        # Detail sheet
        gap_df.to_excel(writer, sheet_name='GAP Details', index=False)
        
        # Filters sheet
        filters_data = pd.DataFrame({
            'Filter': ['Entity', 'Date Range', 'Supply Sources', 'Demand Sources', 
                      'Products', 'Brands', 'Customers', 'Group By'],
            'Value': [
                filters.get('entity', 'All'),
                f"{filters.get('date_range', ['N/A'])[0]} to {filters.get('date_range', ['N/A', 'N/A'])[1]}",
                ', '.join(filters.get('supply_sources', [])),
                ', '.join(filters.get('demand_sources', [])),
                f"{len(filters.get('products', []))} selected" if filters.get('products') else 'All',
                ', '.join(filters.get('brands', [])) or 'All',
                ', '.join(filters.get('customers', [])) or 'All',
                filters.get('group_by', 'product')
            ]
        })
        filters_data.to_excel(writer, sheet_name='Applied Filters', index=False)
    
    output.seek(0)
    return output.getvalue()

def main():
    """Main application logic"""
    # Initialize authentication
    auth_manager = AuthManager()
    
    # Check authentication
    if not auth_manager.check_session():
        st.warning("⚠️ Please login to access this page")
        st.stop()
    
    # Initialize components
    data_loader, calculator, formatter, filters, charts = initialize_components()
    
    # Page header
    st.title("📊 Net GAP Analysis")
    st.markdown("Quick overview of total supply vs demand balance. Select sources to customize your analysis.")
    
    # User info in sidebar
    st.sidebar.markdown(f"👤 **User:** {auth_manager.get_user_display_name()}")
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        auth_manager.logout()
        st.rerun()
    
    # Render filters
    filter_values = filters.render_main_page_filters()
    
    try:
        # Load data with timing
        start_time = time.time()
        
        with st.spinner("Loading supply and demand data..."):
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
        
        load_time = time.time() - start_time
        if load_time > DATA_LOAD_WARNING_SECONDS:
            st.info(f"⏱️ Data loading took {load_time:.1f} seconds. Consider using more specific filters.")
        
        # Validate data
        if not validate_data(supply_df, 'supply') or not validate_data(demand_df, 'demand'):
            st.stop()
        
        # Check if data is available
        if supply_df.empty and demand_df.empty:
            st.warning("No data available for the selected filters. Please adjust your filters.")
            st.stop()
        
        # Calculate GAP
        with st.spinner("Calculating GAP analysis..."):
            gap_df = calculator.calculate_net_gap(
                supply_df=supply_df,
                demand_df=demand_df,
                group_by=filter_values.get('group_by', 'product'),
                selected_supply_sources=filter_values.get('supply_sources'),
                selected_demand_sources=filter_values.get('demand_sources')
            )
        
        # Apply quick filter
        gap_df_filtered = filters.apply_quick_filter(
            gap_df, 
            filter_values.get('quick_filter', 'all')
        )
        
        if gap_df_filtered.empty:
            st.info("No items match the selected quick filter.")
            st.stop()
        
        # Calculate metrics
        metrics = calculator.get_summary_metrics(gap_df_filtered)
        
        # Display configuration summary
        st.info(f"""
        **Analysis Configuration:**
        - Supply: {', '.join(filter_values.get('supply_sources', []))}
        - Demand: {', '.join(filter_values.get('demand_sources', []))}
        - {filters.get_filter_summary(filter_values)}
        """)
        
        # Display KPI cards
        st.subheader("📈 Key Metrics")
        charts.create_kpi_cards(metrics)
        
        st.divider()
        
        # Visualizations
        st.subheader("📊 Visual Analysis")
        display_visualizations(gap_df_filtered, filter_values, charts)
        
        st.divider()
        
        # Detailed data table
        st.subheader("📋 Detailed GAP Analysis")
        display_data_table(gap_df_filtered, filter_values, formatter, metrics)
        
    except Exception as e:
        handle_error(e)
    
    # Footer
    st.divider()
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Net GAP Analysis v2.1")

def display_visualizations(gap_df: pd.DataFrame, filter_values: Dict, charts: Any) -> None:
    """Display visualization tabs"""
    tab1, tab2, tab3, tab4 = st.tabs(["Status Distribution", "Top Shortages", "Supply vs Demand", "GAP Heatmap"])
    
    with tab1:
        if not gap_df.empty:
            fig_pie = charts.create_status_pie_chart(gap_df)
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns([3, 1])
        with col2:
            top_n = st.number_input("Top N items", min_value=5, max_value=20, value=10)
        
        shortage_df = gap_df[gap_df['gap_status'].str.contains('SHORTAGE')]
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
        if not gap_df.empty and filter_values.get('group_by') == 'product':
            fig_heatmap = charts.create_gap_heatmap(gap_df, group_by='brand')
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.info("Heatmap requires product-level grouping")

def display_data_table(gap_df: pd.DataFrame, filter_values: Dict, 
                       formatter: Any, metrics: Dict) -> None:
    """Display the main data table with search and export"""
    
    # Prepare display dataframe
    display_df = prepare_display_dataframe(gap_df, filter_values, formatter)
    
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
            excel_data = export_to_excel(gap_df, metrics, filter_values)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            st.download_button(
                label="📥 Download Excel File",
                data=excel_data,
                file_name=f"gap_analysis_{timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            st.success("✅ Export prepared successfully!")
    
    # Display paginated table
    display_paginated_table(display_df, items_per_page)

def prepare_display_dataframe(gap_df: pd.DataFrame, filter_values: Dict, 
                              formatter: Any) -> pd.DataFrame:
    """Prepare dataframe for display with formatting"""
    display_df = gap_df.copy()
    
    # Status mapping
    status_display = {
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
    
    for col, fmt_func in format_columns.items():
        if col in display_df.columns:
            display_df[f"{col}_display"] = display_df[col].apply(fmt_func)
    
    # Select columns based on grouping
    if filter_values.get('group_by') == 'product':
        columns = ['pt_code', 'product_name', 'brand', 'total_supply_display', 
                  'total_demand_display', 'net_gap_display', 'Status', 'suggested_action']
    elif filter_values.get('group_by') == 'brand':
        columns = ['brand', 'total_supply_display', 'total_demand_display', 
                  'net_gap_display', 'Status', 'suggested_action']
    else:
        columns = ['category', 'total_supply_display', 'total_demand_display', 
                  'net_gap_display', 'Status', 'suggested_action']
    
    # Filter to available columns
    columns = [col for col in columns if col in display_df.columns]
    
    return display_df[columns]

def display_paginated_table(df: pd.DataFrame, items_per_page: int) -> None:
    """Display paginated dataframe"""
    if df.empty:
        st.info("No data matches the current filters")
        return
    
    total_items = len(df)
    total_pages = (total_items + items_per_page - 1) // items_per_page
    
    if total_pages > 1:
        page = st.selectbox(
            f"Page (Total: {total_items} items)",
            range(1, total_pages + 1),
            format_func=lambda x: f"Page {x} of {total_pages}"
        )
    else:
        page = 1
    
    start_idx = (page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, total_items)
    page_df = df.iloc[start_idx:end_idx]
    
    st.dataframe(page_df, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()