# pages/1_ðŸ“Š_Net_GAP.py

"""
Net GAP Analysis Page - Version 3.0
- Removed Category grouping
- Removed Quick Date Range presets
- Support multiselect for products and customers
- Auto date range from data
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
    page_icon="ðŸ“Š",
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
        st.error("ðŸ”Œ Database connection issue. Please refresh the page and try again.")
    elif "permission" in error_msg or "denied" in error_msg:
        st.error("ðŸ”’ Access denied. Please check your permissions.")
    elif "timeout" in error_msg:
        st.error("â±ï¸ Request timed out. Try using more specific filters.")
    else:
        st.error(f"âŒ An error occurred: {error_type}")
    
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
        st.warning(f"âš ï¸ Large dataset ({len(gap_df)} rows). Export limited to {MAX_EXPORT_ROWS} rows.")
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
                'Affected Customers',
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
                metrics['affected_customers'],
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
                f"{len(filters.get('customers', []))} selected" if filters.get('customers') else 'All',
                filters.get('group_by', 'product')
            ]
        })
        filters_data.to_excel(writer, sheet_name='Applied Filters', index=False)
    
    output.seek(0)
    return output.getvalue()

def display_visualizations(gap_df: pd.DataFrame, filter_values: Dict, charts: Any) -> None:
    """Display visualization tabs - REMOVED CATEGORY LOGIC"""
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
        # Only show heatmap for product grouping (by brand)
        if not gap_df.empty and filter_values.get('group_by') == 'product':
            fig_heatmap = charts.create_gap_heatmap(gap_df, group_by='brand')
            st.plotly_chart(fig_heatmap, use_container_width=True)
        elif filter_values.get('group_by') == 'brand':
            st.info("Heatmap not available for brand-level grouping")
        else:
            st.info("Heatmap requires product-level grouping")

def prepare_display_dataframe(gap_df: pd.DataFrame, filter_values: Dict, 
                              formatter: Any) -> pd.DataFrame:
    """Prepare dataframe for display with formatting - REMOVED CATEGORY"""
    display_df = gap_df.copy()
    
    # Status mapping
    status_display = {
        'NO_DEMAND': 'âšª No Demand',
        'SEVERE_SHORTAGE': 'ðŸ”´ Severe Shortage',
        'HIGH_SHORTAGE': 'ðŸŸ  High Shortage',
        'MODERATE_SHORTAGE': 'ðŸŸ¡ Moderate Shortage',
        'BALANCED': 'âœ… Balanced',
        'LIGHT_SURPLUS': 'ðŸ”µ Light Surplus',
        'MODERATE_SURPLUS': 'ðŸŸ£ Moderate Surplus',
        'HIGH_SURPLUS': 'ðŸŸ  High Surplus',
        'SEVERE_SURPLUS': 'ðŸ”´ Severe Surplus'
    }
    
    display_df['Status'] = display_df['gap_status'].map(status_display).fillna('â“ Unknown')
    
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
    
    # Select columns based on grouping - ONLY PRODUCT OR BRAND
    if filter_values.get('group_by') == 'product':
        columns = ['pt_code', 'product_name', 'brand', 'total_supply_display', 
                  'total_demand_display', 'net_gap_display', 'Status', 'suggested_action']
    else:  # brand
        columns = ['brand', 'total_supply_display', 'total_demand_display', 
                  'net_gap_display', 'Status', 'suggested_action']
    
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
    
    # Initialize page in session state if not exists
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
    
    # Display the data table first
    page_df = df.iloc[start_idx:end_idx]
    st.dataframe(page_df, use_container_width=True, hide_index=True)
    
    # Pagination controls
    if total_pages > 1:
        col1, col2, col3, col4, col5 = st.columns([1, 1, 3, 1, 1])
        
        with col1:
            if st.button("â—€â—€", disabled=(page == 1), use_container_width=True, 
                        help="First page"):
                st.session_state.current_page = 1
                st.rerun()
        
        with col2:
            if st.button("â—€", disabled=(page == 1), use_container_width=True,
                        help="Previous page"):
                st.session_state.current_page = page - 1
                st.rerun()
        
        with col3:
            st.markdown(
                f"<div style='text-align: center; padding: 8px; font-size: 14px;'>"
                f"<strong>{start_idx + 1:,}-{end_idx:,}</strong> of <strong>{total_items:,}</strong>"
                f"</div>",
                unsafe_allow_html=True
            )
        
        with col4:
            if st.button("â–¶", disabled=(page == total_pages), use_container_width=True,
                        help="Next page"):
                st.session_state.current_page = page + 1
                st.rerun()
        
        with col5:
            if st.button("â–¶â–¶", disabled=(page == total_pages), use_container_width=True,
                        help="Last page"):
                st.session_state.current_page = total_pages
                st.rerun()
        
        # Optional: Add page jump functionality
        with st.expander("Jump to page", expanded=False):
            col1, col2 = st.columns([3, 1])
            with col1:
                jump_page = st.number_input(
                    "Page number",
                    min_value=1,
                    max_value=total_pages,
                    value=page,
                    label_visibility="collapsed"
                )
            with col2:
                if st.button("Go", use_container_width=True):
                    st.session_state.current_page = jump_page
                    st.rerun()
            
            st.caption(f"Total pages: {total_pages:,}")
    else:
        # Single page - just show the count
        st.markdown(
            f"<div style='text-align: center; padding: 8px; font-size: 14px; color: #666;'>"
            f"Showing all <strong>{total_items:,}</strong> items"
            f"</div>",
            unsafe_allow_html=True
        )

def display_data_table(gap_df: pd.DataFrame, filter_values: Dict, 
                       formatter: Any, metrics: Dict) -> None:
    """Display the main data table with search and export"""
    
    # Prepare display dataframe
    display_df = prepare_display_dataframe(gap_df, filter_values, formatter)
    
    # Controls row
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_term = st.text_input("ðŸ” Search in results", placeholder="Type to filter...")
        if search_term:
            mask = display_df.astype(str).apply(
                lambda x: x.str.contains(search_term, case=False, na=False)
            ).any(axis=1)
            display_df = display_df[mask]
    
    with col2:
        items_per_page = st.selectbox("Items per page", [10, 25, 50, 100], index=1)
    
    with col3:
        if st.button("ðŸ“¥ Export to Excel", type="primary", use_container_width=True):
            excel_data = export_to_excel(gap_df, metrics, filter_values)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            st.download_button(
                label="ðŸ“¥ Download Excel File",
                data=excel_data,
                file_name=f"gap_analysis_{timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            st.success("âœ… Export prepared successfully!")
    
    # Display paginated table
    display_paginated_table(display_df, items_per_page)

def main():
    """Main application logic with Customer Dialog integration"""
    # Initialize authentication
    auth_manager = AuthManager()
    
    # Check authentication
    if not auth_manager.check_session():
        st.warning("âš ï¸ Please login to access this page")
        st.stop()
    
    # Initialize session state for dialog
    if 'show_customer_dialog' not in st.session_state:
        st.session_state.show_customer_dialog = False
    
    # Initialize components
    data_loader, calculator, formatter, filters, charts, customer_dialog = initialize_components()
    
    # Page header
    st.title("ðŸ“Š Net GAP Analysis")
    st.markdown("Quick overview of total supply vs demand balance. Select sources to customize your analysis.")
    
    # User info in sidebar
    st.sidebar.markdown(f"ðŸ‘¤ **User:** {auth_manager.get_user_display_name()}")
    if st.sidebar.button("ðŸšª Logout", use_container_width=True):
        auth_manager.logout()
        st.rerun()
    
    # Render filters - now with multiselect and auto date range
    filter_values = filters.render_main_page_filters()
    
    # Validate group_by - only product or brand allowed
    if filter_values.get('group_by') not in ['product', 'brand']:
        filter_values['group_by'] = 'product'
    
    try:
        # Load data with timing
        start_time = time.time()
        
        with st.spinner("Loading supply and demand data..."):
            supply_df = data_loader.load_supply_data(
                entity_name=filter_values.get('entity'),
                date_from=filter_values.get('date_range')[0],
                date_to=filter_values.get('date_range')[1],
                product_ids=filter_values.get('products'),  # Now multiselect
                brands=filter_values.get('brands')
            )
            
            demand_df = data_loader.load_demand_data(
                entity_name=filter_values.get('entity'),
                date_from=filter_values.get('date_range')[0],
                date_to=filter_values.get('date_range')[1],
                product_ids=filter_values.get('products'),  # Now multiselect
                brands=filter_values.get('brands'),
                customers=filter_values.get('customers')  # Now multiselect
            )
        
        load_time = time.time() - start_time
        if load_time > DATA_LOAD_WARNING_SECONDS:
            st.info(f"â±ï¸ Data loading took {load_time:.1f} seconds. Consider using more specific filters.")
        
        # Validate data
        if not validate_data(supply_df, 'supply') or not validate_data(demand_df, 'demand'):
            st.stop()
        
        # Check if data is available
        if supply_df.empty and demand_df.empty:
            st.warning("No data available for the selected filters. Please adjust your filters.")
            st.stop()
        
        # Calculate GAP - only product or brand grouping
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
        
        # CUSTOMER DIALOG HANDLING
        if st.session_state.show_customer_dialog:
            # Show dialog as popup
            customer_dialog.show_dialog(gap_df_filtered, calculator._filtered_demand_df)
        
        # Display configuration summary
        st.info(f"""
        **Analysis Configuration:**
        - Supply: {', '.join(filter_values.get('supply_sources', []))}
        - Demand: {', '.join(filter_values.get('demand_sources', []))}
        - {filters.get_filter_summary(filter_values)}
        """)
        
        # Display KPI cards with customer dialog button enabled for product-level
        st.subheader("ðŸ“ˆ Key Metrics")
        charts.create_kpi_cards(
            metrics, 
            enable_customer_dialog=(filter_values.get('group_by') == 'product')
        )
        
        st.divider()
        
        # Visualizations
        st.subheader("ðŸ“Š Visual Analysis")
        display_visualizations(gap_df_filtered, filter_values, charts)
        
        st.divider()
        
        # Detailed data table
        st.subheader("ðŸ“‹ Detailed GAP Analysis")
        display_data_table(gap_df_filtered, filter_values, formatter, metrics)
        
    except Exception as e:
        handle_error(e)
    
    # Footer
    st.divider()
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Net GAP Analysis v3.0")

if __name__ == "__main__":
    main()