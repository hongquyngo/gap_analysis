# pages/1_📊_Net_GAP.py

"""
Net GAP Analysis Page
Simple supply-demand GAP analysis without time dimension
Updated with main page filters and source selection
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import logging
from typing import Dict, Any
import io

# Configure page
st.set_page_config(
    page_title="Net GAP Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"  # Collapse sidebar since filters are on main page
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import utilities - Fix the path imports
import sys
from pathlib import Path
# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).parent.parent))

from utils.auth import AuthManager
from utils.net_gap.data_loader import GAPDataLoader
from utils.net_gap.calculator import GAPCalculator
from utils.net_gap.formatters import GAPFormatter
from utils.net_gap.filters import GAPFilters
from utils.net_gap.charts import GAPCharts

# Initialize authentication
auth_manager = AuthManager()

# Check authentication
if not auth_manager.check_session():
    st.warning("⚠️ Please login to access this page")
    st.stop()

# Initialize components
@st.cache_resource
def initialize_components():
    """Initialize all GAP analysis components"""
    data_loader = GAPDataLoader()
    calculator = GAPCalculator()
    formatter = GAPFormatter()
    filters = GAPFilters(data_loader)
    charts = GAPCharts(formatter)
    return data_loader, calculator, formatter, filters, charts

# Load components
data_loader, calculator, formatter, filters, charts = initialize_components()

# Page header
st.title("📊 Net GAP Analysis")
st.markdown("""
Quick overview of total supply vs demand balance. Select supply and demand sources to customize your analysis.
""")

# User info in sidebar (minimal)
st.sidebar.markdown(f"👤 **User:** {auth_manager.get_user_display_name()}")
if st.sidebar.button("🚪 Logout", use_container_width=True):
    auth_manager.logout()
    st.rerun()

# Render filters on main page (not sidebar)
filter_values = filters.render_main_page_filters()

# Main content area
try:
    # Load data based on filters
    with st.spinner("Loading supply data..."):
        supply_df = data_loader.load_supply_data(
            entity_name=filter_values.get('entity'),
            date_from=filter_values.get('date_range')[0],
            date_to=filter_values.get('date_range')[1],
            product_ids=filter_values.get('products'),
            brands=filter_values.get('brands')
        )
    
    with st.spinner("Loading demand data..."):
        demand_df = data_loader.load_demand_data(
            entity_name=filter_values.get('entity'),
            date_from=filter_values.get('date_range')[0],
            date_to=filter_values.get('date_range')[1],
            product_ids=filter_values.get('products'),
            brands=filter_values.get('brands'),
            customers=filter_values.get('customers')
        )
    
    # Check if data is available
    if supply_df.empty and demand_df.empty:
        st.warning("No data available for the selected filters. Please adjust your filters and try again.")
        st.stop()
    
    # Calculate GAP with selected sources
    with st.spinner("Calculating GAP analysis..."):
        gap_df = calculator.calculate_net_gap(
            supply_df=supply_df,
            demand_df=demand_df,
            group_by=filter_values.get('group_by', 'product'),
            selected_supply_sources=filter_values.get('supply_sources'),
            selected_demand_sources=filter_values.get('demand_sources')
        )
    
    # Apply quick filter if selected
    gap_df_filtered = filters.apply_quick_filter(gap_df, filter_values.get('quick_filter', 'all'))
    
    # Calculate summary metrics
    metrics = calculator.get_summary_metrics(gap_df_filtered)
    
    # Display source selection summary
    st.info(f"""
    **Analysis Configuration:**
    - Supply Sources: {', '.join(filter_values.get('supply_sources', []))}
    - Demand Sources: {', '.join(filter_values.get('demand_sources', []))}
    - {filters.get_filter_summary(filter_values)}
    """)
    
    # Display KPI cards
    st.subheader("📈 Key Metrics")
    charts.create_kpi_cards(metrics)
    
    st.divider()
    
    # Visualizations
    st.subheader("📊 Visual Analysis")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["Status Distribution", "Top Shortages", "Supply vs Demand", "GAP Heatmap"])
    
    with tab1:
        if not gap_df_filtered.empty:
            # Update status distribution to show SQL-aligned statuses
            fig_pie = charts.create_status_pie_chart(gap_df_filtered)
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Show status legend
            with st.expander("📖 Status Definitions", expanded=False):
                st.markdown("""
                ### GAP Status Classifications (Aligned with Database Views):
                
                **SHORTAGE Levels:**
                - 🔴 **SEVERE_SHORTAGE**: Coverage < 50% - Emergency action required
                - 🟠 **HIGH_SHORTAGE**: Coverage 50-70% - Urgent orders needed
                - 🟡 **MODERATE_SHORTAGE**: Coverage 70-90% - Plan replenishment
                
                **BALANCED:**
                - ✅ **BALANCED**: Coverage 90-110% - Optimal inventory level
                
                **SURPLUS Levels:**
                - 🔵 **LIGHT_SURPLUS**: Coverage 110-150% - Minor excess
                - 🟣 **MODERATE_SURPLUS**: Coverage 150-200% - Review ordering
                - 🟠 **HIGH_SURPLUS**: Coverage 200-300% - Reduce orders
                - 🔴 **SEVERE_SURPLUS**: Coverage > 300% - Stop ordering
                
                **Special Cases:**
                - ⚪ **NO_DEMAND**: Inventory exists but no demand
                - 🟤 **NO_DEMAND_INCOMING**: PO exists but no demand
                """)
        else:
            st.info("No data to display for status distribution")
    
    with tab2:
        if not gap_df_filtered.empty:
            col1, col2 = st.columns([3, 1])
            with col2:
                top_n = st.number_input("Top N items", min_value=5, max_value=20, value=10)
            # Filter to shortage items for this chart
            shortage_df = gap_df_filtered[gap_df_filtered['gap_status'].str.contains('SHORTAGE')]
            if not shortage_df.empty:
                fig_bar = charts.create_top_shortage_bar_chart(shortage_df, top_n=top_n)
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("No shortage items to display")
        else:
            st.info("No shortage items to display")
    
    with tab3:
        if not gap_df_filtered.empty:
            fig_comparison = charts.create_supply_demand_comparison(gap_df_filtered, top_n=15)
            st.plotly_chart(fig_comparison, use_container_width=True)
        else:
            st.info("No data to compare")
    
    with tab4:
        if not gap_df_filtered.empty and filter_values.get('group_by') == 'product':
            fig_heatmap = charts.create_gap_heatmap(gap_df_filtered, group_by='brand')
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.info("Heatmap requires product-level grouping")
    
    st.divider()
    
    # Detailed data table
    st.subheader("📋 Detailed GAP Analysis")
    
    # Prepare display dataframe with SQL-aligned status
    display_df = gap_df_filtered.copy()
    
    # Format columns for display
    if not display_df.empty:
        # Map status to display format
        status_display = {
            'NO_DEMAND': '⚪ No Demand',
            'NO_DEMAND_INCOMING': '🟤 No Demand (PO Incoming)',
            'SEVERE_SHORTAGE': '🔴 Severe Shortage',
            'HIGH_SHORTAGE': '🟠 High Shortage',
            'MODERATE_SHORTAGE': '🟡 Moderate Shortage',
            'BALANCED': '✅ Balanced',
            'LIGHT_SURPLUS': '🔵 Light Surplus',
            'MODERATE_SURPLUS': '🟣 Moderate Surplus',
            'HIGH_SURPLUS': '🟠 High Surplus',
            'SEVERE_SURPLUS': '🔴 Severe Surplus',
            'UNKNOWN': '❓ Unknown'
        }
        
        display_df['Status'] = display_df['gap_status'].map(status_display).fillna('❓ Unknown')
        
        # Format numeric columns
        numeric_format_cols = {
            'total_supply': 'Supply',
            'total_demand': 'Demand',
            'net_gap': 'Net GAP',
            'coverage_ratio': 'Coverage',
            'gap_percentage': 'GAP %'
        }
        
        for old_col, new_col in numeric_format_cols.items():
            if old_col in display_df.columns:
                if old_col == 'coverage_ratio':
                    display_df[new_col] = display_df[old_col].apply(
                        lambda x: f"{x:.2f}x" if x < 10 else "999x+" if x >= 999 else f"{x:.0f}x"
                    )
                elif 'percentage' in old_col:
                    display_df[new_col] = display_df[old_col].apply(
                        lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A"
                    )
                elif old_col == 'net_gap':
                    display_df[new_col] = display_df[old_col].apply(
                        lambda x: formatter.format_number(x, show_sign=True)
                    )
                else:
                    display_df[new_col] = display_df[old_col].apply(
                        lambda x: formatter.format_number(x)
                    )
        
        # Add source breakdown if available
        source_cols = []
        for col in display_df.columns:
            if col.startswith('supply_') and col != 'supply_breakdown':
                source_name = col.replace('supply_', '').upper()
                if source_name in filter_values.get('supply_sources', []):
                    display_name = f"Supply: {source_name}"
                    display_df[display_name] = display_df[col].apply(formatter.format_number)
                    source_cols.append(display_name)
        
        for col in display_df.columns:
            if col.startswith('demand_') and col != 'demand_breakdown':
                source_name = col.replace('demand_', '').replace('_pending', '').upper()
                if source_name in filter_values.get('demand_sources', []):
                    display_name = f"Demand: {source_name}"
                    display_df[display_name] = display_df[col].apply(formatter.format_number)
                    source_cols.append(display_name)
        
        # Select columns to display based on grouping
        base_columns = ['Supply', 'Demand', 'Net GAP', 'Coverage', 'Status', 'priority', 'suggested_action']
        
        if filter_values.get('group_by') == 'product':
            display_columns = ['pt_code', 'product_name', 'brand'] + source_cols + base_columns
            column_names = {
                'pt_code': 'PT Code',
                'product_name': 'Product Name',
                'brand': 'Brand',
                'priority': 'Priority',
                'suggested_action': 'Action Required'
            }
        elif filter_values.get('group_by') == 'brand':
            display_columns = ['brand'] + source_cols + base_columns
            column_names = {
                'brand': 'Brand',
                'priority': 'Priority',
                'suggested_action': 'Action Required'
            }
        else:  # category
            display_columns = ['category'] + source_cols + base_columns
            column_names = {
                'category': 'Category',
                'priority': 'Priority',
                'suggested_action': 'Action Required'
            }
        
        # Filter to available columns
        display_columns = [col for col in display_columns if col in display_df.columns]
        
        # Create final display dataframe
        final_display_df = display_df[display_columns].copy()
        final_display_df = final_display_df.rename(columns=column_names)
        
        # Sort by priority
        if 'Priority' in final_display_df.columns:
            final_display_df = final_display_df.sort_values('Priority')
        
        # Add row selection and export
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            search_term = st.text_input("🔍 Search in results", placeholder="Type to filter...")
            if search_term:
                mask = final_display_df.astype(str).apply(
                    lambda x: x.str.contains(search_term, case=False, na=False)
                ).any(axis=1)
                final_display_df = final_display_df[mask]
        
        with col2:
            items_per_page = st.selectbox("Items per page", [10, 25, 50, 100], index=1)
        
        with col3:
            if st.button("📥 Export to Excel", type="primary", use_container_width=True):
                # Prepare export data
                export_df = gap_df_filtered.copy()
                
                # Create Excel file in memory
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    # Summary sheet
                    summary_data = {
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
                            metrics['overall_coverage'],
                            metrics['total_shortage'],
                            metrics['total_surplus'],
                            metrics['at_risk_value_usd'],
                            metrics['affected_customers'],
                            ', '.join(filter_values.get('supply_sources', [])),
                            ', '.join(filter_values.get('demand_sources', []))
                        ]
                    }
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                    
                    # Detail sheet
                    export_df.to_excel(writer, sheet_name='GAP Details', index=False)
                    
                    # Filters sheet
                    filters_data = {
                        'Filter': ['Entity', 'Date From', 'Date To', 'Supply Sources', 'Demand Sources', 
                                 'Brands', 'Products', 'Customers', 'Quick Filter', 'Group By'],
                        'Value': [
                            filter_values.get('entity', 'All'),
                            str(filter_values.get('date_range', ['N/A'])[0]),
                            str(filter_values.get('date_range', ['N/A', 'N/A'])[1]),
                            ', '.join(filter_values.get('supply_sources', [])),
                            ', '.join(filter_values.get('demand_sources', [])),
                            ', '.join(filter_values.get('brands', [])) or 'All',
                            f"{len(filter_values.get('products', []))} selected" if filter_values.get('products') else 'All',
                            ', '.join(filter_values.get('customers', [])) or 'All',
                            filter_values.get('quick_filter', 'all'),
                            filter_values.get('group_by', 'product')
                        ]
                    }
                    filters_df = pd.DataFrame(filters_data)
                    filters_df.to_excel(writer, sheet_name='Filters', index=False)
                
                # Prepare download
                output.seek(0)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                st.download_button(
                    label="📥 Download Excel File",
                    data=output.getvalue(),
                    file_name=f"gap_analysis_{timestamp}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                st.success("✅ Export prepared successfully!")
        
        # Display table with pagination
        if not final_display_df.empty:
            # Calculate pagination
            total_items = len(final_display_df)
            total_pages = (total_items + items_per_page - 1) // items_per_page
            
            # Page selector
            if total_pages > 1:
                page = st.selectbox(
                    f"Page (Total: {total_items} items)",
                    range(1, total_pages + 1),
                    format_func=lambda x: f"Page {x} of {total_pages}"
                )
            else:
                page = 1
            
            # Get page data
            start_idx = (page - 1) * items_per_page
            end_idx = min(start_idx + items_per_page, total_items)
            page_df = final_display_df.iloc[start_idx:end_idx]
            
            # Display table
            st.dataframe(
                page_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Net GAP": st.column_config.TextColumn(
                        "Net GAP",
                        help="Supply - Demand. Negative = shortage, Positive = surplus"
                    ),
                    "Coverage": st.column_config.TextColumn(
                        "Coverage",
                        help="Supply / Demand ratio. 1x = perfectly balanced"
                    ),
                    "GAP %": st.column_config.TextColumn(
                        "GAP %",
                        help="(Net GAP / Demand) × 100%. Shows shortage/surplus as percentage"
                    ),
                    "Status": st.column_config.TextColumn(
                        "Status",
                        help="GAP severity classification based on coverage ratio"
                    ),
                    "Priority": st.column_config.NumberColumn(
                        "Priority",
                        help="Action priority (1=Critical, 99=OK)",
                        format="%d"
                    ),
                    "Action Required": st.column_config.TextColumn(
                        "Action Required",
                        help="Recommended action based on GAP analysis"
                    )
                }
            )
        else:
            st.info("No data matches the current filters")
    
except Exception as e:
    logger.error(f"Error in Net GAP analysis: {e}")
    st.error(f"An error occurred during analysis: {str(e)}")
    st.info("Please check your filters and try again. If the problem persists, contact support.")

# Footer
st.divider()
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Net GAP Analysis v2.0 - Aligned with SQL Views")