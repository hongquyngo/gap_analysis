# pages/1_📊_Net_GAP.py

"""
Net GAP Analysis Page
Simple supply-demand GAP analysis without time dimension
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
    initial_sidebar_state="expanded"
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
from utils.gap.data_loader import GAPDataLoader
from utils.gap.calculator import GAPCalculator
from utils.gap.formatters import GAPFormatter
from utils.gap.filters import GAPFilters
from utils.gap.charts import GAPCharts

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
Quick overview of total supply vs demand balance without time dimension.
Ideal for morning checks and executive summaries.
""")

# Display user info
user_display = auth_manager.get_user_display_name()
st.sidebar.markdown(f"👤 **User:** {user_display}")
st.sidebar.divider()

# Render filters in sidebar
filter_values = filters.render_sidebar_filters()

# Display active filters summary
filter_summary = filters.get_filter_summary(filter_values)
st.info(f"🔍 **Active Filters:** {filter_summary}")

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
    
    # Calculate GAP
    with st.spinner("Calculating GAP analysis..."):
        gap_df = calculator.calculate_net_gap(
            supply_df=supply_df,
            demand_df=demand_df,
            group_by=filter_values.get('group_by', 'product')
        )
    
    # Apply quick filter if selected
    gap_df_filtered = filters.apply_quick_filter(gap_df, filter_values.get('quick_filter', 'all'))
    
    # Calculate summary metrics
    metrics = calculator.get_summary_metrics(gap_df_filtered)
    
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
            fig_pie = charts.create_status_pie_chart(gap_df_filtered)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No data to display for status distribution")
    
    with tab2:
        if not gap_df_filtered.empty:
            # Allow user to select number of items
            col1, col2 = st.columns([3, 1])
            with col2:
                top_n = st.number_input("Top N items", min_value=5, max_value=20, value=10)
            fig_bar = charts.create_top_shortage_bar_chart(gap_df_filtered, top_n=top_n)
            st.plotly_chart(fig_bar, use_container_width=True)
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
            # Show heatmap by brand when grouped by product
            fig_heatmap = charts.create_gap_heatmap(gap_df_filtered, group_by='brand')
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.info("Heatmap requires product-level grouping")
    
    st.divider()
    
    # Detailed data table
    st.subheader("📋 Detailed GAP Analysis")
    
    # Prepare display dataframe
    display_df = gap_df_filtered.copy()
    
    # Format columns for display
    if not display_df.empty:
        # Add status emoji
        display_df['Status'] = display_df['gap_status'].apply(
            lambda x: f"{formatter.STATUS_CONFIG.get(x, {}).get('emoji', '')} {formatter.STATUS_CONFIG.get(x, {}).get('label', x)}"
        )
        
        # Format numeric columns
        numeric_format_cols = {
            'total_supply': 'Supply',
            'total_demand': 'Demand',
            'net_gap': 'Net GAP',
            'gap_percentage': 'GAP %',
            'coverage_rate': 'Coverage %'
        }
        
        for old_col, new_col in numeric_format_cols.items():
            if old_col in display_df.columns:
                if 'percentage' in old_col or 'rate' in old_col:
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
        
        # Select columns to display based on grouping
        if filter_values.get('group_by') == 'product':
            display_columns = ['pt_code', 'product_name', 'brand', 'Supply', 'Demand', 
                             'Net GAP', 'GAP %', 'Coverage %', 'Status', 'suggested_action']
            # Rename for display
            column_names = {
                'pt_code': 'PT Code',
                'product_name': 'Product Name',
                'brand': 'Brand',
                'suggested_action': 'Suggested Action'
            }
        elif filter_values.get('group_by') == 'brand':
            display_columns = ['brand', 'Supply', 'Demand', 'Net GAP', 'GAP %', 
                             'Coverage %', 'Status', 'suggested_action']
            column_names = {
                'brand': 'Brand',
                'suggested_action': 'Suggested Action'
            }
        else:  # category
            display_columns = ['category', 'Supply', 'Demand', 'Net GAP', 'GAP %', 
                             'Coverage %', 'Status', 'suggested_action']
            column_names = {
                'category': 'Category',
                'suggested_action': 'Suggested Action'
            }
        
        # Filter to available columns
        display_columns = [col for col in display_columns if col in display_df.columns]
        
        # Create final display dataframe
        final_display_df = display_df[display_columns].copy()
        final_display_df = final_display_df.rename(columns=column_names)
        
        # Add row selection
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            # Search within results
            search_term = st.text_input("🔍 Search in results", placeholder="Type to filter...")
            if search_term:
                mask = final_display_df.astype(str).apply(
                    lambda x: x.str.contains(search_term, case=False, na=False)
                ).any(axis=1)
                final_display_df = final_display_df[mask]
        
        with col2:
            # Items per page
            items_per_page = st.selectbox("Items per page", [10, 25, 50, 100], index=1)
        
        with col3:
            # Export button
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
                            'Affected Customers'
                        ],
                        'Value': [
                            metrics['total_products'],
                            metrics['shortage_items'],
                            metrics['critical_items'],
                            metrics['overall_coverage'],
                            metrics['total_shortage'],
                            metrics['total_surplus'],
                            metrics['at_risk_value_usd'],
                            metrics['affected_customers']
                        ]
                    }
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                    
                    # Detail sheet
                    export_df.to_excel(writer, sheet_name='GAP Details', index=False)
                    
                    # Filters sheet
                    filters_data = {
                        'Filter': ['Entity', 'Date From', 'Date To', 'Brands', 'Products', 'Customers', 'Quick Filter', 'Group By'],
                        'Value': [
                            filter_values.get('entity', 'All'),
                            str(filter_values.get('date_range', ['N/A'])[0]),
                            str(filter_values.get('date_range', ['N/A', 'N/A'])[1]),
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
            
            # Display table with tooltips
            # Note: For full tooltip support, we'd need custom HTML/CSS
            # For now, show basic table with help text
            st.dataframe(
                page_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Net GAP": st.column_config.TextColumn(
                        "Net GAP",
                        help="Supply - Demand. Negative = shortage, Positive = surplus"
                    ),
                    "GAP %": st.column_config.TextColumn(
                        "GAP %",
                        help="(Net GAP / Demand) × 100%. Shows shortage/surplus as percentage of demand"
                    ),
                    "Coverage %": st.column_config.TextColumn(
                        "Coverage %",
                        help="(Supply / Demand) × 100%. Shows how much of demand is covered by supply"
                    ),
                    "Status": st.column_config.TextColumn(
                        "Status",
                        help="GAP severity classification based on percentage"
                    ),
                    "Suggested Action": st.column_config.TextColumn(
                        "Suggested Action",
                        help="Recommended action based on GAP analysis"
                    )
                }
            )
            
            # Show selected item details
            st.divider()
            with st.expander("💡 **Understanding the Data**", expanded=False):
                st.markdown("""
                ### Date Logic Explanations:
                - **Required Date**: ETD/ETA when demand needs fulfillment. Negative days indicate overdue items requiring immediate action.
                - **Availability Date**: Expected date when supply becomes available based on source:
                  - *Inventory*: Available immediately (0 days)
                  - *CAN Pending*: 1-3 days for stock-in processing
                  - *Warehouse Transfer*: 2-5 days transit time
                  - *Purchase Order*: 7-30 days based on vendor location
                
                ### Quantity Explanations:
                - **Supply**: Total available quantity from all sources (Inventory + CAN + Transfers + POs)
                - **Demand**: Total required quantity from confirmed orders and forecasts
                - **Net GAP**: Supply minus Demand. Negative indicates shortage needing action
                - **Allocated Quantity**: Amount already reserved for specific orders
                - **Unallocated Quantity**: Amount still available for allocation
                
                ### Status Classifications:
                - 🔴 **Severe Shortage**: GAP < -50% of demand - Emergency PO needed
                - 🟠 **High Shortage**: GAP -50% to -20% - Urgent action required
                - 🟡 **Low Shortage**: GAP -20% to -5% - Plan replenishment
                - ✅ **Balanced**: GAP -5% to +10% - Optimal inventory level
                - 🔵 **Surplus**: GAP +10% to +50% - Consider demand adjustment
                - 🟣 **High Surplus**: GAP > +50% - Evaluate redistribution options
                """)
        else:
            st.info("No data matches the current filters")
    else:
        st.info("No data available after applying filters")
    
except Exception as e:
    logger.error(f"Error in Net GAP analysis: {e}")
    st.error(f"An error occurred during analysis: {str(e)}")
    st.info("Please check your filters and try again. If the problem persists, contact support.")

# Footer
st.divider()
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Net GAP Analysis v1.0")