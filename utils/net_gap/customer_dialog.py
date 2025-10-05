# utils/net_gap/customer_dialog.py

"""
Customer Impact Dialog for GAP Analysis System - Version 2.1 (Refactored)
- Integrated with SessionStateManager (no DataFrame storage in session)
- Vectorized customer impact calculation for performance
- Optimized pagination handling
- Better memory management
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime
import io

from .session_manager import get_session_manager

logger = logging.getLogger(__name__)

# Constants
ITEMS_PER_PAGE_OPTIONS = [10, 20, 50, 100]
DEFAULT_ITEMS_PER_PAGE = 20
MAX_PRODUCTS_PER_CUSTOMER = 20


class CustomerImpactDialog:
    """Manages customer impact dialog display with optimized data handling"""
    
    def __init__(self, calculator, formatter):
        """
        Initialize dialog with calculator and formatter
        
        Args:
            calculator: GAPCalculator instance
            formatter: GAPFormatter instance
        """
        self.calculator = calculator
        self.formatter = formatter
        self.session_manager = get_session_manager()
    
    def show_dialog(self, gap_df: pd.DataFrame, demand_df: pd.DataFrame) -> None:
        """
        Open the customer impact dialog
        
        Args:
            gap_df: GAP analysis results
            demand_df: Original demand data
        """
        # Extract only shortage product IDs (lightweight)
        shortage_df = gap_df[gap_df['net_gap'] < 0].copy()
        
        if shortage_df.empty:
            st.warning("No shortage items found")
            return
        
        if 'product_id' not in shortage_df.columns:
            st.error("Product-level grouping required for customer impact analysis")
            return
        
        shortage_product_ids = shortage_df['product_id'].tolist()
        
        # Pre-calculate summary metrics (lightweight)
        metrics = {
            'total_shortage_value': shortage_df['at_risk_value_usd'].sum(),
            'shortage_count': len(shortage_product_ids),
            'total_demand': shortage_df['total_demand'].sum(),
            'total_shortage_qty': shortage_df[shortage_df['net_gap'] < 0]['net_gap'].abs().sum()
        }
        
        # Open dialog with minimal data
        self.session_manager.open_customer_dialog(shortage_product_ids, metrics)
        
        logger.info(f"Customer dialog opened with {len(shortage_product_ids)} shortage products")
    
    def calculate_customer_impact(
        self, 
        shortage_product_ids: List[int],
        demand_df: pd.DataFrame,
        gap_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate customer impact using vectorized operations for performance
        
        Args:
            shortage_product_ids: List of product IDs with shortages
            demand_df: Original demand data
            gap_df: GAP analysis results (for shortage details)
            
        Returns:
            DataFrame with customer impact analysis
        """
        try:
            # Build shortage lookup dictionary (vectorized)
            shortage_df = gap_df[gap_df['product_id'].isin(shortage_product_ids)].copy()
            
            # Handle potential duplicates by keeping first occurrence
            # This can happen if data was filtered multiple times
            if shortage_df['product_id'].duplicated().any():
                logger.warning(f"Found {shortage_df['product_id'].duplicated().sum()} duplicate product IDs, keeping first occurrence")
                shortage_df = shortage_df.drop_duplicates(subset=['product_id'], keep='first')
            
            shortage_lookup = shortage_df.set_index('product_id').to_dict('index')
            
            # Filter demand for shortage products
            affected_demand = demand_df[
                demand_df['product_id'].isin(shortage_product_ids)
            ].copy()
            
            if affected_demand.empty:
                logger.warning("No demand found for shortage products")
                return pd.DataFrame()
            
            # Add shortage information to demand (vectorized merge)
            affected_demand['product_net_gap'] = affected_demand['product_id'].map(
                lambda x: shortage_lookup.get(x, {}).get('net_gap', 0)
            )
            affected_demand['product_total_demand'] = affected_demand['product_id'].map(
                lambda x: shortage_lookup.get(x, {}).get('total_demand', 1)
            )
            affected_demand['product_at_risk_value'] = affected_demand['product_id'].map(
                lambda x: shortage_lookup.get(x, {}).get('at_risk_value_usd', 0)
            )
            affected_demand['product_coverage'] = affected_demand['product_id'].map(
                lambda x: shortage_lookup.get(x, {}).get('coverage_ratio', 0)
            )
            
            # Calculate customer's share of shortage (vectorized)
            affected_demand['demand_share'] = np.where(
                affected_demand['product_total_demand'] > 0,
                affected_demand['required_quantity'] / affected_demand['product_total_demand'],
                0
            )
            
            affected_demand['customer_shortage'] = (
                abs(affected_demand['product_net_gap']) * affected_demand['demand_share']
            )
            
            affected_demand['customer_at_risk'] = (
                affected_demand['product_at_risk_value'] * affected_demand['demand_share']
            )
            
            # Group by customer (vectorized aggregation)
            customer_agg = affected_demand.groupby('customer').agg({
                'product_id': 'nunique',
                'required_quantity': 'sum',
                'customer_shortage': 'sum',
                'total_value_usd': 'sum',
                'customer_at_risk': 'sum',
                'demand_source': lambda x: ', '.join(x.unique())
            }).reset_index()
            
            customer_agg.columns = [
                'customer', 'product_count', 'total_required', 
                'total_shortage', 'total_demand_value', 'at_risk_value', 'sources'
            ]
            
            # Get customer codes and urgency (first occurrence per customer)
            customer_info = affected_demand.groupby('customer').first()[
                ['customer_code', 'urgency_level']
            ].reset_index()
            
            customer_agg = customer_agg.merge(customer_info, on='customer', how='left')
            
            # Determine overall urgency per customer
            urgency_priority = {'OVERDUE': 0, 'URGENT': 1, 'UPCOMING': 2, 'FUTURE': 3}
            customer_urgency = affected_demand.groupby('customer')['urgency_level'].apply(
                lambda x: min(x, key=lambda v: urgency_priority.get(v, 999), default='FUTURE')
            ).reset_index()
            customer_urgency.columns = ['customer', 'urgency']
            
            customer_agg = customer_agg.merge(customer_urgency, on='customer', how='left')
            customer_agg.drop('urgency_level', axis=1, inplace=True)
            
            # Build product details for each customer (more efficient than nested loops)
            customer_products = []
            for customer_name in customer_agg['customer'].unique():
                cust_demand = affected_demand[
                    affected_demand['customer'] == customer_name
                ].copy()
                
                # Sort by at-risk value and limit
                cust_demand = cust_demand.sort_values('customer_at_risk', ascending=False)
                cust_demand = cust_demand.head(MAX_PRODUCTS_PER_CUSTOMER)
                
                products = []
                for _, row in cust_demand.iterrows():
                    products.append({
                        'pt_code': row.get('pt_code', ''),
                        'product_name': row.get('product_name', ''),
                        'brand': row.get('brand', ''),
                        'required_quantity': row['required_quantity'],
                        'shortage_quantity': row['customer_shortage'],
                        'demand_value': row.get('total_value_usd', 0),
                        'at_risk_value': row['customer_at_risk'],
                        'coverage': row['product_coverage'] * 100,  # Convert to percentage
                        'urgency': row.get('urgency_level', 'N/A'),
                        'source': row.get('demand_source', '')
                    })
                
                customer_products.append({
                    'customer': customer_name,
                    'products': products
                })
            
            # Merge product details back
            products_df = pd.DataFrame(customer_products)
            customer_agg = customer_agg.merge(products_df, on='customer', how='left')
            
            # Sort by at-risk value (descending)
            customer_agg = customer_agg.sort_values('at_risk_value', ascending=False)
            
            logger.info(f"Calculated impact for {len(customer_agg)} customers")
            return customer_agg
            
        except Exception as e:
            logger.error(f"Error calculating customer impact: {e}", exc_info=True)
            st.error(f"Failed to calculate customer impact: {str(e)}")
            return pd.DataFrame()
    
    def export_excel(self, df: pd.DataFrame) -> Optional[bytes]:
        """
        Export customer impact to Excel
        
        Args:
            df: Customer impact DataFrame
            
        Returns:
            Excel file as bytes, or None on error
        """
        try:
            output = io.BytesIO()
            
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Summary sheet
                summary_df = df[[
                    'customer', 'customer_code', 'product_count',
                    'total_required', 'total_shortage', 
                    'total_demand_value', 'at_risk_value',
                    'urgency', 'sources'
                ]].copy()
                
                summary_df.columns = [
                    'Customer', 'Customer Code', 'Products Affected',
                    'Total Required', 'Total Shortage',
                    'Total Demand Value', 'At Risk Value',
                    'Urgency', 'Demand Sources'
                ]
                
                summary_df.to_excel(writer, sheet_name='Customer Summary', index=False)
                
                # Product details sheet
                details = []
                for _, row in df.iterrows():
                    for prod in row['products']:
                        details.append({
                            'Customer': row['customer'],
                            'Customer Code': row['customer_code'],
                            'PT Code': prod['pt_code'],
                            'Product': prod['product_name'],
                            'Brand': prod['brand'],
                            'Required': prod['required_quantity'],
                            'Shortage': prod['shortage_quantity'],
                            'Demand Value': prod['demand_value'],
                            'At Risk Value': prod['at_risk_value'],
                            'Coverage %': prod['coverage'],
                            'Urgency': prod['urgency']
                        })
                
                if details:
                    pd.DataFrame(details).to_excel(
                        writer, sheet_name='Product Details', index=False
                    )
                
                # Auto-adjust column widths
                for sheet in writer.sheets.values():
                    for column in sheet.columns:
                        max_length = 0
                        column_letter = column[0].column_letter
                        for cell in column:
                            try:
                                if cell.value and len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except:
                                pass
                        adjusted_width = min(max_length + 2, 50)
                        sheet.column_dimensions[column_letter].width = adjusted_width
            
            output.seek(0)
            logger.info("Customer impact exported to Excel successfully")
            return output.getvalue()
            
        except Exception as e:
            logger.error(f"Excel export error: {e}", exc_info=True)
            st.error(f"Failed to export: {str(e)}")
            return None


@st.dialog("Customer Impact Analysis", width="large")
def show_customer_popup():
    """
    Customer impact popup dialog with SessionStateManager integration
    
    This dialog loads data on-demand rather than storing it in session state
    """
    session_manager = get_session_manager()
    
    # Check if dialog should be shown
    if not session_manager.show_customer_dialog():
        return
    
    # Get minimal data from session
    shortage_ids, summary_metrics = session_manager.get_dialog_data()
    
    if not shortage_ids or not summary_metrics:
        st.error("No data available for customer impact analysis")
        if st.button("Close", use_container_width=True):
            session_manager.close_customer_dialog()
            st.rerun()
        return
    
    # Get calculator, formatter, and data from parent context
    # These should be passed via session_state temporarily during dialog initialization
    calculator = st.session_state.get('_temp_calculator')
    formatter = st.session_state.get('_temp_formatter')
    demand_df = st.session_state.get('_temp_demand_df')
    gap_df = st.session_state.get('_temp_gap_df')
    
    if calculator is None or formatter is None or demand_df is None or gap_df is None:
        st.error("Required components not available")
        if st.button("Close", use_container_width=True):
            session_manager.close_customer_dialog()
            st.rerun()
        return
    
    # Initialize dialog instance
    dialog = CustomerImpactDialog(calculator, formatter)
    
    # Calculate customer impact data on-demand
    with st.spinner("Analyzing customer impact..."):
        customer_data = dialog.calculate_customer_impact(
            shortage_ids, demand_df, gap_df
        )
    
    if customer_data.empty:
        st.warning("No affected customers found")
        st.info(
            "This can happen if:\n"
            "• No products have shortages\n"
            "• Filters excluded affected customers"
        )
        if st.button("Close", use_container_width=True):
            session_manager.close_customer_dialog()
            st.rerun()
        return
    
    # Header
    st.markdown("### Customer Impact Analysis")
    st.caption("Customers affected by product shortages")
    
    # Summary metrics (6 columns)
    cols = st.columns(6)
    
    with cols[0]:
        st.metric("Customers", formatter.format_number(len(customer_data)))
    
    with cols[1]:
        st.metric("Products", formatter.format_number(customer_data['product_count'].sum()))
    
    with cols[2]:
        total_demand = customer_data['total_demand_value'].sum()
        st.metric(
            "Total Demand",
            formatter.format_currency(total_demand, abbreviate=True),
            help="Total value of all affected orders"
        )
    
    with cols[3]:
        at_risk_total = customer_data['at_risk_value'].sum()
        st.metric(
            "Value at Risk",
            formatter.format_currency(at_risk_total, abbreviate=True),
            help="Value at risk due to shortages"
        )
    
    with cols[4]:
        st.metric(
            "Total Shortage", 
            formatter.format_number(customer_data['total_shortage'].sum())
        )
    
    with cols[5]:
        urgent = len(customer_data[customer_data['urgency'].isin(['OVERDUE', 'URGENT'])])
        if urgent > 0:
            st.metric("Urgent", urgent, delta="Need attention", delta_color="inverse")
        else:
            st.metric("Urgent", "0")
    
    st.divider()
    
    # Controls
    ctrl_cols = st.columns([3, 1, 1])
    
    with ctrl_cols[0]:
        search = st.text_input(
            "Search", 
            placeholder="Customer name or code...", 
            key="dlg_search"
        )
    
    with ctrl_cols[1]:
        page_size = st.selectbox(
            "Show", 
            ITEMS_PER_PAGE_OPTIONS, 
            index=1, 
            key="dlg_size"
        )
    
    with ctrl_cols[2]:
        excel = dialog.export_excel(customer_data)
        if excel:
            st.download_button(
                "Export",
                excel,
                f"customers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
    
    # Filter data
    if search:
        filtered = customer_data[
            customer_data['customer'].str.contains(search, case=False, na=False) |
            customer_data['customer_code'].astype(str).str.contains(search, case=False, na=False)
        ]
    else:
        filtered = customer_data
    
    if filtered.empty:
        st.info("No matches found")
    else:
        st.divider()
        display_customers(filtered, page_size, formatter, session_manager)
    
    # Footer
    st.divider()
    if st.button("Close", use_container_width=True, type="primary"):
        # Clean up temporary data
        for key in ['_temp_calculator', '_temp_formatter', '_temp_demand_df', '_temp_gap_df']:
            if key in st.session_state:
                del st.session_state[key]
        
        session_manager.close_customer_dialog()
        st.rerun()


def display_customers(
    data: pd.DataFrame, 
    page_size: int, 
    formatter,
    session_manager
) -> None:
    """
    Display customer list with pagination using SessionStateManager
    
    Args:
        data: Customer impact data
        page_size: Number of items per page
        formatter: GAPFormatter instance
        session_manager: SessionStateManager instance
    """
    total = len(data)
    pages = max(1, (total + page_size - 1) // page_size)
    
    # Get and validate current page
    page = session_manager.get_dialog_page()
    session_manager.set_dialog_page(page, pages)
    page = session_manager.get_dialog_page()  # Get validated page
    
    start = (page - 1) * page_size
    end = min(start + page_size, total)
    
    # Current page data
    page_data = data.iloc[start:end]
    
    st.caption(f"**Showing {start + 1}-{end} of {total} customers**")
    
    # Display each customer
    for _, row in page_data.iterrows():
        # Urgency indicator
        urgency_icons = {
            'OVERDUE': '🔴',
            'URGENT': '🟠',
            'UPCOMING': '🟡',
            'FUTURE': '🟢'
        }
        icon = urgency_icons.get(row['urgency'], '⚪')
        
        with st.expander(
            f"{icon} **{row['customer']}** ({row['customer_code']}) - "
            f"{row['product_count']} products affected",
            expanded=False
        ):
            # Metrics row
            m_cols = st.columns(5)
            
            with m_cols[0]:
                st.metric("Required", formatter.format_number(row['total_required']))
            
            with m_cols[1]:
                st.metric("Shortage", formatter.format_number(row['total_shortage']))
            
            with m_cols[2]:
                st.metric("Demand Value", formatter.format_currency(row['total_demand_value']))
            
            with m_cols[3]:
                st.metric("At Risk", formatter.format_currency(row['at_risk_value']))
            
            with m_cols[4]:
                st.metric("Urgency", row['urgency'])
            
            st.divider()
            
            # Product table
            st.markdown("**Affected Products:**")
            
            # Header
            h_cols = st.columns([0.3, 2, 1, 1, 1, 1, 0.8])
            headers = ["#", "Product", "Required", "Shortage", "At Risk", "Coverage", "Urgency"]
            for col, header in zip(h_cols, headers):
                with col:
                    st.caption(header)
            
            # Products
            products = row['products'][:MAX_PRODUCTS_PER_CUSTOMER]
            for i, prod in enumerate(products, 1):
                p_cols = st.columns([0.3, 2, 1, 1, 1, 1, 0.8])
                
                with p_cols[0]:
                    st.text(str(i))
                
                with p_cols[1]:
                    st.text(prod['pt_code'])
                    st.caption(prod['product_name'][:30])
                
                with p_cols[2]:
                    st.text(formatter.format_number(prod['required_quantity']))
                
                with p_cols[3]:
                    st.text(formatter.format_number(prod['shortage_quantity']))
                
                with p_cols[4]:
                    st.text(formatter.format_currency(prod['at_risk_value'], abbreviate=True))
                
                with p_cols[5]:
                    cov = prod['coverage']
                    if cov < 50:
                        st.text(f"🔴 {cov:.0f}%")
                    elif cov < 80:
                        st.text(f"🟡 {cov:.0f}%")
                    else:
                        st.text(f"🟢 {cov:.0f}%")
                
                with p_cols[6]:
                    urg = prod['urgency']
                    urg_icon = urgency_icons.get(urg, '⚪')
                    st.text(urg_icon)
            
            if len(row['products']) > MAX_PRODUCTS_PER_CUSTOMER:
                st.caption(
                    f"... and {len(row['products']) - MAX_PRODUCTS_PER_CUSTOMER} more products"
                )
    
    # Pagination controls
    if pages > 1:
        st.divider()
        p_cols = st.columns([1, 1, 3, 1, 1])
        
        with p_cols[0]:
            if st.button("◀◀", disabled=(page == 1), use_container_width=True, key="dialog_page_first"):
                session_manager.set_dialog_page(1, pages)
                st.rerun()
        
        with p_cols[1]:
            if st.button("◀", disabled=(page == 1), use_container_width=True, key="dialog_page_prev"):
                session_manager.set_dialog_page(page - 1, pages)
                st.rerun()
        
        with p_cols[2]:
            st.markdown(
                f"<center>Page <b>{page}</b> of <b>{pages}</b></center>", 
                unsafe_allow_html=True
            )
        
        with p_cols[3]:
            if st.button("▶", disabled=(page == pages), use_container_width=True, key="dialog_page_next"):
                session_manager.set_dialog_page(page + 1, pages)
                st.rerun()
        
        with p_cols[4]:
            if st.button("▶▶", disabled=(page == pages), use_container_width=True, key="dialog_page_last"):
                session_manager.set_dialog_page(pages, pages)
                st.rerun()