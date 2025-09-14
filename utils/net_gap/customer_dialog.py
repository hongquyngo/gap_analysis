

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
import io

logger = logging.getLogger(__name__)

# Constants
ITEMS_PER_PAGE_OPTIONS = [10, 20, 50, 100]
DEFAULT_ITEMS_PER_PAGE = 20


class CustomerImpactDialog:
    """Manages customer impact dialog display"""
    
    def __init__(self, calculator, formatter):
        """Initialize dialog with calculator and formatter"""
        self.calculator = calculator
        self.formatter = formatter
    
    def show_dialog(self, gap_df: pd.DataFrame, demand_df: pd.DataFrame) -> None:
        """Trigger the customer impact dialog popup"""
        # Store data in session state for the dialog
        st.session_state['dialog_gap_df'] = gap_df
        st.session_state['dialog_demand_df'] = demand_df
        st.session_state['dialog_instance'] = self
        
        # Show the popup dialog
        show_customer_popup()
    
    def calculate_customer_impact(self, gap_df: pd.DataFrame, demand_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate customer impact with correct at-risk value"""
        try:
            # Get shortage products
            shortage_df = gap_df[gap_df['net_gap'] < 0].copy()
            
            if shortage_df.empty:
                return pd.DataFrame()
            
            if 'product_id' not in shortage_df.columns:
                logger.warning("Product-level grouping required")
                return pd.DataFrame()
            
            # Build shortage lookup with all details
            shortage_lookup = {}
            for _, row in shortage_df.iterrows():
                product_id = row['product_id']
                shortage_lookup[product_id] = {
                    'product_name': row.get('product_name', ''),
                    'pt_code': row.get('pt_code', ''),
                    'brand': row.get('brand', ''),
                    'net_gap': abs(row['net_gap']),
                    'total_demand': row.get('total_demand', 0),
                    'total_supply': row.get('total_supply', 0),
                    'coverage_ratio': row.get('coverage_ratio', 0),
                    'at_risk_value_usd': row.get('at_risk_value_usd', 0)  # Get product-level at-risk value
                }
            
            # Filter demand for shortage products
            affected_demand = demand_df[demand_df['product_id'].isin(shortage_lookup.keys())].copy()
            
            if affected_demand.empty:
                return pd.DataFrame()
            
            # Process each customer
            customer_records = []
            
            for customer_name in affected_demand['customer'].unique():
                if pd.isna(customer_name):
                    continue
                
                # Get all demands for this customer
                cust_demands = affected_demand[affected_demand['customer'] == customer_name]
                
                # Calculate metrics
                product_list = []
                total_at_risk = 0
                
                for _, demand_row in cust_demands.iterrows():
                    product_id = demand_row['product_id']
                    product_info = shortage_lookup[product_id]
                    
                    # Calculate proportional shortage for this customer
                    if product_info['total_demand'] > 0:
                        # Customer's share of total demand
                        demand_share = demand_row['required_quantity'] / product_info['total_demand']
                        # Customer's share of shortage
                        customer_shortage = product_info['net_gap'] * demand_share
                        # Customer's share of at-risk value
                        customer_at_risk = product_info['at_risk_value_usd'] * demand_share
                    else:
                        customer_shortage = 0
                        customer_at_risk = 0
                    
                    total_at_risk += customer_at_risk
                    
                    # Build product detail
                    product_list.append({
                        'pt_code': demand_row.get('pt_code', ''),
                        'product_name': demand_row.get('product_name', ''),
                        'brand': demand_row.get('brand', ''),
                        'required_quantity': demand_row['required_quantity'],
                        'shortage_quantity': customer_shortage,
                        'demand_value': demand_row.get('total_value_usd', 0),
                        'at_risk_value': customer_at_risk,
                        'coverage': product_info['coverage_ratio'],
                        'urgency': demand_row.get('urgency_level', 'N/A'),
                        'source': demand_row.get('demand_source', '')
                    })
                
                # Sort products by at-risk value
                product_list.sort(key=lambda x: x['at_risk_value'], reverse=True)
                
                # Determine urgency
                urgency_levels = cust_demands['urgency_level'].tolist()
                if 'OVERDUE' in urgency_levels:
                    urgency = 'OVERDUE'
                elif 'URGENT' in urgency_levels:
                    urgency = 'URGENT'
                elif 'UPCOMING' in urgency_levels:
                    urgency = 'UPCOMING'
                else:
                    urgency = 'FUTURE'
                
                # Create customer record
                customer_records.append({
                    'customer': customer_name,
                    'customer_code': cust_demands.iloc[0].get('customer_code', ''),
                    'product_count': len(cust_demands),
                    'total_required': cust_demands['required_quantity'].sum(),
                    'total_shortage': sum(p['shortage_quantity'] for p in product_list),
                    'total_demand_value': cust_demands['total_value_usd'].sum(),
                    'at_risk_value': total_at_risk,
                    'urgency': urgency,
                    'sources': ', '.join(cust_demands['demand_source'].unique()),
                    'products': product_list
                })
            
            # Create DataFrame
            if not customer_records:
                return pd.DataFrame()
            
            df = pd.DataFrame(customer_records)
            # Sort by at-risk value
            df = df.sort_values('at_risk_value', ascending=False)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating customer impact: {e}", exc_info=True)
            return pd.DataFrame()
    
    def export_excel(self, df: pd.DataFrame) -> bytes:
        """Export to Excel with 2 sheets"""
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
                    pd.DataFrame(details).to_excel(writer, sheet_name='Product Details', index=False)
                
                # Auto-adjust widths
                for sheet in writer.sheets.values():
                    for column in sheet.columns:
                        max_length = 0
                        column_letter = column[0].column_letter
                        for cell in column:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except:
                                pass
                        sheet.column_dimensions[column_letter].width = min(max_length + 2, 50)
            
            output.seek(0)
            return output.getvalue()
            
        except Exception as e:
            logger.error(f"Export error: {e}")
            return None


@st.dialog("👥 Affected Customers Detail", width="large")
def show_customer_popup():
    """Customer impact popup dialog"""
    
    # Get data from session state
    gap_df = st.session_state.get('dialog_gap_df')
    demand_df = st.session_state.get('dialog_demand_df')
    dialog = st.session_state.get('dialog_instance')
    
    if gap_df is None or demand_df is None or dialog is None:
        st.error("No data available")
        if st.button("Close", use_container_width=True):
            cleanup_and_close()
        return
    
    # Calculate customer data
    with st.spinner("Processing customer impact..."):
        data = dialog.calculate_customer_impact(gap_df, demand_df)
    
    if data.empty:
        st.warning("No affected customers found")
        st.info("This can happen if:\n• No products have shortages\n• Filters excluded affected customers")
        if st.button("Close", use_container_width=True):
            cleanup_and_close()
        return
    
    # Header
    st.markdown("### Customer Impact Analysis")
    st.caption("Customers affected by product shortages")
    
    # Summary metrics (6 columns)
    cols = st.columns(6)
    
    with cols[0]:
        st.metric("Customers", dialog.formatter.format_number(len(data)))
    
    with cols[1]:
        st.metric("Products", dialog.formatter.format_number(data['product_count'].sum()))
    
    with cols[2]:
        total_demand = data['total_demand_value'].sum()
        st.metric(
            "Total Demand",
            dialog.formatter.format_currency(total_demand, abbreviate=True),
            help="Total value of all affected orders"
        )
    
    with cols[3]:
        # This should match main page
        at_risk_total = data['at_risk_value'].sum()
        st.metric(
            "Value at Risk",
            dialog.formatter.format_currency(at_risk_total, abbreviate=True),
            help="Value at risk due to shortages"
        )
    
    with cols[4]:
        st.metric("Total Shortage", dialog.formatter.format_number(data['total_shortage'].sum()))
    
    with cols[5]:
        urgent = len(data[data['urgency'].isin(['OVERDUE', 'URGENT'])])
        if urgent > 0:
            st.metric("🔴 Urgent", urgent, delta="Need attention")
        else:
            st.metric("✅ Urgent", "0")
    
    st.divider()
    
    # Controls
    ctrl_cols = st.columns([3, 1, 1])
    
    with ctrl_cols[0]:
        search = st.text_input("🔍 Search", placeholder="Customer name or code...", key="dlg_search")
    
    with ctrl_cols[1]:
        page_size = st.selectbox("Show", ITEMS_PER_PAGE_OPTIONS, index=1, key="dlg_size")
    
    with ctrl_cols[2]:
        excel = dialog.export_excel(data)
        if excel:
            st.download_button(
                "📥 Export",
                excel,
                f"customers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
    
    # Filter
    if search:
        filtered = data[
            data['customer'].str.contains(search, case=False, na=False) |
            data['customer_code'].str.contains(search, case=False, na=False)
        ]
    else:
        filtered = data
    
    if filtered.empty:
        st.info("No matches found")
    else:
        st.divider()
        display_customers(filtered, page_size, dialog.formatter)
    
    # Footer
    st.divider()
    if st.button("Close", use_container_width=True, type="primary"):
        cleanup_and_close()


def display_customers(data: pd.DataFrame, page_size: int, formatter):
    """Display customer list with pagination"""
    
    # Pagination state
    if 'dlg_page' not in st.session_state:
        st.session_state.dlg_page = 1
    
    total = len(data)
    pages = max(1, (total + page_size - 1) // page_size)
    
    # Validate page
    if st.session_state.dlg_page > pages:
        st.session_state.dlg_page = pages
    if st.session_state.dlg_page < 1:
        st.session_state.dlg_page = 1
    
    page = st.session_state.dlg_page
    start = (page - 1) * page_size
    end = min(start + page_size, total)
    
    # Current page data
    page_data = data.iloc[start:end]
    
    st.caption(f"**Showing {start + 1}-{end} of {total} customers**")
    
    # Display each customer
    for _, row in page_data.iterrows():
        # Urgency icon
        icon = "🔴" if row['urgency'] in ['OVERDUE', 'URGENT'] else "🟡" if row['urgency'] == 'UPCOMING' else "🟢"
        
        with st.expander(
            f"{icon} **{row['customer']}** ({row['customer_code']}) - {row['product_count']} products affected",
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
            st.markdown("**📦 Affected Products:**")
            
            # Header
            h_cols = st.columns([0.3, 2, 1, 1, 1, 1, 0.8])
            with h_cols[0]:
                st.caption("#")
            with h_cols[1]:
                st.caption("Product")
            with h_cols[2]:
                st.caption("Required")
            with h_cols[3]:
                st.caption("Shortage")
            with h_cols[4]:
                st.caption("At Risk")
            with h_cols[5]:
                st.caption("Coverage")
            with h_cols[6]:
                st.caption("Urgency")
            
            # Products
            for i, prod in enumerate(row['products'][:20], 1):  # Show max 20
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
                    if urg == 'OVERDUE':
                        st.text("🔴")
                    elif urg == 'URGENT':
                        st.text("🟠")
                    elif urg == 'UPCOMING':
                        st.text("🟡")
                    else:
                        st.text("🟢")
            
            if len(row['products']) > 20:
                st.caption(f"... and {len(row['products']) - 20} more products")
    
    # Pagination
    if pages > 1:
        st.divider()
        p_cols = st.columns([1, 1, 3, 1, 1])
        
        with p_cols[0]:
            if st.button("◀◀", disabled=(page == 1), use_container_width=True):
                st.session_state.dlg_page = 1
                st.rerun()
        
        with p_cols[1]:
            if st.button("◀", disabled=(page == 1), use_container_width=True):
                st.session_state.dlg_page = page - 1
                st.rerun()
        
        with p_cols[2]:
            st.markdown(f"<center>Page <b>{page}</b> of <b>{pages}</b></center>", unsafe_allow_html=True)
        
        with p_cols[3]:
            if st.button("▶", disabled=(page == pages), use_container_width=True):
                st.session_state.dlg_page = page + 1
                st.rerun()
        
        with p_cols[4]:
            if st.button("▶▶", disabled=(page == pages), use_container_width=True):
                st.session_state.dlg_page = pages
                st.rerun()


def cleanup_and_close():
    """Clean up session state and close dialog"""
    keys_to_remove = [
        'dialog_gap_df', 'dialog_demand_df', 'dialog_instance',
        'dlg_page', 'dlg_search', 'dlg_size'
    ]
    for key in keys_to_remove:
        if key in st.session_state:
            del st.session_state[key]
    
    st.session_state.show_customer_dialog = False
    st.rerun()