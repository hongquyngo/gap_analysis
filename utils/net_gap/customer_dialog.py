# utils/net_gap/customer_dialog.py

"""
Customer Affected Dialog - Version 3.0 SIMPLIFIED
- Gets data from GAPCalculationResult (no fallbacks)
- Pre-calculated data (no computation in dialog)
- Removed emergency database reload
- Clean error handling
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
import io

from .session_manager import get_session_manager

logger = logging.getLogger(__name__)

# Constants
ITEMS_PER_PAGE_OPTIONS = [10, 20, 50, 100]
DEFAULT_ITEMS_PER_PAGE = 20
MAX_PRODUCTS_PER_CUSTOMER = 20


class CustomerAffectedDialog:
    """Manages customer affected dialog with pre-calculated data"""
    
    def __init__(self, calculator, formatter):
        """
        Initialize dialog
        
        Args:
            calculator: GAPCalculator instance (for compatibility)
            formatter: GAPFormatter instance
        """
        self.calculator = calculator
        self.formatter = formatter
        self.session_manager = get_session_manager()
    
    def export_excel(self, df: pd.DataFrame) -> Optional[bytes]:
        """
        Export customer affected data to Excel
        
        Args:
            df: Customer affected DataFrame
            
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
            logger.info("Customer affected data exported to Excel")
            return output.getvalue()
            
        except Exception as e:
            logger.error(f"Excel export error: {e}", exc_info=True)
            st.error(f"Failed to export: {str(e)}")
            return None


@st.dialog("Affected Customer Analysis", width="large")
def show_customer_popup():
    """
    Customer affected popup dialog - SIMPLIFIED
    Gets pre-calculated data from GAPCalculationResult
    """
    session_manager = get_session_manager()
    
    # Check if dialog should be shown
    if not session_manager.show_customer_dialog():
        return
    
    # Get calculation result
    result = session_manager.get_gap_result()
    
    if result is None:
        st.error("❌ No calculation result available")
        st.info("Please calculate GAP analysis first.")
        if st.button("Close", use_container_width=True):
            session_manager.close_customer_dialog()
            st.rerun()
        return
    
    # Check if customer impact data exists
    if result.customer_impact is None or result.customer_impact.is_empty():
        st.warning("No customer impact data available")
        st.info(
            "This could be because:\n"
            "• No shortage items found\n"
            "• Analysis not grouped by product\n"
            "• No demand data for shortage products"
        )
        if st.button("Close", use_container_width=True):
            session_manager.close_customer_dialog()
            st.rerun()
        return
    
    # Get pre-calculated customer data
    customer_data = result.customer_impact.customer_summary_df
    
    # Get formatter from session (for display)
    formatter = st.session_state.get('_temp_formatter')
    if formatter is None:
        from .formatters import GAPFormatter
        formatter = GAPFormatter()
    
    # Initialize dialog helper
    from .calculator import GAPCalculator
    calculator = GAPCalculator()
    dialog = CustomerAffectedDialog(calculator, formatter)
    
    # Header
    st.markdown("### 👥 Customer Affected Analysis")
    st.caption("Customers impacted by product shortages")
    
    # Summary metrics (from pre-calculated data)
    cols = st.columns(6)
    with cols[0]:
        st.metric("Customers", formatter.format_number(result.customer_impact.affected_count))
    with cols[1]:
        st.metric("Products", formatter.format_number(customer_data['product_count'].sum()))
    with cols[2]:
        st.metric(
            "Total Demand", 
            formatter.format_currency(customer_data['total_demand_value'].sum(), abbreviate=True)
        )
    with cols[3]:
        st.metric(
            "Value at Risk", 
            formatter.format_currency(result.customer_impact.total_at_risk_value, abbreviate=True)
        )
    with cols[4]:
        st.metric(
            "Total Shortage", 
            formatter.format_number(result.customer_impact.total_shortage_qty)
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
                f"affected_customers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
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
        session_manager.close_customer_dialog()
        st.rerun()


def display_customers(
    data: pd.DataFrame, 
    page_size: int, 
    formatter, 
    session_manager
) -> None:
    """Display customer list with pagination"""
    total = len(data)
    pages = max(1, (total + page_size - 1) // page_size)
    
    page = session_manager.get_dialog_page()
    session_manager.set_dialog_page(page, pages)
    page = session_manager.get_dialog_page()
    
    start = (page - 1) * page_size
    end = min(start + page_size, total)
    page_data = data.iloc[start:end]
    
    st.caption(f"**Showing {start + 1}-{end} of {total} customers**")
    
    urgency_icons = {
        'OVERDUE': '🔴',
        'URGENT': '🟠',
        'UPCOMING': '🟡',
        'FUTURE': '🟢'
    }
    
    for _, row in page_data.iterrows():
        icon = urgency_icons.get(row['urgency'], '⚪')
        with st.expander(
            f"{icon} **{row['customer']}** ({row['customer_code']}) - "
            f"{row['product_count']} products affected", 
            expanded=False
        ):
            # Metrics
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
            
            # Tabs for products and calculations
            tab1, tab2 = st.tabs(["📦 Affected Products", "📊 Calculation Details"])
            
            with tab1:
                st.caption("**Product-level breakdown:**")
                
                # Header row
                h_cols = st.columns([0.3, 2, 1, 1, 1, 1, 0.8])
                headers = ["#", "Product", "Required", "Shortage", "At Risk", "Coverage", "Urgency"]
                for col, header in zip(h_cols, headers):
                    col.caption(header)
                
                # Product rows
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
                        cov_text = f"{cov:.0f}%"
                        if cov < 50:
                            st.markdown(f"🔴 **{cov_text}**")
                        elif cov < 80:
                            st.markdown(f"🟡 **{cov_text}**")
                        else:
                            st.markdown(f"🟢 **{cov_text}**")
                    with p_cols[6]:
                        urg = prod['urgency']
                        urg_icon = urgency_icons.get(urg, '⚪')
                        st.markdown(urg_icon)
                
                if len(row['products']) > MAX_PRODUCTS_PER_CUSTOMER:
                    st.caption(
                        f"... and {len(row['products']) - MAX_PRODUCTS_PER_CUSTOMER} more products"
                    )
            
            with tab2:
                # Calculation explanation
                st.markdown(f"""
                **How these numbers are calculated:**

                ### 1️⃣ Shortage Allocation
                Customer requires **{formatter.format_number(row['total_required'])}** units 
                across **{row['product_count']}** products.

                For each product, the customer's shortage is calculated as:
                    Customer Shortage = Product Shortage × (Customer Demand ÷ Total Product Demand)

                ### 2️⃣ At Risk Value
                - Total order value: **{formatter.format_currency(row['total_demand_value'])}**
                - Unfulfilled quantity: **{formatter.format_number(row['total_shortage'])}**
                - Revenue at risk: **{formatter.format_currency(row['at_risk_value'])}**

                The at-risk value represents the portion of the customer's order value that 
                cannot be fulfilled due to shortages.

                ### 3️⃣ Coverage Calculation
                    Coverage % = (Available Supply ÷ Total Demand) × 100
                    - 🔴 <50%: Severe shortage
                    - 🟡 50-80%: High shortage
                    - 🟢 80-100%: Moderate shortage

                    ### 4️⃣ Urgency Levels
                    Based on required dates:
                    - 🔴 **OVERDUE**: Past due date
                    - 🟠 **URGENT**: Due within 7 days
                    - 🟡 **UPCOMING**: Due within 30 days
                    - 🟢 **FUTURE**: Due after 30 days

                    ### 5️⃣ Demand Sources
                    {row['sources']}
                    """)
    
    # Pagination
    if pages > 1:
        st.divider()
        pag_cols = st.columns([1, 3, 1])
        with pag_cols[0]:
            if st.button("Previous", disabled=(page == 1), use_container_width=True, key="dlg_prev"):
                session_manager.set_dialog_page(page - 1, pages)
                st.rerun()
        with pag_cols[1]:
            st.markdown(
                f"<div style='text-align: center; padding: 8px;'>Page {page} of {pages}</div>", 
                unsafe_allow_html=True
            )
        with pag_cols[2]:
            if st.button("Next", disabled=(page == pages), use_container_width=True, key="dlg_next"):
                session_manager.set_dialog_page(page + 1, pages)
                st.rerun()
