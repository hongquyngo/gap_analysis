# utils/net_gap/customer_dialog.py

"""
Customer Affected Dialog for GAP Analysis System - Version 2.2 FIXED
- Fixed terminology: consistent use of "affected" instead of "impacted"
- Fixed customer count mismatch between KPI and dialog
- Uses same filtered demand data as calculator
- Consistent deduplication logic
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


class CustomerAffectedDialog:
    """Manages customer affected dialog display with consistent data handling"""
    
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
    
    def show_dialog(self, gap_df: pd.DataFrame, demand_df: Optional[pd.DataFrame] = None) -> None:
        """
        Open the customer affected dialog
        
        Args:
            gap_df: GAP analysis results
            demand_df: Original demand data (optional - will use calculator's filtered data)
        """
        # Extract only shortage product IDs
        shortage_df = gap_df[gap_df['net_gap'] < 0].copy()
        
        if shortage_df.empty:
            st.warning("No shortage items found")
            return
        
        if 'product_id' not in shortage_df.columns:
            st.error("Product-level grouping required for customer affected analysis")
            return
        
        shortage_product_ids = shortage_df['product_id'].tolist()
        
        # Pre-calculate summary metrics
        metrics = {
            'total_shortage_value': shortage_df['at_risk_value_usd'].sum(),
            'shortage_count': len(shortage_product_ids),
            'total_demand': shortage_df['total_demand'].sum(),
            'total_shortage_qty': shortage_df[shortage_df['net_gap'] < 0]['net_gap'].abs().sum()
        }
        
        # Open dialog with minimal data
        self.session_manager.open_customer_dialog(shortage_product_ids, metrics)
        
        logger.info(f"Customer dialog opened with {len(shortage_product_ids)} shortage products")
    
    def calculate_customer_affected(
        self, 
        shortage_product_ids: List[int],
        demand_df: Optional[pd.DataFrame] = None,
        gap_df: Optional[pd.DataFrame] = None,
        calculator = None
    ) -> pd.DataFrame:
        """
        Calculate customers affected using consistent logic with main calculation
        FIXED: Use session state as primary data source
        
        Args:
            shortage_product_ids: List of product IDs with shortages
            demand_df: Optional demand data (will prefer session state)
            gap_df: GAP analysis results (for shortage details)
            calculator: Calculator instance (fallback)
            
        Returns:
            DataFrame with customer affected analysis
        """
        try:
            import streamlit as st
            
            # PRIORITY 1: Check session state for filtered demand
            if 'gap_filtered_demand' in st.session_state and st.session_state['gap_filtered_demand'] is not None:
                demand_df = st.session_state['gap_filtered_demand'].copy()
                logger.info(f"Retrieved {len(demand_df)} records from session state 'gap_filtered_demand'")
            
            # PRIORITY 2: Try calculator's filtered data
            elif calculator and hasattr(calculator, 'get_filtered_demand_df'):
                filtered = calculator.get_filtered_demand_df()
                if filtered is not None and not filtered.empty:
                    demand_df = filtered
                    logger.info(f"Retrieved {len(demand_df)} records from calculator")
            
            # PRIORITY 3: Use passed demand_df
            elif demand_df is not None and not demand_df.empty:
                logger.info(f"Using passed demand_df with {len(demand_df)} records")
            
            # EMERGENCY FALLBACK: Reload from database
            else:
                logger.warning("No demand data available from any source, attempting database reload")
                try:
                    from utils.net_gap.data_loader import GAPDataLoader
                    loader = GAPDataLoader()
                    # Get filters from session if available
                    filters = st.session_state.get('last_gap_filters', {})
                    demand_df = loader.load_demand_data(
                        entity_name=filters.get('entity'),
                        date_from=filters.get('date_range', [None])[0],
                        date_to=filters.get('date_range', [None, None])[1],
                        product_ids=tuple(shortage_product_ids) if shortage_product_ids else None
                    )
                    logger.info(f"Emergency reload successful: {len(demand_df)} records")
                except Exception as e:
                    logger.error(f"Emergency reload failed: {e}")
                    return pd.DataFrame()
            
            if demand_df is None or demand_df.empty:
                logger.error("No demand data available after all attempts")
                return pd.DataFrame()
            
            # Log data stats for debugging
            logger.info(f"Demand data stats: {len(demand_df)} total records, "
                    f"{demand_df['product_id'].nunique()} unique products, "
                    f"{demand_df['customer'].nunique()} unique customers")
            
            # Build shortage lookup dictionary
            if gap_df is not None and not gap_df.empty:
                shortage_df = gap_df[gap_df['product_id'].isin(shortage_product_ids)].copy()
                
                # Handle potential duplicates
                if shortage_df['product_id'].duplicated().any():
                    logger.warning(f"Found {shortage_df['product_id'].duplicated().sum()} duplicate product IDs, keeping first")
                    shortage_df = shortage_df.drop_duplicates(subset=['product_id'], keep='first')
                
                shortage_lookup = shortage_df.set_index('product_id').to_dict('index')
            else:
                # Use session state shortage data if available
                if 'gap_shortage_df' in st.session_state:
                    shortage_df = st.session_state['gap_shortage_df']
                    shortage_lookup = shortage_df.set_index('product_id').to_dict('index')
                else:
                    shortage_lookup = {pid: {'net_gap': -1, 'total_demand': 1} for pid in shortage_product_ids}
            
            # Filter demand for shortage products
            affected_demand = demand_df[
                demand_df['product_id'].isin(shortage_product_ids)
            ].copy()
            
            if affected_demand.empty:
                logger.warning(f"No demand found for {len(shortage_product_ids)} shortage products")
                logger.debug(f"Shortage product IDs (first 10): {shortage_product_ids[:10]}")
                logger.debug(f"Available product IDs in demand (first 10): {demand_df['product_id'].unique()[:10].tolist()}")
                return pd.DataFrame()
            
            logger.info(f"Found {len(affected_demand)} demand records for shortage products")
            
            # Apply deduplication for consistency
            dedup_cols = ['customer', 'product_id']
            if 'demand_source' in affected_demand.columns:
                dedup_cols.append('demand_source')
            
            affected_demand_dedup = affected_demand.drop_duplicates(subset=dedup_cols).copy()
            logger.info(f"After deduplication: {len(affected_demand)} -> {len(affected_demand_dedup)} records")
            
            # Calculate metrics
            affected_demand_dedup['product_net_gap'] = affected_demand_dedup['product_id'].map(
                lambda x: shortage_lookup.get(x, {}).get('net_gap', 0)
            )
            affected_demand_dedup['product_total_demand'] = affected_demand_dedup['product_id'].map(
                lambda x: shortage_lookup.get(x, {}).get('total_demand', 1)
            )
            affected_demand_dedup['product_at_risk_value'] = affected_demand_dedup['product_id'].map(
                lambda x: shortage_lookup.get(x, {}).get('at_risk_value_usd', 0)
            )
            affected_demand_dedup['product_coverage'] = affected_demand_dedup['product_id'].map(
                lambda x: shortage_lookup.get(x, {}).get('coverage_ratio', 0)
            )
            
            # Calculate customer's share of shortage
            affected_demand_dedup['demand_share'] = np.where(
                affected_demand_dedup['product_total_demand'] > 0,
                affected_demand_dedup['required_quantity'] / affected_demand_dedup['product_total_demand'],
                0
            )
            
            affected_demand_dedup['customer_shortage'] = (
                abs(affected_demand_dedup['product_net_gap']) * affected_demand_dedup['demand_share']
            )
            
            affected_demand_dedup['customer_at_risk'] = (
                affected_demand_dedup['product_at_risk_value'] * affected_demand_dedup['demand_share']
            )
            
            # Group by customer
            customer_agg = affected_demand_dedup.groupby('customer').agg({
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
            
            # Get customer codes and urgency
            customer_info = affected_demand_dedup.groupby('customer').first()[
                ['customer_code', 'urgency_level']
            ].reset_index()
            
            customer_agg = customer_agg.merge(customer_info, on='customer', how='left')
            
            # Determine overall urgency per customer
            urgency_priority = {'OVERDUE': 0, 'URGENT': 1, 'UPCOMING': 2, 'FUTURE': 3}
            customer_urgency = affected_demand_dedup.groupby('customer')['urgency_level'].apply(
                lambda x: min(x, key=lambda v: urgency_priority.get(v, 999), default='FUTURE')
            ).reset_index()
            customer_urgency.columns = ['customer', 'urgency']
            
            customer_agg = customer_agg.merge(customer_urgency, on='customer', how='left')
            customer_agg.drop('urgency_level', axis=1, inplace=True, errors='ignore')
            
            # Build product details for each customer
            customer_products = []
            for customer_name in customer_agg['customer'].unique():
                cust_demand = affected_demand_dedup[
                    affected_demand_dedup['customer'] == customer_name
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
                        'coverage': row['product_coverage'] * 100,
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
            
            logger.info(f"Successfully calculated affected data for {len(customer_agg)} customers")
            return customer_agg
            
        except Exception as e:
            logger.error(f"Error calculating customer affected data: {e}", exc_info=True)
            st.error(f"Failed to calculate customer affected data: {str(e)}")
            return pd.DataFrame()


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
            logger.info("Customer affected data exported to Excel successfully")
            return output.getvalue()
            
        except Exception as e:
            logger.error(f"Excel export error: {e}", exc_info=True)
            st.error(f"Failed to export: {str(e)}")
            return None

# Updated show_customer_popup function with comprehensive tooltips

@st.dialog("Affected Customer Analysis", width="large")
def show_customer_popup():
    """
    Customer affected popup dialog with calculation tooltips
    Enhanced with formula explanations for all metrics
    """
    session_manager = get_session_manager()
    
    # Check if dialog should be shown
    if not session_manager.show_customer_dialog():
        return
    
    # Validate session state data
    required_keys = ['gap_filtered_demand', 'gap_shortage_products']
    missing_keys = [key for key in required_keys if key not in st.session_state]
    
    if missing_keys:
        st.error(f"Required data not found in session: {', '.join(missing_keys)}")
        st.info("Please recalculate GAP analysis to refresh the data.")
        
        logger.error(f"Missing keys: {missing_keys}")
        logger.debug(f"Available session keys: {list(st.session_state.keys())}")
        
        if st.button("Close", use_container_width=True, type="primary"):
            session_manager.close_customer_dialog()
            st.rerun()
        return
    
    # Get data from session
    shortage_ids, summary_metrics = session_manager.get_dialog_data()
    
    if not shortage_ids:
        shortage_ids = st.session_state.get('gap_shortage_products', [])
        if not shortage_ids:
            st.error("No shortage products found")
            if st.button("Close", use_container_width=True):
                session_manager.close_customer_dialog()
                st.rerun()
            return
    
    # Validate demand data
    demand_df = st.session_state.get('gap_filtered_demand')
    if demand_df is None or demand_df.empty:
        st.error("Demand data is empty or not available")
        if st.button("Close", use_container_width=True):
            session_manager.close_customer_dialog()
            st.rerun()
        return
    
    # Quick validation check
    matching_products = demand_df[demand_df['product_id'].isin(shortage_ids)]
    if matching_products.empty:
        st.warning("No matching products found in demand data")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Shortage Products", len(shortage_ids))
        with col2:
            st.metric("Demand Records", len(demand_df))
        
        st.info(
            "Possible causes:\n"
            "• Filters may have excluded all affected customers\n"
            "• No demand exists for shortage products in selected date range\n"
            "• Data synchronization issue - try recalculating"
        )
        
        if st.button("Close", use_container_width=True):
            session_manager.close_customer_dialog()
            st.rerun()
        return
    
    # Get calculator and formatter from session
    calculator = st.session_state.get('_temp_calculator')
    formatter = st.session_state.get('_temp_formatter')
    gap_df = st.session_state.get('_temp_gap_df')
    
    if calculator is None or formatter is None:
        st.error("Required components not available")
        if st.button("Close", use_container_width=True):
            session_manager.close_customer_dialog()
            st.rerun()
        return
    
    # Initialize dialog instance
    dialog = CustomerAffectedDialog(calculator, formatter)
    
    # Calculate customer affected data
    with st.spinner("Analyzing affected customers..."):
        customer_data = dialog.calculate_customer_affected(
            shortage_ids, 
            demand_df,
            gap_df,
            calculator
        )
    
    if customer_data.empty:
        st.warning("No affected customers found after analysis")
        st.info(
            f"Analysis details:\n"
            f"• Shortage products: {len(shortage_ids)}\n"
            f"• Matching demand records: {len(matching_products)}\n"
            f"• Unique customers in matching records: {matching_products['customer'].nunique()}"
        )
        if st.button("Close", use_container_width=True):
            session_manager.close_customer_dialog()
            st.rerun()
        return
    
    # Header with info icon
    col_title, col_help = st.columns([5, 1])
    with col_title:
        st.markdown("### Customer Affected Analysis")
        st.caption("Customers affected by product shortages")
    with col_help:
        with st.popover("ℹ️ How metrics are calculated", use_container_width=True):
            st.markdown("""
            **Calculation Formulas:**
            
            📊 **Customer Shortage Allocation:**
            ```
            Demand Share = Customer Demand ÷ Total Product Demand
            Customer Shortage = Product Shortage × Demand Share
            ```
            
            💰 **At Risk Value:**
            ```
            At Risk Value = Order Value × (Shortage ÷ Demand)
            ```
            
            📈 **Coverage:**
            ```
            Coverage % = (Available Supply ÷ Total Demand) × 100
            ```
            
            🚨 **Urgency Levels:**
            - **OVERDUE**: Required date has passed
            - **URGENT**: Required within 7 days
            - **UPCOMING**: Required within 30 days
            - **FUTURE**: Required after 30 days
            
            **Allocation Principle:** Fair share - each customer receives shortage proportional to their demand
            """)
    
    # Summary metrics with tooltips (6 columns)
    cols = st.columns(6)
    
    with cols[0]:
        st.metric(
            "Customers", 
            formatter.format_number(len(customer_data)),
            help="Unique customers with demand for shortage products"
        )
    
    with cols[1]:
        st.metric(
            "Products", 
            formatter.format_number(customer_data['product_count'].sum()),
            help="Total unique products affected across all customers"
        )
    
    with cols[2]:
        total_demand = customer_data['total_demand_value'].sum()
        st.metric(
            "Total Demand",
            formatter.format_currency(total_demand, abbreviate=True),
            help="Sum of all order values for affected products"
        )
    
    with cols[3]:
        at_risk_total = customer_data['at_risk_value'].sum()
        st.metric(
            "Value at Risk",
            formatter.format_currency(at_risk_total, abbreviate=True),
            help="Potential revenue loss = Σ(Order Value × Shortage%)"
        )
    
    with cols[4]:
        st.metric(
            "Total Shortage", 
            formatter.format_number(customer_data['total_shortage'].sum()),
            help="Total quantity that cannot be fulfilled"
        )
    
    with cols[5]:
        urgent = len(customer_data[customer_data['urgency'].isin(['OVERDUE', 'URGENT'])])
        if urgent > 0:
            st.metric(
                "Urgent", 
                urgent, 
                delta="Need attention", 
                delta_color="inverse",
                help="Customers with overdue or urgent orders (≤7 days)"
            )
        else:
            st.metric("Urgent", "0", help="No urgent orders")
    
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
        display_customers_with_tooltips(filtered, page_size, formatter, session_manager)
    
    # Footer
    st.divider()
    if st.button("Close", use_container_width=True, type="primary"):
        # Clean up temporary data
        for key in ['_temp_calculator', '_temp_formatter', '_temp_gap_df']:
            if key in st.session_state:
                del st.session_state[key]
        
        session_manager.close_customer_dialog()
        st.rerun()

def display_customers_with_tooltips(
    data: pd.DataFrame, 
    page_size: int, 
    formatter,
    session_manager
) -> None:
    """
    Display customer list with enhanced tooltips for all metrics
    FIXED: No nested expanders
    
    Args:
        data: Customer affected data
        page_size: Number of items per page
        formatter: GAPFormatter instance
        session_manager: SessionStateManager instance
    """
    total = len(data)
    pages = max(1, (total + page_size - 1) // page_size)
    
    # Get and validate current page
    page = session_manager.get_dialog_page()
    session_manager.set_dialog_page(page, pages)
    page = session_manager.get_dialog_page()
    
    start = (page - 1) * page_size
    end = min(start + page_size, total)
    
    # Current page data
    page_data = data.iloc[start:end]
    
    st.caption(f"**Showing {start + 1}-{end} of {total} customers**")
    
    # Display each customer
    for idx, row in page_data.iterrows():
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
            # Metrics row with tooltips
            m_cols = st.columns(5)
            
            with m_cols[0]:
                st.metric(
                    "Required", 
                    formatter.format_number(row['total_required']),
                    help="Total quantity ordered by this customer"
                )
            
            with m_cols[1]:
                shortage_pct = (row['total_shortage'] / row['total_required'] * 100) if row['total_required'] > 0 else 0
                st.metric(
                    "Shortage", 
                    formatter.format_number(row['total_shortage']),
                    help=f"Cannot fulfill: {shortage_pct:.1f}% of total demand\n"
                         f"Formula: Σ(Product Shortage × Demand Share)"
                )
            
            with m_cols[2]:
                st.metric(
                    "Demand Value", 
                    formatter.format_currency(row['total_demand_value']),
                    help="Total value of all orders from this customer"
                )
            
            with m_cols[3]:
                risk_pct = (row['at_risk_value'] / row['total_demand_value'] * 100) if row['total_demand_value'] > 0 else 0
                st.metric(
                    "At Risk", 
                    formatter.format_currency(row['at_risk_value']),
                    help=f"Potential revenue loss: {risk_pct:.1f}% of demand value\n"
                         f"Formula: Σ(Order Value × Shortage/Demand)"
                )
            
            with m_cols[4]:
                urgency_desc = {
                    'OVERDUE': 'Orders are past due date',
                    'URGENT': 'Orders needed within 7 days',
                    'UPCOMING': 'Orders needed within 30 days',
                    'FUTURE': 'Orders needed after 30 days'
                }
                st.metric(
                    "Urgency", 
                    row['urgency'],
                    help=urgency_desc.get(row['urgency'], 'Based on earliest required date')
                )
            
            st.divider()
            
            # Tabs for Product Details and Calculation Details
            tab1, tab2 = st.tabs(["📦 Affected Products", "📊 Calculation Details"])
            
            with tab1:
                # Product table with column headers and tooltips
                st.caption("**Product-level breakdown:**")
                
                # Header with tooltips
                h_cols = st.columns([0.3, 2, 1, 1, 1, 1, 0.8])
                
                with h_cols[0]:
                    st.caption("#")
                with h_cols[1]:
                    st.caption("Product", help="Product code and name")
                with h_cols[2]:
                    st.caption("Required", help="Customer's demand quantity")
                with h_cols[3]:
                    st.caption("Shortage", help="Allocated shortage = Total Shortage × (Customer Demand/Total Demand)")
                with h_cols[4]:
                    st.caption("At Risk", help="Value at risk = Order Value × (Shortage/Demand)")
                with h_cols[5]:
                    st.caption("Coverage", help="Supply availability = (Available Supply/Total Demand) × 100")
                with h_cols[6]:
                    st.caption("Urgency", help="Based on required delivery date")
                
                # Products with detailed tooltips
                products = row['products'][:MAX_PRODUCTS_PER_CUSTOMER]
                for i, prod in enumerate(products, 1):
                    p_cols = st.columns([0.3, 2, 1, 1, 1, 1, 0.8])
                    
                    with p_cols[0]:
                        st.text(str(i))
                    
                    with p_cols[1]:
                        # Product info with hover details
                        st.text(prod['pt_code'])
                        st.caption(prod['product_name'][:30])
                    
                    with p_cols[2]:
                        # Required quantity
                        st.text(formatter.format_number(prod['required_quantity']))
                    
                    with p_cols[3]:
                        # Shortage with percentage
                        shortage_pct = (prod['shortage_quantity'] / prod['required_quantity'] * 100) if prod['required_quantity'] > 0 else 0
                        st.text(formatter.format_number(prod['shortage_quantity']))
                        if shortage_pct > 0:
                            st.caption(f"({shortage_pct:.0f}% short)", help="Percentage of demand that cannot be fulfilled")
                    
                    with p_cols[4]:
                        # At risk value
                        st.text(formatter.format_currency(prod['at_risk_value'], abbreviate=True))
                    
                    with p_cols[5]:
                        # Coverage with color coding
                        cov = prod['coverage']
                        cov_text = f"{cov:.0f}%"
                        cov_help = f"Product has {cov:.0f}% of required supply available"
                        
                        if cov < 50:
                            st.markdown(f"🔴 **{cov_text}**", help=cov_help)
                        elif cov < 80:
                            st.markdown(f"🟡 **{cov_text}**", help=cov_help)
                        else:
                            st.markdown(f"🟢 **{cov_text}**", help=cov_help)
                    
                    with p_cols[6]:
                        # Urgency icon with tooltip
                        urg = prod['urgency']
                        urg_icon = urgency_icons.get(urg, '⚪')
                        urg_help = {
                            'OVERDUE': 'Past due date',
                            'URGENT': 'Due in ≤7 days',
                            'UPCOMING': 'Due in ≤30 days',
                            'FUTURE': 'Due in >30 days'
                        }
                        st.markdown(urg_icon, help=urg_help.get(urg, urg))
                
                if len(row['products']) > MAX_PRODUCTS_PER_CUSTOMER:
                    st.caption(
                        f"... and {len(row['products']) - MAX_PRODUCTS_PER_CUSTOMER} more products"
                    )
            
            with tab2:
                # Calculation details without nested expander
                st.markdown(f"""
                **How these numbers are calculated:**
                
                ### 1️⃣ **Shortage Allocation** 
                This customer demands **{formatter.format_number(row['total_required'])}** units across **{row['product_count']}** products.
                
                Based on **fair-share allocation**, they receive shortage proportional to their demand:
                ```
                Customer Shortage = Product Shortage × (Customer Demand / Total Product Demand)
                ```
                
                ### 2️⃣ **At Risk Value**
                - Total order value: **{formatter.format_currency(row['total_demand_value'])}**
                - Cannot fulfill: **{formatter.format_number(row['total_shortage'])}** units
                - At risk: **{formatter.format_currency(row['at_risk_value'])}** 
                  ({(row['at_risk_value']/row['total_demand_value']*100) if row['total_demand_value'] > 0 else 0:.1f}% of total)
                
                ```
                At Risk = Order Value × (Shortage Quantity / Required Quantity)
                ```
                
                ### 3️⃣ **Coverage Calculation**
                Coverage shows the percentage of demand that can be fulfilled:
                ```
                Coverage % = (Available Supply / Total Demand) × 100
                ```
                
                ### 4️⃣ **Urgency Levels**
                - 🔴 **OVERDUE**: Required date has passed
                - 🟠 **URGENT**: Required within 7 days
                - 🟡 **UPCOMING**: Required within 30 days
                - 🟢 **FUTURE**: Required after 30 days
                
                ### 5️⃣ **Demand Sources**
                Orders from: **{row['sources']}**
                
                ---
                *Note: The allocation principle ensures fairness - each customer receives shortage proportional to their demand share.*
                """)
    
    # Pagination controls
    if pages > 1:
        st.divider()
        pag_cols = st.columns([1, 3, 1])
        
        with pag_cols[0]:
            if st.button("Previous", disabled=(page == 1), use_container_width=True, key="dlg_prev"):
                session_manager.set_dialog_page(page - 1, pages)
                st.rerun()
        
        with pag_cols[1]:
            st.markdown(
                f"<div style='text-align: center; padding: 8px;'>"
                f"Page {page} of {pages}"
                f"</div>",
                unsafe_allow_html=True
            )
        
        with pag_cols[2]:
            if st.button("Next", disabled=(page == pages), use_container_width=True, key="dlg_next"):
                session_manager.set_dialog_page(page + 1, pages)
                st.rerun()