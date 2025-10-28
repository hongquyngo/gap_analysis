# pages/2_📅_Period_GAP_Analysis.py
"""
Period-based Supply-Demand GAP Analysis - Version 3.5
- Analyzes supply-demand gaps by time periods with carry-forward logic
- Enhanced with ETD/ETA selection for OC analysis
- Default to ETA for OC timing analysis
- Quick Add: Bulk PT code import feature for efficient product selection
- Enhanced Export: Metadata sheet and improved period formatting
- New Filters: Net Surplus and Timing Surplus categories
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, Optional, List, Tuple
import sys
from pathlib import Path

# Configure page
st.set_page_config(
    page_title="Period GAP Analysis - SCM",
    page_icon="📅",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import authentication
from utils.auth import AuthManager

# Constants
VERSION = "3.5"  # Enhanced export with metadata and surplus filters
MAX_EXPORT_ROWS = 50000
DATA_LOAD_WARNING_SECONDS = 5
DEFAULT_PERIOD_TYPE = "Weekly"
DEFAULT_TRACK_BACKLOG = True
DEFAULT_OC_DATE_FIELD = "ETA"  # New default: ETA instead of ETD


def initialize_components():
    """Initialize all Period GAP analysis components"""
    from utils.period_gap.data_loader import PeriodGAPDataLoader
    from utils.period_gap.display_components import DisplayComponents
    from utils.period_gap.session_state import initialize_session_state
    
    # Initialize session state
    initialize_session_state()
    
    # Initialize components
    data_loader = PeriodGAPDataLoader()
    display_components = DisplayComponents()
    
    return data_loader, display_components


def handle_error(e: Exception) -> None:
    """Handle errors with appropriate user messages"""
    error_type = type(e).__name__
    error_msg = str(e).lower()
    
    logger.error(f"Error in Period GAP analysis: {e}", exc_info=True)
    
    # Handle specific error types
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


@st.cache_data(ttl=300)
def initialize_filter_data(_data_loader) -> Dict[str, Any]:
    """Pre-load data to populate filter dropdowns with formatted product options"""
    try:
        # Load data from all sources
        demand_df = _data_loader.get_demand_data(
            sources=["OC", "Forecast"],
            include_converted=False,
            oc_date_field="ETA"  # Default to ETA
        )
        supply_df = _data_loader.get_supply_data(
            sources=["Inventory", "Pending CAN", "Pending PO", "Pending WH Transfer"],
            exclude_expired=False
        )
        
        # Extract unique values
        entities = set()
        products = {}  # Use dict to store pt_code: (name, package, brand) mapping
        brands = set()
        
        # Calculate date range
        min_date = datetime.today().date()
        max_date = datetime.today().date()
        
        # Process demand data
        if not demand_df.empty:
            entities.update(demand_df['legal_entity'].dropna().unique())
            brands.update(demand_df['brand'].dropna().unique())
            
            # Get products with complete details
            if 'pt_code' in demand_df.columns:
                for _, row in demand_df.drop_duplicates(subset=['pt_code']).iterrows():
                    pt_code = str(row['pt_code'])
                    if pd.notna(row['pt_code']) and pt_code != 'nan':
                        product_name = str(row.get('product_name', ''))[:30] if pd.notna(row.get('product_name')) else ''
                        package_size = str(row.get('package_size', '')) if pd.notna(row.get('package_size')) else ''
                        brand = str(row.get('brand', '')) if pd.notna(row.get('brand')) else ''
                        
                        # Clean up the values
                        if package_size == 'nan':
                            package_size = ''
                        if brand == 'nan':
                            brand = ''
                        
                        products[pt_code] = (product_name, package_size, brand)
            
            # Update date range from demand - now check both etd and eta
            if 'etd' in demand_df.columns:
                etd_dates = pd.to_datetime(demand_df['etd'], errors='coerce').dropna()
                if len(etd_dates) > 0:
                    min_date = min(min_date, etd_dates.min().date())
                    max_date = max(max_date, etd_dates.max().date())
            
            if 'eta' in demand_df.columns:
                eta_dates = pd.to_datetime(demand_df['eta'], errors='coerce').dropna()
                if len(eta_dates) > 0:
                    min_date = min(min_date, eta_dates.min().date())
                    max_date = max(max_date, eta_dates.max().date())
        
        # Process supply data
        if not supply_df.empty:
            entities.update(supply_df['legal_entity'].dropna().unique())
            brands.update(supply_df['brand'].dropna().unique())
            
            # Get products with complete details (for supply-only products)
            if 'pt_code' in supply_df.columns:
                for _, row in supply_df.drop_duplicates(subset=['pt_code']).iterrows():
                    pt_code = str(row['pt_code'])
                    if pd.notna(row['pt_code']) and pt_code != 'nan' and pt_code not in products:
                        product_name = str(row.get('product_name', ''))[:30] if pd.notna(row.get('product_name')) else ''
                        package_size = str(row.get('package_size', '')) if pd.notna(row.get('package_size')) else ''
                        brand = str(row.get('brand', '')) if pd.notna(row.get('brand')) else ''
                        
                        # Clean up the values
                        if package_size == 'nan':
                            package_size = ''
                        if brand == 'nan':
                            brand = ''
                        
                        products[pt_code] = (product_name, package_size, brand)
            
            # Update date range from supply
            if 'date_ref' in supply_df.columns:
                supply_dates = pd.to_datetime(supply_df['date_ref'], errors='coerce').dropna()
                if len(supply_dates) > 0:
                    min_date = min(min_date, supply_dates.min().date())
                    max_date = max(max_date, supply_dates.max().date())
        
        # Create formatted product options list
        product_options = []
        for pt_code, (name, package, brand) in sorted(products.items()):
            # Format: "PT_CODE | Product Name | Package | Brand"
            parts = [pt_code]
            if name:
                parts.append(name)
            if package:
                parts.append(package)
            if brand:
                parts.append(brand)
            
            formatted_option = " | ".join(parts)
            product_options.append(formatted_option)
        
        return {
            'entities': sorted(list(entities)),
            'products': sorted(list(products.keys())),
            'product_options': product_options,
            'brands': sorted(list(brands)),
            'min_date': min_date,
            'max_date': max_date,
            'demand_df': demand_df,
            'supply_df': supply_df
        }
        
    except Exception as e:
        logger.error(f"Error initializing filter data: {e}")
        today = datetime.today().date()
        return {
            'entities': [],
            'products': [],
            'product_options': [],
            'brands': [],
            'min_date': today,
            'max_date': today,
            'demand_df': pd.DataFrame(),
            'supply_df': pd.DataFrame()
        }


def render_source_selection(filter_data: Dict[str, Any]) -> Dict[str, Any]:
    """Render demand and supply source selection with ETD/ETA option for OC"""
    st.markdown("### 📊 Data Source Selection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📤 Demand Sources")
        
        col1_1, col1_2 = st.columns(2)
        with col1_1:
            # Default to True for OC
            demand_oc = st.checkbox("OC", value=True, key="pgap_demand_oc")
        with col1_2:
            # Default to False for Forecast
            demand_forecast = st.checkbox("Forecast", value=False, key="pgap_demand_forecast")
        
        selected_demand_sources = []
        if demand_oc:
            selected_demand_sources.append("OC")
        if demand_forecast:
            selected_demand_sources.append("Forecast")
        
        # OC Date Field Selection (ETD vs ETA)
        oc_date_field = DEFAULT_OC_DATE_FIELD  # Default to ETA
        if demand_oc:
            st.markdown("##### OC Timing Analysis")
            oc_date_field = st.radio(
                "Analyze OC by:",
                options=["ETA", "ETD"],
                index=0,  # Default to ETA (index 0)
                horizontal=True,
                key="pgap_oc_date_field",
                help="ETA: Estimated Time of Arrival | ETD: Estimated Time of Delivery"
            )
        
        # Forecast conversion option
        include_converted = False
        if demand_forecast:
            include_converted = st.checkbox(
                "Include Converted Forecasts", 
                value=False,
                help="⚠️ May cause double counting if OC is also selected",
                key="pgap_include_converted"
            )
    
    with col2:
        st.markdown("#### 📥 Supply Sources")
        
        col2_1, col2_2 = st.columns(2)
        with col2_1:
            supply_inv = st.checkbox("Inventory", value=True, key="pgap_supply_inv")
            supply_can = st.checkbox("Pending CAN", value=True, key="pgap_supply_can")
        with col2_2:
            supply_po = st.checkbox("Pending PO", value=True, key="pgap_supply_po")
            supply_wht = st.checkbox("Pending WH Transfer", value=True, key="pgap_supply_wht")
        
        exclude_expired = st.checkbox(
            "Exclude Expired", 
            value=True,
            key="pgap_exclude_expired"
        )
        
        selected_supply_sources = []
        if supply_inv:
            selected_supply_sources.append("Inventory")
        if supply_can:
            selected_supply_sources.append("Pending CAN")
        if supply_po:
            selected_supply_sources.append("Pending PO")
        if supply_wht:
            selected_supply_sources.append("Pending WH Transfer")
    
    return {
        "demand": selected_demand_sources,
        "supply": selected_supply_sources,
        "include_converted": include_converted,
        "exclude_expired": exclude_expired,
        "oc_date_field": oc_date_field  # New field for ETD/ETA selection
    }


def render_filters(filter_data: Dict[str, Any]) -> Dict[str, Any]:
    """Render standard filters for GAP analysis with exclude options"""
    with st.expander("🔍 Filters", expanded=True):
        filters = {}
        
        # Main Filters - All on one row with proportional widths
        st.markdown("#### Main Filters")
        
        # Create columns with proportions: Legal Entity (3), Product (5), Brand (2)
        # Total = 10 parts, plus small columns for exclude checkboxes
        filter_cols = st.columns([3, 0.5, 5, 0.5, 2, 0.5])
        
        # Legal Entity filter (3 parts + 0.5 for checkbox)
        with filter_cols[0]:
            all_entities = filter_data.get('entities', [])
            filters['entity'] = st.multiselect(
                "Legal Entity",
                all_entities,
                key="pgap_entity_filter",
                placeholder="All entities" if all_entities else "No entities available"
            )
        with filter_cols[1]:
            filters['exclude_entity'] = st.checkbox(
                "🚫",
                value=False,
                key="pgap_exclude_entity",
                help="Exclude selected legal entities"
            )
        
        # Product filter with Quick Add (5 parts + 0.5 for checkbox)
        with filter_cols[2]:
            # Use the formatted product options
            product_options = filter_data.get('product_options', [])
            
            # Session key for storing selection (separate from widget key)
            session_key = 'pgap_product_selection'
            
            # Handle Quick Add confirmation BEFORE creating widget
            if 'pgap_quick_add_confirmed' in st.session_state:
                new_products = st.session_state.pgap_quick_add_confirmed
                if new_products and isinstance(new_products, list):
                    # Get current selection
                    current_selection = st.session_state.get(session_key, [])
                    # Merge: add new products to existing selection
                    merged_selection = list(set(current_selection + new_products))
                    # Filter to only valid products
                    valid_selected = [p for p in merged_selection if p in product_options]
                    st.session_state[session_key] = valid_selected
                
                # Clear the confirmation flag
                del st.session_state.pgap_quick_add_confirmed
                
                # Increment widget counter to force re-render
                if 'pgap_product_widget_counter' not in st.session_state:
                    st.session_state['pgap_product_widget_counter'] = 0
                st.session_state['pgap_product_widget_counter'] += 1
            
            # Get default selection
            default_selection = st.session_state.get(session_key, [])
            # Filter to only valid products
            default_selection = [p for p in default_selection if p in product_options]
            
            # Clear cancelled flag
            if st.session_state.get('pgap_quick_add_cancelled'):
                del st.session_state.pgap_quick_add_cancelled
            
            # Create sub-columns for multiselect + Quick Add button
            sub_col1, sub_col2 = st.columns([4.5, 1.5])
            
            with sub_col1:
                # Use dynamic key based on counter
                widget_counter = st.session_state.get('pgap_product_widget_counter', 0)
                widget_key = f"pgap_product_filter_{widget_counter}"
                
                selected_products = st.multiselect(
                    "Product",
                    product_options,
                    default=default_selection,
                    key=widget_key,
                    placeholder="All products" if product_options else "No products available"
                )
                
                # Update session state with current selection
                st.session_state[session_key] = selected_products
            
            with sub_col2:
                # Quick Add button
                if st.button("📋 Quick Add", key="pgap_quick_add_btn", use_container_width=True,
                           help="Bulk import PT codes"):
                    st.session_state.pgap_show_quick_add = True
            
            # Show Quick Add dialog if triggered
            if st.session_state.get('pgap_show_quick_add'):
                from utils.period_gap.quick_add_components import show_quick_add_dialog_for_products
                show_quick_add_dialog_for_products(product_options, selected_products, False)
            
            # Extract PT codes from formatted selections
            filters['product'] = []
            for selection in selected_products:
                # Extract PT code (everything before first |)
                if '|' in selection:
                    pt_code = selection.split(' | ')[0].strip()
                else:
                    pt_code = selection.strip()
                filters['product'].append(pt_code)

        
        with filter_cols[3]:
            filters['exclude_product'] = st.checkbox(
                "🚫",
                value=False,
                key="pgap_exclude_product",
                help="Exclude selected products"
            )

        
        # Brand filter (2 parts + 0.5 for checkbox)
        with filter_cols[4]:
            all_brands = filter_data.get('brands', [])
            filters['brand'] = st.multiselect(
                "Brand",
                all_brands,
                key="pgap_brand_filter",
                placeholder="All brands" if all_brands else "No brands available"
            )
        
        with filter_cols[5]:
            filters['exclude_brand'] = st.checkbox(
                "🚫",
                value=False,
                key="pgap_exclude_brand",
                help="Exclude selected brands"
            )
        
        # Date range with proper defaults
        st.markdown("#### 📅 Date Range")
        col_date1, col_date2 = st.columns(2)
        
        # Use date range from filter data
        min_date = filter_data.get('min_date', datetime.today().date())
        max_date = filter_data.get('max_date', datetime.today().date())
        
        with col_date1:
            filters['start_date'] = st.date_input(
                "From Date",
                value=min_date,
                min_value=min_date - timedelta(days=365),
                max_value=max_date + timedelta(days=365),
                key="pgap_start_date"
            )
        
        with col_date2:
            filters['end_date'] = st.date_input(
                "To Date",
                value=max_date,
                min_value=min_date - timedelta(days=365),
                max_value=max_date + timedelta(days=365),
                key="pgap_end_date"
            )
        
        # Show active filters summary
        active_filters = sum(1 for k, v in filters.items() 
                           if k not in ['start_date', 'end_date'] 
                           and not k.startswith('exclude_')
                           and v and v != [])
        excluded_filters = sum(1 for k, v in filters.items()
                             if k.startswith('exclude_') and v)
        
        if active_filters > 0 or excluded_filters > 0:
            status_text = []
            if active_filters > 0:
                status_text.append(f"🔍 {active_filters} filters active")
            if excluded_filters > 0:
                status_text.append(f"🚫 {excluded_filters} exclusions active")
            st.success(" | ".join(status_text))
    
    return filters


def render_calculation_options() -> Dict[str, Any]:
    """Render GAP calculation options"""
    st.markdown("### ⚙️ Calculation Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        period_type = st.selectbox(
            "Group By Period",
            ["Daily", "Weekly", "Monthly"],
            index=1,
            key="pgap_period_select"
        )
    
    with col2:
        exclude_missing_dates = st.checkbox(
            "📅 Exclude missing dates",
            value=True,
            key="pgap_exclude_missing"
        )
    
    with col3:
        track_backlog = st.checkbox(
            "📊 Track Backlog",
            value=DEFAULT_TRACK_BACKLOG,
            key="pgap_track_backlog",
            help="Track negative carry forward (backlog) from shortage periods"
        )
    
    return {
        "period_type": period_type,
        "exclude_missing_dates": exclude_missing_dates,
        "track_backlog": track_backlog
    }


def apply_filters_to_data(
    df_demand: pd.DataFrame,
    df_supply: pd.DataFrame,
    filters: Dict[str, Any],
    oc_date_field: str = "ETA"  # New parameter
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply filters to demand and supply dataframes with exclude logic"""
    
    filtered_demand = df_demand.copy()
    filtered_supply = df_supply.copy()
    
    # Clean product codes
    if 'pt_code' in filtered_demand.columns:
        filtered_demand['pt_code'] = filtered_demand['pt_code'].astype(str).str.strip()
    
    if 'pt_code' in filtered_supply.columns:
        filtered_supply['pt_code'] = filtered_supply['pt_code'].astype(str).str.strip()
    
    # Apply filters to DEMAND with exclude logic
    
    # Legal Entity filter
    if filters.get('entity'):
        if filters.get('exclude_entity', False):
            # Exclude selected entities
            filtered_demand = filtered_demand[~filtered_demand['legal_entity'].isin(filters['entity'])]
        else:
            # Include only selected entities
            filtered_demand = filtered_demand[filtered_demand['legal_entity'].isin(filters['entity'])]
    
    # Product filter
    if filters.get('product'):
        clean_products = [str(p).strip() for p in filters['product']]
        if filters.get('exclude_product', False):
            # Exclude selected products
            filtered_demand = filtered_demand[~filtered_demand['pt_code'].isin(clean_products)]
        else:
            # Include only selected products
            filtered_demand = filtered_demand[filtered_demand['pt_code'].isin(clean_products)]
    
    # Brand filter - Clean brand values for better matching
    if filters.get('brand'):
        # Clean brand values in both filter and dataframe for proper matching
        clean_brands = [str(b).strip().lower() for b in filters['brand']]
        
        # Create a temporary column with cleaned brand values for comparison
        filtered_demand['_brand_clean'] = filtered_demand['brand'].astype(str).str.strip().str.lower()
        
        if filters.get('exclude_brand', False):
            # Exclude selected brands
            filtered_demand = filtered_demand[~filtered_demand['_brand_clean'].isin(clean_brands)]
        else:
            # Include only selected brands
            filtered_demand = filtered_demand[filtered_demand['_brand_clean'].isin(clean_brands)]
        
        # Remove temporary column
        filtered_demand = filtered_demand.drop(columns=['_brand_clean'])
    
    # Apply date filters to demand - now use unified demand_date field
    if 'demand_date' in filtered_demand.columns and filters.get('start_date') and filters.get('end_date'):
        start_date = pd.to_datetime(filters['start_date'])
        end_date = pd.to_datetime(filters['end_date'])
        
        filtered_demand['demand_date'] = pd.to_datetime(filtered_demand['demand_date'], errors='coerce')
        
        date_mask = (
            filtered_demand['demand_date'].isna() |
            ((filtered_demand['demand_date'] >= start_date) & (filtered_demand['demand_date'] <= end_date))
        )
        filtered_demand = filtered_demand[date_mask]
    
    # Apply filters to SUPPLY with same exclude logic
    
    # Legal Entity filter
    if filters.get('entity'):
        if filters.get('exclude_entity', False):
            # Exclude selected entities
            filtered_supply = filtered_supply[~filtered_supply['legal_entity'].isin(filters['entity'])]
        else:
            # Include only selected entities
            filtered_supply = filtered_supply[filtered_supply['legal_entity'].isin(filters['entity'])]
    
    # Product filter
    if filters.get('product'):
        clean_products = [str(p).strip() for p in filters['product']]
        if filters.get('exclude_product', False):
            # Exclude selected products
            filtered_supply = filtered_supply[~filtered_supply['pt_code'].isin(clean_products)]
        else:
            # Include only selected products
            filtered_supply = filtered_supply[filtered_supply['pt_code'].isin(clean_products)]
    
    # Brand filter - Clean brand values for better matching
    if filters.get('brand'):
        # Clean brand values in both filter and dataframe for proper matching
        clean_brands = [str(b).strip().lower() for b in filters['brand']]
        
        # Create a temporary column with cleaned brand values for comparison
        filtered_supply['_brand_clean'] = filtered_supply['brand'].astype(str).str.strip().str.lower()
        
        if filters.get('exclude_brand', False):
            # Exclude selected brands
            filtered_supply = filtered_supply[~filtered_supply['_brand_clean'].isin(clean_brands)]
        else:
            # Include only selected brands
            filtered_supply = filtered_supply[filtered_supply['_brand_clean'].isin(clean_brands)]
        
        # Remove temporary column
        filtered_supply = filtered_supply.drop(columns=['_brand_clean'])
    
    # Apply date filters to supply
    if 'date_ref' in filtered_supply.columns and filters.get('start_date') and filters.get('end_date'):
        start_date = pd.to_datetime(filters['start_date'])
        end_date = pd.to_datetime(filters['end_date'])
        
        filtered_supply['date_ref'] = pd.to_datetime(filtered_supply['date_ref'], errors='coerce')
        
        date_mask = (
            filtered_supply['date_ref'].isna() |
            ((filtered_supply['date_ref'] >= start_date) & (filtered_supply['date_ref'] <= end_date))
        )
        filtered_supply = filtered_supply[date_mask]
    
    # Debug: Show filter status in sidebar
    with st.sidebar:
        st.markdown("### 🔍 Active Filters")
        if filters.get('brand'):
            mode = "Excluding" if filters.get('exclude_brand') else "Including"
            st.info(f"Brand: {mode} {', '.join(filters['brand'])}")
        if filters.get('entity'):
            mode = "Excluding" if filters.get('exclude_entity') else "Including"
            st.info(f"Entity: {mode} {', '.join(filters['entity'])}")
        if filters.get('product'):
            mode = "Excluding" if filters.get('exclude_product') else "Including"
            st.info(f"Products: {mode} {len(filters['product'])} items")
        st.info(f"OC Analysis by: {oc_date_field}")
    
    return filtered_demand, filtered_supply


def render_display_filters(calc_options: Dict[str, Any]) -> Dict[str, Any]:
    """Render display filters for GAP results with improved shortage and surplus categorization"""
    st.markdown("### 🔍 Display Filters")
    st.caption("Filter the calculated results. Changes apply immediately.")
    
    # Product Type Filters
    st.markdown("#### Product Types")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_matched = st.checkbox(
            "🔗 Matched Products",
            value=True,
            key="pgap_show_matched"
        )
    
    with col2:
        show_demand_only = st.checkbox(
            "📤 Demand Only",
            value=True,
            key="pgap_show_demand_only"
        )
    
    with col3:
        show_supply_only = st.checkbox(
            "📥 Supply Only",
            value=True,
            key="pgap_show_supply_only"
        )
    
    # Period/Status Filters - UPDATED OPTIONS
    st.markdown("#### Period & Status Filters")
    
    # Add help text explaining the difference
    with st.expander("ℹ️ Filter Definitions", expanded=False):
        st.markdown("""
        **Main Categories (Mutually Exclusive):**
        - **All**: Show all products and periods
        - **Net Shortage**: Total supply < total demand (need new orders)
        - **Net Surplus**: Total supply > total demand (excess inventory)
        
        **Timing Filters (Cross-cutting):**
        - **Timing Shortage**: Products with shortage periods (need expedite/reschedule)
        - **Timing Surplus**: Products with surplus periods (optimize schedule)
        """)
    
    period_filter = st.radio(
        "Show:",
        options=["All", "Net Shortage", "Timing Shortage", "Net Surplus", "Timing Surplus"],
        horizontal=True,
        key="pgap_period_filter"
    )
    
    # View Options
    col4, col5 = st.columns(2)
    with col4:
        enable_row_highlighting = st.checkbox(
            "🎨 Enable Row Highlighting",
            value=True,
            key="pgap_enable_highlighting"
        )
    
    with col5:
        show_only_nonzero = st.checkbox(
            "🚫 Hide Balanced Products",
            value=False,
            key="pgap_show_nonzero",
            help="Hide products with GAP = 0 (only show shortage/surplus)"
        )
    
    return {
        "show_matched": show_matched,
        "show_demand_only": show_demand_only,
        "show_supply_only": show_supply_only,
        "period_filter": period_filter,
        "enable_row_highlighting": enable_row_highlighting,
        "show_only_nonzero": show_only_nonzero,
        "period_type": calc_options["period_type"],
        "track_backlog": calc_options["track_backlog"]
    }


def apply_display_filters(
    gap_df: pd.DataFrame,
    display_filters: Dict[str, Any],
    df_demand_filtered: pd.DataFrame,
    df_supply_filtered: pd.DataFrame,
    stored_calc_options: Dict[str, Any]
) -> pd.DataFrame:
    """Apply display filters to GAP results with new categorization logic"""
    from utils.period_gap.period_helpers import parse_week_period, parse_month_period
    from utils.period_gap.shortage_analyzer import categorize_main_category, categorize_timing_issues
    
    gap_df_filtered = gap_df.copy()
    
    # Filter by product type
    if not (display_filters['show_matched'] and display_filters['show_demand_only'] and display_filters['show_supply_only']):
        demand_products = set(df_demand_filtered['pt_code'].unique()) if not df_demand_filtered.empty else set()
        supply_products = set(df_supply_filtered['pt_code'].unique()) if not df_supply_filtered.empty else set()
        
        products_to_show = set()
        if display_filters['show_matched']:
            products_to_show.update(demand_products & supply_products)
        if display_filters['show_demand_only']:
            products_to_show.update(demand_products - supply_products)
        if display_filters['show_supply_only']:
            products_to_show.update(supply_products - demand_products)
        
        if products_to_show:
            gap_df_filtered = gap_df_filtered[gap_df_filtered['pt_code'].isin(products_to_show)]
    
    # Filter by period/status - NEW LOGIC with mutually exclusive main categories
    period_filter = display_filters['period_filter']
    period_type = stored_calc_options['period_type']
    
    if period_filter != "All":
        # Get categorizations
        main_cats = categorize_main_category(gap_df_filtered)
        timing_cats = categorize_timing_issues(gap_df_filtered)
        
        if period_filter == "Net Shortage":
            # Main category: Products with net shortage
            gap_df_filtered = gap_df_filtered[gap_df_filtered['pt_code'].isin(main_cats['net_shortage'])]
            
        elif period_filter == "Timing Shortage":
            # Cross-cutting: Products with shortage periods (any main category)
            gap_df_filtered = gap_df_filtered[gap_df_filtered['pt_code'].isin(timing_cats['timing_shortage'])]
        
        elif period_filter == "Net Surplus":
            # Main category: Products with net surplus
            gap_df_filtered = gap_df_filtered[gap_df_filtered['pt_code'].isin(main_cats['net_surplus'])]
        
        elif period_filter == "Timing Surplus":
            # Cross-cutting: Products with surplus periods (any main category)
            gap_df_filtered = gap_df_filtered[gap_df_filtered['pt_code'].isin(timing_cats['timing_surplus'])]
    
    # RE-SORT after filtering to ensure proper order
    if not gap_df_filtered.empty:
        if period_type == "Weekly":
            gap_df_filtered['_sort_product'] = gap_df_filtered['pt_code']
            gap_df_filtered['_sort_period'] = gap_df_filtered['period'].apply(parse_week_period)
        elif period_type == "Monthly":
            gap_df_filtered['_sort_product'] = gap_df_filtered['pt_code']
            gap_df_filtered['_sort_period'] = gap_df_filtered['period'].apply(parse_month_period)
        else:
            gap_df_filtered['_sort_product'] = gap_df_filtered['pt_code']
            gap_df_filtered['_sort_period'] = pd.to_datetime(gap_df_filtered['period'], errors='coerce')
        
        gap_df_filtered = gap_df_filtered.sort_values(['_sort_product', '_sort_period'])
        gap_df_filtered = gap_df_filtered.drop(columns=['_sort_product', '_sort_period'])
        gap_df_filtered = gap_df_filtered.reset_index(drop=True)
    
    return gap_df_filtered


def export_to_excel(
    gap_df: pd.DataFrame, 
    filter_values: Dict[str, Any],
    display_filters: Dict[str, Any],
    calc_options: Dict[str, Any],
    df_demand_filtered: pd.DataFrame,
    df_supply_filtered: pd.DataFrame
) -> bytes:
    """
    Enhanced export with metadata sheet and improved period formatting
    
    Args:
        gap_df: GAP analysis dataframe to export
        filter_values: Data filters that were applied
        display_filters: Display filters that were applied
        calc_options: Calculation options used
        df_demand_filtered: Filtered demand data
        df_supply_filtered: Filtered supply data
    
    Returns:
        Excel file bytes with metadata and formatted data
    """
    from utils.period_gap.helpers import export_gap_with_metadata
    
    return export_gap_with_metadata(
        gap_df=gap_df,
        filter_values=filter_values,
        display_filters=display_filters,
        calc_options=calc_options,
        df_demand_filtered=df_demand_filtered,
        df_supply_filtered=df_supply_filtered
    )


def main():
    """Main application logic for Period GAP Analysis"""
    # Initialize authentication
    auth_manager = AuthManager()
    
    # Check authentication
    if not auth_manager.check_session():
        st.warning("⚠️ Please login to access this page")
        st.stop()
    
    # Initialize components
    data_loader, display_components = initialize_components()
    
    # Import additional modules
    from utils.period_gap.gap_calculator import calculate_gap_with_carry_forward
    from utils.period_gap.gap_display import (
        show_gap_summary,
        show_gap_detail_table,
        show_gap_pivot_view
    )
    from utils.period_gap.helpers import (
        convert_df_to_excel,
        save_to_session_state
    )
    from utils.period_gap.session_state import (
        save_period_gap_state,
        get_period_gap_state,
        update_filter_cache
    )
    
    # Page Header (without dashboard button)
    display_components.show_page_header(
        title="Period-Based GAP Analysis",
        icon="📅",
        prev_page=None,  # Set to None to avoid navigation errors
        next_page=None,
        show_dashboard_button=False  # Disable dashboard button
    )
    
    # User info in sidebar
    st.sidebar.markdown(f"👤 **User:** {auth_manager.get_user_display_name()}")
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        auth_manager.logout()
        st.rerun()
    
    st.markdown("---")
    
    try:
        # Load filter data
        with st.spinner("Initializing filters..."):
            filter_data = initialize_filter_data(data_loader)
            
            # Update filter cache for use by filter functions
            update_filter_cache(
                entities=filter_data['entities'],
                products=filter_data['products'],
                brands=filter_data['brands']
            )
        
        # Source Selection
        selected_sources = render_source_selection(filter_data)
        
        # Show warning if no sources selected
        if not selected_sources['demand'] or not selected_sources['supply']:
            st.warning("⚠️ Please select at least one demand source and one supply source.")
            return
        
        # Filters
        filters = render_filters(filter_data)
        
        # Calculation Options
        calc_options = render_calculation_options()
        
        # Calculate button
        st.markdown("---")
        col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 2])
        with col_btn2:
            calculate_button = st.button(
                "🔄 Calculate GAP",
                type="primary",
                use_container_width=True
            )
        
        # Load and process data when button clicked
        if calculate_button:
            with st.spinner("Loading data..."):
                # Load filtered data based on selections
                df_demand_all = data_loader.get_demand_data(
                    sources=selected_sources['demand'],
                    include_converted=selected_sources['include_converted'],
                    oc_date_field=selected_sources.get("oc_date_field", "ETA")
                )
                
                df_supply_all = data_loader.get_supply_data(
                    sources=selected_sources['supply'],
                    exclude_expired=selected_sources['exclude_expired']
                )
                
                # Apply data filters
                df_demand_filtered, df_supply_filtered = apply_filters_to_data(
                    df_demand_all,
                    df_supply_all,
                    filters,
                    selected_sources.get("oc_date_field", "ETA")
                )
                
                # Apply date exclusion if requested - now use demand_date for demand
                if calc_options.get("exclude_missing_dates", True):
                    if not df_demand_filtered.empty and 'demand_date' in df_demand_filtered.columns:
                        df_demand_filtered = df_demand_filtered[df_demand_filtered['demand_date'].notna()]
                    
                    if not df_supply_filtered.empty and 'date_ref' in df_supply_filtered.columns:
                        df_supply_filtered = df_supply_filtered[df_supply_filtered['date_ref'].notna()]
                
                # Save to session for cross-page access
                save_period_gap_state({
                    'demand': df_demand_filtered,
                    'supply': df_supply_filtered,
                    'calc_options': calc_options,
                    'display_filters': None,
                    'oc_date_field': selected_sources.get("oc_date_field", "ETA")  # Save the selection
                })
                
                # Also save for other pages
                save_to_session_state('gap_analysis_result', None)  # Will be set after GAP calculation
                save_to_session_state('demand_filtered', df_demand_filtered)
                save_to_session_state('supply_filtered', df_supply_filtered)
        
        # Display Results if available
        if get_period_gap_state():
            # Get display filters
            display_filters = render_display_filters(calc_options)
            
            st.markdown("---")
            
            # Get data from state
            state = get_period_gap_state()
            df_demand_filtered = state['demand']
            df_supply_filtered = state['supply']
            stored_calc_options = state['calc_options']
            stored_oc_date_field = state.get('oc_date_field', 'ETA')
            
            # Display which date field is being used
            if 'OC' in selected_sources.get("demand", []):
                st.info(f"📊 OC Analysis using: **{stored_oc_date_field}** (Estimated Time of {'Arrival' if stored_oc_date_field == 'ETA' else 'Delivery'})")
            
            # Create cache key that includes filters to detect changes
            import hashlib
            import json
            
            # Create a unique cache key based on all parameters
            cache_params = {
                'period_type': stored_calc_options['period_type'],
                'track_backlog': stored_calc_options['track_backlog'],
                'oc_date_field': stored_oc_date_field,  # Include in cache key
                'demand_count': len(df_demand_filtered),
                'supply_count': len(df_supply_filtered),
                # Add a hash of the actual data to detect filter changes
                'demand_hash': hashlib.md5(pd.util.hash_pandas_object(df_demand_filtered).values).hexdigest() if not df_demand_filtered.empty else 'empty',
                'supply_hash': hashlib.md5(pd.util.hash_pandas_object(df_supply_filtered).values).hexdigest() if not df_supply_filtered.empty else 'empty'
            }
            
            cache_key = json.dumps(cache_params, sort_keys=True)
            
            if 'pgap_result_cache_key' not in st.session_state or st.session_state['pgap_result_cache_key'] != cache_key:
                with st.spinner("Calculating supply-demand gaps..."):
                    gap_df = calculate_gap_with_carry_forward(
                        df_demand_filtered,
                        df_supply_filtered,
                        stored_calc_options['period_type'],
                        stored_calc_options['track_backlog']
                    )
                    
                    st.session_state['pgap_gap_df'] = gap_df
                    st.session_state['pgap_result_cache_key'] = cache_key
                    
                    # Save for other pages
                    save_to_session_state('gap_analysis_result', gap_df)
                    save_to_session_state('last_gap_analysis', gap_df)
                    save_to_session_state('last_analysis_time', datetime.now().strftime('%Y-%m-%d %H:%M'))
            else:
                gap_df = st.session_state['pgap_gap_df']
            
            if not gap_df.empty:
                # Apply display filters
                gap_df_filtered = apply_display_filters(
                    gap_df, 
                    display_filters,
                    df_demand_filtered,
                    df_supply_filtered,
                    stored_calc_options
                )
                
                if gap_df_filtered.empty:
                    st.warning("No products match the selected display filters.")
                else:
                    # Display results
                    show_gap_summary(
                        gap_df_filtered,
                        display_filters,
                        df_demand_filtered,
                        df_supply_filtered
                    )
                    
                    show_gap_detail_table(
                        gap_df_filtered,
                        display_filters,
                        df_demand_filtered,
                        df_supply_filtered
                    )
                    
                    show_gap_pivot_view(gap_df_filtered, display_filters)
                    
                    # Export section - ENHANCED with dynamic buttons
                    st.markdown("---")
                    st.markdown("### 📤 Export Options")
                    
                    # Define export configurations based on filter type
                    export_configs = {
                        "All": {
                            "label": "📊 Export Full Report",
                            "prefix": "full_report",
                            "description": "Export all data with metadata"
                        },
                        "Net Shortage": {
                            "label": "📦 Export Net Shortage",
                            "prefix": "net_shortage",
                            "description": "Products needing new orders"
                        },
                        "Timing Shortage": {
                            "label": "⏱️ Export Timing Shortage",
                            "prefix": "timing_shortage",
                            "description": "Products with shortage periods (expedite/reschedule)"
                        },
                        "Net Surplus": {
                            "label": "📈 Export Net Surplus",
                            "prefix": "net_surplus",
                            "description": "Products with excess inventory"
                        },
                        "Timing Surplus": {
                            "label": "⏰ Export Timing Surplus",
                            "prefix": "timing_surplus",
                            "description": "Products with surplus periods"
                        }
                    }
                    
                    # Get current filter selection
                    current_filter = display_filters.get('period_filter', 'All')
                    
                    # Create export buttons
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # Always show full report export
                        excel_data = export_to_excel(
                            gap_df_filtered,
                            filters,
                            display_filters,
                            stored_calc_options,
                            df_demand_filtered,
                            df_supply_filtered
                        )
                        st.download_button(
                            export_configs["All"]["label"],
                            data=excel_data,
                            file_name=f"{export_configs['All']['prefix']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            help=export_configs["All"]["description"]
                        )
                    
                    with col2:
                        # Show filtered export if not "All"
                        if current_filter != "All" and current_filter in export_configs:
                            config = export_configs[current_filter]
                            
                            # Data is already filtered
                            filtered_excel = export_to_excel(
                                gap_df_filtered,
                                filters,
                                display_filters,
                                stored_calc_options,
                                df_demand_filtered,
                                df_supply_filtered
                            )
                            
                            st.download_button(
                                config["label"],
                                data=filtered_excel,
                                file_name=f"{config['prefix']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                help=config["description"]
                            )
                    
                    # Show info about export format
                    st.info("📋 **Export includes:** Export_Info (metadata) | GAP_Analysis (detailed data) | Product_Summary (overview)")
                    
            else:
                st.warning("No data available for the selected filters and sources.")
    
    except Exception as e:
        handle_error(e)
    
    # Footer
    st.markdown("---")
    st.caption(f"Period GAP Analysis v{VERSION} | Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()