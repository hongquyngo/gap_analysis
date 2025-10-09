# pages/2_📅_Period_GAP_Analysis.py
"""
Period-based Supply-Demand GAP Analysis
Analyzes supply-demand gaps by time periods with carry-forward logic
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.auth import AuthManager

# Authentication check
auth_manager = AuthManager()
if not auth_manager.check_session():
    st.switch_page("pages/0_🔑_Login.py")
    st.stop()

import pandas as pd
from datetime import datetime, timedelta
import logging

# Import Period GAP modules
from utils.period_gap.data_loader import PeriodGAPDataLoader
from utils.period_gap.gap_calculator import calculate_gap_with_carry_forward
from utils.period_gap.gap_display import (
    show_gap_summary,
    show_gap_detail_table,
    show_gap_pivot_view
)
from utils.period_gap.period_helpers import (
    is_past_period,
    prepare_gap_detail_display,
    format_gap_display_df
)
from utils.period_gap.formatters import (
    format_number,
    format_currency,
    format_percentage,
    check_missing_dates
)
from utils.period_gap.helpers import (
    convert_df_to_excel,
    export_multiple_sheets,
    save_to_session_state,
    create_period_pivot,
    apply_period_indicators
)
from utils.period_gap.display_components import DisplayComponents
from utils.period_gap.session_state import (
    initialize_session_state,
    save_period_gap_state,
    get_period_gap_state,
    clear_period_gap_cache,
    update_filter_cache,
    get_filter_cache
)

logger = logging.getLogger(__name__)

# === Page Config ===
st.set_page_config(
    page_title="Period GAP Analysis - SCM",
    page_icon="📅",
    layout="wide"
)

# === Initialize ===
initialize_session_state()

# === Debug Mode ===
col_debug1, col_debug2 = st.columns([6, 1])
with col_debug2:
    debug_mode = st.checkbox("🛠 Debug Mode", value=False, key="period_gap_debug")

if debug_mode:
    st.info("🛠 Debug Mode is ON")

# === Header ===
DisplayComponents.show_page_header(
    title="Period-Based GAP Analysis",
    icon="📅",
    prev_page="pages/1_📊_Net_GAP.py",
    next_page=None
)

st.markdown("---")

# === Initialize Data Loader ===
@st.cache_resource
def get_data_loader():
    return PeriodGAPDataLoader()

data_loader = get_data_loader()

# === CRITICAL FIX: Pre-load Data for Filters ===
@st.cache_data(ttl=300)  # Cache for 5 minutes
def initialize_filter_data():
    """Pre-load data to populate filter dropdowns with formatted product options"""
    try:
        # Load data from all sources
        demand_df = data_loader.get_demand_data(
            sources=["OC", "Forecast"],
            include_converted=False
        )
        supply_df = data_loader.get_supply_data(
            sources=["Inventory", "Pending CAN", "Pending PO", "Pending WH Transfer"],
            exclude_expired=False
        )
        
        # Extract unique values
        entities = set()
        products = {}  # Use dict to store pt_code: (name, package, brand) mapping
        brands = set()
        customers = set()
        
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
            
            # Get customers
            if 'customer' in demand_df.columns:
                customers.update(demand_df['customer'].dropna().unique())
            
            # Update date range from demand
            if 'etd' in demand_df.columns:
                etd_dates = pd.to_datetime(demand_df['etd'], errors='coerce').dropna()
                if len(etd_dates) > 0:
                    min_date = min(min_date, etd_dates.min().date())
                    max_date = max(max_date, etd_dates.max().date())
        
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
            # Format: PT_CODE | Name | Package (Brand)
            if package and brand:
                option = f"{pt_code} | {name} | {package} ({brand})"
            elif package:
                option = f"{pt_code} | {name} | {package}"
            elif brand:
                option = f"{pt_code} | {name} ({brand})"
            else:
                option = f"{pt_code} | {name}" if name else pt_code
            
            product_options.append(option)
        
        return {
            'entities': sorted(list(entities)),
            'products': sorted(list(products.keys())),
            'product_options': sorted(product_options),
            'brands': sorted(list(brands)),
            'customers': sorted(list(customers)),
            'min_date': min_date,
            'max_date': max_date,
            'demand_df': demand_df,  # Store for immediate use
            'supply_df': supply_df   # Store for immediate use
        }
    except Exception as e:
        logger.error(f"Error initializing filter data: {e}")
        # Return empty defaults with today's date
        today = datetime.today().date()
        return {
            'entities': [],
            'products': [],
            'product_options': [],
            'brands': [],
            'customers': [],
            'min_date': today,
            'max_date': today,
            'demand_df': pd.DataFrame(),
            'supply_df': pd.DataFrame()
        }

# Load filter data on page load
with st.spinner("Initializing filters..."):
    filter_data = initialize_filter_data()
    
    # Update filter cache for use by filter functions
    update_filter_cache(
        entities=filter_data['entities'],
        products=filter_data['products'],
        brands=filter_data['brands'],
        customers=filter_data['customers']
    )
    
    # Store in session state for immediate access
    st.session_state['pgap_filter_data'] = filter_data
    st.session_state['pgap_temp_demand'] = filter_data['demand_df']
    st.session_state['pgap_temp_supply'] = filter_data['supply_df']

# === Source Selection ===
def select_gap_sources():
    """Select demand and supply sources for GAP analysis"""
    st.markdown("### 📊 Data Source Selection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📤 Demand Sources")
        
        col1_1, col1_2 = st.columns(2)
        with col1_1:
            demand_oc = st.checkbox("OC", value=True, key="pgap_demand_oc")
        with col1_2:
            demand_forecast = st.checkbox("Forecast", value=True, key="pgap_demand_forecast")
        
        selected_demand_sources = []
        if demand_oc:
            selected_demand_sources.append("OC")
        if demand_forecast:
            selected_demand_sources.append("Forecast")
        
        if demand_forecast:
            include_converted = st.checkbox(
                "Include Converted Forecasts", 
                value=False,
                help="⚠️ May cause double counting if OC is also selected",
                key="pgap_include_converted"
            )
        else:
            include_converted = False
        
        # Customer filter
        st.markdown("##### Customer Filter")
        all_customers = filter_data.get('customers', [])
        
        selected_customers = st.multiselect(
            "Select Customers", 
            options=all_customers,
            key="pgap_customer",
            placeholder="All customers" if all_customers else "No customers available"
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
        "selected_customers": selected_customers
    }

selected_sources = select_gap_sources()

st.markdown("---")

# === Filters ===
def apply_standard_filters():
    """Apply standard filters for GAP analysis with formatted product display"""
    with st.expander("🔎 Filters", expanded=True):
        filters = {}
        
        # Get pre-loaded filter data
        filter_data = st.session_state.get('pgap_filter_data', {})
        
        # Row 1: Entity, Product, Brand
        col1, col2, col3 = st.columns(3)
        
        with col1:
            all_entities = filter_data.get('entities', [])
            filters['entity'] = st.multiselect(
                "Legal Entity",
                all_entities,
                key="pgap_entity_filter",
                placeholder="All entities" if all_entities else "No entities available"
            )
        
        with col2:
            # Use the formatted product options
            product_options = filter_data.get('product_options', [])
            selected_products = st.multiselect(
                "Product",  # Updated label
                product_options,
                key="pgap_product_filter",
                placeholder="All products" if product_options else "No products available"
            )
            
            # Extract PT codes from formatted selections
            filters['product'] = []
            for selection in selected_products:
                # Extract PT code (everything before first |)
                if '|' in selection:
                    pt_code = selection.split(' | ')[0].strip()
                else:
                    pt_code = selection.strip()
                filters['product'].append(pt_code)
        
        with col3:
            all_brands = filter_data.get('brands', [])
            filters['brand'] = st.multiselect(
                "Brand",
                all_brands,
                key="pgap_brand_filter",
                placeholder="All brands" if all_brands else "No brands available"
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
        
        # Show active filters
        active_filters = sum(1 for k, v in filters.items() 
                           if k not in ['start_date', 'end_date'] and v and v != [])
        if active_filters > 0:
            st.success(f"🔍 {active_filters} filters active")
    
    return filters

filters = apply_standard_filters()

st.markdown("---")

# === Calculation Options ===
def get_calculation_options():
    """Get calculation options"""
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
            value=True,
            key="pgap_track_backlog",
            help="Track negative carry forward (backlog) from shortage periods"
        )
    
    return {
        "period_type": period_type,
        "exclude_missing_dates": exclude_missing_dates,
        "track_backlog": track_backlog
    }

calc_options = get_calculation_options()

st.markdown("---")

# === Apply Filters Function ===
def apply_filters_to_data(df_demand, df_supply, filters, selected_customers):
    """Apply filters to demand and supply dataframes"""
    
    filtered_demand = df_demand.copy()
    filtered_supply = df_supply.copy()
    
    # Clean product codes
    if 'pt_code' in filtered_demand.columns:
        filtered_demand['pt_code'] = filtered_demand['pt_code'].astype(str).str.strip()
    
    if 'pt_code' in filtered_supply.columns:
        filtered_supply['pt_code'] = filtered_supply['pt_code'].astype(str).str.strip()
    
    # Apply filters to DEMAND
    if filters.get('entity'):
        filtered_demand = filtered_demand[filtered_demand['legal_entity'].isin(filters['entity'])]
    
    if filters.get('product'):
        clean_products = [str(p).strip() for p in filters['product']]
        filtered_demand = filtered_demand[filtered_demand['pt_code'].isin(clean_products)]
    
    if filters.get('brand'):
        filtered_demand = filtered_demand[filtered_demand['brand'].isin(filters['brand'])]
    
    if selected_customers and 'customer' in filtered_demand.columns:
        filtered_demand = filtered_demand[filtered_demand['customer'].isin(selected_customers)]
    
    # Apply date filter for demand
    if 'etd' in filtered_demand.columns and filters.get('start_date') and filters.get('end_date'):
        start_date = pd.to_datetime(filters['start_date'])
        end_date = pd.to_datetime(filters['end_date'])
        
        filtered_demand['etd'] = pd.to_datetime(filtered_demand['etd'], errors='coerce')
        
        date_mask = (
            filtered_demand['etd'].isna() |
            ((filtered_demand['etd'] >= start_date) & (filtered_demand['etd'] <= end_date))
        )
        filtered_demand = filtered_demand[date_mask]
    
    # Apply filters to SUPPLY
    if filters.get('entity'):
        filtered_supply = filtered_supply[filtered_supply['legal_entity'].isin(filters['entity'])]
    
    if filters.get('product'):
        clean_products = [str(p).strip() for p in filters['product']]
        filtered_supply = filtered_supply[filtered_supply['pt_code'].isin(clean_products)]
    
    if filters.get('brand'):
        filtered_supply = filtered_supply[filtered_supply['brand'].isin(filters['brand'])]
    
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
    
    return filtered_demand, filtered_supply

# === Run Analysis Button ===
if st.button("🚀 Run Period GAP Analysis", type="primary", use_container_width=True):
    
    if not selected_sources["demand"] or not selected_sources["supply"]:
        st.error("Please select at least one demand source and one supply source.")
    else:
        # Load data
        with st.spinner("Loading demand data..."):
            df_demand_all = data_loader.get_demand_data(
                selected_sources["demand"],
                selected_sources["include_converted"]
            )
        
        with st.spinner("Loading supply data..."):
            df_supply_all = data_loader.get_supply_data(
                selected_sources["supply"],
                selected_sources["exclude_expired"]
            )
        
        # Apply filters
        df_demand_filtered, df_supply_filtered = apply_filters_to_data(
            df_demand_all,
            df_supply_all,
            filters,
            selected_sources.get("selected_customers", [])
        )
        
        # Apply date exclusion if requested
        if calc_options.get("exclude_missing_dates", True):
            if not df_demand_filtered.empty and 'etd' in df_demand_filtered.columns:
                df_demand_filtered = df_demand_filtered[df_demand_filtered['etd'].notna()]
            
            if not df_supply_filtered.empty and 'date_ref' in df_supply_filtered.columns:
                df_supply_filtered = df_supply_filtered[df_supply_filtered['date_ref'].notna()]
        
        # Save to session for cross-page access
        save_period_gap_state({
            'demand': df_demand_filtered,
            'supply': df_supply_filtered,
            'calc_options': calc_options,
            'display_filters': None
        })
        
        # Also save for other pages
        save_to_session_state('gap_analysis_result', None)  # Will be set after GAP calculation
        save_to_session_state('demand_filtered', df_demand_filtered)
        save_to_session_state('supply_filtered', df_supply_filtered)

# === Display Results ===
if get_period_gap_state():
    
    # Get Display Filters
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
    
    # Period/Status Filters
    st.markdown("#### Period & Status Filters")
    period_filter = st.radio(
        "Show:",
        options=["All", "Shortage Only", "Past Periods Only", "Future Periods Only", "Critical Shortage Only"],
        horizontal=True,
        key="pgap_period_filter"
    )
    
    # View Options
    col4, col5 = st.columns(2)
    with col4:
        enable_row_highlighting = st.checkbox(
            "🎨 Enable Row Highlighting",
            value=False,
            key="pgap_row_highlighting"
        )
    
    display_filters = {
        "show_matched": show_matched,
        "show_demand_only": show_demand_only,
        "show_supply_only": show_supply_only,
        "period_filter": period_filter,
        "enable_row_highlighting": enable_row_highlighting,
        "period_type": calc_options["period_type"],
        "track_backlog": calc_options["track_backlog"]
    }
    
    st.markdown("---")
    
    # Get data from state
    state = get_period_gap_state()
    df_demand_filtered = state['demand']
    df_supply_filtered = state['supply']
    stored_calc_options = state['calc_options']
    
    # Calculate GAP if not cached or options changed
    cache_key = f"{stored_calc_options['period_type']}_{stored_calc_options['track_backlog']}"
    
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
        gap_df_filtered = gap_df.copy()
        
        # Filter by product type
        if not (show_matched and show_demand_only and show_supply_only):
            demand_products = set(df_demand_filtered['pt_code'].unique()) if not df_demand_filtered.empty else set()
            supply_products = set(df_supply_filtered['pt_code'].unique()) if not df_supply_filtered.empty else set()
            
            products_to_show = set()
            if show_matched:
                products_to_show.update(demand_products & supply_products)
            if show_demand_only:
                products_to_show.update(demand_products - supply_products)
            if show_supply_only:
                products_to_show.update(supply_products - demand_products)
            
            if products_to_show:
                gap_df_filtered = gap_df_filtered[gap_df_filtered['pt_code'].isin(products_to_show)]
        
        # Filter by period/status
        if period_filter == "Shortage Only":
            gap_df_filtered = gap_df_filtered[gap_df_filtered["gap_quantity"] < 0]
        elif period_filter == "Past Periods Only":
            gap_df_filtered = gap_df_filtered[
                gap_df_filtered['period'].apply(lambda x: is_past_period(str(x), stored_calc_options['period_type']))
            ]
        elif period_filter == "Future Periods Only":
            gap_df_filtered = gap_df_filtered[
                ~gap_df_filtered['period'].apply(lambda x: is_past_period(str(x), stored_calc_options['period_type']))
            ]
        elif period_filter == "Critical Shortage Only":
            gap_df_filtered = gap_df_filtered[
                (gap_df_filtered["gap_quantity"] < 0) & 
                (gap_df_filtered["fulfillment_rate_percent"] < 50)
            ]
        
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
            
            # Export section
            st.markdown("---")
            st.markdown("### 📤 Export Options")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                excel_data = convert_df_to_excel(gap_df_filtered, "GAP_Analysis")
                st.download_button(
                    "📊 Export GAP Details",
                    data=excel_data,
                    file_name=f"period_gap_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            with col2:
                if (gap_df_filtered["gap_quantity"] < 0).any():
                    shortage_df = gap_df_filtered[gap_df_filtered["gap_quantity"] < 0]
                    shortage_excel = convert_df_to_excel(shortage_df, "Shortage")
                    st.download_button(
                        "🚨 Export Shortage Only",
                        data=shortage_excel,
                        file_name=f"shortage_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
    else:
        st.warning("No data available for the selected filters and sources.")

# Footer
st.markdown("---")
st.caption(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")