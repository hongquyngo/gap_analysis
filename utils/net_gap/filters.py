# utils/net_gap/filters.py

"""
Filter components module for GAP Analysis System - Updated Version
- Removed Category grouping
- Removed Quick Date Range presets
- Changed to multiselect for Products and Customers
- Auto-detect date range from data
"""

import streamlit as st
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Filter configuration constants
DEFAULT_DATE_RANGE_DAYS = 30
MAX_MULTISELECT_DISPLAY = 200

QUICK_FILTER_OPTIONS = {
    'all': 'ðŸ“‹ All Items',
    'shortage': 'âš ï¸ Shortage Only',
    'critical': 'ðŸš¨ Critical Only',
    'surplus': 'ðŸ“¦ Surplus Only',
    'balanced': 'âœ… Balanced Only'
}

# REMOVED CATEGORY - Only Product and Brand
GROUP_BY_OPTIONS = {
    'product': 'ðŸ“¦ Product',
    'brand': 'ðŸ·ï¸ Brand'
}

SUPPLY_SOURCES = {
    'INVENTORY': 'ðŸ“¦ Inventory (Available Now)',
    'CAN_PENDING': 'â³ CAN Pending (1-3 days)',
    'WAREHOUSE_TRANSFER': 'ðŸšš Warehouse Transfer (2-5 days)',
    'PURCHASE_ORDER': 'ðŸ“ Purchase Order (7-30 days)'
}

DEMAND_SOURCES = {
    'OC_PENDING': 'ðŸ“‹ Confirmed Orders (OC)',
    'FORECAST': 'ðŸ“Š Customer Forecast'
}


class GAPFilters:
    """Manages filter UI components for GAP analysis"""
    
    def __init__(self, data_loader):
        """
        Initialize filters with data loader
        
        Args:
            data_loader: Instance of GAPDataLoader for fetching reference data
        """
        self.data_loader = data_loader
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize session state for filter persistence"""
        if 'gap_filters' not in st.session_state:
            st.session_state.gap_filters = self._get_default_filters()
    
    def _get_default_filters(self) -> Dict[str, Any]:
        """Get default filter values with auto date range"""
        # Get date range from data
        date_range = self._get_data_date_range()
        
        return {
            'entity': None,
            'date_range': date_range,
            'products': [],  # Multiselect - empty list
            'brands': [],
            'customers': [],  # Multiselect - empty list
            'quick_filter': 'all',
            'group_by': 'product',  # Default to product
            'supply_sources': list(SUPPLY_SOURCES.keys()),
            'demand_sources': list(DEMAND_SOURCES.keys())
        }
    
    def _get_data_date_range(self) -> tuple:
        """Get min/max dates from supply and demand data"""
        try:
            # Try to get date range from data loader
            date_info = self.data_loader.get_date_range()
            if date_info and 'min_date' in date_info and 'max_date' in date_info:
                return (date_info['min_date'], date_info['max_date'])
        except Exception as e:
            logger.warning(f"Could not get date range from data: {e}")
        
        # Fallback to default 30 days
        return (date.today(), date.today() + timedelta(days=DEFAULT_DATE_RANGE_DAYS))
    
    def render_main_page_filters(self) -> Dict[str, Any]:
        """
        Render all filter components on main page
        
        Returns:
            Dictionary containing all filter selections
        """
        filters = {}
        
        with st.expander("ðŸ” **Filters & Settings**", expanded=True):
            # Quick filters and grouping
            self._render_quick_controls(filters)
            st.divider()
            
            # Entity and date range (NO QUICK DATE PRESETS)
            self._render_basic_filters(filters)
            st.divider()
            
            # Source selection
            self._render_source_selection(filters)
            st.divider()
            
            # Product filters (ALL MULTISELECT NOW)
            self._render_product_filters(filters)
            st.divider()
            
            # Filter actions
            self._render_filter_actions(filters)
        
        # Validate group_by - only allow product or brand
        if filters.get('group_by') not in ['product', 'brand']:
            filters['group_by'] = 'product'
        
        # Save to session state
        st.session_state.gap_filters = filters
        return filters
    
    def _render_quick_controls(self, filters: Dict[str, Any]) -> None:
        """Render quick filter and grouping controls"""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Quick Filters")
            filters['quick_filter'] = st.radio(
                "Select preset",
                options=list(QUICK_FILTER_OPTIONS.keys()),
                format_func=lambda x: QUICK_FILTER_OPTIONS[x],
                index=list(QUICK_FILTER_OPTIONS.keys()).index(
                    st.session_state.gap_filters.get('quick_filter', 'all')
                ),
                horizontal=True,
                label_visibility="collapsed"
            )
        
        with col2:
            st.subheader("Group By")
            filters['group_by'] = st.radio(
                "Group by",
                options=list(GROUP_BY_OPTIONS.keys()),
                format_func=lambda x: GROUP_BY_OPTIONS[x],
                index=list(GROUP_BY_OPTIONS.keys()).index(
                    st.session_state.gap_filters.get('group_by', 'product')
                ),
                horizontal=True,
                label_visibility="collapsed"
            )
    
    def _render_basic_filters(self, filters: Dict[str, Any]) -> None:
        """Render entity and date range filters WITHOUT quick date presets"""
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            filters['entity'] = self._render_entity_filter()
        
        # Get current or default date range
        current_range = st.session_state.gap_filters.get('date_range', self._get_data_date_range())
        
        with col2:
            date_from = st.date_input(
                "ðŸ“… From Date",
                value=current_range[0],
                max_value=date.today() + timedelta(days=365),
                help="Start date for analysis"
            )
        
        with col3:
            date_to = st.date_input(
                "ðŸ“… To Date",
                value=current_range[1],
                min_value=date_from,
                max_value=date.today() + timedelta(days=365),
                help="End date for analysis"
            )
        
        filters['date_range'] = (date_from, date_to)
        
        # Show date range info
        days_diff = (date_to - date_from).days
        st.caption(f"ðŸ“… Date range: {days_diff} days selected")
    
    def _render_source_selection(self, filters: Dict[str, Any]) -> None:
        """Render supply and demand source selection"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("ðŸ“¦ Supply Sources")
            filters['supply_sources'] = self._render_supply_sources()
        
        with col2:
            st.subheader("ðŸ“‹ Demand Sources")
            filters['demand_sources'] = self._render_demand_sources()
    
    def _render_supply_sources(self) -> List[str]:
        """Render supply source checkboxes"""
        selected = []
        default_selected = st.session_state.gap_filters.get('supply_sources', list(SUPPLY_SOURCES.keys()))
        
        # Two columns for better layout
        col1, col2 = st.columns(2)
        
        for idx, (source, label) in enumerate(SUPPLY_SOURCES.items()):
            col = col1 if idx < 2 else col2
            with col:
                if st.checkbox(label, value=source in default_selected, key=f"supply_{source}"):
                    selected.append(source)
        
        if not selected:
            st.warning("âš ï¸ Select at least one supply source")
            return ['INVENTORY']  # Default fallback
        
        return selected
    
    def _render_demand_sources(self) -> List[str]:
        """Render demand source checkboxes"""
        selected = []
        default_selected = st.session_state.gap_filters.get('demand_sources', list(DEMAND_SOURCES.keys()))
        
        for source, label in DEMAND_SOURCES.items():
            if st.checkbox(label, value=source in default_selected, key=f"demand_{source}"):
                selected.append(source)
        
        if not selected:
            st.warning("âš ï¸ Select at least one demand source")
            return ['OC_PENDING']  # Default fallback
        
        return selected
    
    def _render_product_filters(self, filters: Dict[str, Any]) -> None:
        """Render product-related filters - ALL MULTISELECT NOW"""
        col1, col2, col3 = st.columns([2, 1, 2])
        
        with col1:
            st.subheader("ðŸ” Product Selection")
            filters['products'] = self._render_product_multiselect(filters.get('entity'))
        
        with col2:
            st.subheader("ðŸ·ï¸ Brands")
            filters['brands'] = self._render_brand_selector(filters.get('entity'))
        
        with col3:
            st.subheader("ðŸ¢ Customers")
            filters['customers'] = self._render_customer_multiselect(filters.get('entity'))
    
    def _render_entity_filter(self) -> Optional[str]:
        """Render entity selection filter"""
        entities = self.data_loader.get_entities()
        
        if not entities:
            st.warning("No entities available")
            return None
        
        entity_options = ["All Entities"] + entities
        current_value = st.session_state.gap_filters.get('entity')
        
        if current_value and current_value in entities:
            default_index = entities.index(current_value) + 1
        else:
            default_index = 0
        
        selected = st.selectbox(
            "ðŸ¢ Entity",
            options=entity_options,
            index=default_index,
            help="Select entity to analyze"
        )
        
        return None if selected == "All Entities" else selected
    
    def _render_product_multiselect(self, entity: Optional[str]) -> List[int]:
        """Render product selection as MULTISELECT"""
        products_df = self.data_loader.get_products(entity)
        
        if products_df.empty:
            st.info("No products available")
            return []
        
        # Create display names for products
        products_df['display_name'] = products_df.apply(
            lambda x: f"{x['pt_code']} - {x['product_name'][:40]}{'...' if len(x['product_name']) > 40 else ''}",
            axis=1
        )
                
        # Get currently selected products
        selected_products = st.session_state.gap_filters.get('products', [])
        
        # Multiselect widget
        selected = st.multiselect(
            "Select products",
            options=products_df['product_id'].tolist(),
            default=[p for p in selected_products if p in products_df['product_id'].tolist()],
            format_func=lambda x: products_df[products_df['product_id'] == x]['display_name'].iloc[0],
            placeholder="All products (leave empty for all)",
            label_visibility="collapsed",
            help="Select specific products or leave empty for all"
        )
        
        # Show count
        if selected:
            st.caption(f"âœ“ {len(selected)} products selected")
        else:
            st.caption("All products selected")
        
        return selected
    
    def _render_brand_selector(self, entity: Optional[str]) -> List[str]:
        """Render brand selection (already multiselect)"""
        brands = self.data_loader.get_brands(entity)
        
        if not brands:
            return []
        
        return st.multiselect(
            "Select brands",
            options=brands,
            default=st.session_state.gap_filters.get('brands', []),
            label_visibility="collapsed",
            placeholder="All brands (leave empty for all)"
        )
    
    def _render_customer_multiselect(self, entity: Optional[str]) -> List[str]:
        """Render customer selection as MULTISELECT"""
        customers = self.data_loader.get_customers(entity)
        
        if not customers:
            return []
        
        # Limit display if too many customers
        if len(customers) > MAX_MULTISELECT_DISPLAY:
            st.info(f"Too many customers ({len(customers)}). Showing first {MAX_MULTISELECT_DISPLAY}")
            customers = customers[:MAX_MULTISELECT_DISPLAY]
        
        selected = st.multiselect(
            "Select customers",
            options=customers,
            default=st.session_state.gap_filters.get('customers', []),
            label_visibility="collapsed",
            placeholder="All customers (leave empty for all)",
            help="Select specific customers or leave empty for all"
        )
        
        # Show count
        if selected:
            st.caption(f"âœ“ {len(selected)} customers selected")
        
        return selected
    
    def _render_filter_actions(self, filters: Dict[str, Any]) -> None:
        """Render filter action buttons"""
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("ðŸ”„ Reset Filters", use_container_width=True):
                st.session_state.gap_filters = self._get_default_filters()
                st.rerun()
        
        with col2:
            if st.button("ðŸ“Š Apply Filters", type="primary", use_container_width=True):
                st.rerun()
        
        with col3:
            active_count = self._count_active_filters(filters)
            if active_count > 0:
                st.info(f"âœ” {active_count} filters active")
    
    def _count_active_filters(self, filters: Dict[str, Any]) -> int:
        """Count number of active (non-default) filters"""
        count = 0
        defaults = self._get_default_filters()
        
        # Check each filter against defaults
        if filters.get('entity') != defaults['entity']:
            count += 1
        if filters.get('products'):  # Not empty list
            count += 1
        if filters.get('brands'):
            count += 1
        if filters.get('customers'):  # Not empty list
            count += 1
        if filters.get('quick_filter') != defaults['quick_filter']:
            count += 1
        if set(filters.get('supply_sources', [])) != set(defaults['supply_sources']):
            count += 1
        if set(filters.get('demand_sources', [])) != set(defaults['demand_sources']):
            count += 1
        
        return count
    
    def apply_quick_filter(self, gap_df: pd.DataFrame, quick_filter: str) -> pd.DataFrame:
        """
        Apply quick filter preset to GAP dataframe
        
        Args:
            gap_df: DataFrame with GAP calculations
            quick_filter: Quick filter type
            
        Returns:
            Filtered DataFrame
        """
        if gap_df.empty or quick_filter == 'all':
            return gap_df
        
        filter_mappings = {
            'shortage': ['SEVERE_SHORTAGE', 'HIGH_SHORTAGE', 'MODERATE_SHORTAGE'],
            'critical': ['SEVERE_SHORTAGE', 'HIGH_SHORTAGE'],
            'surplus': ['SEVERE_SURPLUS', 'HIGH_SURPLUS', 'MODERATE_SURPLUS', 'LIGHT_SURPLUS'],
            'balanced': ['BALANCED']
        }
        
        if quick_filter in filter_mappings:
            return gap_df[gap_df['gap_status'].isin(filter_mappings[quick_filter])]
        
        return gap_df
    
    def get_filter_summary(self, filters: Dict[str, Any]) -> str:
        """Generate a summary string of active filters"""
        summary_parts = []
        
        # Entity
        if filters.get('entity'):
            summary_parts.append(f"Entity: {filters['entity']}")
        
        # Date range
        if filters.get('date_range'):
            date_from, date_to = filters['date_range']
            days_diff = (date_to - date_from).days
            summary_parts.append(f"Period: {days_diff} days ({date_from} to {date_to})")
        
        # Sources
        supply_count = len(filters.get('supply_sources', []))
        demand_count = len(filters.get('demand_sources', []))
        if supply_count < len(SUPPLY_SOURCES):
            summary_parts.append(f"Supply: {supply_count}/{len(SUPPLY_SOURCES)} sources")
        if demand_count < len(DEMAND_SOURCES):
            summary_parts.append(f"Demand: {demand_count}/{len(DEMAND_SOURCES)} sources")
        
        # Product filters
        if filters.get('products'):
            summary_parts.append(f"Products: {len(filters['products'])} selected")
        if filters.get('brands'):
            summary_parts.append(f"Brands: {len(filters['brands'])} selected")
        if filters.get('customers'):
            summary_parts.append(f"Customers: {len(filters['customers'])} selected")
        
        # Quick filter
        if filters.get('quick_filter') != 'all':
            summary_parts.append(f"Filter: {QUICK_FILTER_OPTIONS[filters['quick_filter']]}")
        
        # Grouping
        summary_parts.append(f"Grouped by: {GROUP_BY_OPTIONS[filters.get('group_by', 'product')]}")
        
        return " | ".join(summary_parts) if summary_parts else "No filters applied"