# utils/net_gap/filters.py

"""
Filter components module for GAP Analysis System - Clean Version
Provides reusable filter UI components for main page
"""

import streamlit as st
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Filter configuration constants
DEFAULT_DATE_RANGE_DAYS = 30
MAX_CUSTOMER_DROPDOWN = 100
MAX_PRODUCT_SEARCH_RESULTS = 50

QUICK_FILTER_OPTIONS = {
    'all': '📋 All Items',
    'shortage': '⚠️ Shortage Only',
    'critical': '🚨 Critical Only',
    'surplus': '📦 Surplus Only',
    'balanced': '✅ Balanced Only'
}

GROUP_BY_OPTIONS = {
    'product': '📦 Product',
    'brand': '🏷️ Brand',
    'category': '📂 Category'
}

SUPPLY_SOURCES = {
    'INVENTORY': '📦 Inventory (Available Now)',
    'CAN_PENDING': '⏳ CAN Pending (1-3 days)',
    'WAREHOUSE_TRANSFER': '🚚 Warehouse Transfer (2-5 days)',
    'PURCHASE_ORDER': '📝 Purchase Order (7-30 days)'
}

DEMAND_SOURCES = {
    'OC_PENDING': '📋 Confirmed Orders (OC)',
    'FORECAST': '📊 Customer Forecast'
}

DATE_PRESETS = {
    "Today": 0,
    "7 Days": 7,
    "14 Days": 14,
    "30 Days": 30,
    "60 Days": 60,
    "90 Days": 90
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
        """Get default filter values"""
        return {
            'entity': None,
            'date_range': (date.today(), date.today() + timedelta(days=DEFAULT_DATE_RANGE_DAYS)),
            'products': [],
            'brands': [],
            'customers': [],
            'quick_filter': 'all',
            'group_by': 'product',
            'supply_sources': list(SUPPLY_SOURCES.keys()),
            'demand_sources': list(DEMAND_SOURCES.keys())
        }
    
    def render_main_page_filters(self) -> Dict[str, Any]:
        """
        Render all filter components on main page
        
        Returns:
            Dictionary containing all filter selections
        """
        filters = {}
        
        with st.expander("🔍 **Filters & Settings**", expanded=True):
            # Quick filters and grouping
            self._render_quick_controls(filters)
            st.divider()
            
            # Entity and date range
            self._render_basic_filters(filters)
            st.divider()
            
            # Source selection
            self._render_source_selection(filters)
            st.divider()
            
            # Product filters
            self._render_product_filters(filters)
            st.divider()
            
            # Filter actions
            self._render_filter_actions(filters)
        
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
        """Render entity and date range filters"""
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            filters['entity'] = self._render_entity_filter()
        
        with col2:
            date_from = st.date_input(
                "📅 From Date",
                value=st.session_state.gap_filters['date_range'][0],
                max_value=date.today() + timedelta(days=365),
                help="Start date for analysis"
            )
        
        with col3:
            date_to = st.date_input(
                "📅 To Date",
                value=st.session_state.gap_filters['date_range'][1],
                min_value=date_from,
                max_value=date.today() + timedelta(days=365),
                help="End date for analysis"
            )
        
        filters['date_range'] = (date_from, date_to)
        
        # Date presets
        st.markdown("**Quick Date Range:**")
        preset_cols = st.columns(len(DATE_PRESETS))
        
        for idx, (label, days) in enumerate(DATE_PRESETS.items()):
            with preset_cols[idx]:
                if st.button(label, key=f"date_preset_{days}", use_container_width=True):
                    st.session_state.gap_filters['date_range'] = (
                        date.today(),
                        date.today() + timedelta(days=days)
                    )
                    st.rerun()
    
    def _render_source_selection(self, filters: Dict[str, Any]) -> None:
        """Render supply and demand source selection"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📦 Supply Sources")
            filters['supply_sources'] = self._render_supply_sources()
        
        with col2:
            st.subheader("📋 Demand Sources")
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
            st.warning("⚠️ Select at least one supply source")
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
            st.warning("⚠️ Select at least one demand source")
            return ['OC_PENDING']  # Default fallback
        
        return selected
    
    def _render_product_filters(self, filters: Dict[str, Any]) -> None:
        """Render product-related filters"""
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.subheader("🔍 Product Selection")
            filters['products'] = self._render_product_selector(filters.get('entity'))
        
        with col2:
            st.subheader("🏷️ Brands")
            filters['brands'] = self._render_brand_selector(filters.get('entity'))
        
        with col3:
            st.subheader("🏢 Customers")
            filters['customers'] = self._render_customer_selector(filters.get('entity'))
    
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
            "🏢 Entity",
            options=entity_options,
            index=default_index,
            help="Select entity to analyze"
        )
        
        return None if selected == "All Entities" else selected
    
    def _render_product_selector(self, entity: Optional[str]) -> List[int]:
        """Render product selection with search"""
        products_df = self.data_loader.get_products(entity)
        
        if products_df.empty:
            st.info("No products available")
            return []
        
        # Search input
        search_term = st.text_input(
            "Search",
            placeholder="Enter PT code or product name...",
            label_visibility="collapsed"
        )
        
        selected_products = st.session_state.gap_filters.get('products', [])
        
        if search_term:
            # Filter products based on search
            mask = (
                products_df['pt_code'].str.contains(search_term, case=False, na=False) |
                products_df['product_name'].str.contains(search_term, case=False, na=False)
            )
            filtered_products = products_df[mask].head(MAX_PRODUCT_SEARCH_RESULTS)
            
            if not filtered_products.empty:
                # Create display names
                filtered_products['display'] = filtered_products.apply(
                    lambda x: f"{x['pt_code']} - {x['product_name'][:30]}{'...' if len(x['product_name']) > 30 else ''}", 
                    axis=1
                )
                
                selected_products = st.multiselect(
                    "Select products",
                    options=filtered_products['product_id'].tolist(),
                    format_func=lambda x: filtered_products[
                        filtered_products['product_id'] == x
                    ]['display'].iloc[0],
                    default=[p for p in selected_products if p in filtered_products['product_id'].tolist()],
                    label_visibility="collapsed"
                )
        elif selected_products:
            st.info(f"✓ {len(selected_products)} products selected")
        
        return selected_products
    
    def _render_brand_selector(self, entity: Optional[str]) -> List[str]:
        """Render brand selection"""
        brands = self.data_loader.get_brands(entity)
        
        if not brands:
            return []
        
        return st.multiselect(
            "Select brands",
            options=brands,
            default=st.session_state.gap_filters.get('brands', []),
            label_visibility="collapsed",
            placeholder="All brands"
        )
    
    def _render_customer_selector(self, entity: Optional[str]) -> List[str]:
        """Render customer selection"""
        customers = self.data_loader.get_customers(entity)
        
        if not customers:
            return []
        
        # Handle large customer lists with search
        if len(customers) > MAX_CUSTOMER_DROPDOWN:
            search_customer = st.text_input(
                "Search",
                placeholder="Enter customer name...",
                label_visibility="collapsed"
            )
            
            if search_customer:
                filtered = [c for c in customers if search_customer.lower() in c.lower()]
                return st.multiselect(
                    "Select",
                    options=filtered[:20],
                    default=st.session_state.gap_filters.get('customers', []),
                    label_visibility="collapsed"
                )
            return []
        
        return st.multiselect(
            "Select customers",
            options=customers,
            default=st.session_state.gap_filters.get('customers', []),
            label_visibility="collapsed",
            placeholder="All customers"
        )
    
    def _render_filter_actions(self, filters: Dict[str, Any]) -> None:
        """Render filter action buttons"""
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("🔄 Reset Filters", use_container_width=True):
                st.session_state.gap_filters = self._get_default_filters()
                st.rerun()
        
        with col2:
            if st.button("📊 Apply Filters", type="primary", use_container_width=True):
                st.rerun()
        
        with col3:
            active_count = self._count_active_filters(filters)
            if active_count > 0:
                st.info(f"✔ {active_count} filters active")
    
    def _count_active_filters(self, filters: Dict[str, Any]) -> int:
        """Count number of active (non-default) filters"""
        count = 0
        defaults = self._get_default_filters()
        
        # Check each filter against defaults
        if filters.get('entity') != defaults['entity']:
            count += 1
        if filters.get('products'):
            count += 1
        if filters.get('brands'):
            count += 1
        if filters.get('customers'):
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