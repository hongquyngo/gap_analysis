# utils/net_gap/filters.py

"""
Filter components module for GAP Analysis System - Version 2.1 (Refactored)
- Integrated with SessionStateManager for centralized state management
- Converts lists to tuples for stable cache keys
- Improved validation and error handling
- Moved quick filter to post-calculation (table display)
"""

import streamlit as st
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any, Tuple
import logging

from .session_manager import get_session_manager
from .data_loader import DataLoadError, ValidationError

logger = logging.getLogger(__name__)

# Filter configuration constants
DEFAULT_DATE_RANGE_DAYS = 30
MAX_MULTISELECT_DISPLAY = 200

# Quick filter options (used in table display, not pre-calculation)
QUICK_FILTER_BASE = {
    'all': {'label': 'All Items', 'help': 'Show all products in the analysis'},
    'shortage': {'label': 'Shortage', 'help': 'Products with supply below demand (coverage < 90%)'},
    'balanced': {'label': 'Balanced', 'help': 'Products with balanced supply and demand (90-110% coverage)'},
    'surplus': {'label': 'Surplus', 'help': 'Products with excess inventory (coverage > 110%)'}
}

QUICK_FILTER_SAFETY = {
    'all': {'label': 'All Items', 'help': 'Show all products in the analysis'},
    'shortage': {'label': 'Below Requirements', 'help': 'Products below demand or safety stock requirements'},
    'balanced': {'label': 'Balanced', 'help': 'Products meeting both demand and safety requirements'},
    'surplus': {'label': 'Surplus', 'help': 'Products with excess inventory above safety levels'},
    'reorder': {'label': 'At Reorder Point', 'help': 'Products at or below reorder point - create POs now'}
}

GROUP_BY_OPTIONS = {
    'product': 'Product',
    'brand': 'Brand'
}

SUPPLY_SOURCES = {
    'INVENTORY': 'Inventory (Available Now)',
    'CAN_PENDING': 'CAN Pending (1-3 days)',
    'WAREHOUSE_TRANSFER': 'Warehouse Transfer (2-5 days)',
    'PURCHASE_ORDER': 'Purchase Order (7-30 days)'
}

DEMAND_SOURCES = {
    'OC_PENDING': 'Confirmed Orders (OC)',
    'FORECAST': 'Customer Forecast'
}


class GAPFilters:
    """Manages filter UI components for GAP analysis with SessionStateManager integration"""
    
    def __init__(self, data_loader):
        """
        Initialize filters with data loader and session manager
        
        Args:
            data_loader: Instance of GAPDataLoader for fetching reference data
        """
        self.data_loader = data_loader
        self.session_manager = get_session_manager()
        self._safety_stock_available = False
        self._check_safety_stock_availability()
    
    def _check_safety_stock_availability(self) -> None:
        """Check if safety stock data is available"""
        try:
            self._safety_stock_available = self.data_loader.check_safety_stock_availability()
            logger.info(f"Safety stock available: {self._safety_stock_available}")
        except Exception as e:
            logger.warning(f"Could not check safety stock availability: {e}")
            self._safety_stock_available = False
    
    def _get_data_date_range(self) -> Tuple[date, date]:
        """
        Get min/max dates from supply and demand data
        
        Returns:
            Tuple of (min_date, max_date)
        """
        try:
            date_info = self.data_loader.get_date_range()
            if date_info and 'min_date' in date_info and 'max_date' in date_info:
                return (date_info['min_date'], date_info['max_date'])
        except Exception as e:
            logger.warning(f"Could not get date range from data: {e}")
        
        # Fallback to default 30 days
        return (date.today(), date.today() + timedelta(days=DEFAULT_DATE_RANGE_DAYS))
    
    def render_filters(self) -> Dict[str, Any]:
        """
        Render all filter components for pre-calculation configuration
        
        Returns:
            Dictionary containing all filter selections with tuples for cache stability
        """
        filters = {}
        
        with st.expander("⚙️ **Data Configuration**", expanded=True):
            # Basic filters
            self._render_basic_filters(filters)
            
            st.divider()
            
            # Source selection with safety stock toggle
            self._render_source_selection(filters)
            
            st.divider()
            
            # Product scope filters
            self._render_product_filters(filters)
            
            st.divider()
            
            # Group by selection (quick filter moved to table display)
            self._render_group_by(filters)
        
        # Validate and save to session state
        filters = self._validate_and_convert_filters(filters)
        self.session_manager.set_filters(filters)
        
        return filters
    
    def _render_basic_filters(self, filters: Dict[str, Any]) -> None:
        """Render entity and date range filters"""
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            filters['entity'] = self._render_entity_filter()
        
        # Get current date range from session, or load from data if not set
        current_filters = self.session_manager.get_filters()
        current_range = current_filters.get('date_range')
        
        # If date range is None (first load), get from data
        if current_range is None:
            current_range = self._get_data_date_range()
            logger.info(f"Initialized date range from data: {current_range[0]} to {current_range[1]}")
        
        with col2:
            date_from = st.date_input(
                "From Date",
                value=current_range[0],
                max_value=date.today() + timedelta(days=365),
                help="Start date for analysis",
                key="filter_date_from"
            )
        
        with col3:
            date_to = st.date_input(
                "To Date",
                value=current_range[1],
                min_value=date_from,
                max_value=date.today() + timedelta(days=365),
                help="End date for analysis",
                key="filter_date_to"
            )
        
        filters['date_range'] = (date_from, date_to)
        
        # Show date range info
        days_diff = (date_to - date_from).days
        st.caption(f"Date range: **{days_diff}** days selected")
    
    def _render_source_selection(self, filters: Dict[str, Any]) -> None:
        """Render supply and demand source selection with safety stock toggle"""
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            st.markdown("**Supply Sources**")
            filters['supply_sources'] = self._render_supply_sources()
        
        with col2:
            st.markdown("**Demand Sources**")
            filters['demand_sources'] = self._render_demand_sources()
        
        with col3:
            st.markdown("**Safety Stock**")
            current_filters = self.session_manager.get_filters()
            
            if self._safety_stock_available:
                filters['include_safety_stock'] = st.checkbox(
                    "Include Safety",
                    value=current_filters.get('include_safety_stock', True),
                    help="Consider safety stock requirements in GAP calculation",
                    key="safety_toggle"
                )
                
                if filters['include_safety_stock']:
                    st.caption("✅ Safety rules active")
            else:
                filters['include_safety_stock'] = False
                st.caption("❌ Not configured")
    
    def _render_supply_sources(self) -> List[str]:
        """Render supply source checkboxes"""
        selected = []
        current_filters = self.session_manager.get_filters()
        default_selected = current_filters.get('supply_sources', list(SUPPLY_SOURCES.keys()))
        
        # Compact two-column layout
        col1, col2 = st.columns(2)
        
        for idx, (source, label) in enumerate(SUPPLY_SOURCES.items()):
            col = col1 if idx < 2 else col2
            with col:
                short_label = label.split('(')[0].strip()
                if st.checkbox(
                    short_label, 
                    value=source in default_selected, 
                    key=f"supply_{source}", 
                    help=label
                ):
                    selected.append(source)
        
        if not selected:
            st.warning("⚠️ Select at least one supply source")
            return ['INVENTORY']
        
        return selected
    
    def _render_demand_sources(self) -> List[str]:
        """Render demand source checkboxes"""
        selected = []
        current_filters = self.session_manager.get_filters()
        default_selected = current_filters.get('demand_sources', ['OC_PENDING'])
        
        for source, label in DEMAND_SOURCES.items():
            short_label = label.split('(')[0].strip()
            if st.checkbox(
                short_label, 
                value=source in default_selected, 
                key=f"demand_{source}", 
                help=label
            ):
                selected.append(source)
        
        if not selected:
            st.warning("⚠️ Select at least one demand source")
            return ['OC_PENDING']
        
        return selected
    
    def _render_product_filters(self, filters: Dict[str, Any]) -> None:
        """Render product-related filters"""
        col1, col2, col3 = st.columns([2, 1, 2])
        
        with col1:
            st.markdown("**Product Selection**")
            filters['products'] = self._render_product_multiselect(filters.get('entity'))
        
        with col2:
            st.markdown("**Brands**")
            filters['brands'] = self._render_brand_selector(filters.get('entity'))
        
        with col3:
            st.markdown("**Customers**")
            filters['customers'] = self._render_customer_multiselect(filters.get('entity'))
    
    def _render_group_by(self, filters: Dict[str, Any]) -> None:
        """Render group by selection only (quick filter moved to table display)"""
        st.markdown("**Group By**")
        
        current_filters = self.session_manager.get_filters()
        current_group_by = current_filters.get('group_by', 'product')
        
        try:
            group_by_index = list(GROUP_BY_OPTIONS.keys()).index(current_group_by)
        except ValueError:
            group_by_index = 0
        
        filters['group_by'] = st.radio(
            "Aggregate data by",
            options=list(GROUP_BY_OPTIONS.keys()),
            format_func=lambda x: f"📊 {GROUP_BY_OPTIONS[x]}",
            index=group_by_index,
            horizontal=True,
            label_visibility="collapsed",
            key="group_by_radio",
            help="Product: detailed per-item analysis | Brand: aggregated by brand"
        )
    
    def _render_entity_filter(self) -> Optional[str]:
        """Render entity selection filter with error handling"""
        try:
            entities = self.data_loader.get_entities()
        except DataLoadError as e:
            st.error(f"Failed to load entities: {str(e)}")
            return None
        
        if not entities:
            st.warning("No entities available")
            return None
        
        entity_options = ["All Entities"] + entities
        current_filters = self.session_manager.get_filters()
        current_value = current_filters.get('entity')
        
        if current_value and current_value in entities:
            default_index = entities.index(current_value) + 1
        else:
            default_index = 0
        
        selected = st.selectbox(
            "Entity",
            options=entity_options,
            index=default_index,
            help="Select entity to analyze",
            key="entity_select"
        )
        
        return None if selected == "All Entities" else selected
    
    def _render_product_multiselect(self, entity: Optional[str]) -> List[int]:
        """Render product selection as multiselect with error handling"""
        try:
            products_df = self.data_loader.get_products(entity)
        except (DataLoadError, ValidationError) as e:
            st.error(f"Failed to load products: {str(e)}")
            return []
        
        if products_df.empty:
            st.info("No products available")
            return []
        
        # Create display names
        products_df['display_name'] = products_df.apply(
            lambda x: f"{x['pt_code']} - {x['product_name'][:30]}...",
            axis=1
        )
        
        current_filters = self.session_manager.get_filters()
        selected_products = current_filters.get('products', [])
        
        # Ensure selected products exist in current product list
        valid_selected = [p for p in selected_products if p in products_df['product_id'].tolist()]
        
        selected = st.multiselect(
            "Select products",
            options=products_df['product_id'].tolist(),
            default=valid_selected,
            format_func=lambda x: products_df[products_df['product_id'] == x]['display_name'].iloc[0]
                if x in products_df['product_id'].values else str(x),
            placeholder="All products",
            label_visibility="collapsed",
            help="Leave empty for all products",
            key="products_multiselect"
        )
        
        if selected:
            st.caption(f"✓ {len(selected)} selected")
        
        return selected
    
    def _render_brand_selector(self, entity: Optional[str]) -> List[str]:
        """Render brand selection with error handling"""
        try:
            brands = self.data_loader.get_brands(entity)
        except (DataLoadError, ValidationError) as e:
            st.error(f"Failed to load brands: {str(e)}")
            return []
        
        if not brands:
            return []
        
        current_filters = self.session_manager.get_filters()
        selected_brands = current_filters.get('brands', [])
        
        # Ensure selected brands exist in current brand list
        valid_selected = [b for b in selected_brands if b in brands]
        
        return st.multiselect(
            "Select brands",
            options=brands,
            default=valid_selected,
            label_visibility="collapsed",
            placeholder="All brands",
            key="brands_multiselect"
        )
    
    def _render_customer_multiselect(self, entity: Optional[str]) -> List[str]:
        """Render customer selection with error handling"""
        try:
            customers = self.data_loader.get_customers(entity)
        except (DataLoadError, ValidationError) as e:
            st.error(f"Failed to load customers: {str(e)}")
            return []
        
        if not customers:
            return []
        
        # Limit display if too many
        if len(customers) > MAX_MULTISELECT_DISPLAY:
            st.info(f"Showing first {MAX_MULTISELECT_DISPLAY} customers")
            customers = customers[:MAX_MULTISELECT_DISPLAY]
        
        current_filters = self.session_manager.get_filters()
        selected_customers = current_filters.get('customers', [])
        
        # Ensure selected customers exist in current customer list
        valid_selected = [c for c in selected_customers if c in customers]
        
        selected = st.multiselect(
            "Select customers",
            options=customers,
            default=valid_selected,
            label_visibility="collapsed",
            placeholder="All customers",
            key="customers_multiselect"
        )
        
        if selected:
            st.caption(f"✓ {len(selected)} selected")
        
        return selected
    
    def _validate_and_convert_filters(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and convert filter values (lists to tuples for cache stability)
        
        Args:
            filters: Raw filter dictionary
            
        Returns:
            Validated and converted filter dictionary
        """
        # Ensure group_by is valid
        if filters.get('group_by') not in ['product', 'brand']:
            logger.warning(f"Invalid group_by: {filters.get('group_by')}, defaulting to 'product'")
            filters['group_by'] = 'product'
        
        # Ensure at least one source is selected
        if not filters.get('supply_sources'):
            logger.warning("No supply sources selected, defaulting to INVENTORY")
            filters['supply_sources'] = ['INVENTORY']
        if not filters.get('demand_sources'):
            logger.warning("No demand sources selected, defaulting to OC_PENDING")
            filters['demand_sources'] = ['OC_PENDING']
        
        # Convert lists to tuples for stable cache keys
        filters_converted = filters.copy()
        
        if filters.get('products'):
            filters_converted['products_tuple'] = tuple(filters['products'])
        else:
            filters_converted['products_tuple'] = None
        
        if filters.get('brands'):
            filters_converted['brands_tuple'] = tuple(filters['brands'])
        else:
            filters_converted['brands_tuple'] = None
        
        if filters.get('customers'):
            filters_converted['customers_tuple'] = tuple(filters['customers'])
        else:
            filters_converted['customers_tuple'] = None
        
        # Keep original lists for UI display
        filters_converted['products'] = filters.get('products', [])
        filters_converted['brands'] = filters.get('brands', [])
        filters_converted['customers'] = filters.get('customers', [])
        
        return filters_converted
    
    def apply_quick_filter(
        self, 
        gap_df: pd.DataFrame, 
        quick_filter: str, 
        include_safety: bool = False
    ) -> pd.DataFrame:
        """
        Apply quick filter preset to GAP dataframe (post-calculation)
        Context-aware based on safety stock inclusion
        
        Args:
            gap_df: DataFrame with GAP calculations
            quick_filter: Quick filter type
            include_safety: Whether safety stock is included
            
        Returns:
            Filtered DataFrame
        """
        if gap_df.empty or quick_filter == 'all':
            return gap_df
        
        # Define filter mappings based on safety stock context
        if include_safety:
            filter_mappings = {
                'shortage': [
                    'SEVERE_SHORTAGE', 'HIGH_SHORTAGE', 'MODERATE_SHORTAGE', 
                    'BELOW_SAFETY', 'CRITICAL_BREACH'
                ],
                'surplus': [
                    'SEVERE_SURPLUS', 'HIGH_SURPLUS', 'MODERATE_SURPLUS', 
                    'LIGHT_SURPLUS'
                ],
                'balanced': ['BALANCED'],
                'reorder': ['AT_REORDER', 'BELOW_SAFETY', 'CRITICAL_BREACH']
            }
        else:
            filter_mappings = {
                'shortage': ['SEVERE_SHORTAGE', 'HIGH_SHORTAGE', 'MODERATE_SHORTAGE'],
                'surplus': [
                    'SEVERE_SURPLUS', 'HIGH_SURPLUS', 'MODERATE_SURPLUS', 
                    'LIGHT_SURPLUS'
                ],
                'balanced': ['BALANCED']
            }
        
        if quick_filter in filter_mappings:
            filtered = gap_df[gap_df['gap_status'].isin(filter_mappings[quick_filter])]
            logger.debug(f"Applied quick filter '{quick_filter}': {len(gap_df)} -> {len(filtered)} rows")
            return filtered
        
        logger.warning(f"Unknown quick filter: {quick_filter}")
        return gap_df
    
    def get_filter_summary(self, filters: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary of active filters
        
        Args:
            filters: Filter dictionary
            
        Returns:
            Summary string
        """
        summary_parts = []
        
        # Entity
        if filters.get('entity'):
            summary_parts.append(f"Entity: {filters['entity']}")
        
        # Date range
        if filters.get('date_range'):
            date_from, date_to = filters['date_range']
            days_diff = (date_to - date_from).days
            summary_parts.append(f"Period: {days_diff} days")
        
        # Safety stock
        if filters.get('include_safety_stock'):
            summary_parts.append("✓ Safety stock included")
        
        # Sources
        supply_count = len(filters.get('supply_sources', []))
        demand_count = len(filters.get('demand_sources', []))
        if supply_count < len(SUPPLY_SOURCES):
            summary_parts.append(f"Supply: {supply_count}/{len(SUPPLY_SOURCES)}")
        if demand_count < len(DEMAND_SOURCES):
            summary_parts.append(f"Demand: {demand_count}/{len(DEMAND_SOURCES)}")
        
        # Product filters
        if filters.get('products'):
            summary_parts.append(f"Products: {len(filters['products'])}")
        if filters.get('brands'):
            summary_parts.append(f"Brands: {len(filters['brands'])}")
        if filters.get('customers'):
            summary_parts.append(f"Customers: {len(filters['customers'])}")
        
        # Grouping
        group_by_name = GROUP_BY_OPTIONS.get(filters.get('group_by', 'product'), 'Product')
        summary_parts.append(f"By: {group_by_name}")
        
        return " | ".join(summary_parts) if summary_parts else "No filters applied"
    
    def count_active_filters(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Count number of active (non-default) filters
        
        Args:
            filters: Optional filter dict, uses session state if not provided
            
        Returns:
            Number of active filters
        """
        if filters is None:
            filters = self.session_manager.get_filters()
        
        count = 0
        
        # Check each filter against defaults
        if filters.get('entity'):
            count += 1
        if filters.get('products'):
            count += 1
        if filters.get('brands'):
            count += 1
        if filters.get('customers'):
            count += 1
        if filters.get('include_safety_stock'):
            count += 1
        
        # Check if sources differ from defaults
        default_supply = list(SUPPLY_SOURCES.keys())
        default_demand = ['OC_PENDING']
        
        if set(filters.get('supply_sources', [])) != set(default_supply):
            count += 1
        if set(filters.get('demand_sources', [])) != set(default_demand):
            count += 1
        
        return count