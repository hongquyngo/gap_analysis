# utils/net_gap/filters.py

"""
Filter components module for GAP Analysis System - Version 2.0
- Simplified single-phase filtering
- Added safety stock toggle
- Context-aware quick filters
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

# Dynamic quick filter options (will adapt based on safety stock)
QUICK_FILTER_BASE = {
    'all': 'All Items',
    'shortage': 'Shortage',
    'critical': 'Critical',
    'balanced': 'Balanced',
    'surplus': 'Surplus'
}

QUICK_FILTER_SAFETY = {
    'all': 'All Items',
    'shortage': 'Below Requirements',
    'critical': 'Safety Breach',
    'balanced': 'Balanced',
    'surplus': 'Surplus',
    'reorder': 'At Reorder Point'
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
    """Manages filter UI components for GAP analysis with safety stock support"""
    
    def __init__(self, data_loader):
        """
        Initialize filters with data loader
        
        Args:
            data_loader: Instance of GAPDataLoader for fetching reference data
        """
        self.data_loader = data_loader
        self._safety_stock_available = False
        self._initialize_session_state()
        self._check_safety_stock_availability()
    
    def _initialize_session_state(self):
        """Initialize session state for filter persistence"""
        if 'gap_filters' not in st.session_state:
            st.session_state.gap_filters = self._get_default_filters()
    
    def _check_safety_stock_availability(self):
        """Check if safety stock data is available"""
        try:
            self._safety_stock_available = self.data_loader.check_safety_stock_availability()
        except Exception as e:
            logger.warning(f"Could not check safety stock availability: {e}")
            self._safety_stock_available = False
    
    def _get_default_filters(self) -> Dict[str, Any]:
        """Get default filter values with auto date range"""
        # Get date range from data
        date_range = self._get_data_date_range()
        
        return {
            'entity': None,
            'date_range': date_range,
            'products': [],
            'brands': [],
            'customers': [],
            'quick_filter': 'all',
            'group_by': 'product',
            'supply_sources': list(SUPPLY_SOURCES.keys()),
            'demand_sources': ['OC_PENDING'],  # Only OC by default, no FORECAST
            'include_safety_stock': True  # Default to include safety stock
        }
    
    def _get_data_date_range(self) -> tuple:
        """Get min/max dates from supply and demand data"""
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
        Render all filter components in a single, streamlined interface
        
        Returns:
            Dictionary containing all filter selections
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
            
            # Analysis options
            self._render_analysis_options(filters)
        
        # Validate and save to session state
        filters = self._validate_filters(filters)
        st.session_state.gap_filters = filters
        
        return filters
    
    def _render_basic_filters(self, filters: Dict[str, Any]) -> None:
        """Render entity and date range filters"""
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            filters['entity'] = self._render_entity_filter()
        
        # Get current or default date range
        current_range = st.session_state.gap_filters.get('date_range', self._get_data_date_range())
        
        with col2:
            date_from = st.date_input(
                "From Date",
                value=current_range[0],
                max_value=date.today() + timedelta(days=365),
                help="Start date for analysis"
            )
        
        with col3:
            date_to = st.date_input(
                "To Date",
                value=current_range[1],
                min_value=date_from,
                max_value=date.today() + timedelta(days=365),
                help="End date for analysis"
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
            # Show safety stock toggle if available
            if self._safety_stock_available:
                filters['include_safety_stock'] = st.checkbox(
                    "Include Safety",
                    value=st.session_state.gap_filters.get('include_safety_stock', True),  # Default True
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
        default_selected = st.session_state.gap_filters.get('supply_sources', list(SUPPLY_SOURCES.keys()))
        
        # Compact two-column layout
        col1, col2 = st.columns(2)
        
        for idx, (source, label) in enumerate(SUPPLY_SOURCES.items()):
            col = col1 if idx < 2 else col2
            with col:
                # Shorter labels for compact display
                short_label = label.split('(')[0].strip()
                if st.checkbox(short_label, value=source in default_selected, 
                              key=f"supply_{source}", help=label):
                    selected.append(source)
        
        if not selected:
            st.warning("Select at least one supply source")
            return ['INVENTORY']
        
        return selected
    
    def _render_demand_sources(self) -> List[str]:
        """Render demand source checkboxes"""
        selected = []
        default_selected = st.session_state.gap_filters.get('demand_sources', list(DEMAND_SOURCES.keys()))
        
        for source, label in DEMAND_SOURCES.items():
            short_label = label.split('(')[0].strip()
            if st.checkbox(short_label, value=source in default_selected, 
                          key=f"demand_{source}", help=label):
                selected.append(source)
        
        if not selected:
            st.warning("Select at least one demand source")
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
    
    def _render_analysis_options(self, filters: Dict[str, Any]) -> None:
        """Render analysis options including quick filters and grouping"""
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("**Quick Filter**")
            # Use appropriate filter options based on safety stock
            if filters.get('include_safety_stock', False):
                filter_options = QUICK_FILTER_SAFETY
            else:
                filter_options = QUICK_FILTER_BASE
            
            filters['quick_filter'] = st.radio(
                "Quick filter",
                options=list(filter_options.keys()),
                format_func=lambda x: f"🔍 {filter_options[x]}",
                index=0,  # Default to 'all'
                horizontal=True,
                label_visibility="collapsed"
            )
        
        with col2:
            st.markdown("**Group By**")
            filters['group_by'] = st.radio(
                "Group by",
                options=list(GROUP_BY_OPTIONS.keys()),
                format_func=lambda x: f"📊 {GROUP_BY_OPTIONS[x]}",
                index=list(GROUP_BY_OPTIONS.keys()).index(
                    st.session_state.gap_filters.get('group_by', 'product')
                ),
                horizontal=True,
                label_visibility="collapsed"
            )
    
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
            "Entity",
            options=entity_options,
            index=default_index,
            help="Select entity to analyze"
        )
        
        return None if selected == "All Entities" else selected
    
    def _render_product_multiselect(self, entity: Optional[str]) -> List[int]:
        """Render product selection as multiselect"""
        products_df = self.data_loader.get_products(entity)
        
        if products_df.empty:
            st.info("No products available")
            return []
        
        # Create display names
        products_df['display_name'] = products_df.apply(
            lambda x: f"{x['pt_code']} - {x['product_name'][:30]}...",
            axis=1
        )
        
        selected_products = st.session_state.gap_filters.get('products', [])
        
        selected = st.multiselect(
            "Select products",
            options=products_df['product_id'].tolist(),
            default=[p for p in selected_products if p in products_df['product_id'].tolist()],
            format_func=lambda x: products_df[products_df['product_id'] == x]['display_name'].iloc[0],
            placeholder="All products",
            label_visibility="collapsed",
            help="Leave empty for all products"
        )
        
        if selected:
            st.caption(f"✓ {len(selected)} selected")
        
        return selected
    
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
    
    def _render_customer_multiselect(self, entity: Optional[str]) -> List[str]:
        """Render customer selection"""
        customers = self.data_loader.get_customers(entity)
        
        if not customers:
            return []
        
        # Limit display if too many
        if len(customers) > MAX_MULTISELECT_DISPLAY:
            st.info(f"Showing first {MAX_MULTISELECT_DISPLAY} customers")
            customers = customers[:MAX_MULTISELECT_DISPLAY]
        
        selected = st.multiselect(
            "Select customers",
            options=customers,
            default=st.session_state.gap_filters.get('customers', []),
            label_visibility="collapsed",
            placeholder="All customers"
        )
        
        if selected:
            st.caption(f"✓ {len(selected)} selected")
        
        return selected
    
    def _validate_filters(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean filter values"""
        # Ensure group_by is valid
        if filters.get('group_by') not in ['product', 'brand']:
            filters['group_by'] = 'product'
        
        # Ensure quick_filter is valid
        valid_quick_filters = ['all', 'shortage', 'critical', 'balanced', 'surplus']
        if filters.get('include_safety_stock'):
            valid_quick_filters.append('reorder')
        
        if filters.get('quick_filter') not in valid_quick_filters:
            filters['quick_filter'] = 'all'
        
        # Ensure at least one source is selected
        if not filters.get('supply_sources'):
            filters['supply_sources'] = ['INVENTORY']
        if not filters.get('demand_sources'):
            filters['demand_sources'] = ['OC_PENDING']
        
        return filters
    
    def apply_quick_filter(self, gap_df: pd.DataFrame, quick_filter: str, 
                          include_safety: bool = False) -> pd.DataFrame:
        """
        Apply quick filter preset to GAP dataframe
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
            # Safety-aware filter mappings
            filter_mappings = {
                'shortage': ['SEVERE_SHORTAGE', 'HIGH_SHORTAGE', 'MODERATE_SHORTAGE', 
                           'BELOW_SAFETY', 'CRITICAL_BREACH'],
                'critical': ['SEVERE_SHORTAGE', 'HIGH_SHORTAGE', 'CRITICAL_BREACH', 
                           'BELOW_SAFETY', 'HAS_EXPIRED'],
                'surplus': ['SEVERE_SURPLUS', 'HIGH_SURPLUS', 'MODERATE_SURPLUS', 
                          'LIGHT_SURPLUS'],
                'balanced': ['BALANCED'],
                'reorder': ['AT_REORDER', 'BELOW_SAFETY', 'CRITICAL_BREACH']
            }
        else:
            # Traditional filter mappings
            filter_mappings = {
                'shortage': ['SEVERE_SHORTAGE', 'HIGH_SHORTAGE', 'MODERATE_SHORTAGE'],
                'critical': ['SEVERE_SHORTAGE', 'HIGH_SHORTAGE'],
                'surplus': ['SEVERE_SURPLUS', 'HIGH_SURPLUS', 'MODERATE_SURPLUS', 
                          'LIGHT_SURPLUS'],
                'balanced': ['BALANCED']
            }
        
        if quick_filter in filter_mappings:
            return gap_df[gap_df['gap_status'].isin(filter_mappings[quick_filter])]
        
        return gap_df
    
    def get_filter_summary(self, filters: Dict[str, Any]) -> str:
        """Generate a human-readable summary of active filters"""
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
        
        # Quick filter
        if filters.get('quick_filter') != 'all':
            if filters.get('include_safety_stock'):
                filter_name = QUICK_FILTER_SAFETY[filters['quick_filter']]
            else:
                filter_name = QUICK_FILTER_BASE[filters['quick_filter']]
            summary_parts.append(f"Filter: {filter_name}")
        
        # Grouping
        summary_parts.append(f"By: {GROUP_BY_OPTIONS[filters.get('group_by', 'product')]}")
        
        return " | ".join(summary_parts) if summary_parts else "No filters applied"
    
    def render_action_buttons(self) -> tuple[bool, bool]:
        """
        Render action buttons for filter operations
        
        Returns:
            Tuple of (reset_clicked, apply_clicked)
        """
        col1, col2, col3 = st.columns([1, 1, 2])
        
        reset_clicked = False
        apply_clicked = False
        
        with col1:
            if st.button("🔄 Reset", use_container_width=True, help="Reset all filters"):
                st.session_state.gap_filters = self._get_default_filters()
                reset_clicked = True
        
        with col2:
            if st.button("📊 Calculate GAP", type="primary", use_container_width=True):
                apply_clicked = True
        
        with col3:
            # Show filter count
            active_count = self._count_active_filters(st.session_state.gap_filters)
            if active_count > 0:
                st.success(f"✓ {active_count} filters active")
        
        return reset_clicked, apply_clicked
    
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
        if filters.get('include_safety_stock', False):
            count += 1
        if set(filters.get('supply_sources', [])) != set(defaults['supply_sources']):
            count += 1
        if set(filters.get('demand_sources', [])) != set(defaults['demand_sources']):
            count += 1
        
        return count