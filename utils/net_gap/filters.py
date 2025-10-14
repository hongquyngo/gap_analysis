# utils/net_gap/filters.py

"""
Filter Components for GAP Analysis - Version 3.0 REFACTORED
- REMOVED: Date range filter (uses all data)
- REMOVED: Customer filter (incorrect for GAP calculation)
- Simplified filter configuration for accurate Net GAP analysis
"""

import streamlit as st
import pandas as pd
from typing import Optional, List, Dict, Any, Tuple
import logging

from .session_manager import get_session_manager
from .data_loader import DataLoadError, ValidationError

logger = logging.getLogger(__name__)

MAX_MULTISELECT_DISPLAY = 200

QUICK_FILTER_BASE = {
    'all': {'label': 'All Items', 'help': 'Show all products in the analysis'},
    'shortage': {'label': 'Shortage', 'help': 'Products with supply below demand'},
    'balanced': {'label': 'Balanced', 'help': 'Products with balanced supply and demand'},
    'surplus': {'label': 'Surplus', 'help': 'Products with excess inventory'}
}

QUICK_FILTER_SAFETY = {
    'all': {'label': 'All Items', 'help': 'Show all products'},
    'shortage': {'label': 'Below Requirements', 'help': 'Below demand or safety stock'},
    'balanced': {'label': 'Balanced', 'help': 'Meeting both demand and safety requirements'},
    'surplus': {'label': 'Surplus', 'help': 'Excess inventory above safety levels'},
    'reorder': {'label': 'At Reorder Point', 'help': 'At or below reorder point'}
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
    """Manages filter UI for GAP analysis"""
    
    def __init__(self, data_loader):
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
    
    def render_filters(self) -> Dict[str, Any]:
        """Render all filter components"""
        filters = {}
        
        with st.expander("⚙️ **Data Configuration**", expanded=True):
            # Basic filters
            self._render_basic_filters(filters)
            
            st.divider()
            
            # Source selection
            self._render_source_selection(filters)
            
            st.divider()
            
            # Product scope
            self._render_product_filters(filters)
            
            st.divider()
            
            # Group by
            self._render_group_by(filters)
        
        filters = self._validate_and_convert_filters(filters)
        self.session_manager.set_filters(filters)
        
        return filters
    
    def _render_basic_filters(self, filters: Dict[str, Any]) -> None:
        """Render entity filter and data range info"""
        col1, col2 = st.columns([1, 2])
        
        with col1:
            filters['entity'] = self._render_entity_filter()
        
        with col2:
            # Show data range info (not editable)
            date_range = self.data_loader.get_date_range()
            st.info(
                f"📅 **Data Range:** {date_range['min_date']} to {date_range['max_date']}\n\n"
                f"Analysis uses complete dataset for accurate Net GAP calculation."
            )
    
    def _render_source_selection(self, filters: Dict[str, Any]) -> None:
        """Render supply and demand source selection"""
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
                    help="Consider safety stock requirements",
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
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**Product Selection**")
            filters['products'] = self._render_product_multiselect(filters.get('entity'))
        
        with col2:
            st.markdown("**Brands**")
            filters['brands'] = self._render_brand_selector(filters.get('entity'))
    
    def _render_group_by(self, filters: Dict[str, Any]) -> None:
        """Render group by selection"""
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
            help="Product: detailed per-item | Brand: aggregated by brand"
        )
    
    def _render_entity_filter(self) -> Optional[str]:
        """Render entity selection"""
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
    
    def _format_product_display(self, row: pd.Series) -> str:
        """Format: pt_code | name | package size (Brand)"""
        pt_code = str(row.get('pt_code', 'N/A'))
        product_name = str(row.get('product_name', 'Unknown'))
        package_size = row.get('package_size', '')
        brand = str(row.get('brand', 'No Brand'))
        
        if len(product_name) > 30:
            product_name = product_name[:30] + "..."
        
        display_parts = [pt_code, product_name]
        
        if package_size and str(package_size).strip() and str(package_size) != 'nan':
            display_parts.append(str(package_size))
        
        display_text = " | ".join(display_parts)
        display_text += f" ({brand})"
        
        return display_text
    
    def _render_product_multiselect(self, entity: Optional[str]) -> List[int]:
        """Render product selection"""
        try:
            products_df = self.data_loader.get_products(entity)
        except (DataLoadError, ValidationError) as e:
            st.error(f"Failed to load products: {str(e)}")
            return []
        
        if products_df.empty:
            st.info("No products available")
            return []
        
        products_df['display_name'] = products_df.apply(self._format_product_display, axis=1)
        
        current_filters = self.session_manager.get_filters()
        selected_products = current_filters.get('products', [])
        valid_selected = [p for p in selected_products if p in products_df['product_id'].tolist()]
        
        product_display_map = dict(zip(products_df['product_id'], products_df['display_name']))
        
        selected = st.multiselect(
            "Select products",
            options=products_df['product_id'].tolist(),
            default=valid_selected,
            format_func=lambda x: product_display_map.get(x, f"Product ID: {x}"),
            placeholder="All products",
            label_visibility="collapsed",
            help="Leave empty for all products",
            key="products_multiselect"
        )
        
        if selected:
            st.caption(f"✔ {len(selected)} selected")
        
        return selected
    
    def _render_brand_selector(self, entity: Optional[str]) -> List[str]:
        """Render brand selection"""
        try:
            brands = self.data_loader.get_brands(entity)
        except (DataLoadError, ValidationError) as e:
            st.error(f"Failed to load brands: {str(e)}")
            return []
        
        if not brands:
            return []
        
        current_filters = self.session_manager.get_filters()
        selected_brands = current_filters.get('brands', [])
        valid_selected = [b for b in selected_brands if b in brands]
        
        return st.multiselect(
            "Select brands",
            options=brands,
            default=valid_selected,
            label_visibility="collapsed",
            placeholder="All brands",
            key="brands_multiselect"
        )
    
    def _validate_and_convert_filters(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and convert filter values"""
        if filters.get('group_by') not in ['product', 'brand']:
            logger.warning(f"Invalid group_by: {filters.get('group_by')}")
            filters['group_by'] = 'product'
        
        if not filters.get('supply_sources'):
            filters['supply_sources'] = ['INVENTORY']
        if not filters.get('demand_sources'):
            filters['demand_sources'] = ['OC_PENDING']
        
        filters_converted = filters.copy()
        
        filters_converted['products_tuple'] = tuple(filters['products']) if filters.get('products') else None
        filters_converted['brands_tuple'] = tuple(filters['brands']) if filters.get('brands') else None
        
        filters_converted['products'] = filters.get('products', [])
        filters_converted['brands'] = filters.get('brands', [])
        
        return filters_converted
    
    def apply_quick_filter(
        self, 
        gap_df: pd.DataFrame, 
        quick_filter: str, 
        include_safety: bool = False
    ) -> pd.DataFrame:
        """Apply quick filter to results"""
        if gap_df.empty or quick_filter == 'all':
            return gap_df
        
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
            logger.debug(f"Quick filter '{quick_filter}': {len(gap_df)} -> {len(filtered)}")
            return filtered
        
        return gap_df
    
    def get_filter_summary(self, filters: Dict[str, Any]) -> str:
        """Generate filter summary"""
        summary_parts = []
        
        if filters.get('entity'):
            summary_parts.append(f"Entity: {filters['entity']}")
        
        if filters.get('include_safety_stock'):
            summary_parts.append("✔ Safety stock included")
        
        supply_count = len(filters.get('supply_sources', []))
        demand_count = len(filters.get('demand_sources', []))
        if supply_count < len(SUPPLY_SOURCES):
            summary_parts.append(f"Supply: {supply_count}/{len(SUPPLY_SOURCES)}")
        if demand_count < len(DEMAND_SOURCES):
            summary_parts.append(f"Demand: {demand_count}/{len(DEMAND_SOURCES)}")
        
        if filters.get('products'):
            summary_parts.append(f"Products: {len(filters['products'])}")
        if filters.get('brands'):
            summary_parts.append(f"Brands: {len(filters['brands'])}")
        
        group_by_name = GROUP_BY_OPTIONS.get(filters.get('group_by', 'product'), 'Product')
        summary_parts.append(f"By: {group_by_name}")
        
        return " | ".join(summary_parts) if summary_parts else "No filters applied"
    
    def count_active_filters(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count active filters"""
        if filters is None:
            filters = self.session_manager.get_filters()
        
        count = 0
        
        if filters.get('entity'):
            count += 1
        if filters.get('products'):
            count += 1
        if filters.get('brands'):
            count += 1
        if filters.get('include_safety_stock'):
            count += 1
        
        default_supply = list(SUPPLY_SOURCES.keys())
        default_demand = ['OC_PENDING']
        
        if set(filters.get('supply_sources', [])) != set(default_supply):
            count += 1
        if set(filters.get('demand_sources', [])) != set(default_demand):
            count += 1
        
        return count