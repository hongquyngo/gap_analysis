# utils/net_gap/filters.py

"""
Filter Components for GAP Analysis - Version 3.1 REDESIGNED
- Compact Card Layout for better UX
- Organized sections: Scope, Data Sources, Analysis
- Improved visual hierarchy and space efficiency
"""

import streamlit as st
import pandas as pd
from typing import Optional, List, Dict, Any, Tuple
import logging

from .session_manager import get_session_manager
from .data_loader import DataLoadError, ValidationError

logger = logging.getLogger(__name__)

# Constants
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
    'INVENTORY': {'name': 'Inventory', 'icon': '📦', 'days': '0 days'},
    'CAN_PENDING': {'name': 'CAN Pending', 'icon': '📋', 'days': '1-3 days'},
    'WAREHOUSE_TRANSFER': {'name': 'Transfer', 'icon': '🚛', 'days': '2-5 days'},
    'PURCHASE_ORDER': {'name': 'Purchase Order', 'icon': '📝', 'days': '7-30 days'}
}

DEMAND_SOURCES = {
    'OC_PENDING': {'name': 'Confirmed Orders', 'icon': '✓', 'desc': 'Customer orders'},
    'FORECAST': {'name': 'Customer Forecast', 'icon': '📊', 'desc': 'Predicted demand'}
}


class GAPFilters:
    """Manages filter UI for GAP analysis with Compact Card Layout"""
    
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
        """Render all filter components with Compact Card Layout"""
        filters = {}
        
        # Main configuration card
        with st.expander("🔧 **Data Configuration**", expanded=True):
            # Apply custom CSS for better styling
            self._apply_custom_css()
            
            # Section 1: Scope
            self._render_scope_section(filters)
            
            # Section 2: Data Sources
            self._render_data_sources_section(filters)
            
            # Section 3: Analysis Options
            self._render_analysis_section(filters)
            
            # Action buttons at the bottom of the card
            st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        
        # Validate and convert filters
        filters = self._validate_and_convert_filters(filters)
        self.session_manager.set_filters(filters)
        
        return filters
    
    def _apply_custom_css(self) -> None:
        """Apply custom CSS for better layout"""
        st.markdown("""
            <style>
            .section-header {
                font-size: 14px;
                font-weight: 600;
                color: #1f2937;
                margin-bottom: 10px;
                padding: 8px 12px;
                background: linear-gradient(90deg, #f3f4f6 0%, transparent 100%);
                border-left: 3px solid #3b82f6;
            }
            .subsection-box {
                background: #fafafa;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                padding: 12px;
                margin-bottom: 12px;
            }
            .info-badge {
                display: inline-block;
                padding: 2px 8px;
                background: #eff6ff;
                color: #1e40af;
                border-radius: 12px;
                font-size: 12px;
                margin-left: 8px;
            }
            .source-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 10px;
            }
            </style>
        """, unsafe_allow_html=True)
    
    def _render_scope_section(self, filters: Dict[str, Any]) -> None:
        """Render Scope section with entity, products, and brands"""
        st.markdown("<div class='section-header'>📍 Scope</div>", unsafe_allow_html=True)
        
        with st.container():
            col1, col2, col3 = st.columns([2, 2, 2])
            
            with col1:
                filters['entity'] = self._render_entity_selector()
            
            with col2:
                filters['products'] = self._render_product_selector_compact(filters.get('entity'))
            
            with col3:
                filters['brands'] = self._render_brand_selector_compact(filters.get('entity'))
    
    def _render_data_sources_section(self, filters: Dict[str, Any]) -> None:
        """Render Data Sources section with supply, demand, and safety options"""
        st.markdown("<div class='section-header'>📊 Data Sources</div>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([2.5, 2.5, 2])
        
        with col1:
            self._render_supply_sources_compact(filters)
        
        with col2:
            self._render_demand_sources_compact(filters)
        
        with col3:
            self._render_safety_stock_toggle(filters)
    
    def _render_analysis_section(self, filters: Dict[str, Any]) -> None:
        """Render Analysis section with grouping and date range info"""
        st.markdown("<div class='section-header'>📈 Analysis</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            self._render_group_by_compact(filters)
        
        with col2:
            self._render_date_range_info()
    
    def _render_entity_selector(self) -> Optional[str]:
        """Render compact entity selector"""
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
            key="entity_select",
            help="Select specific entity or analyze all"
        )
        
        return None if selected == "All Entities" else selected
    
    def _render_product_selector_compact(self, entity: Optional[str]) -> List[int]:
        """Render compact product selector with count"""
        try:
            products_df = self.data_loader.get_products(entity)
        except (DataLoadError, ValidationError) as e:
            st.error(f"Failed to load products: {str(e)}")
            return []
        
        if products_df.empty:
            st.info("No products available")
            return []
        
        current_filters = self.session_manager.get_filters()
        selected_products = current_filters.get('products', [])
        valid_selected = [p for p in selected_products if p in products_df['product_id'].tolist()]
        
        # Create display mapping
        products_df['display'] = products_df.apply(
            lambda x: f"{x['pt_code']} - {x['product_name'][:20]}...", axis=1
        )
        product_map = dict(zip(products_df['product_id'], products_df['display']))
        
        selected = st.multiselect(
            f"Products",
            options=products_df['product_id'].tolist(),
            default=valid_selected,
            format_func=lambda x: product_map.get(x, f"ID: {x}"),
            placeholder=f"All ({len(products_df)} available)",
            key="products_multiselect"
        )
        
        if selected:
            st.caption(f"✓ {len(selected)} selected")
        
        return selected
    
    def _render_brand_selector_compact(self, entity: Optional[str]) -> List[str]:
        """Render compact brand selector"""
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
        
        selected = st.multiselect(
            "Brands",
            options=brands,
            default=valid_selected,
            placeholder=f"All ({len(brands)} available)",
            key="brands_multiselect"
        )
        
        if selected:
            st.caption(f"✓ {len(selected)} selected")
        
        return selected
    
    def _render_supply_sources_compact(self, filters: Dict[str, Any]) -> None:
        """Render supply sources as compact checkbox group"""
        st.markdown("**Supply**", help="Select data sources for supply calculation")
        
        selected = []
        current_filters = self.session_manager.get_filters()
        default_selected = current_filters.get('supply_sources', list(SUPPLY_SOURCES.keys()))
        
        # Create 2x2 grid
        col1, col2 = st.columns(2)
        cols = [col1, col2, col1, col2]
        
        for idx, (source_key, source_info) in enumerate(SUPPLY_SOURCES.items()):
            with cols[idx]:
                if st.checkbox(
                    f"{source_info['icon']} {source_info['name']}",
                    value=source_key in default_selected,
                    key=f"supply_{source_key}",
                    help=f"Lead time: {source_info['days']}"
                ):
                    selected.append(source_key)
        
        filters['supply_sources'] = selected if selected else ['INVENTORY']
        
        # Show summary
        if len(selected) == len(SUPPLY_SOURCES):
            st.caption("✅ All sources included")
        elif selected:
            st.caption(f"📊 {len(selected)}/{len(SUPPLY_SOURCES)} sources")
    
    def _render_demand_sources_compact(self, filters: Dict[str, Any]) -> None:
        """Render demand sources as compact checkbox group"""
        st.markdown("**Demand**", help="Select data sources for demand calculation")
        
        selected = []
        current_filters = self.session_manager.get_filters()
        default_selected = current_filters.get('demand_sources', ['OC_PENDING'])
        
        for source_key, source_info in DEMAND_SOURCES.items():
            if st.checkbox(
                f"{source_info['icon']} {source_info['name']}",
                value=source_key in default_selected,
                key=f"demand_{source_key}",
                help=source_info['desc']
            ):
                selected.append(source_key)
        
        filters['demand_sources'] = selected if selected else ['OC_PENDING']
        
        # Show summary
        if len(selected) == len(DEMAND_SOURCES):
            st.caption("✅ All sources included")
        elif selected:
            st.caption(f"📊 {len(selected)}/{len(DEMAND_SOURCES)} sources")
    
    def _render_safety_stock_toggle(self, filters: Dict[str, Any]) -> None:
        """Render safety stock toggle with status"""
        st.markdown("**Safety Stock**")
        
        current_filters = self.session_manager.get_filters()
        
        if self._safety_stock_available:
            filters['include_safety_stock'] = st.toggle(
                "Include Safety",
                value=current_filters.get('include_safety_stock', True),
                key="safety_toggle",
                help="Consider minimum inventory requirements"
            )
            
            if filters['include_safety_stock']:
                st.caption("✅ Safety rules active")
            else:
                st.caption("⚪ Not included")
        else:
            filters['include_safety_stock'] = False
            st.caption("❌ Not configured")
            st.caption("Safety stock data unavailable")
    
    def _render_group_by_compact(self, filters: Dict[str, Any]) -> None:
        """Render compact group by selector"""
        current_filters = self.session_manager.get_filters()
        current_group_by = current_filters.get('group_by', 'product')
        
        filters['group_by'] = st.radio(
            "Group by",
            options=list(GROUP_BY_OPTIONS.keys()),
            format_func=lambda x: f"📊 {GROUP_BY_OPTIONS[x]}",
            index=list(GROUP_BY_OPTIONS.keys()).index(current_group_by),
            horizontal=True,
            key="group_by_radio",
            help="Choose aggregation level"
        )
    
    def _render_date_range_info(self) -> None:
        """Display date range information"""
        date_range = self.data_loader.get_date_range()
        
        st.info(
            f"📅 **Data Range**: {date_range['min_date']} to {date_range['max_date']}\n\n"
            f"Analysis uses complete dataset for accurate Net GAP calculation."
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
        
        # Convert to tuples for caching
        filters_converted['products_tuple'] = tuple(filters['products']) if filters.get('products') else None
        filters_converted['brands_tuple'] = tuple(filters['brands']) if filters.get('brands') else None
        
        # Keep lists for display
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
        """Generate concise filter summary"""
        summary_parts = []
        
        # Entity
        if filters.get('entity'):
            summary_parts.append(f"Entity: {filters['entity']}")
        else:
            summary_parts.append("All Entities")
        
        # Products and Brands count
        if filters.get('products'):
            summary_parts.append(f"{len(filters['products'])} products")
        if filters.get('brands'):
            summary_parts.append(f"{len(filters['brands'])} brands")
        
        # Sources summary
        supply_count = len(filters.get('supply_sources', []))
        demand_count = len(filters.get('demand_sources', []))
        summary_parts.append(f"Supply: {supply_count}/4, Demand: {demand_count}/2")
        
        # Safety stock
        if filters.get('include_safety_stock'):
            summary_parts.append("✓ Safety")
        
        # Group by
        group_by = GROUP_BY_OPTIONS.get(filters.get('group_by', 'product'))
        summary_parts.append(f"By {group_by}")
        
        return " | ".join(summary_parts)
    
    def count_active_filters(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count number of active (non-default) filters"""
        if filters is None:
            filters = self.session_manager.get_filters()
        
        count = 0
        defaults = self._get_default_filters()
        
        # Check each filter against defaults
        if filters.get('entity') != defaults['entity']:
            count += 1
        if filters.get('products', []) != defaults['products']:
            count += 1
        if filters.get('brands', []) != defaults['brands']:
            count += 1
        if set(filters.get('supply_sources', [])) != set(defaults['supply_sources']):
            count += 1
        if set(filters.get('demand_sources', [])) != set(defaults['demand_sources']):
            count += 1
        if filters.get('include_safety_stock') != defaults['include_safety_stock']:
            count += 1
        if filters.get('group_by') != defaults['group_by']:
            count += 1
        
        return count
    
    def _get_default_filters(self) -> Dict[str, Any]:
        """Get default filter configuration"""
        return {
            'entity': None,
            'products': [],
            'brands': [],
            'group_by': 'product',
            'supply_sources': ['INVENTORY', 'CAN_PENDING', 'WAREHOUSE_TRANSFER', 'PURCHASE_ORDER'],
            'demand_sources': ['OC_PENDING'],
            'include_safety_stock': True
        }


def render_action_buttons(session_manager, filters) -> Tuple[bool, bool]:
    """Render action buttons with improved layout"""
    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
    
    force_recalc = False
    
    with col1:
        if st.button("🔄 Reset", use_container_width=True, help="Reset all filters to defaults"):
            session_manager.reset_filters()
            session_manager.clear_gap_calculation()
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear", use_container_width=True, help="Clear all selections"):
            session_manager.reset_filters()
            st.rerun()
    
    with col3:
        active_count = filters.count_active_filters()
        if active_count > 0:
            st.info(f"✓ {active_count} active")
        else:
            st.info("Default config")
    
    with col4:
        calculate_clicked = st.button(
            f"📊 Calculate GAP →",
            type="primary",
            use_container_width=True,
            help="Run GAP analysis with current configuration"
        )
        if calculate_clicked:
            force_recalc = True
    
    return calculate_clicked, force_recalc