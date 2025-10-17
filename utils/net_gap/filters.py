# utils/net_gap/filters.py

"""
Filter Components for GAP Analysis - Version 3.3 IMPROVED LAYOUT
- Cleaner, more compact filter layout
- Entity exclusion support added
- Expired inventory moved to Scope section
- Simplified labels and better visual hierarchy
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
    """Manages filter UI for GAP analysis with improved layout"""
    
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
        """Render all filter components with improved layout"""
        filters = {}
        
        # Main configuration card
        with st.expander("🔧 **Data Configuration**", expanded=True):
            self._apply_custom_css()
            
            # Section 1: Scope (includes exclusions)
            self._render_scope_section(filters)
            
            # Section 2: Data Sources
            self._render_data_sources_section(filters)
            
            # Section 3: Analysis Options
            self._render_analysis_section(filters)
            
            st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        
        # Validate and convert filters
        filters = self._validate_and_convert_filters(filters)
        self.session_manager.set_filters(filters)
        
        return filters
    
    def _apply_custom_css(self) -> None:
        """Apply custom CSS for compact layout"""
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
            .filter-row {
                display: flex;
                align-items: center;
                gap: 8px;
                margin-bottom: 12px;
            }
            .excl-checkbox {
                min-width: 80px;
                font-size: 12px;
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
            </style>
        """, unsafe_allow_html=True)
    
    def _render_scope_section(self, filters: Dict[str, Any]) -> None:
        """Render Scope section with all exclusions"""
        st.markdown("<div class='section-header'>🔍 Scope</div>", unsafe_allow_html=True)
        
        current_filters = self.session_manager.get_filters()
        
        # Entity filter with exclusion
        entity_data = self._render_entity_selector_compact(current_filters)
        filters['entity'] = entity_data['selected']
        filters['exclude_entity'] = entity_data['exclude']
        
        # Product filter with exclusion
        product_data = self._render_product_selector_compact(
            filters.get('entity'), current_filters
        )
        filters['products'] = product_data['selected']
        filters['exclude_products'] = product_data['exclude']
        
        # Brand filter with exclusion
        brand_data = self._render_brand_selector_compact(
            filters.get('entity'), current_filters
        )
        filters['brands'] = brand_data['selected']
        filters['exclude_brands'] = brand_data['exclude']
        
        # Expired inventory exclusion
        st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
        filters['exclude_expired_inventory'] = st.checkbox(
            "🗑️ Exclude Expired Inventory",
            value=current_filters.get('exclude_expired_inventory', True),
            key="exclude_expired_checkbox",
            help="Remove expired items from supply calculation"
        )
        
        if filters['exclude_expired_inventory']:
            st.caption("✅ Expired items excluded from analysis")
    
    def _render_entity_selector_compact(
        self, 
        current_filters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Render entity selector with exclusion checkbox on same row"""
        try:
            entities_df = self.data_loader.get_entities_formatted()
        except DataLoadError as e:
            st.error(f"Failed to load entities: {str(e)}")
            return {'selected': None, 'exclude': False}
        
        if entities_df.empty:
            st.warning("No entities available")
            return {'selected': None, 'exclude': False}
        
        # Create display mapping
        entity_display = {}
        entity_map = {}
        
        for _, row in entities_df.iterrows():
            display_text = f"{row['company_code']} | {row['english_name']}"
            entity_display[row['english_name']] = display_text
            entity_map[display_text] = row['english_name']
        
        display_options = ["All Entities"] + list(entity_display.values())
        
        # Get current values
        current_value = current_filters.get('entity')
        current_exclude = current_filters.get('exclude_entity', False)
        
        # Set default index
        if current_value and current_value in entity_display:
            default_index = display_options.index(entity_display[current_value])
        else:
            default_index = 0
        
        # Render in columns
        col1, col2 = st.columns([5, 1])
        
        with col1:
            selected_display = st.selectbox(
                "Entity",
                options=display_options,
                index=default_index,
                key="entity_select",
                help="Select entity or 'All Entities'"
            )
        
        with col2:
            st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
            exclude = st.checkbox(
                "Excl",
                value=current_exclude,
                key="exclude_entity_checkbox",
                help="Exclude selected entity",
                disabled=(selected_display == "All Entities")
            )
        
        # Status caption
        if selected_display != "All Entities":
            mode = "excluded" if exclude else "selected"
            st.caption(f"{'🚫' if exclude else '✓'} Entity {mode}")
        
        selected_entity = entity_map.get(selected_display) if selected_display != "All Entities" else None
        
        return {'selected': selected_entity, 'exclude': exclude}
    
    def _render_product_selector_compact(
        self, 
        entity: Optional[str],
        current_filters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Render product selector with exclusion checkbox on same row"""
        try:
            products_df = self.data_loader.get_products(entity)
        except (DataLoadError, ValidationError) as e:
            st.error(f"Failed to load products: {str(e)}")
            return {'selected': [], 'exclude': False}
        
        if products_df.empty:
            st.info("No products available")
            return {'selected': [], 'exclude': False}
        
        selected_products = current_filters.get('products', [])
        exclude_mode = current_filters.get('exclude_products', False)
        
        valid_selected = [p for p in selected_products if p in products_df['product_id'].tolist()]
        
        # Format display
        products_df['display'] = products_df.apply(
            lambda x: self._format_product_display(
                x['pt_code'], 
                x['product_name'], 
                x.get('package_size', ''),
                x['brand']
            ), 
            axis=1
        )
        product_map = dict(zip(products_df['product_id'], products_df['display']))
        
        # Render in columns
        col1, col2 = st.columns([5, 1])
        
        with col1:
            selected = st.multiselect(
                "Product",
                options=products_df['product_id'].tolist(),
                default=valid_selected,
                format_func=lambda x: product_map.get(x, f"ID: {x}"),
                placeholder=f"All ({len(products_df)} available)",
                key="products_multiselect"
            )
        
        with col2:
            st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
            exclude = st.checkbox(
                "Excl",
                value=exclude_mode,
                key="exclude_products_checkbox",
                help="Exclude selected products",
                disabled=(len(selected) == 0)
            )
        
        # Status caption
        if selected:
            mode = "excluded" if exclude else "selected"
            st.caption(f"{'🚫' if exclude else '✓'} {len(selected)} products {mode}")
        
        return {'selected': selected, 'exclude': exclude}
    
    def _render_brand_selector_compact(
        self, 
        entity: Optional[str],
        current_filters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Render brand selector with exclusion checkbox on same row"""
        try:
            brands = self.data_loader.get_brands(entity)
        except (DataLoadError, ValidationError) as e:
            st.error(f"Failed to load brands: {str(e)}")
            return {'selected': [], 'exclude': False}
        
        if not brands:
            return {'selected': [], 'exclude': False}
        
        selected_brands = current_filters.get('brands', [])
        exclude_mode = current_filters.get('exclude_brands', False)
        
        valid_selected = [b for b in selected_brands if b in brands]
        
        # Render in columns
        col1, col2 = st.columns([5, 1])
        
        with col1:
            selected = st.multiselect(
                "Brand",
                options=brands,
                default=valid_selected,
                placeholder=f"All ({len(brands)} available)",
                key="brands_multiselect"
            )
        
        with col2:
            st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
            exclude = st.checkbox(
                "Excl",
                value=exclude_mode,
                key="exclude_brands_checkbox",
                help="Exclude selected brands",
                disabled=(len(selected) == 0)
            )
        
        # Status caption
        if selected:
            mode = "excluded" if exclude else "selected"
            st.caption(f"{'🚫' if exclude else '✓'} {len(selected)} brands {mode}")
        
        return {'selected': selected, 'exclude': exclude}
    
    def _format_product_display(
        self, 
        pt_code: str, 
        name: str, 
        package_size: str, 
        brand: str,
        max_name_length: int = 30
    ) -> str:
        """Format product display: pt_code | name | package_size (brand)"""
        display_name = name[:max_name_length] + "..." if len(name) > max_name_length else name
        
        parts = [pt_code, display_name]
        
        if package_size and str(package_size).strip():
            parts.append(package_size)
        
        display = " | ".join(parts) + f" ({brand})"
        
        return display
    
    def _render_data_sources_section(self, filters: Dict[str, Any]) -> None:
        """Render Data Sources section"""
        st.markdown("<div class='section-header'>📊 Data Sources</div>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([2.5, 2.5, 2])
        
        with col1:
            self._render_supply_sources_compact(filters)
        
        with col2:
            self._render_demand_sources_compact(filters)
        
        with col3:
            self._render_safety_stock_toggle(filters)
    
    def _render_analysis_section(self, filters: Dict[str, Any]) -> None:
        """Render Analysis section"""
        st.markdown("<div class='section-header'>📈 Analysis</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            self._render_group_by_compact(filters)
        
        with col2:
            self._render_date_range_info()
    
    def _render_supply_sources_compact(self, filters: Dict[str, Any]) -> None:
        """Render supply sources as compact checkbox group"""
        st.markdown("**Supply**", help="Select data sources for supply calculation")
        
        selected = []
        current_filters = self.session_manager.get_filters()
        default_selected = current_filters.get('supply_sources', list(SUPPLY_SOURCES.keys()))
        
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
        
        # Ensure exclusion flags
        filters_converted['exclude_entity'] = filters.get('exclude_entity', False)
        filters_converted['exclude_products'] = filters.get('exclude_products', False)
        filters_converted['exclude_brands'] = filters.get('exclude_brands', False)
        filters_converted['exclude_expired_inventory'] = filters.get('exclude_expired_inventory', True)
        
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
        """Generate concise filter summary with exclusions"""
        summary_parts = []
        
        # Entity
        if filters.get('entity'):
            mode = "excluded" if filters.get('exclude_entity') else "selected"
            summary_parts.append(f"Entity: {filters['entity']} ({mode})")
        else:
            summary_parts.append("All Entities")
        
        # Products and Brands with exclusion indicators
        if filters.get('products'):
            mode = "excluded" if filters.get('exclude_products') else "selected"
            summary_parts.append(f"{len(filters['products'])} products {mode}")
        
        if filters.get('brands'):
            mode = "excluded" if filters.get('exclude_brands') else "selected"
            summary_parts.append(f"{len(filters['brands'])} brands {mode}")
        
        # Expired inventory
        if filters.get('exclude_expired_inventory'):
            summary_parts.append("No expired")
        
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
        
        if filters.get('entity') != defaults['entity']:
            count += 1
        if filters.get('exclude_entity', False):
            count += 1
        if filters.get('products', []) != defaults['products']:
            count += 1
        if filters.get('brands', []) != defaults['brands']:
            count += 1
        if filters.get('exclude_products', False):
            count += 1
        if filters.get('exclude_brands', False):
            count += 1
        if filters.get('exclude_expired_inventory', True) != defaults['exclude_expired_inventory']:
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
            'exclude_entity': False,
            'products': [],
            'brands': [],
            'exclude_products': False,
            'exclude_brands': False,
            'exclude_expired_inventory': True,
            'group_by': 'product',
            'supply_sources': ['INVENTORY', 'CAN_PENDING', 'WAREHOUSE_TRANSFER', 'PURCHASE_ORDER'],
            'demand_sources': ['OC_PENDING'],
            'include_safety_stock': True
        }