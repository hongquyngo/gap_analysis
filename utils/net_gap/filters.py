# utils/net_gap/filters.py

"""
Filter Components for GAP Analysis - Cleaned Version
- Improved UI layout for brand selector
- Added date range display
- Enhanced product/entity display formats
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import logging

from .constants import SUPPLY_SOURCES, DEMAND_SOURCES
from .state import get_state

logger = logging.getLogger(__name__)


class GAPFilters:
    """Filter management with improved UI layout"""
    
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.state = get_state()
        self._safety_available = data_loader.check_safety_stock_availability()
    
    def render_filters(self) -> Dict[str, Any]:
        """Render all filters with improved layout"""
        
        with st.container():
            # Apply compact CSS with tooltip support
            self._apply_compact_css()
            
            # Get current filters from state
            current = self.state.get_filters()
            filters = {}
            
            # Date Range Display (non-editable, informational)
            self._render_date_range_info()
            
            # Scope Section - Adjusted column widths
            st.markdown("#### 🔍 Scope")
            # Adjusted: reduced product width, increased brand width
            col_entity, col_product, col_brand, col_options = st.columns([3, 4, 2, 1])
            
            with col_entity:
                entity_data = self._render_entity_selector(current)
                filters['entity'] = entity_data['selected']
                filters['exclude_entity'] = entity_data['exclude']
            
            with col_product:
                product_data = self._render_product_selector(filters.get('entity'), current)
                filters['products'] = product_data['selected']
                filters['exclude_products'] = product_data['exclude']
            
            with col_brand:
                brand_data = self._render_brand_selector(filters.get('entity'), current)
                filters['brands'] = brand_data['selected']
                filters['exclude_brands'] = brand_data['exclude']
            
            with col_options:
                filters['exclude_expired'] = st.checkbox(
                    "No Exp",
                    value=current.get('exclude_expired', True),
                    key="exclude_expired",
                    help="Exclude expired inventory"
                )
            
            # Data Sources Section
            st.markdown("#### 📊 Data Sources")
            col_supply, col_demand, col_safety = st.columns([4, 3, 3])
            
            with col_supply:
                filters['supply_sources'] = self._render_supply_sources(current)
            
            with col_demand:
                filters['demand_sources'] = self._render_demand_sources(current)
            
            with col_safety:
                filters['include_safety'] = self._render_safety_toggle(current)
            
            # Analysis Options
            col_group, col_info = st.columns([3, 7])
            
            with col_group:
                filters['group_by'] = st.radio(
                    "Group by",
                    options=['product', 'brand'],
                    format_func=lambda x: f"📊 By {x.title()}",
                    index=0 if current.get('group_by') == 'product' else 1,
                    horizontal=True,
                    key="group_by"
                )
            
            with col_info:
                active_filters = self._count_active_filters(filters)
                if active_filters > 0:
                    st.info(f"✓ {active_filters} filters active")
        
        # Convert lists to tuples for caching
        filters['products_tuple'] = tuple(filters.get('products', []))
        filters['brands_tuple'] = tuple(filters.get('brands', []))
        
        return filters
    
    def _apply_compact_css(self):
        """Apply CSS for compact layout with tooltip support"""
        st.markdown("""
            <style>
            /* Compact multiselect */
            .stMultiSelect > div {
                max-height: 38px;
            }
            /* Reduce column gaps */
            [data-testid="column"] {
                padding: 0 0.5rem;
            }
            /* Align elements */
            .stCheckbox {
                margin-top: 28px;
            }
            /* Tooltip support for truncated text */
            .truncate {
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            }
            .truncate:hover {
                overflow: visible;
                white-space: normal;
                background: #f0f0f0;
                z-index: 1000;
                position: relative;
            }
            </style>
        """, unsafe_allow_html=True)
    
    def _render_date_range_info(self):
        """Display data date range information (non-editable)"""
        try:
            # Get date range from data
            today = datetime.now().date()
            # Default range if no data
            min_date = today - timedelta(days=90)
            max_date = today + timedelta(days=90)
            
            st.info(
                f"📅 Data Range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')} "
                f"(All available data will be included)"
            )
        except Exception as e:
            logger.warning(f"Could not display date range: {e}")
    
    def _render_entity_selector(self, current: Dict) -> Dict[str, Any]:
        """Render entity selector with improved format: Code | English name"""
        try:
            entities_df = self.data_loader.get_entities_formatted()
            
            if entities_df.empty:
                st.warning("No entities")
                return {'selected': None, 'exclude': False}
            
            entity_count = len(entities_df)
            
            # Sub-columns
            sub1, sub2 = st.columns([5, 1])
            
            with sub1:
                options = [f"All ({entity_count} available)"]
                entity_map = {}
                
                for _, row in entities_df.iterrows():
                    # Format: Code | English name
                    code = row.get('company_code', 'N/A')
                    name = row['english_name']
                    
                    # Truncate if too long
                    display_name = name[:40] + "..." if len(name) > 40 else name
                    display = f"{code} | {display_name}"
                    
                    options.append(display)
                    entity_map[display] = row['english_name']
                
                # Get current selection
                current_entity = current.get('entity')
                default_idx = 0
                if current_entity:
                    for idx, (display, name) in enumerate(entity_map.items(), 1):
                        if name == current_entity:
                            default_idx = idx
                            break
                
                selected_display = st.selectbox(
                    "Entity",
                    options=options,
                    index=default_idx,
                    key="entity_select",
                    help="Select entity or leave as 'All'"
                )
            
            with sub2:
                st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
                exclude = st.checkbox(
                    "Excl",
                    value=current.get('exclude_entity', False),
                    key="entity_excl",
                    help="Exclude this entity"
                )
            
            # Parse selection
            if selected_display.startswith("All"):
                return {'selected': None, 'exclude': False}
            else:
                entity = entity_map.get(selected_display)
                if entity and exclude:
                    st.caption("🚫 Excluded")
                elif entity:
                    st.caption("✓ Only")
                return {'selected': entity, 'exclude': exclude}
                
        except Exception as e:
            logger.error(f"Error loading entities: {e}")
            st.error("Failed to load entities")
            return {'selected': None, 'exclude': False}
    
    def _render_product_selector(self, entity: Optional[str], current: Dict) -> Dict[str, Any]:
        """Render product multiselect with format: pt_code | Name | Package size (Brand)"""
        try:
            products_df = self.data_loader.get_products(entity)
            
            if products_df.empty:
                return {'selected': [], 'exclude': False}
            
            # Format display with full information
            def format_product_display(row):
                pt_code = row.get('pt_code', 'N/A')
                name = row.get('product_name', 'N/A')
                package = row.get('package_size', '')
                brand = row.get('brand', '')
                
                # Truncate name if too long
                name_display = name[:25] + "..." if len(name) > 25 else name
                
                # Build display string
                display = f"{pt_code} | {name_display}"
                if package:
                    display += f" | {package}"
                if brand:
                    display += f" ({brand})"
                
                return display
            
            products_df['display'] = products_df.apply(format_product_display, axis=1)
            
            # Create mapping with full info for tooltip
            product_map = {}
            for _, row in products_df.iterrows():
                product_map[row['product_id']] = {
                    'display': row['display'],
                    'full_name': row.get('product_name', ''),
                    'pt_code': row.get('pt_code', ''),
                    'package': row.get('package_size', ''),
                    'brand': row.get('brand', '')
                }
            
            # Current selection
            current_products = current.get('products', [])
            valid_selected = [p for p in current_products if p in product_map]
            
            # Sub-columns
            sub1, sub2 = st.columns([5, 1])
            
            with sub1:
                selected = st.multiselect(
                    "Products",
                    options=list(product_map.keys()),
                    default=valid_selected,
                    format_func=lambda x: product_map[x]['display'],
                    placeholder=f"All ({len(products_df)} available)",
                    key="products_multi",
                    help="Select products or leave empty for all"
                )
            
            with sub2:
                st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
                exclude = st.checkbox(
                    "Excl",
                    value=current.get('exclude_products', False),
                    key="products_excl",
                    help="Exclude selected"
                )
            
            if selected:
                st.caption(f"{len(selected)} {'excluded' if exclude else 'selected'}")
            
            return {'selected': selected, 'exclude': exclude}
            
        except Exception as e:
            logger.error(f"Error loading products: {e}")
            return {'selected': [], 'exclude': False}
    
    def _render_brand_selector(self, entity: Optional[str], current: Dict) -> Dict[str, Any]:
        """Render brand multiselect with better layout"""
        try:
            brands = self.data_loader.get_brands(entity)
            
            if not brands:
                return {'selected': [], 'exclude': False}
            
            current_brands = current.get('brands', [])
            valid_selected = [b for b in current_brands if b in brands]
            
            # Sub-columns with better proportion
            sub1, sub2 = st.columns([4, 2])
            
            with sub1:
                selected = st.multiselect(
                    "Brands",
                    options=brands,
                    default=valid_selected,
                    placeholder=f"All ({len(brands)})",
                    key="brands_multi",
                    help="Select brands or leave empty for all"
                )
            
            with sub2:
                st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
                exclude = st.checkbox(
                    "Excl",
                    value=current.get('exclude_brands', False),
                    key="brands_excl",
                    help="Exclude selected brands"
                )
            
            return {'selected': selected, 'exclude': exclude}
            
        except Exception as e:
            logger.error(f"Error loading brands: {e}")
            return {'selected': [], 'exclude': False}
    
    def _render_supply_sources(self, current: Dict) -> List[str]:
        """Render supply source checkboxes inline"""
        st.markdown("**Supply Sources**")
        
        selected = []
        default_selected = current.get('supply_sources', list(SUPPLY_SOURCES.keys()))
        
        cols = st.columns(2)
        sources = list(SUPPLY_SOURCES.items())
        
        for idx, (key, config) in enumerate(sources):
            col_idx = idx % 2
            with cols[col_idx]:
                if st.checkbox(
                    f"{config['icon']} {config['name']}",
                    value=key in default_selected,
                    key=f"supply_{key}",
                    help=f"Lead: {config['lead_days']} days"
                ):
                    selected.append(key)
        
        return selected if selected else ['INVENTORY']
    
    def _render_demand_sources(self, current: Dict) -> List[str]:
        """Render demand source checkboxes"""
        st.markdown("**Demand Sources**")
        
        selected = []
        default_selected = current.get('demand_sources', ['OC_PENDING'])
        
        for key, config in DEMAND_SOURCES.items():
            if st.checkbox(
                f"{config['icon']} {config['name']}",
                value=key in default_selected,
                key=f"demand_{key}"
            ):
                selected.append(key)
        
        return selected if selected else ['OC_PENDING']
    
    def _render_safety_toggle(self, current: Dict) -> bool:
        """Render safety stock toggle"""
        st.markdown("**Safety Stock**")
        
        if self._safety_available:
            return st.toggle(
                "Include Safety",
                value=current.get('include_safety', True),
                key="safety_toggle"
            )
        else:
            st.caption("⚠️ Not configured")
            return False
    
    def _count_active_filters(self, filters: Dict) -> int:
        """Count non-default filters"""
        count = 0
        defaults = self.state.get_default_filters()
        
        # Check each filter
        if filters.get('entity') != defaults['entity']:
            count += 1
        if filters.get('products', []) != defaults['products']:
            count += 1
        if filters.get('brands', []) != defaults['brands']:
            count += 1
        if filters.get('exclude_expired') != defaults['exclude_expired']:
            count += 1
        if set(filters.get('supply_sources', [])) != set(defaults['supply_sources']):
            count += 1
        if set(filters.get('demand_sources', [])) != set(defaults['demand_sources']):
            count += 1
        
        return count