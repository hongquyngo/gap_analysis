# utils/gap/filters.py

"""
Filter components module for GAP Analysis System
Provides reusable filter UI components for main page
"""

import streamlit as st
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Optional, List, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


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
            st.session_state.gap_filters = {
                'entity': None,
                'date_range': (date.today(), date.today() + timedelta(days=30)),
                'products': [],
                'brands': [],
                'customers': [],
                'quick_filter': 'all',
                'group_by': 'product',
                'supply_sources': ['INVENTORY', 'CAN_PENDING', 'WAREHOUSE_TRANSFER', 'PURCHASE_ORDER'],
                'demand_sources': ['OC_PENDING', 'FORECAST']
            }
    
    def render_main_page_filters(self) -> Dict[str, Any]:
        """
        Render all filter components on main page
        
        Returns:
            Dictionary containing all filter selections
        """
        filters = {}
        
        # Create expandable filter section
        with st.expander("🔍 **Filters & Settings**", expanded=True):
            
            # First row: Quick filters and grouping
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Quick Filters")
                filters['quick_filter'] = self._render_quick_filters_horizontal()
            
            with col2:
                st.subheader("Group By")
                filters['group_by'] = self._render_grouping_option_horizontal()
            
            st.divider()
            
            # Second row: Entity and Date Range
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                filters['entity'] = self._render_entity_filter()
            
            with col2:
                date_from = st.date_input(
                    "📅 From Date",
                    value=st.session_state.gap_filters.get('date_range', (date.today(), date.today() + timedelta(30)))[0],
                    max_value=date.today() + timedelta(days=365),
                    help="Start date for analysis"
                )
            
            with col3:
                date_to = st.date_input(
                    "📅 To Date",
                    value=st.session_state.gap_filters.get('date_range', (date.today(), date.today() + timedelta(30)))[1],
                    min_value=date_from,
                    max_value=date.today() + timedelta(days=365),
                    help="End date for analysis"
                )
            
            filters['date_range'] = (date_from, date_to)
            
            # Date presets
            st.markdown("**Quick Date Range:**")
            preset_cols = st.columns(6)
            preset_ranges = {
                "Today": 0,
                "7 Days": 7,
                "14 Days": 14,
                "30 Days": 30,
                "60 Days": 60,
                "90 Days": 90
            }
            
            for idx, (label, days) in enumerate(preset_ranges.items()):
                with preset_cols[idx]:
                    if st.button(label, key=f"date_preset_{days}", use_container_width=True):
                        st.session_state.gap_filters['date_range'] = (
                            date.today(),
                            date.today() + timedelta(days=days)
                        )
                        st.rerun()
            
            st.divider()
            
            # Third row: Supply and Demand Source Selection
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📦 Supply Sources")
                filters['supply_sources'] = self._render_supply_source_selection()
            
            with col2:
                st.subheader("📋 Demand Sources")
                filters['demand_sources'] = self._render_demand_source_selection()
            
            st.divider()
            
            # Fourth row: Product filters
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                # Product search and selection
                st.subheader("🔍 Product Selection")
                filters['products'] = self._render_product_filter_compact(filters['entity'])
            
            with col2:
                # Brand filter
                st.subheader("🏷️ Brands")
                filters['brands'] = self._render_brand_filter_compact(filters['entity'])
            
            with col3:
                # Customer filter
                st.subheader("🏢 Customers")
                filters['customers'] = self._render_customer_filter_compact(filters['entity'])
            
            st.divider()
            
            # Filter actions
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                if st.button("🔄 Reset Filters", use_container_width=True):
                    self._reset_filters()
                    st.rerun()
            
            with col2:
                if st.button("📊 Apply Filters", type="primary", use_container_width=True):
                    st.rerun()
            
            with col3:
                # Show filter summary
                active_filters = self._count_active_filters(filters)
                if active_filters > 0:
                    st.info(f"✓ {active_filters} filters active")
        
        # Save to session state
        st.session_state.gap_filters = filters
        
        return filters
    
    def _render_quick_filters_horizontal(self) -> str:
        """Render quick filter preset buttons horizontally"""
        quick_options = {
            'all': '📋 All Items',
            'shortage': '⚠️ Shortage Only',
            'critical': '🚨 Critical Only',
            'surplus': '📦 Surplus Only',
            'balanced': '✅ Balanced Only'
        }
        
        selected = st.radio(
            "Select preset",
            options=list(quick_options.keys()),
            format_func=lambda x: quick_options[x],
            index=list(quick_options.keys()).index(st.session_state.gap_filters.get('quick_filter', 'all')),
            horizontal=True,
            label_visibility="collapsed"
        )
        
        return selected
    
    def _render_grouping_option_horizontal(self) -> str:
        """Render grouping option horizontally"""
        group_options = {
            'product': '📦 Product',
            'brand': '🏷️ Brand',
            'category': '📂 Category'
        }
        
        selected_group = st.radio(
            "Group by",
            options=list(group_options.keys()),
            format_func=lambda x: group_options[x],
            index=list(group_options.keys()).index(
                st.session_state.gap_filters.get('group_by', 'product')
            ),
            horizontal=True,
            label_visibility="collapsed"
        )
        
        return selected_group
    
    def _render_supply_source_selection(self) -> List[str]:
        """Render supply source checkboxes"""
        supply_sources = {
            'INVENTORY': '📦 Inventory (Available Now)',
            'CAN_PENDING': '⏳ CAN Pending (1-3 days)',
            'WAREHOUSE_TRANSFER': '🚚 Warehouse Transfer (2-5 days)',
            'PURCHASE_ORDER': '📝 Purchase Order (7-30 days)'
        }
        
        # Create two columns for checkboxes
        col1, col2 = st.columns(2)
        selected = []
        
        default_selected = st.session_state.gap_filters.get('supply_sources', list(supply_sources.keys()))
        
        for idx, (source, label) in enumerate(supply_sources.items()):
            col = col1 if idx < 2 else col2
            with col:
                if st.checkbox(label, value=source in default_selected, key=f"supply_{source}"):
                    selected.append(source)
        
        if not selected:
            st.warning("⚠️ Select at least one supply source")
            selected = ['INVENTORY']  # Default to inventory if nothing selected
        
        return selected
    
    def _render_demand_source_selection(self) -> List[str]:
        """Render demand source checkboxes"""
        demand_sources = {
            'OC_PENDING': '📋 Confirmed Orders (OC)',
            'FORECAST': '📊 Customer Forecast'
        }
        
        selected = []
        default_selected = st.session_state.gap_filters.get('demand_sources', list(demand_sources.keys()))
        
        for source, label in demand_sources.items():
            if st.checkbox(label, value=source in default_selected, key=f"demand_{source}"):
                selected.append(source)
        
        if not selected:
            st.warning("⚠️ Select at least one demand source")
            selected = ['OC_PENDING']  # Default to OC if nothing selected
        
        return selected
    
    def _render_entity_filter(self) -> Optional[str]:
        """Render entity selection filter"""
        entities = self.data_loader.get_entities()
        
        if not entities:
            st.warning("No entities available")
            return None
        
        # Add "All Entities" option
        entity_options = ["All Entities"] + entities
        
        selected_entity = st.selectbox(
            "🏢 Entity",
            options=entity_options,
            index=0 if st.session_state.gap_filters.get('entity') is None 
                   else entity_options.index(st.session_state.gap_filters.get('entity', 'All Entities')),
            help="Select entity to analyze"
        )
        
        return None if selected_entity == "All Entities" else selected_entity
    
    def _render_product_filter_compact(self, entity: Optional[str]) -> List[int]:
        """Render compact product selection"""
        products_df = self.data_loader.get_products(entity)
        
        if products_df.empty:
            st.info("No products available")
            return []
        
        # Create display strings for products
        products_df['display'] = products_df.apply(
            lambda x: f"{x['pt_code']} - {x['product_name'][:30]}", 
            axis=1
        )
        
        # Search box for products
        search_term = st.text_input(
            "Search",
            placeholder="PT code or name...",
            label_visibility="collapsed"
        )
        
        # Filter products based on search
        if search_term:
            mask = (
                products_df['pt_code'].str.contains(search_term, case=False, na=False) |
                products_df['product_name'].str.contains(search_term, case=False, na=False)
            )
            filtered_products = products_df[mask].head(50)
        else:
            filtered_products = pd.DataFrame()  # Show nothing until search
        
        # Multi-select for products
        if not filtered_products.empty:
            selected_products = st.multiselect(
                "Select products",
                options=filtered_products['product_id'].tolist(),
                format_func=lambda x: filtered_products[
                    filtered_products['product_id'] == x
                ]['display'].iloc[0],
                default=st.session_state.gap_filters.get('products', []),
                label_visibility="collapsed"
            )
        else:
            selected_products = st.session_state.gap_filters.get('products', [])
            if selected_products:
                st.info(f"{len(selected_products)} products selected")
        
        return selected_products
    
    def _render_brand_filter_compact(self, entity: Optional[str]) -> List[str]:
        """Render compact brand selection"""
        brands = self.data_loader.get_brands(entity)
        
        if not brands:
            return []
        
        selected_brands = st.multiselect(
            "Select brands",
            options=brands,
            default=st.session_state.gap_filters.get('brands', []),
            label_visibility="collapsed",
            placeholder="All brands"
        )
        
        return selected_brands
    
    def _render_customer_filter_compact(self, entity: Optional[str]) -> List[str]:
        """Render compact customer selection"""
        customers = self.data_loader.get_customers(entity)
        
        if not customers or len(customers) > 100:
            # Too many customers, use search
            search_customer = st.text_input(
                "Search",
                placeholder="Customer name...",
                label_visibility="collapsed"
            )
            
            if search_customer and customers:
                filtered = [c for c in customers if search_customer.lower() in c.lower()]
                return st.multiselect(
                    "Select",
                    options=filtered[:20],
                    default=st.session_state.gap_filters.get('customers', []),
                    label_visibility="collapsed"
                )
            return []
        else:
            return st.multiselect(
                "Select customers",
                options=customers,
                default=st.session_state.gap_filters.get('customers', []),
                label_visibility="collapsed",
                placeholder="All customers"
            )
    
    def _reset_filters(self):
        """Reset all filters to default values"""
        st.session_state.gap_filters = {
            'entity': None,
            'date_range': (date.today(), date.today() + timedelta(days=30)),
            'products': [],
            'brands': [],
            'customers': [],
            'quick_filter': 'all',
            'group_by': 'product',
            'supply_sources': ['INVENTORY', 'CAN_PENDING', 'WAREHOUSE_TRANSFER', 'PURCHASE_ORDER'],
            'demand_sources': ['OC_PENDING', 'FORECAST']
        }
    
    def _count_active_filters(self, filters: Dict[str, Any]) -> int:
        """Count number of active filters"""
        count = 0
        if filters.get('entity'):
            count += 1
        if filters.get('products'):
            count += 1
        if filters.get('brands'):
            count += 1
        if filters.get('customers'):
            count += 1
        if filters.get('quick_filter') != 'all':
            count += 1
        if len(filters.get('supply_sources', [])) < 4:
            count += 1
        if len(filters.get('demand_sources', [])) < 2:
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
        if quick_filter == 'all':
            return gap_df
        elif quick_filter == 'shortage':
            shortage_statuses = ['SEVERE_SHORTAGE', 'HIGH_SHORTAGE', 'MODERATE_SHORTAGE']
            return gap_df[gap_df['gap_status'].isin(shortage_statuses)]
        elif quick_filter == 'critical':
            critical_statuses = ['SEVERE_SHORTAGE', 'HIGH_SHORTAGE']
            return gap_df[gap_df['gap_status'].isin(critical_statuses)]
        elif quick_filter == 'surplus':
            surplus_statuses = ['SEVERE_SURPLUS', 'HIGH_SURPLUS', 'MODERATE_SURPLUS', 'LIGHT_SURPLUS']
            return gap_df[gap_df['gap_status'].isin(surplus_statuses)]
        elif quick_filter == 'balanced':
            return gap_df[gap_df['gap_status'] == 'BALANCED']
        else:
            return gap_df
    
    def get_filter_summary(self, filters: Dict[str, Any]) -> str:
        """
        Generate a summary string of active filters
        """
        summary_parts = []
        
        if filters.get('entity'):
            summary_parts.append(f"Entity: {filters['entity']}")
        
        if filters.get('date_range'):
            date_from, date_to = filters['date_range']
            summary_parts.append(f"Dates: {date_from} to {date_to}")
        
        supply_count = len(filters.get('supply_sources', []))
        if supply_count < 4:
            summary_parts.append(f"Supply: {supply_count}/4 sources")
        
        demand_count = len(filters.get('demand_sources', []))
        if demand_count < 2:
            summary_parts.append(f"Demand: {demand_count}/2 sources")
        
        if filters.get('brands'):
            summary_parts.append(f"Brands: {len(filters['brands'])} selected")
        
        if filters.get('products'):
            summary_parts.append(f"Products: {len(filters['products'])} selected")
        
        if filters.get('customers'):
            summary_parts.append(f"Customers: {len(filters['customers'])} selected")
        
        if filters.get('quick_filter') != 'all':
            summary_parts.append(f"Filter: {filters['quick_filter'].replace('_', ' ').title()}")
        
        return " | ".join(summary_parts) if summary_parts else "All data - no filters applied"