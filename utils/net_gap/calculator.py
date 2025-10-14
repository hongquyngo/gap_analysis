# utils/net_gap/calculator.py

"""
GAP Calculator - Version 3.0 REFACTORED
- Returns GAPCalculationResult (single source of truth)
- Pre-calculates customer impact during GAP calculation
- Removed defensive type conversions (data_loader ensures types)
- Fixed customer count calculation (no fallbacks)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

from .calculation_result import GAPCalculationResult, SourceData, CustomerImpactData

logger = logging.getLogger(__name__)

# GAP thresholds
COVERAGE_THRESHOLDS = {
    'SEVERE_SURPLUS': 3.0,
    'HIGH_SURPLUS': 2.0,
    'MODERATE_SURPLUS': 1.5,
    'LIGHT_SURPLUS': 1.1,
    'BALANCED_HIGH': 1.1,
    'BALANCED_LOW': 0.9,
    'MODERATE_SHORTAGE': 0.7,
    'HIGH_SHORTAGE': 0.5,
    'SEVERE_SHORTAGE': 0.0
}

SAFETY_THRESHOLDS = {
    'CRITICAL_BREACH': 0.5,
    'BELOW_SAFETY': 1.0,
    'EXCESS_STOCK': 3.0
}

PRIORITY_LEVELS = {
    'CRITICAL': 1,
    'HIGH': 2,
    'MEDIUM': 3,
    'LOW': 4,
    'OK': 99
}


class GAPCalculator:
    """Handles GAP calculations with safety stock support"""
    
    def __init__(self):
        self.supply_sources = ['INVENTORY', 'CAN_PENDING', 'WAREHOUSE_TRANSFER', 'PURCHASE_ORDER']
        self.demand_sources = ['OC_PENDING', 'FORECAST']
        self._include_safety = False
    
    def calculate_net_gap(
        self,
        supply_df: pd.DataFrame,
        demand_df: pd.DataFrame,
        safety_stock_df: Optional[pd.DataFrame] = None,
        group_by: str = 'product',
        selected_supply_sources: Optional[List[str]] = None,
        selected_demand_sources: Optional[List[str]] = None,
        include_safety_stock: bool = False
    ) -> GAPCalculationResult:
        """
        Calculate net GAP and return complete result object
        
        Returns:
            GAPCalculationResult with gap_df, metrics, source_data, and customer_impact
        """
        try:
            self._include_safety = include_safety_stock and safety_stock_df is not None and not safety_stock_df.empty
            
            if group_by not in ['product', 'brand']:
                logger.warning(f"Invalid group_by: {group_by}, defaulting to 'product'")
                group_by = 'product'
            
            # Filter by selected sources
            if selected_supply_sources:
                supply_df = supply_df[supply_df['supply_source'].isin(selected_supply_sources)].copy()
            
            if selected_demand_sources:
                demand_df = demand_df[demand_df['demand_source'].isin(selected_demand_sources)].copy()
            
            # Store source data
            source_data = SourceData(
                supply_df=supply_df.copy(),
                demand_df=demand_df.copy(),
                safety_stock_df=safety_stock_df.copy() if self._include_safety else None
            )
            
            # Get grouping columns
            group_cols = self._get_group_columns(group_by)
            join_keys = self._get_join_keys(group_by)
            
            # Aggregate supply and demand
            supply_agg = self._aggregate_supply(supply_df, group_cols)
            demand_agg = self._aggregate_demand(demand_df, group_cols)
            
            # Merge
            gap_df = self._smart_merge(supply_agg, demand_agg, join_keys, group_cols)
            
            # Merge safety stock if included
            if self._include_safety:
                gap_df = self._merge_safety_stock(gap_df, safety_stock_df, join_keys)
            
            # Fill NaN values
            numeric_cols = gap_df.select_dtypes(include=[np.number]).columns.tolist()
            gap_df[numeric_cols] = gap_df[numeric_cols].fillna(0)
            
            # Calculate GAP metrics
            gap_df = self._calculate_gap_metrics(gap_df)
            
            # Sort by priority
            gap_df = gap_df.sort_values(
                by=['priority', 'net_gap'],
                ascending=[True, True]
            )
            
            # Calculate summary metrics
            metrics = self.get_summary_metrics(gap_df, demand_df)
            
            # Pre-calculate customer impact
            customer_impact = self._calculate_customer_impact(gap_df, demand_df)
            
            # Store filters used for this calculation
            filters_used = {
                'group_by': group_by,
                'supply_sources': selected_supply_sources or self.supply_sources,
                'demand_sources': selected_demand_sources or self.demand_sources,
                'include_safety_stock': include_safety_stock
            }
            
            logger.info(f"GAP calculation completed: {len(gap_df)} items, "
                       f"{metrics.get('affected_customers', 0)} affected customers")
            
            return GAPCalculationResult(
                gap_df=gap_df,
                metrics=metrics,
                source_data=source_data,
                customer_impact=customer_impact,
                filters_used=filters_used
            )
            
        except Exception as e:
            logger.error(f"Error calculating net GAP: {e}", exc_info=True)
            raise
    
    def _get_group_columns(self, group_by: str) -> List[str]:
        """Get grouping columns"""
        if group_by == 'product':
            return ['product_id', 'product_name', 'pt_code', 'brand', 'standard_uom']
        elif group_by == 'brand':
            return ['brand']
        else:
            return ['product_id', 'product_name', 'pt_code', 'brand', 'standard_uom']
    
    def _get_join_keys(self, group_by: str) -> List[str]:
        """Get JOIN key columns (primary keys only)"""
        if group_by == 'product':
            return ['product_id']
        elif group_by == 'brand':
            return ['brand']
        else:
            return ['product_id']
    
    def _smart_merge(
        self, 
        supply_agg: pd.DataFrame, 
        demand_agg: pd.DataFrame,
        join_keys: List[str],
        group_cols: List[str]
    ) -> pd.DataFrame:
        """Smart merge handling mismatched descriptive columns"""
        desc_cols = [col for col in group_cols if col not in join_keys]
        
        gap_df = pd.merge(
            supply_agg,
            demand_agg,
            on=join_keys,
            how='outer',
            suffixes=('_supply', '_demand')
        )
        
        # Handle descriptive columns - prefer supply, fallback to demand
        for col in desc_cols:
            supply_col = f'{col}_supply'
            demand_col = f'{col}_demand'
            
            if supply_col in gap_df.columns and demand_col in gap_df.columns:
                gap_df[col] = gap_df[supply_col].fillna(gap_df[demand_col])
                gap_df.drop([supply_col, demand_col], axis=1, inplace=True)
            elif supply_col in gap_df.columns:
                gap_df[col] = gap_df[supply_col]
                gap_df.drop([supply_col], axis=1, inplace=True)
            elif demand_col in gap_df.columns:
                gap_df[col] = gap_df[demand_col]
                gap_df.drop([demand_col], axis=1, inplace=True)
        
        return gap_df
    
    def _aggregate_supply(self, supply_df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
        """Aggregate supply data"""
        if supply_df.empty:
            return pd.DataFrame(columns=group_cols + ['total_supply'])
        
        supply_df = supply_df.copy()
        
        # Filter out expired items
        today = pd.Timestamp.now().normalize()
        if 'expiry_date' in supply_df.columns:
            supply_df['expiry_date'] = pd.to_datetime(supply_df['expiry_date'], errors='coerce')
            supply_df = supply_df[
                (supply_df['expiry_date'].isna()) | 
                (supply_df['expiry_date'] > today)
            ].copy()
        
        # Aggregation dictionary
        agg_dict = {
            'available_quantity': 'sum',
            'total_value_usd': 'sum'
        }
        
        # Source-specific columns
        for source in self.supply_sources:
            col_name = f'supply_{source.lower()}'
            supply_df[col_name] = np.where(
                supply_df['supply_source'] == source,
                supply_df['available_quantity'],
                0
            )
            agg_dict[col_name] = 'sum'
        
        # Weighted average cost
        if 'unit_cost_usd' in supply_df.columns:
            supply_df['cost_x_qty'] = supply_df['unit_cost_usd'] * supply_df['available_quantity']
            agg_dict['cost_x_qty'] = 'sum'
        
        # Expiry tracking
        if 'days_to_expiry' in supply_df.columns:
            supply_df['expired_qty'] = np.where(
                supply_df['days_to_expiry'] <= 0,
                supply_df['available_quantity'],
                0
            )
            supply_df['near_expiry_qty'] = np.where(
                (supply_df['days_to_expiry'] > 0) & (supply_df['days_to_expiry'] <= 30),
                supply_df['available_quantity'],
                0
            )
            agg_dict['expired_qty'] = 'sum'
            agg_dict['near_expiry_qty'] = 'sum'
        
        # Aggregate
        supply_agg = supply_df.groupby(group_cols, as_index=False).agg(agg_dict)
        
        # Rename
        supply_agg.rename(columns={
            'available_quantity': 'total_supply',
            'total_value_usd': 'supply_value_usd'
        }, inplace=True)
        
        # Weighted average unit cost
        if 'cost_x_qty' in supply_agg.columns:
            supply_agg['avg_unit_cost_usd'] = np.where(
                supply_agg['total_supply'] > 0,
                supply_agg['cost_x_qty'] / supply_agg['total_supply'],
                0
            )
            supply_agg.drop(columns=['cost_x_qty'], inplace=True)
        
        return supply_agg
    
    def _aggregate_demand(self, demand_df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
        """Aggregate demand data"""
        if demand_df.empty:
            return pd.DataFrame(columns=group_cols + ['total_demand'])
        
        demand_df = demand_df.copy()
        
        agg_dict = {
            'required_quantity': 'sum',
            'total_value_usd': 'sum',
            'customer': 'nunique'
        }
        
        # Optional columns
        optional_aggs = {
            'allocated_quantity': 'sum',
            'unallocated_quantity': 'sum',
            'over_committed_qty_standard': 'sum',
            'days_to_required': 'mean'
        }
        
        for col, agg_func in optional_aggs.items():
            if col in demand_df.columns:
                agg_dict[col] = agg_func
        
        # Source-specific columns
        for source in self.demand_sources:
            col_name = f'demand_{source.lower()}'
            demand_df[col_name] = np.where(
                demand_df['demand_source'] == source,
                demand_df['required_quantity'],
                0
            )
            agg_dict[col_name] = 'sum'
        
        # Urgency levels
        if 'urgency_level' in demand_df.columns:
            for urgency in ['OVERDUE', 'URGENT', 'UPCOMING']:
                col_name = f'count_{urgency.lower()}'
                demand_df[col_name] = (demand_df['urgency_level'] == urgency).astype(int)
                agg_dict[col_name] = 'sum'
        
        # Aggregate
        demand_agg = demand_df.groupby(group_cols, as_index=False).agg(agg_dict)
        
        # Rename
        demand_agg.rename(columns={
            'required_quantity': 'total_demand',
            'total_value_usd': 'demand_value_usd',
            'customer': 'customer_count',
            'days_to_required': 'avg_days_to_required'
        }, inplace=True)
        
        # Weighted average selling price
        demand_agg['avg_selling_price_usd'] = np.where(
            demand_agg['total_demand'] > 0,
            demand_agg['demand_value_usd'] / demand_agg['total_demand'],
            0
        )
        
        return demand_agg
    
    def _merge_safety_stock(
        self, 
        gap_df: pd.DataFrame, 
        safety_stock_df: pd.DataFrame, 
        join_keys: List[str]
    ) -> pd.DataFrame:
        """Merge safety stock data"""
        if safety_stock_df.empty:
            gap_df['safety_stock_qty'] = 0
            gap_df['reorder_point'] = 0
            gap_df['avg_daily_demand'] = 0
            return gap_df
        
        safety_cols = ['product_id', 'safety_stock_qty', 'reorder_point', 'avg_daily_demand']
        safety_data = safety_stock_df[safety_cols].copy()
        
        if 'product_id' in join_keys:
            gap_df = pd.merge(gap_df, safety_data, on='product_id', how='left')
        else:
            gap_df['safety_stock_qty'] = 0
            gap_df['reorder_point'] = 0
            gap_df['avg_daily_demand'] = 0
        
        gap_df['safety_stock_qty'] = gap_df['safety_stock_qty'].fillna(0)
        gap_df['reorder_point'] = gap_df['reorder_point'].fillna(0)
        gap_df['avg_daily_demand'] = gap_df['avg_daily_demand'].fillna(0)
        
        return gap_df
    
    def _calculate_gap_metrics(self, gap_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate GAP metrics"""
        # Determine effective available supply
        if self._include_safety:
            gap_df['available_supply'] = np.maximum(
                0,
                gap_df['total_supply'] - gap_df.get('safety_stock_qty', 0)
            )
        else:
            gap_df['available_supply'] = gap_df['total_supply']
        
        # Basic GAP
        gap_df['net_gap'] = gap_df['available_supply'] - gap_df['total_demand']
        
        # Coverage ratio
        gap_df['coverage_ratio'] = np.where(
            gap_df['total_demand'] > 0,
            gap_df['available_supply'] / gap_df['total_demand'],
            np.where(gap_df['available_supply'] > 0, 999, 0)
        )
        
        # GAP percentage
        gap_df['gap_percentage'] = np.where(
            gap_df['total_demand'] > 0,
            (gap_df['net_gap'] / gap_df['total_demand']) * 100,
            np.where(gap_df['available_supply'] > 0, 100, 0)
        )
        
        # GAP type
        gap_df['gap_type'] = np.where(
            gap_df['net_gap'] > 0, 'SURPLUS',
            np.where(gap_df['net_gap'] < 0, 'SHORTAGE', 'BALANCED')
        )
        
        # Safety stock metrics
        if self._include_safety:
            gap_df['safety_coverage'] = np.where(
                gap_df.get('safety_stock_qty', 0) > 0,
                gap_df.get('supply_inventory', 0) / gap_df['safety_stock_qty'],
                999
            )
            
            gap_df['below_reorder'] = (
                gap_df.get('supply_inventory', 0) <= gap_df.get('reorder_point', 0)
            ) & (gap_df.get('reorder_point', 0) > 0)
            
            gap_df['days_of_supply'] = np.where(
                gap_df.get('avg_daily_demand', 0) > 0,
                gap_df.get('supply_inventory', 0) / gap_df['avg_daily_demand'],
                999
            )
        
        # Status classification
        gap_df['gap_status'] = gap_df.apply(
            lambda row: self._classify_gap_status(row), 
            axis=1
        )
        
        # Priority
        gap_df['priority'] = gap_df.apply(
            lambda row: self._calculate_priority(row),
            axis=1
        )
        
        # Suggested action
        gap_df['suggested_action'] = gap_df.apply(
            lambda row: self._get_suggested_action(row),
            axis=1
        )
        
        # Value calculations
        if 'avg_unit_cost_usd' in gap_df.columns:
            gap_df['gap_value_usd'] = gap_df['net_gap'] * gap_df['avg_unit_cost_usd']
        
        # At-risk value
        if 'demand_value_usd' in gap_df.columns:
            gap_df['shortage_ratio'] = np.where(
                (gap_df['net_gap'] < 0) & (gap_df['total_demand'] > 0),
                abs(gap_df['net_gap']) / gap_df['total_demand'],
                0
            )
            gap_df['at_risk_value_usd'] = gap_df['shortage_ratio'] * gap_df['demand_value_usd']
            gap_df.drop(columns=['shortage_ratio'], inplace=True)
        elif 'avg_selling_price_usd' in gap_df.columns:
            gap_df['at_risk_value_usd'] = np.where(
                gap_df['net_gap'] < 0,
                abs(gap_df['net_gap']) * gap_df['avg_selling_price_usd'],
                0
            )
        else:
            gap_df['at_risk_value_usd'] = 0
        
        return gap_df
    
    def _classify_gap_status(self, row: pd.Series) -> str:
        """Classify GAP status with safety awareness"""
        coverage = row.get('coverage_ratio', 0)
        demand = row.get('total_demand', 0)
        inventory = row.get('supply_inventory', 0)
        
        # Safety stock specific statuses
        if self._include_safety:
            safety_stock = row.get('safety_stock_qty', 0)
            reorder_point = row.get('reorder_point', 0)
            
            if safety_stock > 0 and inventory < safety_stock * SAFETY_THRESHOLDS['CRITICAL_BREACH']:
                return 'CRITICAL_BREACH'
            
            if safety_stock > 0 and inventory < safety_stock:
                return 'BELOW_SAFETY'
            
            if reorder_point > 0 and inventory <= reorder_point:
                return 'AT_REORDER'
            
            if row.get('expired_qty', 0) > 0:
                return 'HAS_EXPIRED'
            
            if row.get('near_expiry_qty', 0) > demand * 0.5:
                return 'EXPIRY_RISK'
        
        # Standard classification
        if demand == 0:
            if inventory > 0:
                return 'NO_DEMAND'
            elif row.get('supply_purchase_order', 0) > 0:
                return 'NO_DEMAND_INCOMING'
            return 'NO_DEMAND'
        
        if coverage > COVERAGE_THRESHOLDS['SEVERE_SURPLUS']:
            return 'SEVERE_SURPLUS'
        elif coverage > COVERAGE_THRESHOLDS['HIGH_SURPLUS']:
            return 'HIGH_SURPLUS'
        elif coverage > COVERAGE_THRESHOLDS['MODERATE_SURPLUS']:
            return 'MODERATE_SURPLUS'
        elif coverage > COVERAGE_THRESHOLDS['LIGHT_SURPLUS']:
            return 'LIGHT_SURPLUS'
        elif coverage >= COVERAGE_THRESHOLDS['BALANCED_LOW']:
            return 'BALANCED'
        elif coverage >= COVERAGE_THRESHOLDS['MODERATE_SHORTAGE']:
            return 'MODERATE_SHORTAGE'
        elif coverage >= COVERAGE_THRESHOLDS['HIGH_SHORTAGE']:
            return 'HIGH_SHORTAGE'
        else:
            return 'SEVERE_SHORTAGE'
    
    def _calculate_priority(self, row: pd.Series) -> int:
        """Calculate action priority"""
        status = row.get('gap_status', 'UNKNOWN')
        
        critical_statuses = [
            'CRITICAL_BREACH', 'HAS_EXPIRED', 'SEVERE_SHORTAGE', 'BELOW_SAFETY'
        ]
        if status in critical_statuses:
            return PRIORITY_LEVELS['CRITICAL']
        
        high_statuses = [
            'AT_REORDER', 'HIGH_SHORTAGE', 'EXPIRY_RISK',
            'SEVERE_SURPLUS', 'HIGH_SURPLUS'
        ]
        if status in high_statuses:
            return PRIORITY_LEVELS['HIGH']
        
        medium_statuses = ['MODERATE_SHORTAGE', 'MODERATE_SURPLUS']
        if status in medium_statuses:
            return PRIORITY_LEVELS['MEDIUM']
        
        if status == 'LIGHT_SURPLUS':
            return PRIORITY_LEVELS['LOW']
        
        return PRIORITY_LEVELS['OK']
    
    def _get_suggested_action(self, row: pd.Series) -> str:
        """Generate suggested action"""
        status = row.get('gap_status', 'UNKNOWN')
        gap = row.get('net_gap', 0)
        inventory = row.get('supply_inventory', 0)
        coverage = row.get('coverage_ratio', 0)
        
        # Safety stock specific actions
        if self._include_safety:
            safety_stock = row.get('safety_stock_qty', 0)
            reorder_point = row.get('reorder_point', 0)
            
            if status == 'CRITICAL_BREACH':
                return f"🚨 CRITICAL: Inventory ({inventory:.0f}) below safety minimum ({safety_stock:.0f}). EXPEDITE NOW!"
            
            if status == 'BELOW_SAFETY':
                shortage = safety_stock - inventory
                return f"📦 Below safety stock by {shortage:.0f} units. Create PO immediately"
            
            if status == 'AT_REORDER':
                order_qty = max(safety_stock * 2 - inventory, 0)
                return f"🔄 At reorder point. Order {order_qty:.0f} units"
        
        # Expiry actions
        if status == 'HAS_EXPIRED':
            expired = row.get('expired_qty', 0)
            return f"❌ {expired:.0f} units expired. Dispose immediately"
        
        if status == 'EXPIRY_RISK':
            near_expiry = row.get('near_expiry_qty', 0)
            return f"⏰ {near_expiry:.0f} units expiring soon. Prioritize or discount"
        
        # Standard actions
        if status == 'NO_DEMAND':
            return f"📊 No demand. {inventory:.0f} units in stock. Review forecast"
        
        if coverage < COVERAGE_THRESHOLDS['HIGH_SHORTAGE']:
            return f"🚨 URGENT: Need {abs(gap):.0f} units. Expedite orders!"
        elif coverage < COVERAGE_THRESHOLDS['MODERATE_SHORTAGE']:
            return f"⚡ Need {abs(gap):.0f} units. Create PO within 2 days"
        elif coverage < COVERAGE_THRESHOLDS['BALANCED_LOW']:
            return f"📋 Need {abs(gap):.0f} units. Plan replenishment"
        elif coverage <= COVERAGE_THRESHOLDS['BALANCED_HIGH']:
            return "✅ Supply-demand balanced"
        elif coverage <= COVERAGE_THRESHOLDS['MODERATE_SURPLUS']:
            return f"📈 Minor surplus ({gap:.0f} units). Monitor"
        elif coverage <= COVERAGE_THRESHOLDS['HIGH_SURPLUS']:
            return f"📦 Surplus {gap:.0f} units. Reduce orders"
        elif coverage <= COVERAGE_THRESHOLDS['SEVERE_SURPLUS']:
            return f"⚠️ High surplus {gap:.0f} units. Stop ordering"
        else:
            return f"🛑 SEVERE SURPLUS {gap:.0f} units. Cancel orders!"
    
    def _calculate_customer_impact(
        self, 
        gap_df: pd.DataFrame, 
        demand_df: pd.DataFrame
    ) -> Optional[CustomerImpactData]:
        """
        Pre-calculate customer impact during GAP calculation
        No fallbacks - demand_df must be available
        """
        try:
            shortage_df = gap_df[gap_df['net_gap'] < 0].copy()
            
            if shortage_df.empty or 'product_id' not in shortage_df.columns:
                logger.info("No shortages or product-level data not available")
                return None
            
            shortage_product_ids = shortage_df['product_id'].tolist()
            
            if demand_df.empty:
                logger.warning("Demand data is empty, cannot calculate customer impact")
                return None
            
            # Filter demand for shortage products
            affected_demand = demand_df[
                demand_df['product_id'].isin(shortage_product_ids)
            ].copy()
            
            if affected_demand.empty:
                logger.warning(f"No demand found for {len(shortage_product_ids)} shortage products")
                return None
            
            # Build shortage lookup
            shortage_lookup = shortage_df.set_index('product_id').to_dict('index')
            
            # Deduplicate
            dedup_cols = ['customer', 'product_id', 'demand_source']
            affected_demand = affected_demand.drop_duplicates(subset=dedup_cols).copy()
            
            # Calculate metrics
            affected_demand['product_net_gap'] = affected_demand['product_id'].map(
                lambda x: shortage_lookup.get(x, {}).get('net_gap', 0)
            )
            affected_demand['product_total_demand'] = affected_demand['product_id'].map(
                lambda x: shortage_lookup.get(x, {}).get('total_demand', 1)
            )
            affected_demand['product_at_risk_value'] = affected_demand['product_id'].map(
                lambda x: shortage_lookup.get(x, {}).get('at_risk_value_usd', 0)
            )
            affected_demand['product_coverage'] = affected_demand['product_id'].map(
                lambda x: shortage_lookup.get(x, {}).get('coverage_ratio', 0)
            )
            
            # Customer's share
            affected_demand['demand_share'] = np.where(
                affected_demand['product_total_demand'] > 0,
                affected_demand['required_quantity'] / affected_demand['product_total_demand'],
                0
            )
            
            affected_demand['customer_shortage'] = (
                abs(affected_demand['product_net_gap']) * affected_demand['demand_share']
            )
            
            affected_demand['customer_at_risk'] = (
                affected_demand['product_at_risk_value'] * affected_demand['demand_share']
            )
            
            # Group by customer
            customer_agg = affected_demand.groupby('customer').agg({
                'product_id': 'nunique',
                'required_quantity': 'sum',
                'customer_shortage': 'sum',
                'total_value_usd': 'sum',
                'customer_at_risk': 'sum',
                'demand_source': lambda x: ', '.join(x.unique())
            }).reset_index()
            
            customer_agg.columns = [
                'customer', 'product_count', 'total_required', 
                'total_shortage', 'total_demand_value', 'at_risk_value', 'sources'
            ]
            
            # Get customer info
            customer_info = affected_demand.groupby('customer').first()[
                ['customer_code', 'urgency_level']
            ].reset_index()
            
            customer_agg = customer_agg.merge(customer_info, on='customer', how='left')
            
            # Overall urgency
            urgency_priority = {'OVERDUE': 0, 'URGENT': 1, 'UPCOMING': 2, 'FUTURE': 3}
            customer_urgency = affected_demand.groupby('customer')['urgency_level'].apply(
                lambda x: min(x, key=lambda v: urgency_priority.get(v, 999), default='FUTURE')
            ).reset_index()
            customer_urgency.columns = ['customer', 'urgency']
            
            customer_agg = customer_agg.merge(customer_urgency, on='customer', how='left')
            customer_agg.drop('urgency_level', axis=1, inplace=True, errors='ignore')
            
            # Build product details
            customer_products = []
            for customer_name in customer_agg['customer'].unique():
                cust_demand = affected_demand[
                    affected_demand['customer'] == customer_name
                ].copy()
                
                cust_demand = cust_demand.sort_values('customer_at_risk', ascending=False).head(20)
                
                products = []
                for _, row in cust_demand.iterrows():
                    products.append({
                        'pt_code': row.get('pt_code', ''),
                        'product_name': row.get('product_name', ''),
                        'brand': row.get('brand', ''),
                        'required_quantity': row['required_quantity'],
                        'shortage_quantity': row['customer_shortage'],
                        'demand_value': row.get('total_value_usd', 0),
                        'at_risk_value': row['customer_at_risk'],
                        'coverage': row['product_coverage'] * 100,
                        'urgency': row.get('urgency_level', 'N/A'),
                        'source': row.get('demand_source', '')
                    })
                
                customer_products.append({
                    'customer': customer_name,
                    'products': products
                })
            
            products_df = pd.DataFrame(customer_products)
            customer_agg = customer_agg.merge(products_df, on='customer', how='left')
            
            # Sort by at-risk value
            customer_agg = customer_agg.sort_values('at_risk_value', ascending=False)
            
            affected_count = len(customer_agg)
            total_at_risk = customer_agg['at_risk_value'].sum()
            total_shortage = customer_agg['total_shortage'].sum()
            
            logger.info(f"Customer impact calculated: {affected_count} customers, "
                       f"${total_at_risk:,.2f} at risk")
            
            return CustomerImpactData(
                customer_summary_df=customer_agg,
                affected_count=affected_count,
                total_at_risk_value=total_at_risk,
                total_shortage_qty=total_shortage
            )
            
        except Exception as e:
            logger.error(f"Error calculating customer impact: {e}", exc_info=True)
            return None
    
    def get_summary_metrics(self, gap_df: pd.DataFrame, demand_df: pd.DataFrame) -> Dict[str, any]:
        """Calculate summary metrics"""
        # Status groups
        if self._include_safety:
            shortage_statuses = ['SEVERE_SHORTAGE', 'HIGH_SHORTAGE', 'MODERATE_SHORTAGE', 
                                'BELOW_SAFETY', 'CRITICAL_BREACH']
            critical_statuses = ['SEVERE_SHORTAGE', 'HIGH_SHORTAGE', 'CRITICAL_BREACH', 
                                'BELOW_SAFETY', 'HAS_EXPIRED']
        else:
            shortage_statuses = ['SEVERE_SHORTAGE', 'HIGH_SHORTAGE', 'MODERATE_SHORTAGE']
            critical_statuses = ['SEVERE_SHORTAGE', 'HIGH_SHORTAGE']
        
        surplus_statuses = ['SEVERE_SURPLUS', 'HIGH_SURPLUS', 'MODERATE_SURPLUS', 'LIGHT_SURPLUS']
        
        # Calculate unique affected customers directly from demand_df
        affected_customers = self._calculate_unique_affected_customers(gap_df, demand_df)
        
        metrics = {
            'total_products': len(gap_df),
            'total_supply': gap_df['total_supply'].sum(),
            'total_demand': gap_df['total_demand'].sum(),
            'net_gap': gap_df['net_gap'].sum(),
            'shortage_items': len(gap_df[gap_df['gap_status'].isin(shortage_statuses)]),
            'total_shortage': abs(gap_df[gap_df['net_gap'] < 0]['net_gap'].sum()),
            'surplus_items': len(gap_df[gap_df['gap_status'].isin(surplus_statuses)]),
            'total_surplus': gap_df[gap_df['net_gap'] > 0]['net_gap'].sum(),
            'critical_items': len(gap_df[gap_df['gap_status'].isin(critical_statuses)]),
            'overall_coverage': self._calculate_overall_coverage(gap_df),
            'at_risk_value_usd': gap_df.get('at_risk_value_usd', pd.Series([0])).sum(),
            'gap_value_usd': gap_df.get('gap_value_usd', pd.Series([0])).sum(),
            'total_demand_value_usd': gap_df.get('demand_value_usd', pd.Series([0])).sum(),
            'total_supply_value_usd': gap_df.get('supply_value_usd', pd.Series([0])).sum(),
            'affected_customers': affected_customers,
            'overdue_items': int(gap_df.get('count_overdue', pd.Series([0])).sum()),
            'urgent_items': int(gap_df.get('count_urgent', pd.Series([0])).sum())
        }
        
        # Safety stock specific metrics
        if self._include_safety:
            safety_metrics = {
                'below_safety_count': len(gap_df[gap_df['gap_status'] == 'BELOW_SAFETY']),
                'at_reorder_count': len(gap_df[gap_df.get('below_reorder', False) == True]),
                'safety_stock_value': (gap_df.get('safety_stock_qty', pd.Series([0])) * 
                                      gap_df.get('avg_unit_cost_usd', pd.Series([0]))).sum(),
                'avg_safety_coverage': gap_df[gap_df.get('safety_coverage', 0) < 999].get('safety_coverage', pd.Series([0])).mean(),
                'has_expired_count': len(gap_df[gap_df['gap_status'] == 'HAS_EXPIRED']),
                'expiry_risk_count': len(gap_df[gap_df['gap_status'] == 'EXPIRY_RISK'])
            }
            metrics.update(safety_metrics)
        
        return metrics
    
    def _calculate_unique_affected_customers(
        self, 
        gap_df: pd.DataFrame, 
        demand_df: pd.DataFrame
    ) -> int:
        """
        Calculate unique affected customers - NO FALLBACKS
        Must use demand_df for accurate count
        """
        try:
            shortage_df = gap_df[gap_df['net_gap'] < 0]
            
            if shortage_df.empty or 'product_id' not in shortage_df.columns:
                return 0
            
            if demand_df.empty:
                logger.error("Demand data is empty, cannot calculate customer count")
                return 0
            
            shortage_product_ids = shortage_df['product_id'].unique().tolist()
            
            affected_demand = demand_df[
                demand_df['product_id'].isin(shortage_product_ids)
            ]
            
            if 'customer' not in affected_demand.columns:
                logger.error("Customer column not found in demand data")
                return 0
            
            unique_customers = affected_demand['customer'].nunique()
            
            logger.info(f"Calculated {unique_customers} unique affected customers from demand data")
            return int(unique_customers)
            
        except Exception as e:
            logger.error(f"Error calculating affected customers: {e}", exc_info=True)
            raise  # Don't fallback, raise error
    
    def _calculate_overall_coverage(self, gap_df: pd.DataFrame) -> float:
        """Calculate overall coverage percentage"""
        if self._include_safety:
            total_available = gap_df['available_supply'].sum()
        else:
            total_available = gap_df['total_supply'].sum()
        
        total_demand = gap_df['total_demand'].sum()
        
        if total_demand > 0:
            return (total_available / total_demand) * 100
        return 100.0 if total_available == 0 else 999.0