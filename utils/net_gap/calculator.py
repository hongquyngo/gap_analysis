# utils/net_gap/calculator.py

"""
Calculator module for GAP Analysis System - Version 2.0
- Integrated safety stock calculations
- Single unified GAP metric that adapts based on safety stock
- Context-aware status classification
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Constants for GAP thresholds
COVERAGE_THRESHOLDS = {
    'SEVERE_SURPLUS': 3.0,      # > 300%
    'HIGH_SURPLUS': 2.0,         # 200-300%
    'MODERATE_SURPLUS': 1.5,     # 150-200%
    'LIGHT_SURPLUS': 1.1,        # 110-150%
    'BALANCED_HIGH': 1.1,        # 90-110%
    'BALANCED_LOW': 0.9,         # 90-110%
    'MODERATE_SHORTAGE': 0.7,    # 70-90%
    'HIGH_SHORTAGE': 0.5,        # 50-70%
    'SEVERE_SHORTAGE': 0.0       # < 50%
}

# Safety stock specific thresholds
SAFETY_THRESHOLDS = {
    'CRITICAL_BREACH': 0.5,     # < 50% of safety stock
    'BELOW_SAFETY': 1.0,         # < 100% of safety stock
    'EXCESS_STOCK': 3.0          # > 300% of safety stock
}

PRIORITY_LEVELS = {
    'CRITICAL': 1,
    'HIGH': 2,
    'MEDIUM': 3,
    'LOW': 4,
    'OK': 99
}


class GAPCalculator:
    """Handles GAP calculations for supply-demand analysis with safety stock support"""
    
    def __init__(self):
        """Initialize calculator with default settings"""
        self.supply_sources = ['INVENTORY', 'CAN_PENDING', 'WAREHOUSE_TRANSFER', 'PURCHASE_ORDER']
        self.demand_sources = ['OC_PENDING', 'FORECAST']
        self._filtered_demand_df = None  # Store for customer impact analysis
        self._safety_stock_df = None  # Store safety stock data
        self._include_safety = False  # Track if safety stock is included
    
    def calculate_net_gap(
        self,
        supply_df: pd.DataFrame,
        demand_df: pd.DataFrame,
        safety_stock_df: Optional[pd.DataFrame] = None,
        group_by: str = 'product',
        selected_supply_sources: Optional[List[str]] = None,
        selected_demand_sources: Optional[List[str]] = None,
        include_safety_stock: bool = False
    ) -> pd.DataFrame:
        """
        Calculate net GAP with optional safety stock consideration
        
        Args:
            supply_df: Supply data from unified_supply_view
            demand_df: Demand data from unified_demand_view
            safety_stock_df: Safety stock requirements (optional)
            group_by: Aggregation level ('product' or 'brand')
            selected_supply_sources: Supply sources to include
            selected_demand_sources: Demand sources to include
            include_safety_stock: Whether to include safety stock in calculations
            
        Returns:
            DataFrame with GAP calculations (adjusted for safety stock if enabled)
        """
        try:
            # Store configuration
            self._include_safety = include_safety_stock and safety_stock_df is not None and not safety_stock_df.empty
            
            # Validate group_by parameter
            if group_by not in ['product', 'brand']:
                logger.warning(f"Invalid group_by value: {group_by}, defaulting to 'product'")
                group_by = 'product'
            
            # Filter by selected sources
            if selected_supply_sources:
                supply_df = supply_df[supply_df['supply_source'].isin(selected_supply_sources)].copy()
            
            if selected_demand_sources:
                demand_df = demand_df[demand_df['demand_source'].isin(selected_demand_sources)].copy()
            
            # Store filtered data for later use
            self._filtered_demand_df = demand_df.copy()
            if self._include_safety:
                self._safety_stock_df = safety_stock_df.copy()
            
            # Get group columns
            group_cols = self._get_group_columns(group_by)
            
            # Aggregate supply and demand
            supply_agg = self._aggregate_supply(supply_df, group_cols)
            demand_agg = self._aggregate_demand(demand_df, group_cols)
            
            # Merge safety stock if included
            if self._include_safety:
                supply_agg = self._merge_safety_stock(supply_agg, safety_stock_df, group_cols)
            
            # Merge supply and demand
            gap_df = pd.merge(
                supply_agg,
                demand_agg,
                on=group_cols,
                how='outer',
                suffixes=('_supply', '_demand')
            )
            
            # Fill NaN values
            numeric_cols = gap_df.select_dtypes(include=[np.number]).columns
            gap_df[numeric_cols] = gap_df[numeric_cols].fillna(0)
            
            # Calculate GAP metrics (will adapt based on safety stock)
            gap_df = self._calculate_gap_metrics(gap_df)
            
            # Sort by priority
            gap_df = gap_df.sort_values(
                by=['priority', 'net_gap'],
                ascending=[True, True]
            )
            
            logger.info(f"Calculated GAP for {len(gap_df)} {group_by} groups "
                       f"(safety stock: {'included' if self._include_safety else 'excluded'})")
            return gap_df
            
        except Exception as e:
            logger.error(f"Error calculating net GAP: {e}", exc_info=True)
            raise
    
    def _get_group_columns(self, group_by: str) -> List[str]:
        """Get grouping columns based on aggregation level"""
        if group_by == 'product':
            return ['product_id', 'product_name', 'pt_code', 'brand', 'standard_uom']
        elif group_by == 'brand':
            return ['brand']
        else:
            logger.error(f"Unexpected group_by value: {group_by}")
            return ['product_id', 'product_name', 'pt_code', 'brand', 'standard_uom']
    
    def _aggregate_supply(self, supply_df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
        """Aggregate supply data by specified columns"""
        if supply_df.empty:
            return pd.DataFrame(columns=group_cols + ['total_supply'])
        
        supply_df = supply_df.copy()
        
        # Filter out expired items
        today = pd.Timestamp.now().date()
        if 'expiry_date' in supply_df.columns:
            supply_df = supply_df[
                (supply_df['expiry_date'].isna()) | 
                (pd.to_datetime(supply_df['expiry_date']).dt.date > today)
            ].copy()
        
        # Create aggregation dictionary
        agg_dict = {
            'available_quantity': 'sum',
            'total_value_usd': 'sum'
        }
        
        # Add source-specific columns for breakdown
        for source in self.supply_sources:
            col_name = f'supply_{source.lower()}'
            supply_df.loc[:, col_name] = np.where(
                supply_df['supply_source'] == source,
                supply_df['available_quantity'],
                0
            )
            agg_dict[col_name] = 'sum'
        
        # Add weighted average cost calculation
        if 'unit_cost_usd' in supply_df.columns:
            supply_df.loc[:, 'cost_x_qty'] = supply_df['unit_cost_usd'] * supply_df['available_quantity']
            agg_dict['cost_x_qty'] = 'sum'
        
        # Track expired and near-expiry quantities
        if 'days_to_expiry' in supply_df.columns:
            supply_df.loc[:, 'expired_qty'] = np.where(
                supply_df['days_to_expiry'] <= 0,
                supply_df['available_quantity'],
                0
            )
            supply_df.loc[:, 'near_expiry_qty'] = np.where(
                (supply_df['days_to_expiry'] > 0) & (supply_df['days_to_expiry'] <= 30),
                supply_df['available_quantity'],
                0
            )
            agg_dict['expired_qty'] = 'sum'
            agg_dict['near_expiry_qty'] = 'sum'
        
        # Perform aggregation
        supply_agg = supply_df.groupby(group_cols, as_index=False).agg(agg_dict)
        
        # Rename columns
        supply_agg.rename(columns={
            'available_quantity': 'total_supply',
            'total_value_usd': 'supply_value_usd'
        }, inplace=True)
        
        # Calculate weighted average unit cost
        if 'cost_x_qty' in supply_agg.columns:
            supply_agg['avg_unit_cost_usd'] = np.where(
                supply_agg['total_supply'] > 0,
                supply_agg['cost_x_qty'] / supply_agg['total_supply'],
                0
            )
            supply_agg.drop(columns=['cost_x_qty'], inplace=True)
        
        return supply_agg
    
    def _aggregate_demand(self, demand_df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
        """Aggregate demand data by specified columns"""
        if demand_df.empty:
            return pd.DataFrame(columns=group_cols + ['total_demand'])
        
        demand_df = demand_df.copy()
        
        # Create aggregation dictionary
        agg_dict = {
            'required_quantity': 'sum',
            'total_value_usd': 'sum',
            'customer': 'nunique'
        }
        
        # Add optional columns if they exist
        optional_aggs = {
            'allocated_quantity': 'sum',
            'unallocated_quantity': 'sum',
            'over_committed_qty_standard': 'sum',
            'days_to_required': 'mean'
        }
        
        for col, agg_func in optional_aggs.items():
            if col in demand_df.columns:
                agg_dict[col] = agg_func
        
        # Add source-specific columns
        for source in self.demand_sources:
            col_name = f'demand_{source.lower()}'
            demand_df.loc[:, col_name] = np.where(
                demand_df['demand_source'] == source,
                demand_df['required_quantity'],
                0
            )
            agg_dict[col_name] = 'sum'
        
        # Count urgency levels
        if 'urgency_level' in demand_df.columns:
            for urgency in ['OVERDUE', 'URGENT', 'UPCOMING']:
                col_name = f'count_{urgency.lower()}'
                demand_df.loc[:, col_name] = (demand_df['urgency_level'] == urgency).astype(int)
                agg_dict[col_name] = 'sum'
        
        # Perform aggregation
        demand_agg = demand_df.groupby(group_cols, as_index=False).agg(agg_dict)
        
        # Rename columns
        demand_agg.rename(columns={
            'required_quantity': 'total_demand',
            'total_value_usd': 'demand_value_usd',
            'customer': 'customer_count',
            'days_to_required': 'avg_days_to_required'
        }, inplace=True)
        
        # Calculate weighted average selling price
        demand_agg['avg_selling_price_usd'] = np.where(
            demand_agg['total_demand'] > 0,
            demand_agg['demand_value_usd'] / demand_agg['total_demand'],
            0
        )
        
        return demand_agg
    
    def _merge_safety_stock(
        self, 
        supply_agg: pd.DataFrame, 
        safety_stock_df: pd.DataFrame, 
        group_cols: List[str]
    ) -> pd.DataFrame:
        """Merge safety stock data with supply aggregation"""
        if safety_stock_df.empty:
            supply_agg['safety_stock_qty'] = 0
            supply_agg['reorder_point'] = 0
            supply_agg['avg_daily_demand'] = 0
            return supply_agg
        
        # Select relevant columns from safety stock
        safety_cols = ['product_id', 'safety_stock_qty', 'reorder_point', 'avg_daily_demand']
        safety_data = safety_stock_df[safety_cols].copy()
        
        # Merge based on group level
        if 'product_id' in group_cols:
            supply_agg = pd.merge(
                supply_agg,
                safety_data,
                on='product_id',
                how='left'
            )
        else:
            # For brand-level grouping, sum safety stock by brand
            # This requires joining with product master first
            supply_agg['safety_stock_qty'] = 0
            supply_agg['reorder_point'] = 0
            supply_agg['avg_daily_demand'] = 0
        
        # Fill NaN values
        supply_agg['safety_stock_qty'] = supply_agg['safety_stock_qty'].fillna(0)
        supply_agg['reorder_point'] = supply_agg['reorder_point'].fillna(0)
        supply_agg['avg_daily_demand'] = supply_agg['avg_daily_demand'].fillna(0)
        
        return supply_agg
    
    def _calculate_gap_metrics(self, gap_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate GAP metrics with safety stock awareness
        """
        # Determine effective available supply
        if self._include_safety:
            # When safety stock is included, adjust available supply
            gap_df['available_supply'] = np.maximum(
                0,
                gap_df['total_supply'] - gap_df.get('safety_stock_qty', 0)
            )
        else:
            # Traditional calculation
            gap_df['available_supply'] = gap_df['total_supply']
        
        # Basic GAP calculation (using available supply)
        gap_df['net_gap'] = gap_df['available_supply'] - gap_df['total_demand']
        
        # Coverage ratio (using available supply)
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
        
        # Safety stock metrics (if included)
        if self._include_safety:
            # Safety stock coverage
            gap_df['safety_coverage'] = np.where(
                gap_df.get('safety_stock_qty', 0) > 0,
                gap_df.get('supply_inventory', 0) / gap_df['safety_stock_qty'],
                999
            )
            
            # Below reorder point flag
            gap_df['below_reorder'] = (
                gap_df.get('supply_inventory', 0) <= gap_df.get('reorder_point', 0)
            ) & (gap_df.get('reorder_point', 0) > 0)
            
            # Days of supply
            gap_df['days_of_supply'] = np.where(
                gap_df.get('avg_daily_demand', 0) > 0,
                gap_df.get('supply_inventory', 0) / gap_df['avg_daily_demand'],
                999
            )
        
        # GAP status (context-aware)
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
        
        # At-risk value calculation
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
        """
        Classify GAP status with safety stock awareness
        """
        coverage = row.get('coverage_ratio', 0)
        demand = row.get('total_demand', 0)
        inventory = row.get('supply_inventory', 0)
        
        # Safety stock specific statuses (when enabled)
        if self._include_safety:
            safety_stock = row.get('safety_stock_qty', 0)
            reorder_point = row.get('reorder_point', 0)
            safety_coverage = row.get('safety_coverage', 999)
            
            # Critical safety breaches
            if safety_stock > 0 and inventory < safety_stock * SAFETY_THRESHOLDS['CRITICAL_BREACH']:
                return 'CRITICAL_BREACH'
            
            # Below safety stock
            if safety_stock > 0 and inventory < safety_stock:
                return 'BELOW_SAFETY'
            
            # At reorder point
            if reorder_point > 0 and inventory <= reorder_point:
                return 'AT_REORDER'
            
            # Check for expiry issues
            if row.get('expired_qty', 0) > 0:
                return 'HAS_EXPIRED'
            
            if row.get('near_expiry_qty', 0) > demand * 0.5:
                return 'EXPIRY_RISK'
        
        # Standard coverage-based classification
        if demand == 0:
            if inventory > 0:
                return 'NO_DEMAND'
            elif row.get('supply_purchase_order', 0) > 0:
                return 'NO_DEMAND_INCOMING'
            return 'NO_DEMAND'
        
        # Coverage-based classification (same thresholds but using adjusted coverage)
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
        """Calculate action priority with safety stock awareness"""
        status = row.get('gap_status', 'UNKNOWN')
        coverage = row.get('coverage_ratio', 0)
        
        # Critical priorities
        critical_statuses = [
            'CRITICAL_BREACH', 'HAS_EXPIRED', 'SEVERE_SHORTAGE',
            'BELOW_SAFETY'
        ]
        if status in critical_statuses:
            return PRIORITY_LEVELS['CRITICAL']
        
        # High priorities
        high_statuses = [
            'AT_REORDER', 'HIGH_SHORTAGE', 'EXPIRY_RISK',
            'SEVERE_SURPLUS', 'HIGH_SURPLUS'
        ]
        if status in high_statuses:
            return PRIORITY_LEVELS['HIGH']
        
        # Medium priorities
        medium_statuses = [
            'MODERATE_SHORTAGE', 'MODERATE_SURPLUS'
        ]
        if status in medium_statuses:
            return PRIORITY_LEVELS['MEDIUM']
        
        # Low priorities
        if status == 'LIGHT_SURPLUS':
            return PRIORITY_LEVELS['LOW']
        
        return PRIORITY_LEVELS['OK']
    
    def _get_suggested_action(self, row: pd.Series) -> str:
        """Generate suggested action with safety stock awareness"""
        status = row.get('gap_status', 'UNKNOWN')
        gap = row.get('net_gap', 0)
        demand = row.get('total_demand', 0)
        inventory = row.get('supply_inventory', 0)
        coverage = row.get('coverage_ratio', 0)
        
        # Safety stock specific actions
        if self._include_safety:
            safety_stock = row.get('safety_stock_qty', 0)
            reorder_point = row.get('reorder_point', 0)
            
            if status == 'CRITICAL_BREACH':
                return f"⚠️ CRITICAL: Inventory ({inventory:.0f}) below safety minimum ({safety_stock:.0f}). EXPEDITE NOW!"
            
            if status == 'BELOW_SAFETY':
                shortage = safety_stock - inventory
                return f"📦 Below safety stock by {shortage:.0f} units. Create PO immediately"
            
            if status == 'AT_REORDER':
                order_qty = max(safety_stock * 2 - inventory, 0)
                return f"🔄 At reorder point. Order {order_qty:.0f} units"
        
        # Expiry related actions
        if status == 'HAS_EXPIRED':
            expired = row.get('expired_qty', 0)
            return f"❌ {expired:.0f} units expired. Dispose immediately"
        
        if status == 'EXPIRY_RISK':
            near_expiry = row.get('near_expiry_qty', 0)
            return f"⏰ {near_expiry:.0f} units expiring soon. Prioritize or discount"
        
        # Standard GAP actions
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
    
    def _calculate_unique_affected_customers(self, gap_df: pd.DataFrame) -> int:
        """Calculate unique number of customers affected by shortages"""
        try:
            if not hasattr(self, '_filtered_demand_df') or self._filtered_demand_df is None:
                logger.warning("Filtered demand data not available")
                return int(gap_df[gap_df['net_gap'] < 0].get('customer_count', pd.Series([0])).sum())
            
            shortage_df = gap_df[gap_df['net_gap'] < 0]
            
            if shortage_df.empty:
                return 0
            
            # Get shortage items based on grouping
            if 'product_id' in shortage_df.columns:
                shortage_items = shortage_df['product_id'].unique().tolist()
                affected_demand = self._filtered_demand_df[
                    self._filtered_demand_df['product_id'].isin(shortage_items)
                ]
            elif 'brand' in shortage_df.columns and len(shortage_df.columns) <= 15:
                shortage_brands = shortage_df['brand'].unique().tolist()
                affected_demand = self._filtered_demand_df[
                    self._filtered_demand_df['brand'].isin(shortage_brands)
                ]
            else:
                return int(gap_df[gap_df['net_gap'] < 0].get('customer_count', pd.Series([0])).sum())
            
            if 'customer' in affected_demand.columns:
                unique_customers = affected_demand['customer'].nunique()
            else:
                return 0
            
            return int(unique_customers)
            
        except Exception as e:
            logger.error(f"Error calculating affected customers: {e}", exc_info=True)
            return int(gap_df[gap_df['net_gap'] < 0].get('customer_count', pd.Series([0])).sum())
    
    def get_summary_metrics(self, gap_df: pd.DataFrame) -> Dict[str, any]:
        """
        Calculate summary metrics from GAP analysis
        Enhanced with safety stock metrics when enabled
        """
        # Define status groups
        if self._include_safety:
            # Safety-aware status groups
            shortage_statuses = ['SEVERE_SHORTAGE', 'HIGH_SHORTAGE', 'MODERATE_SHORTAGE', 
                                'BELOW_SAFETY', 'CRITICAL_BREACH']
            critical_statuses = ['SEVERE_SHORTAGE', 'HIGH_SHORTAGE', 'CRITICAL_BREACH', 
                                'BELOW_SAFETY', 'HAS_EXPIRED']
        else:
            # Traditional status groups
            shortage_statuses = ['SEVERE_SHORTAGE', 'HIGH_SHORTAGE', 'MODERATE_SHORTAGE']
            critical_statuses = ['SEVERE_SHORTAGE', 'HIGH_SHORTAGE']
        
        surplus_statuses = ['SEVERE_SURPLUS', 'HIGH_SURPLUS', 'MODERATE_SURPLUS', 'LIGHT_SURPLUS']
        
        # Calculate unique affected customers
        affected_customers = self._calculate_unique_affected_customers(gap_df)
        
        # Basic metrics (same for both modes)
        metrics = {
            'total_products': len(gap_df),
            'total_supply': gap_df['total_supply'].sum(),
            'total_demand': gap_df['total_demand'].sum(),
            'net_gap': gap_df['net_gap'].sum(),
            
            # Shortage metrics
            'shortage_items': len(gap_df[gap_df['gap_status'].isin(shortage_statuses)]),
            'total_shortage': abs(gap_df[gap_df['net_gap'] < 0]['net_gap'].sum()),
            
            # Surplus metrics
            'surplus_items': len(gap_df[gap_df['gap_status'].isin(surplus_statuses)]),
            'total_surplus': gap_df[gap_df['net_gap'] > 0]['net_gap'].sum(),
            
            # Critical items
            'critical_items': len(gap_df[gap_df['gap_status'].isin(critical_statuses)]),
            
            # Coverage
            'overall_coverage': self._calculate_overall_coverage(gap_df),
            
            # Value metrics
            'at_risk_value_usd': gap_df.get('at_risk_value_usd', pd.Series([0])).sum(),
            'gap_value_usd': gap_df.get('gap_value_usd', pd.Series([0])).sum(),
            'total_demand_value_usd': gap_df.get('demand_value_usd', pd.Series([0])).sum(),
            'total_supply_value_usd': gap_df.get('supply_value_usd', pd.Series([0])).sum(),
            
            # Customer impact
            'affected_customers': affected_customers,
            
            # Urgency counts
            'overdue_items': int(gap_df.get('count_overdue', pd.Series([0])).sum()),
            'urgent_items': int(gap_df.get('count_urgent', pd.Series([0])).sum())
        }
        
        # Add safety stock specific metrics if enabled
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
        
        logger.info(f"Summary Metrics: {metrics['shortage_items']} shortage items, "
                   f"{affected_customers} affected customers, "
                   f"At-risk value: ${metrics['at_risk_value_usd']:,.2f}")
        
        return metrics
    
    def _calculate_overall_coverage(self, gap_df: pd.DataFrame) -> float:
        """Calculate overall coverage percentage"""
        # Use available supply if safety stock is included
        if self._include_safety:
            total_available = gap_df['available_supply'].sum()
        else:
            total_available = gap_df['total_supply'].sum()
        
        total_demand = gap_df['total_demand'].sum()
        
        if total_demand > 0:
            return (total_available / total_demand) * 100
        return 100.0 if total_available == 0 else 999.0