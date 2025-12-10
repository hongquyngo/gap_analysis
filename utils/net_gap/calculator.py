# utils/net_gap/calculator.py - IMPROVED VERSION v4.4.0

"""
GAP Calculator - Production Ready Complete Version
IMPROVEMENTS in v4.4:
- Added safety_gap column (Supply - Safety Stock, can be negative)
- Added shortage_cause column (explains WHY shortage exists)
- Better insights for users
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

from .calculation_result import GAPCalculationResult, CustomerImpact
from .constants import THRESHOLDS, GAP_CATEGORIES

logger = logging.getLogger(__name__)


class GAPCalculator:
    """GAP calculation engine - complete with all fixes and improvements"""
    
    def calculate_net_gap(
        self,
        supply_df: pd.DataFrame,
        demand_df: pd.DataFrame,
        safety_stock_df: Optional[pd.DataFrame] = None,
        expired_inventory_df: Optional[pd.DataFrame] = None,
        group_by: str = 'product',
        selected_supply_sources: Optional[List[str]] = None,
        selected_demand_sources: Optional[List[str]] = None,
        include_safety_stock: bool = False
    ) -> GAPCalculationResult:
        """Calculate net GAP analysis"""
        try:
            # Log incoming data structure
            logger.info(f"GAP Calculation started:")
            logger.info(f"  - Supply rows: {len(supply_df)}, columns: {list(supply_df.columns) if not supply_df.empty else 'EMPTY'}")
            logger.info(f"  - Demand rows: {len(demand_df)}, columns: {list(demand_df.columns) if not demand_df.empty else 'EMPTY'}")
            logger.info(f"  - Selected supply sources: {selected_supply_sources}")
            logger.info(f"  - Selected demand sources: {selected_demand_sources}")
            
            # Validate group_by
            if group_by not in ['product', 'brand']:
                group_by = 'product'
            
            # Filter by selected sources (with defensive checks)
            if selected_supply_sources and not supply_df.empty:
                if 'supply_source' in supply_df.columns:
                    supply_df = supply_df[supply_df['supply_source'].isin(selected_supply_sources)]
                    logger.info(f"Filtered supply by sources: {selected_supply_sources}")
                else:
                    logger.warning("supply_source column not found in supply_df, skipping source filter")
            
            if selected_demand_sources and not demand_df.empty:
                if 'demand_source' in demand_df.columns:
                    demand_df = demand_df[demand_df['demand_source'].isin(selected_demand_sources)]
                    logger.info(f"Filtered demand by sources: {selected_demand_sources}")
                else:
                    logger.warning("demand_source column not found in demand_df, skipping source filter")
            
            # Get grouping columns
            group_cols = self._get_group_columns(group_by)
            
            # Aggregate data
            supply_agg = self._aggregate_supply(supply_df, group_cols, group_by)
            demand_agg = self._aggregate_demand(demand_df, group_cols, group_by)
            
            # Merge data
            gap_df = self._merge_data(supply_agg, demand_agg, group_by)
            
            # Add safety stock if needed
            if include_safety_stock and safety_stock_df is not None:
                gap_df = self._add_safety_stock(gap_df, safety_stock_df, group_by)
            
            # Calculate GAP metrics
            gap_df = self._calculate_metrics(gap_df, include_safety_stock)
            
            # Add expired inventory data if provided
            if expired_inventory_df is not None and not expired_inventory_df.empty and group_by == "product":
                gap_df = gap_df.merge(
                    expired_inventory_df[["product_id", "expired_quantity", "expired_batches_info"]],
                    on="product_id",
                    how="left"
                )
                gap_df["expired_quantity"] = gap_df["expired_quantity"].fillna(0)
                gap_df["expired_batches_info"] = gap_df["expired_batches_info"].fillna("")
            else:
                # Ensure columns exist even when no expired data
                gap_df["expired_quantity"] = 0
                gap_df["expired_batches_info"] = ""
            
            # Sort by priority
            gap_df = gap_df.sort_values(['priority', 'net_gap'], ascending=[True, True])
            
            # Calculate summary metrics
            metrics = self._calculate_summary_metrics(gap_df, demand_df, include_safety_stock)
            
            # Calculate customer impact
            customer_impact = None
            if group_by == 'product':
                customer_impact = self._calculate_customer_impact(gap_df, demand_df)
            
            # Create result
            filters_used = {
                'group_by': group_by,
                'supply_sources': selected_supply_sources or ['ALL'],
                'demand_sources': selected_demand_sources or ['ALL'],
                'include_safety_stock': include_safety_stock
            }
            
            result = GAPCalculationResult(
                gap_df=gap_df,
                metrics=metrics,
                customer_impact=customer_impact,
                filters_used=filters_used,
                supply_df=supply_df,
                demand_df=demand_df,
                safety_df=safety_stock_df
            )
            
            logger.info(f"GAP calculation completed: {len(gap_df)} items")
            return result
            
        except Exception as e:
            logger.error(f"GAP calculation failed: {e}", exc_info=True)
            raise
    
    def _get_group_columns(self, group_by: str) -> List[str]:
        """Get columns for grouping"""
        if group_by == 'product':
            # FIXED: Only group by product_id to prevent duplicates
            return ['product_id']
        else:  # brand
            return ['brand']
    
    def _aggregate_supply(self, df: pd.DataFrame, group_cols: List[str], group_by: str) -> pd.DataFrame:
        """Aggregate supply data - FIXED to handle product_id only grouping"""
        if df.empty:
            return pd.DataFrame()
        
        supply_df = df.copy()
        
        # Basic aggregation
        agg_dict = {
            'available_quantity': 'sum',
            'total_value_usd': 'sum'
        }
        
        # If grouping by product, add text fields with 'first' aggregation
        if group_by == 'product':
            agg_dict.update({
                'product_name': 'first',
                'pt_code': 'first',
                'brand': 'first',
                'standard_uom': 'first'
            })
        
        # Add source-specific columns
        for source in supply_df['supply_source'].unique():
            col_name = f'supply_{source.lower()}'
            supply_df[col_name] = np.where(
                supply_df['supply_source'] == source,
                supply_df['available_quantity'],
                0
            )
            agg_dict[col_name] = 'sum'
        
        # Calculate weighted average cost
        if 'unit_cost_usd' in supply_df.columns:
            supply_df['cost_x_qty'] = supply_df['unit_cost_usd'] * supply_df['available_quantity']
            agg_dict['cost_x_qty'] = 'sum'
        
        # Aggregate
        supply_agg = supply_df.groupby(group_cols, as_index=False).agg(agg_dict)
        
        # Rename columns
        supply_agg.rename(columns={
            'available_quantity': 'total_supply',
            'total_value_usd': 'supply_value_usd'
        }, inplace=True)
        
        # Calculate average unit cost
        if 'cost_x_qty' in supply_agg.columns:
            supply_agg['avg_unit_cost_usd'] = np.where(
                supply_agg['total_supply'] > 0,
                supply_agg['cost_x_qty'] / supply_agg['total_supply'],
                0
            )
            supply_agg.drop('cost_x_qty', axis=1, inplace=True)
        
        return supply_agg
    
    def _aggregate_demand(self, df: pd.DataFrame, group_cols: List[str], group_by: str) -> pd.DataFrame:
        """Aggregate demand data - FIXED for product_id grouping and avg_selling_price"""
        if df.empty:
            return pd.DataFrame()
        
        demand_df = df.copy()
        
        # Basic aggregation
        agg_dict = {
            'required_quantity': 'sum',
            'total_value_usd': 'sum',
            'customer': 'nunique'
        }
        
        # If grouping by product, add text fields with 'first' aggregation
        if group_by == 'product':
            agg_dict.update({
                'product_name': 'first',
                'pt_code': 'first',
                'brand': 'first',
                'standard_uom': 'first'
            })
        
        # Add source-specific columns
        for source in demand_df['demand_source'].unique():
            col_name = f'demand_{source.lower()}'
            demand_df[col_name] = np.where(
                demand_df['demand_source'] == source,
                demand_df['required_quantity'],
                0
            )
            agg_dict[col_name] = 'sum'
        
        # NOTE: Removed incorrect selling_unit_price calculation
        # We use total_value_usd which is already in USD
        
        # Urgency aggregation
        if 'urgency_level' in demand_df.columns:
            urgency_map = {'OVERDUE': 1, 'URGENT': 2, 'UPCOMING': 3, 'FUTURE': 4}
            demand_df['urgency_numeric'] = demand_df['urgency_level'].map(urgency_map).fillna(5)
            agg_dict['urgency_numeric'] = 'min'
        
        # Aggregate
        demand_agg = demand_df.groupby(group_cols, as_index=False).agg(agg_dict)
        
        # Rename columns
        demand_agg.rename(columns={
            'required_quantity': 'total_demand',
            'total_value_usd': 'demand_value_usd',
            'customer': 'customer_count'
        }, inplace=True)
        
        # FIXED: Calculate average selling price using total_value_usd (already in USD)
        demand_agg['avg_selling_price_usd'] = np.where(
            demand_agg['total_demand'] > 0,
            demand_agg['demand_value_usd'] / demand_agg['total_demand'],
            0
        )
        
        # Map urgency back
        if 'urgency_numeric' in demand_agg.columns:
            urgency_reverse = {1: 'OVERDUE', 2: 'URGENT', 3: 'UPCOMING', 4: 'FUTURE', 5: 'N/A'}
            demand_agg['urgency'] = demand_agg['urgency_numeric'].map(urgency_reverse)
            demand_agg.drop('urgency_numeric', axis=1, inplace=True)
        
        return demand_agg
    
    def _merge_data(
        self,
        supply_df: pd.DataFrame,
        demand_df: pd.DataFrame,
        group_by: str
    ) -> pd.DataFrame:
        """Merge supply and demand data - FIXED to handle product_id only merge"""
        
        # Determine merge columns based on group_by
        if group_by == 'product':
            merge_cols = ['product_id']
        else:
            merge_cols = ['brand']
        
        # Perform outer merge
        if not supply_df.empty and not demand_df.empty:
            gap_df = pd.merge(supply_df, demand_df, on=merge_cols, how='outer', suffixes=('_supply', '_demand'))
            
            # Consolidate text fields after merge (prefer non-null values)
            if group_by == 'product':
                for col in ['product_name', 'pt_code', 'brand', 'standard_uom']:
                    if f'{col}_supply' in gap_df.columns and f'{col}_demand' in gap_df.columns:
                        # Use supply value if available, otherwise demand
                        gap_df[col] = gap_df[f'{col}_supply'].fillna(gap_df[f'{col}_demand'])
                        gap_df.drop([f'{col}_supply', f'{col}_demand'], axis=1, inplace=True)
                    elif f'{col}_supply' in gap_df.columns:
                        gap_df.rename(columns={f'{col}_supply': col}, inplace=True)
                    elif f'{col}_demand' in gap_df.columns:
                        gap_df.rename(columns={f'{col}_demand': col}, inplace=True)
        elif not supply_df.empty:
            gap_df = supply_df.copy()
        elif not demand_df.empty:
            gap_df = demand_df.copy()
        else:
            return pd.DataFrame()
        
        # Fill NaN values for numeric columns
        numeric_cols = gap_df.select_dtypes(include=[np.number]).columns.tolist()
        gap_df[numeric_cols] = gap_df[numeric_cols].fillna(0)
        
        return gap_df
    
    def _add_safety_stock(
        self,
        gap_df: pd.DataFrame,
        safety_df: pd.DataFrame,
        group_by: str
    ) -> pd.DataFrame:
        """Add safety stock data with safe column handling"""
        
        if safety_df.empty or group_by != 'product':
            gap_df['safety_stock_qty'] = 0
            gap_df['reorder_point'] = 0
            return gap_df
        
        # Check which columns are actually available in safety_df
        available_cols = ['product_id']
        desired_cols = ['safety_stock_qty', 'reorder_point', 'avg_daily_demand']
        
        for col in desired_cols:
            if col in safety_df.columns:
                available_cols.append(col)
            else:
                logger.warning(f"Column {col} not found in safety_stock_df, using default value 0")
        
        # Only select columns that exist
        safety_data = safety_df[available_cols].copy()
        
        # Ensure no duplicates in safety data (take first if duplicate exists)
        safety_data = safety_data.groupby('product_id').first().reset_index()
        
        # Merge safety data
        gap_df = pd.merge(gap_df, safety_data, on='product_id', how='left')
        
        # Fill NaN values for safety columns with 0
        for col in desired_cols:
            if col not in gap_df.columns:
                gap_df[col] = 0
            else:
                gap_df[col] = gap_df[col].fillna(0)
        
        return gap_df
    
    def _calculate_metrics(self, gap_df: pd.DataFrame, include_safety: bool) -> pd.DataFrame:
        """
        Calculate GAP metrics with improved no-demand handling
        UPDATED v4.4: Added safety_gap and shortage_cause columns
        """
        
        # Ensure required columns exist (defensive programming)
        if 'total_supply' not in gap_df.columns:
            gap_df['total_supply'] = 0
            logger.warning("total_supply column missing, initialized to 0")
        
        if 'total_demand' not in gap_df.columns:
            gap_df['total_demand'] = 0
            logger.warning("total_demand column missing, initialized to 0")
        
        # Available supply (considering safety stock)
        if include_safety and 'safety_stock_qty' in gap_df.columns:
            gap_df['available_supply'] = np.maximum(
                0,
                gap_df['total_supply'] - gap_df['safety_stock_qty']
            )
        else:
            gap_df['available_supply'] = gap_df['total_supply']
        
        # Net GAP
        gap_df['net_gap'] = gap_df['available_supply'] - gap_df['total_demand']
        
        # True GAP (always ignores safety stock)
        gap_df['true_gap'] = gap_df['total_supply'] - gap_df['total_demand']
        
        # NEW v4.4: Safety Gap = Total Supply - Safety Stock (can be negative)
        # This shows if supply is above or below safety requirement
        if include_safety and 'safety_stock_qty' in gap_df.columns:
            gap_df['safety_gap'] = gap_df['total_supply'] - gap_df['safety_stock_qty']
        else:
            gap_df['safety_gap'] = np.nan
        
        # Coverage ratio - Use NaN for no-demand (from stable version)
        gap_df['coverage_ratio'] = np.where(
            gap_df['total_demand'] > 0,
            gap_df['available_supply'] / gap_df['total_demand'],
            np.nan  # Use NaN for undefined coverage when no demand
        )
        
        # GAP percentage - Also use NaN for no-demand
        gap_df['gap_percentage'] = np.where(
            gap_df['total_demand'] > 0,
            (gap_df['net_gap'] / gap_df['total_demand']) * 100,
            np.nan  # Use NaN when no demand
        )
        
        # Safety metrics
        if include_safety and 'safety_stock_qty' in gap_df.columns:
            gap_df['safety_coverage'] = np.where(
                gap_df['safety_stock_qty'] > 0,
                gap_df.get('supply_inventory', gap_df['total_supply']) / gap_df['safety_stock_qty'],
                np.nan  # NaN when no safety stock defined
            )
            
            gap_df['below_reorder'] = (
                gap_df.get('supply_inventory', gap_df['total_supply']) <= gap_df['reorder_point']
            ) & (gap_df['reorder_point'] > 0)
        
        # Status and priority
        gap_df['gap_status'] = gap_df.apply(self._classify_status, axis=1)
        gap_df['priority'] = gap_df.apply(self._get_priority, axis=1)
        gap_df['suggested_action'] = gap_df.apply(self._get_action, axis=1)
        
        # NEW v4.4: Shortage Cause - explains WHY there's a shortage
        gap_df['shortage_cause'] = gap_df.apply(
            lambda row: self._classify_shortage_cause(row, include_safety), 
            axis=1
        )
        
        # Financial metrics - ensure cost/price columns exist
        if 'avg_unit_cost_usd' not in gap_df.columns:
            gap_df['avg_unit_cost_usd'] = 0
        
        if 'avg_selling_price_usd' not in gap_df.columns:
            gap_df['avg_selling_price_usd'] = 0
        
        if 'supply_value_usd' not in gap_df.columns:
            gap_df['supply_value_usd'] = 0
        
        if 'demand_value_usd' not in gap_df.columns:
            gap_df['demand_value_usd'] = 0
        
        # Calculate GAP value
        gap_df['gap_value_usd'] = gap_df['net_gap'] * gap_df['avg_unit_cost_usd']
        
        # Calculate at-risk value
        gap_df['at_risk_value_usd'] = np.where(
            gap_df['net_gap'] < 0,
            abs(gap_df['net_gap']) * gap_df['avg_selling_price_usd'],
            0
        )
        
        return gap_df
    
    def _classify_status(self, row: pd.Series) -> str:
        """Classify GAP status - improved for no-demand"""
        
        coverage = row.get('coverage_ratio')
        demand = row.get('total_demand', 0)
        supply = row.get('total_supply', 0)
        
        # No demand cases
        if demand == 0:
            if supply > 0:
                return 'NO_DEMAND'  # Have supply but no demand
            else:
                return 'NO_ACTIVITY'  # No supply and no demand
        
        # When coverage is NaN (shouldn't happen after above check, but safety)
        if pd.isna(coverage):
            return 'NO_DEMAND'
        
        # Safety stock checks (if available)
        if 'safety_stock_qty' in row and row['safety_stock_qty'] > 0:
            inventory = row.get('supply_inventory', supply)
            if inventory < row['safety_stock_qty'] * THRESHOLDS['safety']['critical_breach']:
                return 'CRITICAL_BREACH'
            if inventory < row['safety_stock_qty']:
                return 'BELOW_SAFETY'
        
        # Coverage-based classification - FIXED: Use correct key names
        if coverage < THRESHOLDS['coverage']['severe_shortage']:
            return 'SEVERE_SHORTAGE'
        elif coverage < THRESHOLDS['coverage']['high_shortage']:
            return 'HIGH_SHORTAGE'
        elif coverage < THRESHOLDS['coverage']['moderate_shortage']:
            return 'MODERATE_SHORTAGE'
        elif coverage <= THRESHOLDS['coverage']['balanced_high']:  # FIXED from 'balanced_max'
            return 'BALANCED'
        elif coverage <= THRESHOLDS['coverage']['light_surplus']:
            return 'LIGHT_SURPLUS'
        elif coverage <= THRESHOLDS['coverage']['moderate_surplus']:
            return 'MODERATE_SURPLUS'
        elif coverage <= THRESHOLDS['coverage']['high_surplus']:
            return 'HIGH_SURPLUS'
        else:
            return 'SEVERE_SURPLUS'
    
    def _get_priority(self, row: pd.Series) -> int:
        """Get priority level"""
        
        status = row.get('gap_status', '')
        
        # Critical priority
        if status in ['CRITICAL_BREACH', 'SEVERE_SHORTAGE', 'BELOW_SAFETY']:
            return THRESHOLDS['priority']['critical']
        # High priority
        elif status in ['HIGH_SHORTAGE', 'AT_REORDER', 'SEVERE_SURPLUS']:
            return THRESHOLDS['priority']['high']
        # Medium priority
        elif status in ['MODERATE_SHORTAGE', 'HIGH_SURPLUS', 'MODERATE_SURPLUS']:
            return THRESHOLDS['priority']['medium']
        # Low priority
        elif status in ['LIGHT_SURPLUS']:
            return THRESHOLDS['priority']['low']
        # OK
        else:
            return THRESHOLDS['priority']['ok']
    
    def _get_action(self, row: pd.Series) -> str:
        """Get suggested action"""
        
        status = row.get('gap_status', '')
        gap = row.get('net_gap', 0)
        
        actions = {
            'CRITICAL_BREACH': f"🚨 CRITICAL: Expedite all orders NOW!",
            'SEVERE_SHORTAGE': f"🚨 Need {abs(gap):.0f} units urgently",
            'HIGH_SHORTAGE': f"⚠️ Need {abs(gap):.0f} units within 2 days",
            'MODERATE_SHORTAGE': f"📋 Plan to replenish {abs(gap):.0f} units",
            'BELOW_SAFETY': f"🔒 Below safety stock, order immediately",
            'BALANCED': "✅ Supply-demand balanced",
            'LIGHT_SURPLUS': f"📦 Minor surplus ({gap:.0f} units)",
            'MODERATE_SURPLUS': f"📦 Surplus {gap:.0f} units, reduce orders",
            'HIGH_SURPLUS': f"⚠️ High surplus {gap:.0f} units, stop ordering",
            'SEVERE_SURPLUS': f"🛑 Severe surplus {gap:.0f} units, cancel orders",
            'NO_DEMAND': f"⭕ No demand, {gap:.0f} units in stock",
            'NO_ACTIVITY': "⚪ No supply or demand"
        }
        
        return actions.get(status, "Review manually")
    
    def _classify_shortage_cause(self, row: pd.Series, include_safety: bool) -> str:
        """
        NEW v4.4: Classify the cause of shortage for better user understanding
        
        Returns:
            str: Descriptive string explaining WHY there's a shortage
        """
        net_gap = row.get('net_gap', 0)
        true_gap = row.get('true_gap', 0)
        total_supply = row.get('total_supply', 0)
        total_demand = row.get('total_demand', 0)
        safety_stock = row.get('safety_stock_qty', 0) if include_safety else 0
        
        # No demand case
        if total_demand == 0:
            if total_supply > 0:
                return "⚪ No Demand"
            return "⚪ Inactive"
        
        # No shortage
        if net_gap >= 0:
            return "✅ OK"
        
        # Has shortage - determine cause
        if true_gap < 0:
            # Real shortage - not enough supply even without safety
            if include_safety and safety_stock > 0 and total_supply < safety_stock:
                return "🚨 Real + Below Safety"
            return "🚨 Real Shortage"
        elif true_gap >= 0 and net_gap < 0:
            # Safety-induced shortage
            if include_safety and total_supply < safety_stock:
                return "🔒 Supply < Safety Req."
            return "🔒 Safety Requirement"
        
        return "⚠️ Review"
    
    def _calculate_summary_metrics(
        self,
        gap_df: pd.DataFrame,
        demand_df: pd.DataFrame,
        include_safety: bool
    ) -> Dict[str, any]:
        """Calculate summary metrics"""
        
        # Count by simplified categories
        shortage_statuses = GAP_CATEGORIES['SHORTAGE']['statuses']
        surplus_statuses = GAP_CATEGORIES['SURPLUS']['statuses']
        
        # Calculate affected customers
        affected_customers = 0
        if 'product_id' in gap_df.columns and not demand_df.empty:
            shortage_products = gap_df[gap_df['net_gap'] < 0]['product_id'].tolist()
            if shortage_products:
                affected_demand = demand_df[demand_df['product_id'].isin(shortage_products)]
                affected_customers = affected_demand['customer'].nunique()
        
        metrics = {
            'total_products': len(gap_df),
            'total_supply': gap_df.get('total_supply', pd.Series([0])).sum(),
            'total_demand': gap_df.get('total_demand', pd.Series([0])).sum(),
            'net_gap': gap_df.get('net_gap', pd.Series([0])).sum(),
            
            'shortage_items': len(gap_df[gap_df['gap_status'].isin(shortage_statuses)]),
            'surplus_items': len(gap_df[gap_df['gap_status'].isin(surplus_statuses)]),
            'critical_items': len(gap_df[gap_df.get('priority', 99) == THRESHOLDS['priority']['critical']]),
            
            'total_shortage': abs(gap_df[gap_df['net_gap'] < 0]['net_gap'].sum()) if 'net_gap' in gap_df.columns else 0,
            'total_surplus': gap_df[gap_df['net_gap'] > 0]['net_gap'].sum() if 'net_gap' in gap_df.columns else 0,
            
            'overall_coverage': self._calculate_overall_coverage(gap_df),
            'at_risk_value_usd': gap_df.get('at_risk_value_usd', pd.Series([0])).sum(),
            'gap_value_usd': gap_df.get('gap_value_usd', pd.Series([0])).sum(),
            'total_supply_value_usd': gap_df.get('supply_value_usd', pd.Series([0])).sum(),
            'total_demand_value_usd': gap_df.get('demand_value_usd', pd.Series([0])).sum(),
            
            'affected_customers': affected_customers
        }
        
        # Add safety metrics if included
        if include_safety and 'below_reorder' in gap_df.columns:
            metrics.update({
                'below_safety_count': len(gap_df[gap_df['gap_status'] == 'BELOW_SAFETY']),
                'at_reorder_count': len(gap_df[gap_df['below_reorder'] == True]),
                'has_expired_count': 0,  # Would need expiry tracking
                'expiry_risk_count': 0   # Would need expiry tracking
            })
        
        return metrics
    
    def _calculate_overall_coverage(self, gap_df: pd.DataFrame) -> float:
        """Calculate overall coverage percentage - excluding no-demand items"""
        
        # Check if required columns exist
        if 'total_demand' not in gap_df.columns or 'total_supply' not in gap_df.columns:
            logger.warning("Missing total_demand or total_supply columns for coverage calculation")
            return 0.0
        
        # Filter to items with demand > 0
        items_with_demand = gap_df[gap_df['total_demand'] > 0]
        
        if items_with_demand.empty:
            return 100.0  # If no items have demand, coverage is not meaningful
        
        # Use available_supply if exists, otherwise total_supply
        if 'available_supply' in items_with_demand.columns:
            total_supply = items_with_demand['available_supply'].sum()
        else:
            total_supply = items_with_demand['total_supply'].sum()
        
        total_demand = items_with_demand['total_demand'].sum()
        
        if total_demand > 0:
            return (total_supply / total_demand) * 100
        return 100.0
    
    def _calculate_customer_impact(
        self,
        gap_df: pd.DataFrame,
        demand_df: pd.DataFrame
    ) -> Optional[CustomerImpact]:
        """Calculate customer impact for shortage items using optimized groupby"""
        
        try:
            # Get shortage products
            shortage_df = gap_df[gap_df['net_gap'] < 0]
            
            if shortage_df.empty or demand_df.empty:
                return None
            
            shortage_products = shortage_df['product_id'].tolist()
            
            # Get affected demand
            affected_demand = demand_df[demand_df['product_id'].isin(shortage_products)].copy()
            
            if affected_demand.empty:
                return None
            
            # Build shortage lookup
            shortage_lookup = shortage_df.set_index('product_id')[
                ['net_gap', 'total_demand', 'at_risk_value_usd', 'coverage_ratio']
            ].to_dict('index')
            
            # Calculate shortage allocation for each demand row
            affected_demand['product_shortage'] = affected_demand.apply(
                lambda row: abs(shortage_lookup.get(row['product_id'], {}).get('net_gap', 0)) *
                (row['required_quantity'] / shortage_lookup.get(row['product_id'], {}).get('total_demand', 1))
                if row['product_id'] in shortage_lookup and shortage_lookup[row['product_id']]['total_demand'] > 0
                else 0,
                axis=1
            )
            
            affected_demand['product_risk'] = affected_demand.apply(
                lambda row: shortage_lookup.get(row['product_id'], {}).get('at_risk_value_usd', 0) *
                (row['required_quantity'] / shortage_lookup.get(row['product_id'], {}).get('total_demand', 1))
                if row['product_id'] in shortage_lookup and shortage_lookup[row['product_id']]['total_demand'] > 0
                else 0,
                axis=1
            )
            
            # Use groupby for efficient aggregation
            customer_agg = affected_demand.groupby('customer').agg({
                'required_quantity': 'sum',
                'product_id': 'nunique',
                'total_value_usd': 'sum',
                'product_shortage': 'sum',
                'product_risk': 'sum',
                'urgency_level': 'min',  # Get most urgent
                'customer_code': 'first'
            }).reset_index()
            
            # Rename columns for output
            customer_agg.rename(columns={
                'required_quantity': 'total_required',
                'product_id': 'product_count',
                'total_value_usd': 'total_demand_value',
                'product_shortage': 'total_shortage',
                'product_risk': 'at_risk_value',
                'urgency_level': 'urgency'
            }, inplace=True)
            
            # Sort by at risk value
            customer_df = customer_agg.sort_values('at_risk_value', ascending=False)
            
            # Add empty products list for compatibility
            customer_df['products'] = [[] for _ in range(len(customer_df))]
            
            return CustomerImpact(
                customer_df=customer_df,
                affected_count=len(customer_df),
                at_risk_value=customer_df['at_risk_value'].sum(),
                shortage_qty=customer_df['total_shortage'].sum()
            )
            
        except Exception as e:
            logger.error(f"Error calculating customer impact: {e}")
            return None