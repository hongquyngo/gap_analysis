# utils/gap/calculator.py

"""
Calculator module for GAP Analysis System
Handles all GAP calculations and aggregations
Aligned with SQL view logic from net_gap_view_with_fc
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, date
import logging

logger = logging.getLogger(__name__)


class GAPCalculator:
    """Handles GAP calculations for supply-demand analysis"""
    
    def __init__(self):
        """Initialize calculator with default settings"""
        self.aggregation_levels = ['product', 'brand', 'category']
        self.supply_sources = ['INVENTORY', 'CAN_PENDING', 'WAREHOUSE_TRANSFER', 'PURCHASE_ORDER']
        self.demand_sources = ['OC_PENDING', 'FORECAST']
    
    def calculate_net_gap(
        self,
        supply_df: pd.DataFrame,
        demand_df: pd.DataFrame,
        group_by: str = 'product',
        selected_supply_sources: List[str] = None,
        selected_demand_sources: List[str] = None
    ) -> pd.DataFrame:
        """
        Calculate simple net GAP (Supply - Demand) without time dimension
        Aligned with SQL view logic
        
        Args:
            supply_df: Supply data from unified_supply_view
            demand_df: Demand data from unified_demand_view
            group_by: Aggregation level ('product', 'brand', 'category')
            selected_supply_sources: List of supply sources to include (None = all)
            selected_demand_sources: List of demand sources to include (None = all)
            
        Returns:
            DataFrame with GAP calculations
        """
        try:
            # Filter by selected sources
            if selected_supply_sources:
                supply_df = supply_df[supply_df['supply_source'].isin(selected_supply_sources)].copy()
            
            if selected_demand_sources:
                demand_df = demand_df[demand_df['demand_source'].isin(selected_demand_sources)].copy()
            
            # Prepare aggregation columns based on group_by level
            if group_by == 'product':
                group_cols = ['product_id', 'product_name', 'pt_code', 'brand', 'standard_uom']
                sort_cols = ['product_name']
            elif group_by == 'brand':
                group_cols = ['brand']
                sort_cols = ['brand']
            else:  # category - use PT code prefix as category
                supply_df = supply_df.copy()
                demand_df = demand_df.copy()
                supply_df['category'] = supply_df['pt_code'].str[:2] if 'pt_code' in supply_df.columns else 'Unknown'
                demand_df['category'] = demand_df['pt_code'].str[:2] if 'pt_code' in demand_df.columns else 'Unknown'
                group_cols = ['category']
                sort_cols = ['category']
            
            # Aggregate supply by group
            supply_agg = self._aggregate_supply(supply_df, group_cols)
            
            # Aggregate demand by group
            demand_agg = self._aggregate_demand(demand_df, group_cols)
            
            # Merge supply and demand
            gap_df = pd.merge(
                supply_agg,
                demand_agg,
                on=group_cols,
                how='outer',
                suffixes=('_supply', '_demand')
            )
            
            # Fill NaN values with 0
            numeric_cols = gap_df.select_dtypes(include=[np.number]).columns
            gap_df[numeric_cols] = gap_df[numeric_cols].fillna(0)
            
            # Calculate GAP metrics
            gap_df['net_gap'] = gap_df['total_supply'] - gap_df['total_demand']
            
            # Calculate coverage ratio (aligned with SQL view)
            gap_df['coverage_ratio'] = np.where(
                gap_df['total_demand'] > 0,
                gap_df['total_supply'] / gap_df['total_demand'],
                np.where(gap_df['total_supply'] > 0, 999, 0)
            )
            
            # Calculate GAP percentage (for display)
            gap_df['gap_percentage'] = np.where(
                gap_df['total_demand'] > 0,
                (gap_df['net_gap'] / gap_df['total_demand']) * 100,
                np.where(gap_df['total_supply'] > 0, 100, 0)
            )
            
            # Add gap type (aligned with SQL)
            gap_df['gap_type'] = np.where(
                gap_df['total_supply'] > gap_df['total_demand'], 'SURPLUS',
                np.where(gap_df['total_supply'] < gap_df['total_demand'], 'SHORTAGE', 'BALANCED')
            )
            
            # Add status classification (aligned with SQL view thresholds)
            gap_df['gap_status'] = gap_df.apply(
                lambda row: self._classify_gap_status_sql_aligned(
                    row['total_supply'], 
                    row['total_demand'],
                    row.get('supply_inventory', 0),
                    row.get('supply_purchase_order', 0)
                ),
                axis=1
            )
            
            # Add priority (aligned with SQL)
            gap_df['priority'] = gap_df.apply(
                lambda row: self._calculate_priority(
                    row['total_supply'],
                    row['total_demand'],
                    row.get('supply_inventory', 0)
                ),
                axis=1
            )
            
            # Add suggested action (aligned with SQL)
            gap_df['suggested_action'] = gap_df.apply(
                lambda row: self._get_action_required(
                    row['total_supply'],
                    row['total_demand'],
                    row.get('supply_inventory', 0),
                    row.get('avg_days_to_required', None)
                ),
                axis=1
            )
            
            # Calculate value impact (if cost/price data available)
            if 'avg_unit_cost_usd' in gap_df.columns:
                gap_df['gap_value_usd'] = gap_df['net_gap'] * gap_df['avg_unit_cost_usd']
                gap_df['abs_gap_value_usd'] = abs(gap_df['gap_value_usd'])
                gap_df['at_risk_value_usd'] = np.where(
                    gap_df['net_gap'] < 0,
                    abs(gap_df['net_gap']) * gap_df.get('avg_selling_price_usd', gap_df['avg_unit_cost_usd']),
                    0
                )
            
            # Sort by priority and absolute gap value
            gap_df = gap_df.sort_values(
                by=['priority', 'abs_gap_value_usd'] if 'abs_gap_value_usd' in gap_df.columns else ['priority', 'net_gap'],
                ascending=[True, False]
            )
            
            logger.info(f"Calculated GAP for {len(gap_df)} {group_by} groups")
            
            return gap_df
            
        except Exception as e:
            logger.error(f"Error calculating net GAP: {e}")
            raise
    
    def _classify_gap_status_sql_aligned(self, supply: float, demand: float, 
                                         inventory: float = 0, po_qty: float = 0) -> str:
        """
        Classify GAP status aligned with SQL view logic
        """
        # No demand scenarios
        if demand == 0:
            if inventory > 0:
                return 'NO_DEMAND'
            elif po_qty > 0:
                return 'NO_DEMAND_INCOMING'
            else:
                return 'NO_DEMAND'
        
        # Calculate coverage ratio
        coverage = supply / demand
        
        # Surplus levels
        if coverage > 3:
            return 'SEVERE_SURPLUS'
        elif coverage > 2:
            return 'HIGH_SURPLUS'
        elif coverage > 1.5:
            return 'MODERATE_SURPLUS'
        elif coverage > 1.1:
            return 'LIGHT_SURPLUS'
        
        # Balanced
        elif coverage >= 0.9 and coverage <= 1.1:
            return 'BALANCED'
        
        # Shortage levels
        elif coverage < 0.5:
            return 'SEVERE_SHORTAGE'
        elif coverage < 0.7:
            return 'HIGH_SHORTAGE'
        elif coverage < 0.9:
            return 'MODERATE_SHORTAGE'
        else:
            return 'UNKNOWN'
    
    def _calculate_priority(self, supply: float, demand: float, inventory: float = 0) -> int:
        """
        Calculate action priority aligned with SQL view
        """
        # Critical priorities
        if demand == 0 and inventory > 0:
            return 1
        if demand > 0 and supply < demand * 0.5:
            return 1
        
        # High priorities
        if demand > 0 and supply > demand * 3:
            return 2
        if demand > 0 and supply < demand * 0.7:
            return 2
        
        # Medium priorities
        if demand > 0 and supply > demand * 2:
            return 3
        if demand > 0 and supply < demand * 0.9:
            return 3
        
        # Low priorities
        if demand > 0 and supply > demand * 1.5:
            return 4
        
        # OK
        return 99
    
    def _get_action_required(self, supply: float, demand: float, 
                            inventory: float = 0, avg_days_to_required: Optional[float] = None) -> str:
        """
        Generate action required message aligned with SQL view
        """
        gap = supply - demand
        
        # SURPLUS ACTIONS
        if demand == 0 and inventory > 0:
            return f"NO DEMAND: {inventory:.0f} units in stock need liquidation"
        
        if demand > 0:
            coverage = supply / demand
            
            if coverage > 3:
                return f"SEVERE SURPLUS: {gap:.0f} units excess. Stop ordering & promote aggressively"
            elif coverage > 2:
                return f"HIGH SURPLUS: {gap:.0f} units excess. Reduce orders & plan promotions"
            elif coverage > 1.5:
                return f"MODERATE SURPLUS: {gap:.0f} units excess. Review ordering"
            elif coverage > 1.1:
                return "LIGHT SURPLUS: Minor excess, monitor closely"
            
            # BALANCED
            elif coverage >= 0.9 and coverage <= 1.1:
                return "BALANCED: Supply matches demand well"
            
            # SHORTAGE ACTIONS
            elif coverage < 0.5:
                shortage = abs(gap)
                return f"SEVERE SHORTAGE: Need {shortage:.0f} units URGENTLY. Expedite orders!"
            elif coverage < 0.7:
                shortage = abs(gap)
                return f"HIGH SHORTAGE: Need {shortage:.0f} units. Rush orders required"
            elif coverage < 0.9:
                shortage = abs(gap)
                return f"MODERATE SHORTAGE: Need {shortage:.0f} more units"
        
        return "Status unclear - review manually"
    
    def _aggregate_supply(self, supply_df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
        """
        Aggregate supply data by specified columns
        """
        # Filter out expired items
        today = pd.Timestamp.now().date()
        supply_df = supply_df.copy()
        
        if 'expiry_date' in supply_df.columns:
            supply_df = supply_df[
                (supply_df['expiry_date'].isna()) | 
                (pd.to_datetime(supply_df['expiry_date']).dt.date > today)
            ]
        
        # Create aggregation dictionary
        agg_dict = {
            'available_quantity': 'sum',
            'total_value_usd': 'sum'
        }
        
        # Add source-specific aggregations
        for source in self.supply_sources:
            source_df = supply_df[supply_df['supply_source'] == source].copy()
            if not source_df.empty:
                supply_df[f'supply_{source.lower()}'] = np.where(
                    supply_df['supply_source'] == source,
                    supply_df['available_quantity'],
                    0
                )
                agg_dict[f'supply_{source.lower()}'] = 'sum'
        
        # Add timing metrics
        if 'days_to_available' in supply_df.columns:
            agg_dict['days_to_available'] = 'mean'
        
        if 'unit_cost_usd' in supply_df.columns:
            # Calculate weighted average cost
            supply_df['cost_x_qty'] = supply_df['unit_cost_usd'] * supply_df['available_quantity']
            agg_dict['cost_x_qty'] = 'sum'
        
        # Perform aggregation
        supply_agg = supply_df.groupby(group_cols, as_index=False).agg(agg_dict)
        
        # Rename columns
        supply_agg = supply_agg.rename(columns={
            'available_quantity': 'total_supply',
            'total_value_usd': 'supply_value_usd',
            'days_to_available': 'avg_days_to_available'
        })
        
        # Calculate weighted average unit cost
        if 'cost_x_qty' in supply_agg.columns:
            supply_agg['avg_unit_cost_usd'] = np.where(
                supply_agg['total_supply'] > 0,
                supply_agg['cost_x_qty'] / supply_agg['total_supply'],
                0
            )
            supply_agg = supply_agg.drop(columns=['cost_x_qty'])
        
        # Add supply source breakdown for tooltips
        source_cols = [col for col in supply_agg.columns if col.startswith('supply_')]
        if source_cols:
            supply_agg['supply_breakdown'] = supply_agg[source_cols].to_dict('records')
        
        return supply_agg
    
    def _aggregate_demand(self, demand_df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
        """
        Aggregate demand data by specified columns
        """
        demand_df = demand_df.copy()
        
        # Create aggregation dictionary
        agg_dict = {
            'required_quantity': 'sum',
            'total_value_usd': 'sum',
            'allocated_quantity': 'sum',
            'unallocated_quantity': 'sum',
            'customer': 'nunique'  # Count unique customers
        }
        
        # Add source-specific aggregations
        for source in self.demand_sources:
            source_df = demand_df[demand_df['demand_source'] == source].copy()
            if not source_df.empty:
                demand_df[f'demand_{source.lower()}'] = np.where(
                    demand_df['demand_source'] == source,
                    demand_df['required_quantity'],
                    0
                )
                agg_dict[f'demand_{source.lower()}'] = 'sum'
        
        # Add timing and urgency metrics
        if 'days_to_required' in demand_df.columns:
            agg_dict['days_to_required'] = 'mean'
            
            # Count urgency levels
            for urgency in ['OVERDUE', 'URGENT', 'UPCOMING', 'FUTURE']:
                demand_df[f'count_{urgency.lower()}'] = np.where(
                    demand_df['urgency_level'] == urgency, 1, 0
                )
                agg_dict[f'count_{urgency.lower()}'] = 'sum'
        
        if 'selling_unit_price' in demand_df.columns:
            # Calculate weighted average price
            demand_df['price_x_qty'] = demand_df['selling_unit_price'] * demand_df['required_quantity']
            agg_dict['price_x_qty'] = 'sum'
        
        # Add over-commitment metrics
        if 'over_committed_qty_standard' in demand_df.columns:
            agg_dict['over_committed_qty_standard'] = 'sum'
        
        # Perform aggregation
        demand_agg = demand_df.groupby(group_cols, as_index=False).agg(agg_dict)
        
        # Rename columns
        demand_agg = demand_agg.rename(columns={
            'required_quantity': 'total_demand',
            'total_value_usd': 'demand_value_usd',
            'days_to_required': 'avg_days_to_required',
            'customer': 'customer_count'
        })
        
        # Calculate weighted average selling price
        if 'price_x_qty' in demand_agg.columns:
            demand_agg['avg_selling_price_usd'] = np.where(
                demand_agg['total_demand'] > 0,
                demand_agg['price_x_qty'] / demand_agg['total_demand'],
                0
            )
            demand_agg = demand_agg.drop(columns=['price_x_qty'])
        
        # Calculate allocation coverage
        demand_agg['allocation_coverage'] = np.where(
            demand_agg['total_demand'] > 0,
            (demand_agg['allocated_quantity'] / demand_agg['total_demand']) * 100,
            0
        )
        
        return demand_agg
    
    def get_summary_metrics(self, gap_df: pd.DataFrame) -> Dict[str, any]:
        """
        Calculate summary metrics from GAP analysis
        """
        # Count different statuses
        shortage_statuses = ['SEVERE_SHORTAGE', 'HIGH_SHORTAGE', 'MODERATE_SHORTAGE']
        critical_statuses = ['SEVERE_SHORTAGE', 'HIGH_SHORTAGE']
        surplus_statuses = ['SEVERE_SURPLUS', 'HIGH_SURPLUS', 'MODERATE_SURPLUS', 'LIGHT_SURPLUS']
        
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
            
            # Critical items (severe + high shortage)
            'critical_items': len(gap_df[gap_df['gap_status'].isin(critical_statuses)]),
            
            # Coverage
            'overall_coverage': (gap_df['total_supply'].sum() / gap_df['total_demand'].sum() * 100) 
                                if gap_df['total_demand'].sum() > 0 else 100,
            
            # Value metrics (if available)
            'at_risk_value_usd': gap_df.get('at_risk_value_usd', pd.Series([0])).sum(),
            'gap_value_usd': gap_df.get('gap_value_usd', pd.Series([0])).sum(),
            
            # Customer impact
            'affected_customers': gap_df[gap_df['net_gap'] < 0].get('customer_count', pd.Series([0])).sum(),
            
            # Urgency breakdown
            'overdue_items': gap_df.get('count_overdue', pd.Series([0])).sum(),
            'urgent_items': gap_df.get('count_urgent', pd.Series([0])).sum()
        }
        
        return metrics