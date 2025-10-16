# utils/net_gap/calculation_result.py

"""
GAP Calculation Result Container - Version 3.2 ENHANCED
- Added exclusion filter tracking
- Improved filter hash generation
- Complete metadata for reproducibility
"""

import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class SourceData:
    """Raw source data used in calculation"""
    supply_df: pd.DataFrame
    demand_df: pd.DataFrame
    safety_stock_df: Optional[pd.DataFrame] = None
    
    def __post_init__(self):
        """Validate data on creation"""
        if self.supply_df is None or self.demand_df is None:
            raise ValueError("Supply and demand dataframes are required")


@dataclass
class CustomerImpactData:
    """Pre-calculated customer impact analysis"""
    customer_summary_df: pd.DataFrame
    affected_count: int
    total_at_risk_value: float
    total_shortage_qty: float
    
    def is_empty(self) -> bool:
        return self.customer_summary_df.empty or self.affected_count == 0


@dataclass
class GAPCalculationResult:
    """
    Complete GAP calculation result with all associated data
    Single source of truth for the entire calculation
    Version 3.2: Enhanced with exclusion filter tracking
    """
    gap_df: pd.DataFrame
    metrics: Dict[str, Any]
    source_data: SourceData
    customer_impact: Optional[CustomerImpactData]
    filters_used: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate result on creation"""
        if self.gap_df is None or self.gap_df.empty:
            logger.warning("GAP calculation result is empty")
        
        if self.metrics is None:
            raise ValueError("Metrics dictionary is required")
        
        # Ensure exclusion flags exist in filters
        if 'exclude_products' not in self.filters_used:
            self.filters_used['exclude_products'] = False
        if 'exclude_brands' not in self.filters_used:
            self.filters_used['exclude_brands'] = False
        if 'exclude_expired_inventory' not in self.filters_used:
            self.filters_used['exclude_expired_inventory'] = True
        
        logger.info(f"GAPCalculationResult created: {len(self.gap_df)} items, "
                   f"{self.metrics.get('affected_customers', 0)} affected customers, "
                   f"exclusions: products={self.filters_used.get('exclude_products')}, "
                   f"brands={self.filters_used.get('exclude_brands')}, "
                   f"expired={self.filters_used.get('exclude_expired_inventory')}")
    
    def get_shortage_products(self) -> list:
        """Get list of product IDs with shortages"""
        if self.gap_df.empty or 'product_id' not in self.gap_df.columns:
            return []
        
        shortage_df = self.gap_df[self.gap_df['net_gap'] < 0]
        return shortage_df['product_id'].tolist()
    
    def get_demand_for_products(self, product_ids: list) -> pd.DataFrame:
        """Get demand data for specific products"""
        if not product_ids:
            return pd.DataFrame()
        
        return self.source_data.demand_df[
            self.source_data.demand_df['product_id'].isin(product_ids)
        ].copy()
    
    def get_filter_hash(self) -> str:
        """
        Generate hash for filter comparison
        Enhanced: Includes exclusion flags
        """
        def safe_str(val):
            if val is None:
                return 'None'
            if isinstance(val, (list, tuple)):
                return str(sorted(val) if val else [])
            return str(val)
        
        key_parts = [
            str(self.filters_used.get('entity')),
            safe_str(self.filters_used.get('products', [])),
            safe_str(self.filters_used.get('brands', [])),
            str(self.filters_used.get('exclude_products', False)),
            str(self.filters_used.get('exclude_brands', False)),
            str(self.filters_used.get('exclude_expired_inventory', True)),
            safe_str(self.filters_used.get('supply_sources', [])),
            safe_str(self.filters_used.get('demand_sources', [])),
            str(self.filters_used.get('include_safety_stock', False)),
            str(self.filters_used.get('group_by', 'product'))
        ]
        return '|'.join(key_parts)
    
    def get_exclusion_summary(self) -> Dict[str, Any]:
        """Get summary of exclusion filters applied"""
        return {
            'products_excluded': self.filters_used.get('exclude_products', False),
            'products_count': len(self.filters_used.get('products', [])),
            'brands_excluded': self.filters_used.get('exclude_brands', False),
            'brands_count': len(self.filters_used.get('brands', [])),
            'expired_excluded': self.filters_used.get('exclude_expired_inventory', True)
        }
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """Get summary for logging/debugging"""
        exclusion_summary = self.get_exclusion_summary()
        
        return {
            'items_count': len(self.gap_df),
            'shortage_items': self.metrics.get('shortage_items', 0),
            'affected_customers': self.metrics.get('affected_customers', 0),
            'at_risk_value': self.metrics.get('at_risk_value_usd', 0),
            'calculation_time': self.timestamp.isoformat(),
            'has_customer_impact': self.customer_impact is not None and not self.customer_impact.is_empty(),
            'filters': {
                'entity': self.filters_used.get('entity'),
                'products_count': len(self.filters_used.get('products', [])),
                'brands_count': len(self.filters_used.get('brands', [])),
                'group_by': self.filters_used.get('group_by'),
                'safety_stock': self.filters_used.get('include_safety_stock', False),
                'exclusions': exclusion_summary
            }
        }
    
    def to_export_metadata(self) -> Dict[str, Any]:
        """Get metadata for Excel export"""
        exclusion_summary = self.get_exclusion_summary()
        
        metadata = {
            'Generated': self.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'Total Items': len(self.gap_df),
            'Shortage Items': self.metrics.get('shortage_items', 0),
            'Critical Items': self.metrics.get('critical_items', 0),
            'Coverage Rate': f"{self.metrics.get('overall_coverage', 0):.1f}%",
            'Affected Customers': self.metrics.get('affected_customers', 0),
            'At Risk Value': f"${self.metrics.get('at_risk_value_usd', 0):,.2f}",
            '_separator_1': '',
            'Filter Configuration': '',
            'Entity': self.filters_used.get('entity', 'All'),
            'Group By': self.filters_used.get('group_by', 'product'),
            'Safety Stock': 'Yes' if self.filters_used.get('include_safety_stock') else 'No',
            '_separator_2': '',
            'Exclusion Filters': '',
            'Products Mode': 'EXCLUDED' if exclusion_summary['products_excluded'] else 'INCLUDED',
            'Products Count': exclusion_summary['products_count'],
            'Brands Mode': 'EXCLUDED' if exclusion_summary['brands_excluded'] else 'INCLUDED',
            'Brands Count': exclusion_summary['brands_count'],
            'Expired Inventory': 'EXCLUDED' if exclusion_summary['expired_excluded'] else 'INCLUDED',
            '_separator_3': '',
            'Data Sources': '',
            'Supply Sources': ', '.join(self.filters_used.get('supply_sources', [])),
            'Demand Sources': ', '.join(self.filters_used.get('demand_sources', []))
        }
        
        return metadata