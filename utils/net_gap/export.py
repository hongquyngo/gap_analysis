# utils/net_gap/export.py

"""
Enhanced Excel export with cost transparency
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
import io
import logging

from .constants import EXPORT_CONFIG, GAP_CATEGORIES
from .formatters import GAPFormatter

logger = logging.getLogger(__name__)


def export_to_excel(
    result,
    filters: Dict[str, Any],
    include_cost_breakdown: bool = True
) -> bytes:
    """
    Export GAP analysis to Excel with cost transparency
    
    Args:
        result: GAPCalculationResult object
        filters: Filters used for calculation
        include_cost_breakdown: Include detailed cost analysis
    
    Returns:
        Excel file as bytes
    """
    
    formatter = GAPFormatter()
    output = io.BytesIO()
    
    try:
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # 1. Summary Sheet
            summary_df = _create_summary_sheet(result, filters, formatter)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # 2. GAP Details with cost columns
            details_df = _create_details_sheet(result.gap_df, formatter)
            details_df.to_excel(writer, sheet_name='GAP Details', index=False)
            
            # 3. Cost Breakdown (if requested)
            if include_cost_breakdown and 'avg_unit_cost_usd' in result.gap_df.columns:
                cost_df = _create_cost_breakdown(result.gap_df, formatter)
                cost_df.to_excel(writer, sheet_name='Cost Analysis', index=False)
            
            # 4. Calculation Guide
            guide_df = _create_calculation_guide()
            guide_df.to_excel(writer, sheet_name='Calculation Guide', index=False)
            
            # 5. Customer Impact (if available)
            if result.customer_impact and not result.customer_impact.is_empty():
                customer_df = _create_customer_sheet(result.customer_impact)
                customer_df.to_excel(writer, sheet_name='Customer Impact', index=False)
            
            # Format worksheets
            _format_excel_sheets(writer)
        
        output.seek(0)
        logger.info("Excel export generated successfully")
        return output.getvalue()
        
    except Exception as e:
        logger.error(f"Excel export failed: {e}", exc_info=True)
        raise


def _create_summary_sheet(result, filters: Dict, formatter) -> pd.DataFrame:
    """Create summary sheet with key metrics and filters"""
    
    metrics = result.metrics
    categories = result.get_category_summary()
    
    summary_data = []
    
    # Report Info
    summary_data.extend([
        ['Report Information', ''],
        ['Generated', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        ['', '']
    ])
    
    # Filters Applied
    summary_data.extend([
        ['Filters Applied', ''],
        ['Entity', filters.get('entity', 'All Entities')],
        ['Products', f"{len(filters.get('products', []))} selected" if filters.get('products') else 'All'],
        ['Brands', ', '.join(filters.get('brands', [])) or 'All'],
        ['Expired Inventory', 'Excluded' if filters.get('exclude_expired') else 'Included'],
        ['Safety Stock', 'Included' if filters.get('include_safety') else 'Not Included'],
        ['', '']
    ])
    
    # Key Metrics
    summary_data.extend([
        ['Key Metrics', ''],
        ['Total Products', f"{metrics['total_products']:,}"],
        ['Coverage Rate', f"{metrics['overall_coverage']:.1f}%"],
        ['', ''],
        ['Shortage Items', f"{metrics['shortage_items']:,}"],
        ['Optimal Items', f"{categories.get('optimal', 0):,}"],
        ['Surplus Items', f"{metrics['surplus_items']:,}"],
        ['Inactive Items', f"{categories.get('inactive', 0):,}"],
        ['', '']
    ])
    
    # Financial Impact
    summary_data.extend([
        ['Financial Impact', ''],
        ['Total Supply Value', formatter.format_currency(metrics.get('total_supply_value_usd', 0))],
        ['Total Demand Value', formatter.format_currency(metrics.get('total_demand_value_usd', 0))],
        ['Revenue at Risk', formatter.format_currency(metrics['at_risk_value_usd'])],
        ['Total Shortage Value', formatter.format_currency(metrics['total_shortage'])],
        ['Total Surplus Value', formatter.format_currency(metrics['total_surplus'])],
        ['', '']
    ])
    
    # Customer Impact
    if metrics.get('affected_customers', 0) > 0:
        summary_data.extend([
            ['Customer Impact', ''],
            ['Affected Customers', f"{metrics['affected_customers']:,}"],
            ['Overdue Items', f"{metrics.get('overdue_items', 0):,}"],
            ['Urgent Items', f"{metrics.get('urgent_items', 0):,}"]
        ])
    
    return pd.DataFrame(summary_data, columns=['Metric', 'Value'])


def _create_details_sheet(gap_df: pd.DataFrame, formatter) -> pd.DataFrame:
    """Create details sheet with essential columns and cost info"""
    
    if gap_df.empty:
        return pd.DataFrame()
    
    # Select columns to export
    export_columns = [
        'product_id', 'pt_code', 'product_name', 'brand',
        'total_supply', 'total_demand', 'net_gap',
        'coverage_ratio', 'gap_percentage', 'gap_status',
        'avg_unit_cost_usd', 'avg_selling_price_usd',
        'gap_value_usd', 'at_risk_value_usd',
        'priority', 'suggested_action'
    ]
    
    # Add safety columns if present
    safety_columns = ['safety_stock_qty', 'available_supply', 'below_reorder']
    for col in safety_columns:
        if col in gap_df.columns:
            export_columns.append(col)
    
    # Filter to available columns
    export_columns = [col for col in export_columns if col in gap_df.columns]
    
    # Create export dataframe
    export_df = gap_df[export_columns].copy()
    
    # Format numeric columns
    if 'coverage_ratio' in export_df.columns:
        export_df['coverage_ratio'] = export_df['coverage_ratio'].apply(
            lambda x: x if pd.notna(x) and x < 10 else None
        )
    
    # Limit rows if needed
    if len(export_df) > EXPORT_CONFIG['max_rows']:
        export_df = export_df.head(EXPORT_CONFIG['max_rows'])
        logger.warning(f"Export limited to {EXPORT_CONFIG['max_rows']} rows")
    
    return export_df


def _create_cost_breakdown(gap_df: pd.DataFrame, formatter) -> pd.DataFrame:
    """Create detailed cost breakdown sheet"""
    
    breakdown_data = []
    
    for _, row in gap_df.iterrows():
        breakdown_data.append({
            'Product Code': row.get('pt_code', ''),
            'Product Name': row.get('product_name', ''),
            'Brand': row.get('brand', ''),
            
            # Quantities
            'Supply Qty': row.get('total_supply', 0),
            'Demand Qty': row.get('total_demand', 0),
            'GAP Qty': row.get('net_gap', 0),
            
            # Unit Costs
            'Avg Unit Cost (USD)': row.get('avg_unit_cost_usd', 0),
            'Avg Selling Price (USD)': row.get('avg_selling_price_usd', 0),
            'Margin per Unit': row.get('avg_selling_price_usd', 0) - row.get('avg_unit_cost_usd', 0),
            
            # Value Calculations
            'Supply Value': row.get('total_supply', 0) * row.get('avg_unit_cost_usd', 0),
            'Demand Value': row.get('total_demand', 0) * row.get('avg_selling_price_usd', 0),
            'GAP Value': row.get('gap_value_usd', 0),
            
            # Risk Analysis
            'Shortage Qty': abs(row.get('net_gap', 0)) if row.get('net_gap', 0) < 0 else 0,
            'Revenue at Risk': row.get('at_risk_value_usd', 0),
            'Lost Margin': (abs(row.get('net_gap', 0)) * 
                          (row.get('avg_selling_price_usd', 0) - row.get('avg_unit_cost_usd', 0))
                          if row.get('net_gap', 0) < 0 else 0),
            
            # Status
            'Status': row.get('gap_status', ''),
            'Priority': row.get('priority', 99)
        })
    
    return pd.DataFrame(breakdown_data)


def _create_calculation_guide() -> pd.DataFrame:
    """Create calculation guide sheet"""
    
    guide_data = [
        {
            'Metric': 'Net GAP',
            'Formula': 'Total Supply - Total Demand',
            'Purpose': 'Identifies shortage (negative) or surplus (positive)',
            'Example': '100 - 150 = -50 (shortage of 50 units)'
        },
        {
            'Metric': 'Coverage Ratio',
            'Formula': 'Total Supply ÷ Total Demand',
            'Purpose': 'Supply as percentage of demand',
            'Example': '100 ÷ 150 = 0.67 (67% coverage)'
        },
        {
            'Metric': 'GAP Value',
            'Formula': 'Net GAP × Average Unit Cost',
            'Purpose': 'Inventory value of the gap',
            'Example': '-50 × $10 = -$500 (shortage value)'
        },
        {
            'Metric': 'At Risk Value',
            'Formula': '|Net GAP| × Selling Price (when GAP < 0)',
            'Purpose': 'Revenue at risk due to shortage',
            'Example': '50 × $15 = $750 revenue at risk'
        },
        {
            'Metric': 'Lost Margin',
            'Formula': '|Net GAP| × (Selling Price - Unit Cost)',
            'Purpose': 'Profit margin lost due to shortage',
            'Example': '50 × ($15 - $10) = $250 lost margin'
        },
        {
            'Metric': 'Available Supply',
            'Formula': 'Total Supply - Safety Stock',
            'Purpose': 'Supply available after safety requirement',
            'Example': '100 - 20 = 80 units available'
        },
        {
            'Metric': 'Safety Coverage',
            'Formula': 'Current Inventory ÷ Safety Stock',
            'Purpose': 'How many times safety stock is covered',
            'Example': '50 ÷ 20 = 2.5x coverage'
        }
    ]
    
    return pd.DataFrame(guide_data)


def _create_customer_sheet(customer_impact) -> pd.DataFrame:
    """Create customer impact sheet"""
    
    if customer_impact.customer_df.empty:
        return pd.DataFrame()
    
    # Select relevant columns
    customer_df = customer_impact.customer_df[[
        'customer', 'customer_code', 'product_count',
        'total_required', 'total_shortage', 
        'total_demand_value', 'at_risk_value',
        'urgency'
    ]].copy()
    
    # Sort by at risk value
    customer_df = customer_df.sort_values('at_risk_value', ascending=False)
    
    return customer_df


def _format_excel_sheets(writer):
    """Apply formatting to Excel sheets"""
    
    workbook = writer.book
    
    # Format each sheet
    for sheet_name in writer.sheets:
        worksheet = writer.sheets[sheet_name]
        
        # Auto-adjust column widths
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            
            for cell in column:
                try:
                    if cell.value:
                        max_length = max(max_length, len(str(cell.value)))
                except:
                    pass
            
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column_letter].width = adjusted_width