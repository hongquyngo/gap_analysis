# utils/period_gap/helpers.py
"""
General Helper Functions
Excel export, period manipulation, session state management
Version 2.0 - Enhanced with metadata export and improved period formatting
"""

import pandas as pd
import streamlit as st
from io import BytesIO
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
import logging

logger = logging.getLogger(__name__)

# === CONSTANTS ===
EXCEL_SHEET_NAME_LIMIT = 31
DEFAULT_EXCEL_ENGINE = "xlsxwriter"

EXCEL_HEADER_FORMAT = {
    'bold': True,
    'text_wrap': True,
    'valign': 'top',
    'fg_color': '#D7E4BD',
    'border': 1
}

# === EXCEL EXPORT FUNCTIONS ===

def convert_df_to_excel(df: pd.DataFrame, sheet_name: str = "Data") -> bytes:
    """Convert dataframe to Excel bytes with auto-formatting"""
    if df.empty:
        logger.warning("Attempting to convert empty DataFrame to Excel")
        return BytesIO().getvalue()
    
    output = BytesIO()
    
    try:
        with pd.ExcelWriter(output, engine=DEFAULT_EXCEL_ENGINE) as writer:
            sheet_name = sheet_name[:EXCEL_SHEET_NAME_LIMIT]
            df.to_excel(writer, index=False, sheet_name=sheet_name)
            
            workbook = writer.book
            worksheet = writer.sheets[sheet_name]
            
            header_format = workbook.add_format(EXCEL_HEADER_FORMAT)
            
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)
            
            for i, col in enumerate(df.columns):
                try:
                    max_len = df[col].astype(str).map(len).max()
                    max_len = max(max_len, len(str(col))) + 2
                    worksheet.set_column(i, i, min(max_len, 50))
                except Exception as e:
                    logger.debug(f"Could not calculate width for column {col}: {e}")
                    worksheet.set_column(i, i, 15)
        
        return output.getvalue()
        
    except Exception as e:
        logger.error(f"Error converting DataFrame to Excel: {e}")
        raise


def export_multiple_sheets(dataframes_dict: Dict[str, pd.DataFrame]) -> bytes:
    """Export multiple dataframes to different sheets in one Excel file"""
    if not dataframes_dict:
        logger.warning("No DataFrames provided for multi-sheet export")
        return BytesIO().getvalue()
    
    output = BytesIO()
    
    try:
        with pd.ExcelWriter(output, engine=DEFAULT_EXCEL_ENGINE) as writer:
            for sheet_name, df in dataframes_dict.items():
                if df is None or df.empty:
                    logger.debug(f"Skipping empty sheet: {sheet_name}")
                    continue
                    
                truncated_name = sheet_name[:EXCEL_SHEET_NAME_LIMIT]
                df.to_excel(writer, index=False, sheet_name=truncated_name)
                
                workbook = writer.book
                worksheet = writer.sheets[truncated_name]
                
                header_format = workbook.add_format(EXCEL_HEADER_FORMAT)
                
                for col_num, value in enumerate(df.columns.values):
                    worksheet.write(0, col_num, value, header_format)
                    
                for i, col in enumerate(df.columns):
                    try:
                        max_len = df[col].astype(str).map(len).max()
                        max_len = max(max_len, len(str(col))) + 2
                        worksheet.set_column(i, i, min(max_len, 50))
                    except:
                        worksheet.set_column(i, i, 15)
        
        return output.getvalue()
        
    except Exception as e:
        logger.error(f"Error exporting multiple sheets: {e}")
        raise


def create_metadata_sheet(
    filter_values: Dict[str, Any],
    calc_options: Dict[str, Any],
    gap_df: pd.DataFrame,
    display_filters: Dict[str, Any],
    df_demand_filtered: pd.DataFrame,
    df_supply_filtered: pd.DataFrame
) -> pd.DataFrame:
    """
    Create Export_Info metadata sheet with analysis parameters and summary statistics
    
    Args:
        filter_values: Data filters applied
        calc_options: Calculation options used
        gap_df: GAP analysis results
        display_filters: Display filters applied
        df_demand_filtered: Filtered demand data
        df_supply_filtered: Filtered supply data
    
    Returns:
        DataFrame formatted for metadata sheet
    """
    from .shortage_analyzer import categorize_products
    
    metadata_rows = []
    
    # === EXPORT INFORMATION ===
    metadata_rows.append(['EXPORT INFORMATION', ''])
    metadata_rows.append(['Export Date', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
    metadata_rows.append(['Report Type', 'Period GAP Analysis'])
    metadata_rows.append(['', ''])
    
    # === CALCULATION PARAMETERS ===
    metadata_rows.append(['CALCULATION PARAMETERS', ''])
    metadata_rows.append(['Period Type', calc_options.get('period_type', 'Weekly')])
    metadata_rows.append(['Track Backlog', 'Yes' if calc_options.get('track_backlog', True) else 'No'])
    metadata_rows.append(['Exclude Missing Dates', 'Yes' if calc_options.get('exclude_missing_dates', True) else 'No'])
    metadata_rows.append(['', ''])
    
    # === DATA FILTERS ===
    metadata_rows.append(['DATA FILTERS', ''])
    
    if filter_values.get('entity'):
        entity_mode = "Excluded" if filter_values.get('exclude_entity', False) else "Included"
        metadata_rows.append(['Legal Entity', f"{entity_mode}: {', '.join(filter_values['entity'])}"])
    else:
        metadata_rows.append(['Legal Entity', 'All'])
    
    if filter_values.get('brand'):
        brand_mode = "Excluded" if filter_values.get('exclude_brand', False) else "Included"
        metadata_rows.append(['Brand', f"{brand_mode}: {', '.join(filter_values['brand'])}"])
    else:
        metadata_rows.append(['Brand', 'All'])
    
    if filter_values.get('product'):
        product_mode = "Excluded" if filter_values.get('exclude_product', False) else "Included"
        metadata_rows.append(['Products', f"{product_mode}: {len(filter_values['product'])} products"])
    else:
        metadata_rows.append(['Products', 'All'])
    
    if filter_values.get('start_date') and filter_values.get('end_date'):
        metadata_rows.append(['Date Range', f"{filter_values['start_date']} to {filter_values['end_date']}"])
    
    metadata_rows.append(['', ''])
    
    # === DISPLAY FILTERS ===
    metadata_rows.append(['DISPLAY FILTERS', ''])
    metadata_rows.append(['Period Filter', display_filters.get('period_filter', 'All')])
    
    product_types = []
    if display_filters.get('show_matched', True):
        product_types.append('Matched')
    if display_filters.get('show_demand_only', True):
        product_types.append('Demand Only')
    if display_filters.get('show_supply_only', True):
        product_types.append('Supply Only')
    metadata_rows.append(['Product Types', ', '.join(product_types)])
    metadata_rows.append(['', ''])
    
    # === SUMMARY STATISTICS ===
    metadata_rows.append(['SUMMARY STATISTICS', ''])
    
    if not gap_df.empty:
        total_products = gap_df['pt_code'].nunique()
        total_periods = gap_df['period'].nunique()
        
        metadata_rows.append(['Total Products', total_products])
        metadata_rows.append(['Total Periods', total_periods])
        metadata_rows.append(['Total Records', len(gap_df)])
        metadata_rows.append(['', ''])
        
        # Shortage & Surplus categorization
        categorization = categorize_products(gap_df)
        
        metadata_rows.append(['CATEGORIZATION', ''])
        metadata_rows.append(['Net Shortage Products', len(categorization['net_shortage'])]),
        metadata_rows.append(['Balanced Products', len(categorization['balanced'])]),
        metadata_rows.append(['Net Surplus Products', len(categorization['net_surplus'])]),
        metadata_rows.append(['', ''])
        metadata_rows.append(['TIMING FLAGS', ''])
        metadata_rows.append(['Timing Shortage Products', len(categorization['timing_shortage'])]),
        metadata_rows.append(['Timing Surplus Products', len(categorization['timing_surplus'])]),
        metadata_rows.append(['', ''])
        
        # Supply vs Demand totals
        total_demand = gap_df['total_demand_qty'].sum()
        total_supply = gap_df['supply_in_period'].sum()
        net_position = total_supply - total_demand
        
        metadata_rows.append(['SUPPLY vs DEMAND', ''])
        metadata_rows.append(['Total Demand', f"{total_demand:,.2f}"])
        metadata_rows.append(['Total Supply', f"{total_supply:,.2f}"])
        metadata_rows.append(['Net Position', f"{net_position:,.2f}"])
        
        if total_demand > 0:
            fill_rate = min(100, total_supply / total_demand * 100)
            metadata_rows.append(['Overall Fill Rate', f"{fill_rate:.1f}%"])
        
        metadata_rows.append(['', ''])
        
        # Shortage/Surplus quantities
        total_shortage = abs(gap_df[gap_df['gap_quantity'] < 0]['gap_quantity'].sum())
        total_surplus = gap_df[gap_df['gap_quantity'] > 0]['gap_quantity'].sum()
        
        metadata_rows.append(['Total Shortage Quantity', f"{total_shortage:,.2f}"])
        metadata_rows.append(['Total Surplus Quantity', f"{total_surplus:,.2f}"])
        
        # Backlog info if tracking
        if calc_options.get('track_backlog', True) and 'backlog_to_next' in gap_df.columns:
            final_backlog = gap_df.groupby('pt_code')['backlog_to_next'].last().sum()
            products_with_backlog = (gap_df.groupby('pt_code')['backlog_to_next'].last() > 0).sum()
            
            metadata_rows.append(['', ''])
            metadata_rows.append(['Final Backlog', f"{final_backlog:,.2f}"])
            metadata_rows.append(['Products with Backlog', products_with_backlog])
    
    metadata_rows.append(['', ''])
    
    # === SOURCE DATA COUNTS ===
    metadata_rows.append(['SOURCE DATA COUNTS', ''])
    metadata_rows.append(['Demand Records', len(df_demand_filtered)])
    metadata_rows.append(['Supply Records', len(df_supply_filtered)])
    
    # Convert to DataFrame
    metadata_df = pd.DataFrame(metadata_rows, columns=['Parameter', 'Value'])
    
    return metadata_df


def export_gap_with_metadata(
    gap_df: pd.DataFrame,
    filter_values: Dict[str, Any],
    display_filters: Dict[str, Any],
    calc_options: Dict[str, Any],
    df_demand_filtered: pd.DataFrame,
    df_supply_filtered: pd.DataFrame
) -> bytes:
    """
    Export GAP analysis with metadata sheet and enhanced period formatting
    
    Args:
        gap_df: GAP analysis results
        filter_values: Data filters applied
        display_filters: Display filters applied
        calc_options: Calculation options
        df_demand_filtered: Filtered demand data
        df_supply_filtered: Filtered supply data
    
    Returns:
        Excel file bytes with multiple sheets
    """
    from .period_helpers import format_period_with_dates
    
    if gap_df.empty:
        logger.warning("Empty GAP dataframe for export")
        return BytesIO().getvalue()
    
    # Prepare GAP data for export with enhanced period formatting
    export_df = gap_df.copy()
    
    # Format period column with date ranges
    period_type = calc_options.get('period_type', 'Weekly')
    if 'period' in export_df.columns:
        export_df['period'] = export_df['period'].apply(
            lambda x: format_period_with_dates(x, period_type)
        )
    
    # Create metadata sheet
    metadata_df = create_metadata_sheet(
        filter_values=filter_values,
        calc_options=calc_options,
        gap_df=gap_df,
        display_filters=display_filters,
        df_demand_filtered=df_demand_filtered,
        df_supply_filtered=df_supply_filtered
    )
    
    # Create product summary sheet
    summary_df = create_product_summary(gap_df, calc_options)
    
    # Prepare sheets dictionary
    sheets_dict = {
        'Export_Info': metadata_df,
        'GAP_Analysis': export_df,
        'Product_Summary': summary_df
    }
    
    # Export to Excel
    return export_multiple_sheets(sheets_dict)


def create_product_summary(gap_df: pd.DataFrame, calc_options: Dict[str, Any]) -> pd.DataFrame:
    """
    Create product-level summary for export
    
    Args:
        gap_df: GAP analysis dataframe
        calc_options: Calculation options
    
    Returns:
        Product summary dataframe
    """
    from .shortage_analyzer import categorize_products
    
    if gap_df.empty:
        return pd.DataFrame()
    
    # Get categorization
    categorization = categorize_products(gap_df)
    
    summary_data = []
    
    for pt_code in gap_df['pt_code'].unique():
        product_df = gap_df[gap_df['pt_code'] == pt_code]
        
        # Basic info
        product_name = product_df['product_name'].iloc[0] if 'product_name' in product_df.columns else ''
        brand = product_df['brand'].iloc[0] if 'brand' in product_df.columns else ''
        package_size = product_df['package_size'].iloc[0] if 'package_size' in product_df.columns else ''
        standard_uom = product_df['standard_uom'].iloc[0] if 'standard_uom' in product_df.columns else ''
        
        # Totals
        total_demand = product_df['total_demand_qty'].sum()
        total_supply = product_df['supply_in_period'].sum()
        net_position = total_supply - total_demand
        
        # Period counts
        total_periods = len(product_df)
        shortage_periods = (product_df['gap_quantity'] < 0).sum()
        surplus_periods = (product_df['gap_quantity'] > 0).sum()
        balanced_periods = total_periods - shortage_periods - surplus_periods
        
        # Max shortage/surplus
        max_shortage = abs(product_df[product_df['gap_quantity'] < 0]['gap_quantity'].min()) if shortage_periods > 0 else 0
        max_surplus = product_df[product_df['gap_quantity'] > 0]['gap_quantity'].max() if surplus_periods > 0 else 0
        
        # Main categorization (mutually exclusive)
        if pt_code in categorization['net_shortage']:
            category = "Net Shortage"
        elif pt_code in categorization['net_surplus']:
            category = "Net Surplus"
        elif pt_code in categorization['balanced']:
            category = "Balanced"
        else:
            category = "Unknown"
        
        # Timing flags
        timing_flags = []
        if pt_code in categorization['timing_shortage']:
            timing_flags.append("Timing Shortage")
        if pt_code in categorization['timing_surplus']:
            timing_flags.append("Timing Surplus")
        timing_flag_str = " | ".join(timing_flags) if timing_flags else "None"
        
        # Fill rate
        fill_rate = min(100, (total_supply / total_demand * 100)) if total_demand > 0 else 100
        
        # Backlog info if tracking
        if calc_options.get('track_backlog', True) and 'backlog_to_next' in product_df.columns:
            final_backlog = product_df['backlog_to_next'].iloc[-1] if not product_df.empty else 0
        else:
            final_backlog = 0
        
        summary_data.append({
            'PT Code': pt_code,
            'Product Name': product_name,
            'Brand': brand,
            'Package Size': package_size,
            'UOM': standard_uom,
            'Category': category,
            'Timing Flags': timing_flag_str,
            'Total Demand': total_demand,
            'Total Supply': total_supply,
            'Net Position': net_position,
            'Fill Rate %': fill_rate,
            'Total Periods': total_periods,
            'Shortage Periods': shortage_periods,
            'Surplus Periods': surplus_periods,
            'Balanced Periods': balanced_periods,
            'Max Shortage': max_shortage,
            'Max Surplus': max_surplus,
            'Final Backlog': final_backlog
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Sort by category priority and net position
    category_order = {
        'Net Shortage': 1,
        'Balanced': 2,
        'Net Surplus': 3,
        'Unknown': 4
    }
    
    if not summary_df.empty:
        summary_df['_sort_order'] = summary_df['Category'].map(category_order)
        summary_df = summary_df.sort_values(['_sort_order', 'Net Position'])
        summary_df = summary_df.drop(columns=['_sort_order'])
    
    return summary_df


# === SESSION STATE HELPERS ===

def save_to_session_state(key: str, value: Any, add_timestamp: bool = True):
    """Save value to session state with optional timestamp"""
    st.session_state[key] = value
    if add_timestamp:
        st.session_state[f"{key}_timestamp"] = datetime.now()


def get_from_session_state(key: str, default: Any = None) -> Any:
    """Get value from session state"""
    return st.session_state.get(key, default)


def clear_session_state_pattern(pattern: str):
    """Clear session state keys matching pattern"""
    keys_to_clear = [key for key in st.session_state.keys() if pattern in key]
    for key in keys_to_clear:
        del st.session_state[key]
    
    if keys_to_clear:
        logger.debug(f"Cleared {len(keys_to_clear)} session state keys matching '{pattern}'")


# === STANDARDIZED PERIOD HANDLING ===

def create_period_pivot(
    df: pd.DataFrame,
    group_cols: List[str],
    period_col: str,
    value_col: str,
    agg_func: str = "sum",
    period_type: str = "Weekly",
    show_only_nonzero: bool = True,
    fill_value: Any = 0
) -> pd.DataFrame:
    """Create standardized pivot table for any analysis page"""
    from .period_helpers import parse_week_period, parse_month_period
    
    if df.empty:
        return pd.DataFrame()
    
    missing_cols = [col for col in group_cols + [period_col, value_col] if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing columns in dataframe: {missing_cols}")
        return pd.DataFrame()
    
    try:
        pivot_df = df.pivot_table(
            index=group_cols,
            columns=period_col,
            values=value_col,
            aggfunc=agg_func,
            fill_value=fill_value
        ).reset_index()
        
        if show_only_nonzero and len(pivot_df.columns) > len(group_cols):
            numeric_cols = [col for col in pivot_df.columns if col not in group_cols]
            if numeric_cols:
                row_sums = pivot_df[numeric_cols].sum(axis=1)
                pivot_df = pivot_df[row_sums > 0]
        
        # Sort columns by period
        info_cols = group_cols
        period_cols = [col for col in pivot_df.columns if col not in info_cols]
        
        valid_period_cols = [col for col in period_cols 
                            if pd.notna(col) and str(col).strip() != "" and str(col) != "nan"]
        
        try:
            if period_type == "Weekly":
                sorted_periods = sorted(valid_period_cols, key=parse_week_period)
            elif period_type == "Monthly":
                sorted_periods = sorted(valid_period_cols, key=parse_month_period)
            else:
                sorted_periods = sorted(valid_period_cols)
        except Exception as e:
            logger.error(f"Error sorting period columns: {e}")
            sorted_periods = valid_period_cols
        
        return pivot_df[info_cols + sorted_periods]
        
    except Exception as e:
        logger.error(f"Error creating pivot: {str(e)}")
        return pd.DataFrame()


def create_download_button(df: pd.DataFrame, filename: str, 
                         button_label: str = "📥 Download Excel",
                         key: Optional[str] = None) -> None:
    """Create a download button for dataframe"""
    if df.empty:
        st.warning("No data available for download")
        return
        
    try:
        excel_data = convert_df_to_excel(df)
        
        st.download_button(
            label=button_label,
            data=excel_data,
            file_name=f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=key
        )
    except Exception as e:
        st.error(f"Error creating download: {str(e)}")


# === ANALYSIS FUNCTIONS ===

def calculate_fulfillment_rate(available: float, demand: float) -> float:
    """Calculate fulfillment rate percentage"""
    if demand <= 0:
        return 100.0 if available >= 0 else 0.0
    return min(100.0, max(0.0, (available / demand) * 100))


def calculate_days_of_supply(inventory: float, daily_demand: float) -> float:
    """Calculate days of supply"""
    if daily_demand <= 0:
        return float('inf') if inventory > 0 else 0.0
    return max(0.0, inventory / daily_demand)


def calculate_working_days(start_date: datetime, end_date: datetime, 
                         working_days_per_week: int = 5) -> int:
    """Calculate number of working days between two dates"""
    if pd.isna(start_date) or pd.isna(end_date):
        return 0
    
    if start_date > end_date:
        start_date, end_date = end_date, start_date
    
    working_days_per_week = max(1, min(7, working_days_per_week))
    
    total_days = (end_date - start_date).days + 1
    
    if working_days_per_week == 7:
        return total_days
    
    full_weeks = total_days // 7
    remaining_days = total_days % 7
    
    working_days = full_weeks * working_days_per_week
    
    current_date = start_date + timedelta(days=full_weeks * 7)
    for _ in range(remaining_days):
        if current_date.weekday() < working_days_per_week:
            working_days += 1
        current_date += timedelta(days=1)
    
    return max(0, working_days)


# === NOTIFICATION HELPERS ===

def show_success_message(message: str, duration: int = 3):
    """Show success message that auto-disappears"""
    placeholder = st.empty()
    placeholder.success(message)
    
    import time
    time.sleep(duration)
    placeholder.empty()


# === EXPORT HELPERS ===

def create_multi_sheet_export(
    sheets_config: List[Dict[str, Any]],
    filename_prefix: str
) -> Tuple[Optional[bytes], Optional[str]]:
    """Create multi-sheet Excel export"""
    sheets_dict = {}
    
    for config in sheets_config:
        if 'name' not in config or 'data' not in config:
            logger.warning(f"Invalid sheet config: {config}")
            continue
            
        df = config['data']
        if df is not None and not df.empty:
            if 'formatter' in config and callable(config['formatter']):
                try:
                    df = config['formatter'](df)
                except Exception as e:
                    logger.error(f"Error applying formatter to sheet '{config['name']}': {e}")
            
            sheet_name = str(config['name'])[:EXCEL_SHEET_NAME_LIMIT]
            sheets_dict[sheet_name] = df
    
    if sheets_dict:
        try:
            excel_data = export_multiple_sheets(sheets_dict)
            filename = f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            return excel_data, filename
        except Exception as e:
            logger.error(f"Error creating multi-sheet export: {e}")
            return None, None
    
    return None, None