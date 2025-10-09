# utils/period_gap/helpers.py
"""
General Helper Functions
Excel export, period manipulation, session state management
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