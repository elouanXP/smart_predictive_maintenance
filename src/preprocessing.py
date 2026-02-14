import pandas as pd
from src.config import CLIP_RUL, VAR_THRESHOLD

def compute_rul(
    df:pd.DataFrame, 
    group_col:str = 'unit_number', 
    time_col:str = 'time_cycles', 
    clip:int = CLIP_RUL
) -> pd.Series:  
    """
    Compute Remaining Useful Life (RUL) for each unit and apply clipping.
    """
    df = df.copy()
    max_cycles = df.groupby(group_col)[time_col].transform('max')
    df['RUL'] = max_cycles - df[time_col]
    return df['RUL'].clip(upper=clip)


def remove_low_variance_features(
    df:pd.DataFrame,
    var_threshold:float = VAR_THRESHOLD
) -> pd.DataFrame:
    """
    Remove sensor/setting columns with variance below a threshold.
    """
    df = df.copy()
    cols = [c for c in df.columns if "sensor" in c or "setting" in c]
    variances = df[cols].var()
    low_var_cols = variances[variances < var_threshold].index
    return df.drop(columns=low_var_cols)