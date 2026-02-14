import pandas as pd
from src.config import CLIP_RUL, VAR_THRESHOLD, WINDOW, CORR_THRESHOLD

def compute_rul(
    df:pd.DataFrame, 
    group_col:str = 'unit_number', 
    time_col:str = 'time_cycles', 
    clip:int = CLIP_RUL
) -> pd.Series:  
    """
    Compute Remaining Useful Life (RUL) for each unit and apply clipping.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    group_col : str, default="unit_number"
        Column identifying each unit.
    time_col : str, default="time_cycles"
        Column representing the time/cycle progression.
    clip : int, default=CLIP_RUL
        Maximum value allowed for the RUL.

    Returns
    -------
    pd.Series
        Clipped RUL values aligned with the input dataframe index.
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
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    var_threshold : float, default=VAR_THRESHOLD
        Minimum variance required to keep a feature.

    Returns
    -------
    pd.DataFrame
        Dataframe with low-variance sensor/setting columns removed.
    """
    df = df.copy()
    cols = [c for c in df.columns if "sensor" in c or "setting" in c]
    variances = df[cols].var()
    low_var_cols = variances[variances < var_threshold].index
    return df.drop(columns=low_var_cols)


def add_features(df:pd.DataFrame, window:int=WINDOW, group_col:str='unit_number')->pd.DataFrame:
    """
    Add rolling_mean, rolling_std and diff_inst columns for each unit
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    window : int, default=WINDOW
        Rolling window size used for mean and standard deviation.
    group_col : str, default="unit_number"
        Column identifying each unit.

    Returns
    -------
    pd.DataFrame
        Dataframe enriched with additional time-series features.
    """
    df=df.copy()
    cols_clean = [c for c in df.columns if 'sensor' in c or 'setting' in c]
    grouped = df.groupby(group_col)

    for col in cols_clean :
        df[f'{col}_mean_{window}'] = grouped[col].transform(
            lambda x : x.rolling(window, min_periods=1).mean()
        )
        df[f'{col}_std_{window}'] = grouped[col].transform(
            lambda x : x.rolling(window, min_periods=1).std()
        )
        df[f'{col}_diff'] = grouped[col].transform(
            lambda x : x.diff()
        )

    return df.fillna(0)


def remove_low_corr_features(
    df:pd.DataFrame, 
    target_col:str = 'RUL', 
    corr_threshold: float = CORR_THRESHOLD
) -> pd.DataFrame:

    """
    Remove features whose correlation with the target variable is below or equal to a given threshold.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    target_col : str, default="RUL"
        Target correlation column.
    corr_threshold : float, default=CORR_THRESHOLD
        Minimum correlation required to keep a feature.

    Returns
    -------
    pd.DataFrame
        DataFrame with low-correlation features removed.
    """
    df = df.copy()
    cols = [c for c in df.columns if "sensor" in c or "setting" in c]
    corr = df[cols+[target_col]].corr()[target_col].abs()
    low_corr_features = corr[corr <= corr_threshold].index
    return df.drop(columns=low_corr_features)