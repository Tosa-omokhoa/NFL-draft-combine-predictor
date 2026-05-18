"""
src/feature_engineering.py
NFL Draft Combine Predictor — Feature Engineering Pipeline
Author: Omokhoa Oshose Tosayoname (Tosa)
GitHub: github.com/Tosa9/NFL-draft-combine-predictor
GCI World 2026 April · In-Class Competition
"""

import pandas as pd
import numpy as np


# ── Constants ────────────────────────────────────────────────────────────────

MISSING_COLS = [
    'Age', 'Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps',
    'Broad_Jump', 'Agility_3cone', 'Shuttle'
]

PERF_COLS = [
    'Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps', 'Broad_Jump',
    'Agility_3cone', 'Shuttle', 'Height', 'Weight', 'Age'
]

DROP_COLS = ['Id', 'School', 'Player_Type', 'Position_Type', 'Position']

GROUP_RATE_COLS = [
    ('Position_rate',    'Position'),
    ('PositionType_rate','Position_Type'),
    ('PlayerType_rate',  'Player_Type'),
]


# ── Step 1: Missing value flags ──────────────────────────────────────────────

def add_missing_flags(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """
    Add binary missing-value indicator columns before imputation.

    The absence of a combine measurement is itself informative.
    Age_missing alone ranks #3 in feature importance, above all
    raw performance metrics.

    Parameters
    ----------
    train, test : pd.DataFrame
        Raw data frames before imputation.

    Returns
    -------
    train, test with new binary flag columns added.
    """
    for col in MISSING_COLS:
        train[col + '_missing'] = train[col].isnull().astype(int)
        test[col + '_missing']  = test[col].isnull().astype(int)
    return train, test


# ── Step 2: Position-wise group mean imputation ──────────────────────────────

def position_wise_impute(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """
    Impute missing values using the position-group mean from training data.

    A missing WR sprint time is filled with the average WR sprint time,
    not the global mean across all positions. This preserves the
    position-specific physiological profile of each athlete.

    Falls back to the global training mean if a position group has
    no valid observations for a given metric.

    Parameters
    ----------
    train, test : pd.DataFrame
        Data frames after missing flags have been added.

    Returns
    -------
    train, test with missing values filled.
    """
    for col in MISSING_COLS:
        group_means = train.groupby('Position')[col].mean()
        global_mean = train[col].mean()

        def _fill(row, gm=group_means, glb=global_mean, c=col):
            if pd.isnull(row[c]):
                return gm.get(row['Position'], glb)
            return row[c]

        train[col] = train.apply(_fill, axis=1)
        test[col]  = test.apply(_fill, axis=1)
    return train, test


# ── Step 3: Smoothed school target encoding ──────────────────────────────────

def school_target_encoding(train: pd.DataFrame, test: pd.DataFrame,
                            k: int = 10) -> tuple:
    """
    Encode the School column as a smoothed historical draft rate.

    Formula:
        encoded = (school_count * school_mean + k * global_mean)
                  / (school_count + k)

    The smoothing factor k=10 prevents overfitting to schools with
    few combine attendees. Schools not seen in training are assigned
    the global draft rate.

    Parameters
    ----------
    train, test : pd.DataFrame
    k : int
        Smoothing strength. Higher k = more shrinkage toward global mean.

    Returns
    -------
    train, test with School_encoded column added.
    """
    school_counts     = train.groupby('School')['Drafted'].count()
    school_means      = train.groupby('School')['Drafted'].mean()
    global_draft_rate = train['Drafted'].mean()

    encoded = (school_counts * school_means + k * global_draft_rate) \
              / (school_counts + k)

    train['School_encoded'] = train['School'].map(encoded).fillna(global_draft_rate)
    test['School_encoded']  = test['School'].map(encoded).fillna(global_draft_rate)
    return train, test


# ── Step 4: Group draft-rate encodings ──────────────────────────────────────

def group_rate_encoding(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """
    Encode Position, Position_Type, and Player_Type as their
    historical draft selection rate from training data.

    This gives the model a direct prior probability per athlete type.
    OLBs have a 75.9% prior; Punters have 22.7%.

    Returns
    -------
    train, test with Position_rate, PositionType_rate,
    PlayerType_rate columns added.
    """
    for feat_name, grp_col in GROUP_RATE_COLS:
        rate = train.groupby(grp_col)['Drafted'].mean()
        train[feat_name] = train[grp_col].map(rate)
        test[feat_name]  = test[grp_col].map(rate)
    return train, test


# ── Step 5: Physical composite features ─────────────────────────────────────

def add_physical_composites(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive composite physical performance features from raw combine metrics.

    Features created:
    - BMI               : Weight / Height²
    - Agility_composite : -(Agility_3cone + Shuttle)  [higher = more agile]
    - Power_composite   : Vertical_Jump + Broad_Jump/10 + Bench_Press_Reps
    - Speed_power_ratio : Broad_Jump / Sprint_40yd
    - Vert_Broad_ratio  : Vertical_Jump / Broad_Jump
    - Cone_Shuttle_diff : Agility_3cone - Shuttle
    """
    df = df.copy()
    df['BMI']               = df['Weight'] / (df['Height'] ** 2)
    df['Agility_composite'] = -df['Agility_3cone'] - df['Shuttle']
    df['Power_composite']   = (df['Vertical_Jump']
                               + df['Broad_Jump'] / 10
                               + df['Bench_Press_Reps'])
    df['Speed_power_ratio'] = df['Broad_Jump'] / (df['Sprint_40yd'] + 1e-6)
    df['Vert_Broad_ratio']  = df['Vertical_Jump'] / (df['Broad_Jump'] + 1e-6)
    df['Cone_Shuttle_diff'] = df['Agility_3cone'] - df['Shuttle']
    return df


# ── Step 6: Position-normalised z-scores ────────────────────────────────────

def position_zscores(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """
    Compute position-normalised z-scores for all performance metrics.

    For each metric, subtract the mean and divide by the std of the
    athlete's own position group. This captures how an athlete performs
    relative to their positional peers — which is far more informative
    than raw absolute values.

    For example:
    - A 4.45s 40-yard dash is elite for an OT but below average for a WR.
    - Sprint_40yd_pos_z ranks #1 in feature importance, vs Sprint_40yd
      which ranks #12.

    All statistics are computed from training data only to prevent leakage.

    Returns
    -------
    train, test with _pos_z suffix columns added for each metric.
    """
    for col in PERF_COLS:
        pos_mean = train.groupby('Position')[col].mean()
        pos_std  = train.groupby('Position')[col].std().replace(0, 1)

        global_mean = train[col].mean()
        global_std  = train[col].std()

        def _zscore(row, pm=pos_mean, ps=pos_std,
                    c=col, gm=global_mean, gs=global_std):
            mean = pm.get(row['Position'], gm)
            std  = ps.get(row['Position'], gs)
            return (row[c] - mean) / std

        train[col + '_pos_z'] = train.apply(_zscore, axis=1)
        test[col + '_pos_z']  = test.apply(_zscore, axis=1)
    return train, test


# ── Master pipeline ──────────────────────────────────────────────────────────

def engineer_features(df_train: pd.DataFrame,
                       df_test:  pd.DataFrame,
                       school_k: int = 10) -> tuple:
    """
    Full feature engineering pipeline for the NFL Draft Combine Predictor.

    Executes all 6 steps in sequence:
        1. Missing value flags
        2. Position-wise group mean imputation
        3. Smoothed school target encoding
        4. Group draft-rate encodings
        5. Physical composite features
        6. Position-normalised z-scores
        7. Drop original categorical and ID columns

    All statistics are computed exclusively from df_train and then
    applied to df_test. This prevents any data leakage from the
    test set into the training pipeline.

    Parameters
    ----------
    df_train : pd.DataFrame
        Raw training data including the 'Drafted' target column.
    df_test : pd.DataFrame
        Raw test data without the 'Drafted' column.
    school_k : int
        Smoothing factor for school target encoding (default: 10).

    Returns
    -------
    train : pd.DataFrame
        Engineered training data including 'Drafted' column.
    test : pd.DataFrame
        Engineered test data ready for model prediction.

    Example
    -------
    >>> train_fe, test_fe = engineer_features(train_raw, test_raw)
    >>> X = train_fe.drop(columns=['Drafted'])
    >>> y = train_fe['Drafted']
    >>> X_test = test_fe.copy()
    """
    train = df_train.copy()
    test  = df_test.copy()

    # 1. Missing flags
    train, test = add_missing_flags(train, test)

    # 2. Position-wise imputation
    train, test = position_wise_impute(train, test)

    # 3. School target encoding
    train, test = school_target_encoding(train, test, k=school_k)

    # 4. Group rate encodings
    train, test = group_rate_encoding(train, test)

    # 5. Physical composites
    train = add_physical_composites(train)
    test  = add_physical_composites(test)

    # 6. Position z-scores
    train, test = position_zscores(train, test)

    # 7. Drop originals
    train = train.drop(columns=DROP_COLS)
    test  = test.drop(columns=DROP_COLS)

    return train, test


def fill_residual_nulls(X_train: pd.DataFrame,
                         X_test:  pd.DataFrame) -> tuple:
    """
    Fill any remaining NaN values with the column median from training data.

    This is a final safety net after the main imputation pass, handling
    edge cases where a position group had no valid observations for a metric.

    Parameters
    ----------
    X_train : pd.DataFrame
        Feature matrix (training set, no target column).
    X_test : pd.DataFrame
        Feature matrix (test set).

    Returns
    -------
    X_train, X_test with all NaN filled.
    """
    for col in X_train.columns:
        med = X_train[col].median()
        X_train[col] = X_train[col].fillna(med)
        X_test[col]  = X_test[col].fillna(med)
    return X_train, X_test
