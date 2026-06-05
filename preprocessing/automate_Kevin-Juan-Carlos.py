import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
 
 
def cap_outliers_iqr(df, numerical_cols):
    """
    Caps outlier values in specified columns using the IQR method
    """
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower=lower, upper=upper)

    return df
 
 
def preprocess_data(input_path, output_path):
    """
    Preprocesses the dataset by cleaning, encoding, capping outliers, binning, and standardizing features
    """
    # Load the data
    df = pd.read_csv(input_path)
    print(f'Data successfully loaded from: {input_path}')
 
    # Drop duplicate rows
    df = df.drop_duplicates()
    if df.empty:
        raise ValueError('Dataset is empty after removing duplicates')
    else:
        print(f'Duplicate data after cleaning: {df.duplicated().sum()}')
 
    # Drop columns which not useful for modeling
    drop_cols = ['TransactionID', 'AccountID', 'DeviceID', 'IP Address', 'MerchantID']
    existing_drop = [c for c in drop_cols if c in df.columns]

    df.drop(columns=existing_drop, inplace=True)
    print(f'Identifier columns dropped: {existing_drop}')
 
    # Feature engineering for TransactionDate column
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'], format='mixed', dayfirst=False)
    df['TransactionMonth'] = df['TransactionDate'].dt.month
    df['TransactionDayOfWeek'] = df['TransactionDate'].dt.dayofweek
    df['TransactionYear'] = df['TransactionDate'].dt.year

    df.drop(columns=['TransactionDate'], inplace=True)
    print('Date features extracted')
 
    # Encode categorical features
    df['TransactionType'] = df['TransactionType'].map({'Debit': 0, 'Credit': 1})
    le = LabelEncoder()
    for col in ['Channel', 'CustomerOccupation', 'Location']:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])

    print('Categorical features encoded')
 
    # Login Attempts binning
    df['LoginAttempts_Bin'] = (df['LoginAttempts'] > 1).astype(int)
 
    # Cap outliers using IQR method
    outlier_cols = ['TransactionAmount', 'AccountBalance', 'TransactionDuration']
    df = cap_outliers_iqr(df, outlier_cols)
 
    if df.empty:
        raise ValueError('Dataset is empty after outlier handling')
 
    # Feature standardization
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    print('Feature standardization done')
 
    # Convert to new CSV
    df_scaled.to_csv(output_path, index=False)
    df_scaled.info()
 
    return df_scaled
 
 
if __name__ == "__main__":
    preprocess_data(
        input_path='../bank_transactions_raw.csv',
        output_path='bank_transactions_preprocessing.csv',
    )
