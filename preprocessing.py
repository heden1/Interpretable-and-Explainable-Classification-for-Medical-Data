import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer

PREPROCESSED_COLUMN_NAMES = ['Age', 'Sex - F', 'Sex - M', 'ChestPainType - ASY', 'ChestPainType - ATA', 'ChestPainType - NAP', 'ChestPainType - TA', 'RestingBP', 'Cholesterol', 'FastingBS',
       'RestingECG - LVH', 'RestingECG - Normal', 'RestingECG - ST', 'MaxHR', 'ExerciseAngina - N', 'ExerciseAngina - Y', 'Oldpeak', 'ST_Slope - Down', 'ST_Slope - Flat',  'ST_Slope - Up',]

# assumes heart failures have been put in without the heart_disease label column
def preprocess_hf_for_lasso(heart_failures: pd.DataFrame):
    categorical = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
    numerical = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR" , "Oldpeak"]
    preprocessed = heart_failures.copy()

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        # ('labelencoder', LabelEncoder()),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical),
            ('cat', categorical_transformer, categorical)
        ])
    preprocessor.fit(preprocessed)
    preprocessed = preprocessor.fit_transform(preprocessed)
    return preprocessed

def preprocess_hf_for_nam(heart_failures: pd.DataFrame):
    labels = heart_failures.pop("HeartDisease")
    preprocessed = preprocess_hf_for_lasso(heart_failures)
    # pd_preprocessed = pd.DataFrame(preprocessed, columns=PREPROCESSED_COLUMN_NAMES)
    return preprocessed, labels
    
