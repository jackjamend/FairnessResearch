"""
Formats data for ML training.

Formats the UCI Adult data into numeric values.

Jack Amend
3/3/2020
"""
import pandas as pd

class DataLoader:
    def __init__(self, file_name):
        df = pd.read_csv(file_name, header=None)

        cols = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
                "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
                "hours-per-week", "native-country", "label"]

        df = df.rename(columns={i: c for i, c in enumerate(cols)})
        df = df.drop('fnlwgt', axis="columns")

        self.df = df

    def get_numeric_data(self, balanced=False):
        numeric_col = ['age', 'education-num', 'capital-gain', 'capital-loss',
                       'hours-per-week']
        boolean_cols = ['sex', 'label']
        category_cols = ['workclass', 'marital-status', 'occupation',
                         'relationship',
                         'race', 'native-country']
        df = self.df.copy()
        df = df.drop(['education'], axis=1)

        if balanced:

            df.sex = df.sex.str.strip()
            gender_counts = df.sex.value_counts()
            diff_needed = gender_counts['Male'] - gender_counts['Female']
            df = df.append(df[df.sex == 'Female'].sample(diff_needed, replace=True, random_state=7215))

        for bol in boolean_cols:
            df[bol] = df[bol].apply(lambda x: 0.0 if x == df[bol][0] else 1.0)

        for col in numeric_col:
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

        df = pd.get_dummies(df, columns=category_cols)
        data = df.drop('label', axis=1).values
        labels = df['label'].values
        protected = df['sex'].values
        return data, labels, protected



def get_data_labels(file_name, verbose=1):
    if verbose:
        print("reading in data")

    df = pd.read_csv(file_name, header=None)

    cols = ["age", "workclass", "fnlwgt", "education", "education-num",
            "marital-status",
            "occupation", "relationship", "race", "sex", "capital-gain",
            "capital-loss",
            "hours-per-week", "native-country", "label"]

    df = df.rename(columns={i: c for i, c in enumerate(cols)})
    df = df.drop('fnlwgt', axis="columns")
    numeric_col = ['age', 'education-num', 'capital-gain', 'capital-loss',
                   'hours-per-week']
    boolean_cols = ['sex', 'label']
    category_cols = ['workclass', 'marital-status', 'occupation',
                     'relationship',
                     'race', 'native-country']

    df = df.drop(['education'], axis=1)

    for col in numeric_col:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    for bol in boolean_cols:
        df[bol] = df[bol].apply(lambda x: 0.0 if x == df[bol][0] else 1.0)

    df = pd.get_dummies(df, columns=category_cols)
    print(df.head())
    print("shape =", df.shape)

    data = df.drop('label', axis=1).values
    labels = df['label'].values
    protected = df['sex'].values
    return data, labels, protected

def get_balanced_data(file_name, verbose=1):
    if verbose:
        print("reading in data")

    df = pd.read_csv(file_name, header=None)

    cols = ["age", "workclass", "fnlwgt", "education", "education-num",
            "marital-status",
            "occupation", "relationship", "race", "sex", "capital-gain",
            "capital-loss",
            "hours-per-week", "native-country", "label"]

    df = df.rename(columns={i: c for i, c in enumerate(cols)})
    df = df.drop('fnlwgt', axis="columns")
    # Sampling to make balanced data
    df.sex = df.sex.str.strip()
    gender_counts = df.sex.value_counts()
    diff_needed = gender_counts['Male'] - gender_counts['Female']
    df = df.append(df[df.sex == 'Female'].sample(diff_needed, replace=True, random_state=7215))

    numeric_col = ['age', 'education-num', 'capital-gain', 'capital-loss',
                   'hours-per-week']
    boolean_cols = ['sex', 'label']
    category_cols = ['workclass', 'marital-status', 'occupation',
                     'relationship',
                     'race', 'native-country']

    df = df.drop(['education'], axis=1)

    for col in numeric_col:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    for bol in boolean_cols:
        df[bol] = df[bol].apply(lambda x: 0.0 if x == df[bol][0] else 1.0)

    df = pd.get_dummies(df, columns=category_cols)
    print(df.head())
    print("shape =", df.shape)

    data = df.drop('label', axis=1).values
    labels = df['label'].values
    protected = df['sex'].values
    return data, labels, protected