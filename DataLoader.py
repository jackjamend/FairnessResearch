import pandas as pd

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
    print(df['label'].value_counts())

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
