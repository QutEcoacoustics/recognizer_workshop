import pandas as pd
import sys 

sys.path.append('..')

import utils

def test_balance_dataset():

    print("test balance dataset")

    df = pd.read_csv('test_data.csv')

    balanced_df = utils.balance_dataset(df)


test_balance_dataset()