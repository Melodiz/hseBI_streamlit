import pandas as pd
import os
PATH_DATA = './dataset'

transactions = pd.read_csv(
    os.path.join(PATH_DATA, 'transactions.csv')
)

step = 600_000

tr_1 = transactions.iloc[0:step]
tr_2 = transactions.iloc[step:2*step]
tr_3 = transactions.iloc[2*step:3*step]
tr_4 = transactions.iloc[3*step:4*step]
tr_5 = transactions.iloc[4*step:5*step]
tr_6 = transactions.iloc[5*step:]


tr_1.to_csv('dataset/transaction_1.csv')
tr_2.to_csv('dataset/transaction_2.csv')
tr_3.to_csv('dataset/transaction_3.csv')
tr_4.to_csv('dataset/transaction_4.csv')
tr_5.to_csv('dataset/transaction_5.csv')
tr_6.to_csv('dataset/transaction_6.csv')


print('finished')