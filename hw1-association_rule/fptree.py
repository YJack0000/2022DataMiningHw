from os import path
import pandas
import json
from fpgrowth_py import fpgrowth
import time

df_main = pandas.DataFrame(columns=['customer_id', 'product_id'])

if not path.exists('data/main_data.csv'):
    # Read the data
    df_fact = pandas.read_csv('data/sales_fact_1998.csv')

    customer_id = df_fact.iloc[0]['customer_id']
    product_id_list = []
    for idx, row in df_fact.iterrows():
        if row['customer_id'] == customer_id:
            product_id_list.append(row['product_id'])
        else:
            df_main.loc[len(df_main)] = [customer_id, product_id_list]
            customer_id = row['customer_id']
            product_id_list = [row['product_id']]

    df_dec = pandas.read_csv('data/sales_fact_dec_1998.csv')

    customer_id = df_dec.iloc[0]['customer_id']
    product_id_list = []
    for idx, row in df_dec.iterrows():
        if row['customer_id'] == customer_id:
            product_id_list.append(row['product_id'])
        else:
            df_main.loc[len(df_main)] = [customer_id, product_id_list]
            customer_id = row['customer_id']
            product_id_list = [row['product_id']]

    # save data as a transcation file
    df_main.to_csv('data/main_data.csv', index=False)
else:
    df_main = pandas.read_csv('data/main_data.csv')
    df_main["product_id"] = df_main["product_id"].apply(lambda x: json.loads(x))

# print(df_main.head())

# Convert the data into a file that can be read by apriori
records = []
for idx, row in df_main.iterrows():
    records.append(row['product_id'])
# print(records)

start = time.time()
# Run apriori
freqItemSet, rules = fpgrowth(records, minSupRatio=0.0001, minConf=0.9)
print("fp-tree Time: ", time.time() - start)

# Top 10 confidence rules
rules.sort(key=lambda x: x[2], reverse=True)
print("Top 10 confidence rules:")
for i in range(10):
    print(rules[i])

# Top 10 lift rules
freqItemSet