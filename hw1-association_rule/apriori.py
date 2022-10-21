from os import path
import pandas
import json
from apyori import apriori
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
association_rules = apriori(records, min_support=0.0001, min_confidence=0.9)
print("Apriori Time: ", time.time() - start)

start = time.time()
# Print the result
association_results = list(association_rules)
print("Convert to list Time: ", time.time() - start)

print(association_results[0])

# sort the result by confidence and list the top 10
association_results.sort(key=lambda x: x[2][0][2], reverse=True)
print("Top 10 confidence rules:")
for item in association_results[:10]:
    pair = item[0]
    items = [x for x in pair]
    print(items)
    print("Rule: " + str(items[0]) + " -> " + str(items[1]))
    print("Support: " + str(item[1]))
    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")

print("")
# sort the result by lift and list the top 10
association_results.sort(key=lambda x: x[2][0][3], reverse=True)
print("Top 10 lift rules:")
for item in association_results[:10]:
    pair = item[0]
    items = [x for x in pair]
    print(items)
    print("Rule: " + str(items[0]) + " -> " + str(items[1]))
    print("Support: " + str(item[1]))
    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")

