import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder
dataset=pd.read_csv(r'C:\Users\Elite\Desktop\data mining\students_activities.csv')
transactions=dataset["student_activities"].apply(lambda  x:[i.strip() for i in x.split(';')]).tolist()
te=TransactionEncoder()
te_data=te.fit(transactions).transform(transactions)
df_encoded=pd.DataFrame(te_data,columns=te.columns_)

frequent_items=apriori(df_encoded,min_support=0.2,use_colnames=True)

rules=association_rules(frequent_items,metric="confidence",min_threshold=0.5)

print(f"frquent_items :{frequent_items}")
print(f"association rules :{rules[['antecedents','consequents','support','confidence','lift']]}")