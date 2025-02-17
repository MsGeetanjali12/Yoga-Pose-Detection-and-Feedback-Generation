import pandas as pd
df = pd.read_csv('coords_final.csv')
print(df)
print(df.head())
i=33
l=[]
for i in range(1,i+1):
	s='v'+str(i)
	l.append(s)
print(l)
for j in l:
    index_names = df[ df[j]<0.9].index


    df.drop(index_names, inplace = True)

print(df)
df.to_csv('pdcoord0.csv')