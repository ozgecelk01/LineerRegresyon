import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

data = pd.read_csv("ev_fiyat_tahmini.csv",sep=";")

x = data['MetreKare'].values.reshape((-1, 1))
y = data['fiyatlar'].values.reshape((-1, 1))

model = LinearRegression()
model.fit(x,y)

sns.regplot(x="MetreKare", y="fiyatlar", data=data,line_kws={"color": "red"});
plt.show()
