import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
import matplotlib.pyplot as plt

# Baca data dari file CSV
data = pd.read_csv("Dataset/ANIME_CLEANED_1000data.csv", sep=';', engine='python', encoding='utf-8')

# Misalkan 'data' memiliki kolom 'variabel_dependen' dan 'variabel_independen'
variabel_dependen = 'Members'
variabel_independen = ['Ranked', 'Popularity', 'Favorites', 'Watching', 'Completed']

# Tambahkan kolom konstan (intercept) ke dalam data
data['const'] = 1  # Tambahkan kolom konstan dengan nilai 1

# Membangun model regresi
model = sm.OLS(data[variabel_dependen], data[variabel_independen + ['const']])
hasil = model.fit()

# Menghitung residual dari model regresi
residuals = hasil.resid

# Melakukan uji Durbin-Watson untuk autokorelasi
durbin_watson_stat = durbin_watson(residuals)
print("Nilai statistik Durbin-Watson:", durbin_watson_stat)

# Interpretasi nilai statistik Durbin-Watson
if durbin_watson_stat < 1.5:
    print("Terdapat indikasi positif kuat terhadap autokorelasi positif.")
elif durbin_watson_stat > 2.5:
    print("Terdapat indikasi negatif kuat terhadap autokorelasi negatif.")
else:
    print("Tidak terdapat bukti kuat untuk menolak asumsi tidak ada autokorelasi dalam residual.")

# Visualisasi plot residual
plt.figure(figsize=(8, 4))
plt.plot(residuals)
plt.title('Plot Residual Model Regresi')
plt.xlabel('Observasi')
plt.ylabel('Residual')
plt.grid(True)
plt.show()