import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.compat import lzip
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

# Melakukan uji Breusch-Pagan untuk heteroskedastisitas
uji_breusch_pagan = het_breuschpagan(hasil.resid, data[variabel_independen + ['const']])
hasil_uji = lzip(['Lagrange Multiplier statistic', 'LM p-value', 'F-value', 'F p-value'], uji_breusch_pagan)

# Menampilkan hasil uji Breusch-Pagan
print("Hasil uji Breusch-Pagan untuk heteroskedastisitas:")
print(hasil_uji)

# Interpretasi hasil uji dengan alpha 0,05
alpha = 0.05

LM_p_value = hasil_uji[1][1]
F_p_value = hasil_uji[3][1]

if LM_p_value < alpha or F_p_value < alpha:
    print("Terdapat bukti yang cukup untuk menolak hipotesis nol.")
    print("Artinya, model mengalami heteroskedastisitas.")
else:
    print("Tidak terdapat bukti yang cukup untuk menolak hipotesis nol.")
    print("Artinya, tidak ada bukti kuat untuk menyatakan bahwa terjadi heteroskedastisitas pada model.")

# Visualisasi scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(data['Ranked'], data[variabel_dependen], label='Ranked', alpha=0.7)
plt.scatter(data['Popularity'], data[variabel_dependen], label='Popularity', alpha=0.7)
plt.scatter(data['Favorites'], data[variabel_dependen], label='Favorites', alpha=0.7)
plt.scatter(data['Watching'], data[variabel_dependen], label='Watching', alpha=0.7)
plt.scatter(data['Completed'], data[variabel_dependen], label='Completed', alpha=0.7)
plt.xlabel('Variabel Independen')
plt.ylabel('Variabel Dependen')
plt.title('Scatter Plot Variabel Independen vs Variabel Dependen')
plt.legend()
plt.show()