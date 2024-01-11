import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt

# Baca data dari file CSV
data = pd.read_csv("Dataset/ANIME_CLEANED_1000data.csv", sep=';', engine='python', encoding='utf-8')

# Misalkan 'data' memiliki kolom 'Income', 'Recency', 'Response', dan 'Customer_Days'
variabel_dependen = 'Members'
variabel_independen1 = 'Ranked'
variabel_independen2 = 'Popularity'
variabel_independen3 = 'Favorites'
variabel_independen4 = 'Watching'
variabel_independen5 = 'Completed'

# Melakukan regresi linear berganda
X = data[[variabel_independen1, variabel_independen2, variabel_independen3, variabel_independen4,variabel_independen5]]
y = data[variabel_dependen]

# Melakukan regresi linear sederhana untuk variabel_independen1
slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(data[variabel_independen1], y)

# Melakukan regresi linear sederhana untuk variabel_independen2
slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(data[variabel_independen2], y)

# Melakukan regresi linear sederhana untuk variabel_independen3
slope3, intercept3, r_value3, p_value3, std_err3 = stats.linregress(data[variabel_independen3], y)

# Melakukan regresi linear sederhana untuk variabel_independen3
slope4, intercept4, r_value4, p_value4, std_err4 = stats.linregress(data[variabel_independen4], y)

# Melakukan regresi linear sederhana untuk variabel_independen3
slope5, intercept5, r_value5, p_value5, std_err5 = stats.linregress(data[variabel_independen5], y)

# Melakukan prediksi berdasarkan model regresi
predicted1 = slope1 * data[variabel_independen1] + intercept1
predicted2 = slope2 * data[variabel_independen2] + intercept2
predicted3 = slope3 * data[variabel_independen3] + intercept3
predicted4 = slope4 * data[variabel_independen4] + intercept4
predicted5 = slope5 * data[variabel_independen5] + intercept5

# Menghitung residu dari model regresi
residuals1 = y - predicted1
residuals2 = y - predicted2
residuals3 = y - predicted3
residuals4 = y - predicted4
residuals5 = y - predicted5

# Melakukan uji normalitas terhadap residu menggunakan metode Shapiro-Wilk
shapiro_test1 = stats.shapiro(residuals1)
shapiro_test2 = stats.shapiro(residuals2)
shapiro_test3 = stats.shapiro(residuals3)
shapiro_test4 = stats.shapiro(residuals4)
shapiro_test5 = stats.shapiro(residuals5)

print("Hasil uji normalitas Shapiro-Wilk (Residu Variabel Independen1): p-value =", shapiro_test1.pvalue)
print("Hasil uji normalitas Shapiro-Wilk (Residu Variabel Independen2): p-value =", shapiro_test2.pvalue)
print("Hasil uji normalitas Shapiro-Wilk (Residu Variabel Independen3): p-value =", shapiro_test3.pvalue)
print("Hasil uji normalitas Shapiro-Wilk (Residu Variabel Independen4): p-value =", shapiro_test4.pvalue)
print("Hasil uji normalitas Shapiro-Wilk (Residu Variabel Independen5): p-value =", shapiro_test5.pvalue)

# Interpretasi nilai p-value untuk level signifikansi 0,05
alpha = 0.05

if shapiro_test1.pvalue < alpha:
    print("Residu Variabel Independen1 tidak terdistribusi secara normal (Tolak H0)")
else:
    print("Residu Variabel Independen1 terdistribusi secara normal (Tidak cukup bukti untuk menolak H0)")

if shapiro_test2.pvalue < alpha:
    print("Residu Variabel Independen2 tidak terdistribusi secara normal (Tolak H0)")
else:
    print("Residu Variabel Independen2 terdistribusi secara normal (Tidak cukup bukti untuk menolak H0)")

if shapiro_test3.pvalue < alpha:
    print("Residu Variabel Independen3 tidak terdistribusi secara normal (Tolak H0)")
else:
    print("Residu Variabel Independen3 terdistribusi secara normal (Tidak cukup bukti untuk menolak H0)")

if shapiro_test4.pvalue < alpha:
    print("Residu Variabel Independen4 tidak terdistribusi secara normal (Tolak H0)")
else:
    print("Residu Variabel Independen4 terdistribusi secara normal (Tidak cukup bukti untuk menolak H0)")

if shapiro_test5.pvalue < alpha:
    print("Residu Variabel Independen5 tidak terdistribusi secara normal (Tolak H0)")
else:
    print("Residu Variabel Independen5 terdistribusi secara normal (Tidak cukup bukti untuk menolak H0)")
skewness1 = stats.skew(residuals1)
skewness2 = stats.skew(residuals2)
skewness3 = stats.skew(residuals3)
skewness4 = stats.skew(residuals4)
skewness5 = stats.skew(residuals5)

# Visualisasi distribusi residu
sns.set(style="whitegrid")
plt.figure(figsize=(15, 6))

# Plot distribusi residu variabel_independen1
plt.subplot(1, 5, 1)
sns.histplot(residuals1, kde=True, color='skyblue')
plt.title('Distribusi Residu (Variabel Independen1) {:.2f}'.format(skewness1))
plt.xlabel('Nilai Residu')
plt.ylabel('Frekuensi')
plt.text(0.5, 50, 'Skewness Negatif' if skewness1 < 0 else 'Skewness Positif', fontsize=10, color='red')

# Plot distribusi residu variabel_independen2
plt.subplot(1, 5, 2)
sns.histplot(residuals2, kde=True, color='coral')
plt.title('Distribusi Residu (Variabel Independen2) {:.2f}'.format(skewness2))
plt.xlabel('Nilai Residu')
plt.ylabel('Frekuensi')
plt.text(0.5, 50, 'Skewness Negatif' if skewness2 < 0 else 'Skewness Positif', fontsize=10, color='red')

# Plot distribusi residu variabel_independen3
plt.subplot(1, 5, 3)
sns.histplot(residuals3, kde=True, color='cyan')
plt.title('Distribusi Residu (Variabel Independen3) {:.2f}'.format(skewness3))
plt.xlabel('Nilai Residu')
plt.ylabel('Frekuensi')
plt.text(0.5, 50, 'Skewness Negatif' if skewness3 < 0 else 'Skewness Positif', fontsize=10, color='red')

# Plot distribusi residu variabel_independen4
plt.subplot(1, 5, 4)
sns.histplot(residuals4, kde=True, color='cyan')
plt.title('Distribusi Residu (Variabel Independen4) {:.2f}'.format(skewness4))
plt.xlabel('Nilai Residu')
plt.ylabel('Frekuensi')
plt.text(0.5, 50, 'Skewness Negatif' if skewness4 < 0 else 'Skewness Positif', fontsize=10, color='red')

# Plot distribusi residu variabel_independen5
plt.subplot(1, 5, 5)
sns.histplot(residuals5, kde=True, color='cyan')
plt.title('Distribusi Residu (Variabel Independen5) {:.2f}'.format(skewness5))
plt.xlabel('Nilai Residu')
plt.ylabel('Frekuensi')
plt.text(0.5, 50, 'Skewness Negatif' if skewness5 < 0 else 'Skewness Positif', fontsize=10, color='red')
plt.tight_layout()
plt.show()