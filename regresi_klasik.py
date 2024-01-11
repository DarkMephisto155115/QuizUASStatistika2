import pandas as pd
import statsmodels.api as sm
from scipy import stats

# Membaca dataset 1000 data
df = pd.read_csv("Dataset/ANIME_CLEANED_1000data.csv", sep=';', engine='python', encoding='utf-8')
df.dropna()
# Membaca dataset 100 data
df_sample = pd.read_csv("Dataset/ANIME_CLEANED_100data.csv", sep=';', engine='python', encoding='utf-8')

# 1. Analisis dengan 5 variabel numerik menggunakan Python
X_sample = df_sample[['Ranked', 'Popularity', 'Favorites', 'Watching', 'Completed']]
y_sample = df_sample['Members']

# Menambahkan konstanta untuk termasuk intercept
X_sample = sm.add_constant(X_sample)

# Membuat model regresi
model_sample = sm.OLS(y_sample, X_sample).fit()

# Menampilkan hasil regresi
print("Hasil Regresi untuk 5 Variabel Numerik:")
print(model_sample.summary())

# 2. Analisis dengan 10 variabel numerik menggunakan Python
X_full = df[['Score', 'Episodes', 'Premiered', 'Ranked', 'Popularity', 'Favorites', 'Watching', 'Completed', 'On-Hold', 'Dropped']]
y_full = df['Members']

# Menambahkan konstanta untuk termasuk intercept
X_full = sm.add_constant(X_full)

# Membuat model regresi
model_full = sm.OLS(y_full, X_full).fit()

# Menampilkan hasil regresi
print("\nHasil Regresi untuk 10 Variabel Numerik:")
print(model_full.summary())

# 3. Uji Parsial (t-test) untuk variabel 'Popularity' pada data 1000 sample
t_stat, p_value = stats.ttest_ind(df['Popularity'], df['Members'])
print(f"\nUji Parsial (t-test) untuk variabel 'Popularity' pada data 1000 sample:")
print(f"T-Stat: {t_stat}, P-Value: {p_value}")
if p_value < 0.05:
    print("Variabel 'Popularity' signifikan secara parsial.")
else:
    print("Variabel 'Popularity' tidak signifikan secara parsial.")

# 4. Uji Simultan Regresi (F-test) untuk model dengan 10 variabel numerik
f_test = model_full.wald_test("Score = Episodes = Premiered = Ranked = Popularity = Favorites = Watching = Completed = On-Hold = Dropped = 0")
f_stat = f_test.statistic[0][0]
f_p_value = f_test.pvalue

print(f"\nUji Simultan Regresi (F-test) untuk model dengan 10 variabel numerik:")
format(f_p_value, '.6f')
print(f"F-Stat: {f_stat}, P-Value: {f_p_value}")
if f_p_value < 0.05:
    print("Model regresi secara keseluruhan signifikan.")
else:
    print("Model regresi secara keseluruhan tidak signifikan.")

# 5. Uji Kebaikan Model menggunakan R-squared
r_squared = model_full.rsquared
print(f"\nR-Squared (Koefisien Determinasi) untuk model dengan 10 variabel numerik:")
print(f"R-Squared: {r_squared}")