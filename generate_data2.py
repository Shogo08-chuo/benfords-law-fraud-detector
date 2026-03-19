import pandas as pd
import numpy as np

# A. 正常データ：市区町村の人口（自然な統計データ）
# ベンフォードの法則にきれいに従うはず
np.random.seed(1)
populations = 10 ** np.random.uniform(3, 6, 1500) # 1,000人〜1,000,000人の分布
df_pop = pd.DataFrame({"amount": populations})
df_pop.to_csv("city_population.csv", index=False)

# B. 異常データ：経費精算（キリの良い数字や承認上限を意識したデータ）
# 「9」から始まる数字（例：9,800円）が不自然に多い設定
np.random.seed(2)
normal_expenses = 10 ** np.random.uniform(2, 4.5, 800)
fraud_expenses = np.random.choice([9500, 9800, 9900, 48000], size=200)
all_expenses = np.concatenate([normal_expenses, fraud_expenses])
df_exp = pd.DataFrame({"amount": all_expenses})
df_exp.to_csv("expense_data.csv", index=False)

print("2種類のテストデータ（city_population.csv, expense_data.csv）を生成しました。")