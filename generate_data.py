import numpy as np
import pandas as pd

np.random.seed(42)

# 正常っぽいデータ（対数分布に近い）
normal_data = 10 ** np.random.uniform(2, 6, 2000)

# 人為的に作った不自然なデータ
fraud_data = np.random.choice(
    [500, 800, 1500, 2500, 5800, 8800],
    size=300
)

data = np.concatenate([normal_data, fraud_data])

df = pd.DataFrame({"amount": data})
df.to_csv("financial_data.csv", index=False)

print("データ生成完了")
