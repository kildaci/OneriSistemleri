import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split as surprise_train_test_split
from surprise import accuracy

# Veri setini yükleyin
df = pd.read_csv('Reviews.csv')

# Boş yorumları temizleyin
df.dropna(subset=['Text'], inplace=True)

# Ürün öneri sistemi için veri hazırlığı
df_recommend = df[['UserId', 'ProductId', 'Score']].dropna()
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df_recommend[['UserId', 'ProductId', 'Score']], reader)

# Eğitim ve test setlerine ayırın
trainset, testset = surprise_train_test_split(data, test_size=0.2)

# SVD algoritması ile model oluşturun
algo_svd = SVD()
algo_svd.fit(trainset)

# Test işlemi ve performans ölçümleri
predictions = algo_svd.test(testset)
rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)

print(f"RMSE: {rmse}")
print(f"MAE: {mae}")

# Kullanıcı için öneri yapma fonksiyonu
def get_recommendations_for_user(algo, user_id, item_ids, n=5):
    predictions = [algo.predict(user_id, item_id) for item_id in item_ids]
    recommendations = sorted(predictions, key=lambda x: x.est, reverse=True)[:n]
    return recommendations

# Belirli kullanıcıya ait yorumları filtreleme
user_id = 'A395BORC6FGVXV'
user_reviews = df[df['UserId'] == user_id]

# Kullanıcının yaptığı toplam yorum sayısı
num_reviews = len(user_reviews)

print(f"Kullanıcı {user_id} toplam {num_reviews} adet yorum yapmıştır.")

# Örnek kullanıcı için öneriler
user_id = 'A3SGXH7AUHU8GW'  # Kullanıcı ID'si
item_ids = df['ProductId'].unique()  # Tüm ürün ID'leri
recommendations = get_recommendations_for_user(algo_svd, user_id, item_ids, n=5)

print(f'\nÖnerilen Ürünler (SVD) Kullanıcı için ({user_id}):')
for rec in recommendations:
    print(f'Ürün: {rec.iid}, Tahmin Edilen Puan: {rec.est}')
