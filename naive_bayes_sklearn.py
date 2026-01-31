import pandas as pd
import re
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

# ===============================
# DOWNLOAD NLTK
# ===============================
nltk.download('punkt')
nltk.download('stopwords')

print("===================================")
print(" UAS KLASIFIKASI SENTIMEN TEKS ")
print(" Nama : TS.Rahmat Karim")
print("===================================\n")

# ===============================
# DATA SENTIMEN (BUAT CSV)
# ===============================
data_sentimen = {
    "komentar": [
        "Produk ini sangat bagus dan berkualitas tinggi",
        "Pelayanan yang diberikan sangat memuaskan sekali",
        "Harganya terlalu mahal untuk kualitas seperti ini",
        "Pengiriman cepat dan barang sesuai deskripsi",
        "Kualitas produk biasa saja tidak istimewa",
        "Sangat kecewa dengan pelayanan customer service",
        "Barang yang diterima rusak dan tidak bisa dipakai",
        "Sesuai dengan ekspektasi saya lumayan bagus",
        "Recommended banget pokoknya mantap jiwa",
        "Pengalaman belanja yang sangat menyenangkan",
        "Tidak sesuai dengan gambar yang ditampilkan",
        "Produknya standar tidak ada yang spesial",
        "Packing rapih dan aman terima kasih seller",
        "Kecewa banget barang tidak sesuai pesanan",
        "Harga sebanding dengan kualitas yang didapat",
        "Pelayanan ramah dan responsif sangat membantu",
        "Pengiriman lama sekali tidak sesuai estimasi",
        "Produk lumayan bagus untuk harga segini",
        "Sangat puas dengan pembelian kali ini",
        "Kualitas jelek tidak recommend sama sekali",
        "Barang original dan packaging mewah sekali",
        "Ukurannya tidak sesuai dengan size chart",
        "Produk cukup bagus sesuai harga",
        "Pelayanan buruk dan tidak profesional",
        "Terima kasih barang sudah sampai dengan selamat",
        "Warna tidak sesuai dengan foto di katalog",
        "Kualitas oke untuk harga yang terjangkau",
        "Sangat merekomendasikan produk ini ke teman",
        "Bahan produk kasar dan tidak nyaman dipakai",
        "Pengalaman berbelanja yang cukup menyenangkan",
        "Produk berkualitas premium worth it banget",
        "Pesanan tidak lengkap ada yang kurang",
        "Standar saja tidak mengecewakan tapi biasa",
        "Customer service sangat helpful dan sabar",
        "Produk cacat dan penjual tidak bertanggung jawab",
        "Lumayan bagus sesuai dengan harganya",
        "Pengiriman super cepat dan aman terimakasih",
        "Kualitas mengecewakan tidak sesuai harga",
        "Produk sesuai deskripsi dan foto",
        "Sangat puas recommended seller terpercaya",
        "Barang palsu bukan original kecewa banget",
        "Cukup memuaskan untuk pembelian pertama",
        "Pelayanan ramah dan pengiriman cepat mantap",
        "Ukuran kekecilan tidak bisa dipakai",
        "Produk standar sesuai ekspektasi",
        "Kualitas bagus banget melebihi ekspektasi",
        "Warna pudar setelah dicuci pertama kali",
        "Lumayan worth it untuk dicoba",
        "Seller responsif dan barang cepat sampai",
        "Bau tidak enak dan bahan murahan",
        "Sesuai dengan harga yang dibayarkan",
        "Produk original packaging rapih terimakasih",
        "Tidak akan beli lagi sangat mengecewakan",
        "Cukup bagus tidak terlalu istimewa",
        "Sangat memuaskan akan repeat order lagi",
        "Pengiriman lama dan barang rusak parah",
        "Produk standar tidak lebih tidak kurang",
        "Pelayanan excellent dan produk berkualitas",
        "Bahan tipis mudah robek tidak awet",
        "Lumayan oke sesuai dengan review",
        "Packaging mewah dan produk sangat bagus",
        "Kecewa total buang-buang uang saja",
        "Cukup puas dengan kualitas produk",
        "Recommended banget seller terpercaya dan amanah",
        "Ukuran tidak standar sangat mengecewakan",
        "Produk biasa saja sesuai harganya",
        "Sangat puas pengalaman belanja terbaik",
        "Warna luntur setelah beberapa kali pakai",
        "Lumayan bagus untuk harga segitu",
        "Kualitas premium pelayanan memuaskan mantap"
    ],
    "label": [
        2, 2, 0, 2, 1, 0, 0, 1, 2, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 2, 0, 1, 0, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2
    ]
}

# Buat DataFrame & simpan CSV
df = pd.DataFrame(data_sentimen)
df.to_csv("data_sentimen.csv", index=False)

print("File data_sentimen.csv berhasil dibuat\n")

# ===============================
# LOAD DATASET
# ===============================
data = pd.read_csv("data_sentimen.csv")

# Stopword & stemmer
stop_words = set(stopwords.words('indonesian'))
stemmer = StemmerFactory().create_stemmer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)

# Preprocessing
data['clean_komentar'] = data['komentar'].apply(clean_text)

X = data['clean_komentar']
y = data['label']

print(data[['komentar', 'clean_komentar']])

# Vectorisasi
vectorizer = CountVectorizer()
X_vector = vectorizer.fit_transform(X)

# Split data 80:20
X_train, X_test, y_train, y_test = train_test_split(
    X_vector, y, test_size=0.2, random_state=1
)

# Model Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# Prediksi
y_pred = model.predict(X_test)

# Evaluasi
print("=== HASIL EVALUASI MODEL ===")
print(classification_report(y_test, y_pred))

print("=== CONFUSION MATRIX ===")
print(confusion_matrix(y_test, y_pred))
