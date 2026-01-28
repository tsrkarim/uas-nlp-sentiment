import math

# =========================
# DATASET (diambil dari dataset_sentimen.pdf)
# label: 0 = negatif, 1 = netral, 2 = positif
# =========================
data = [
    ("Materinya goks abis auto paham instruktornya juga asik parah gak ngebosenin fix ini mah worth it", 2),
    ("Platformnya anjay interaktif banget jadi gak mager belajar bisa diakses kapan aja mood booster", 2),
    ("Mentornya pro player ngajarnya santuy tapi ngena ilmu daging semua gak kaleng kaleng", 2),
    ("Yah materinya B aja kurang update gitu loh agak laen ekspektasi gue", 0),
    ("Instrukturnya kureng ngejelasinnya muter muter bikin ngantuk skip dulu deh", 0),
    ("Platformnya sering nge lag ganggu banget pas lagi fokus belajar capek deh", 0),
    ("Materinya standar lah ya gak yang gimana gimana banget tapi oke buat dasar", 1)
]

# =========================
# PREPROCESSING
# =========================
def preprocess(text):
    text = text.lower()
    text = text.replace(".", "")
    text = text.replace(",", "")
    return text.split()

# =========================
# PERSIAPAN DATA
# =========================
classes = set(label for _, label in data)

texts = []
labels = []

for text, label in data:
    texts.append(preprocess(text))
    labels.append(label)

# =========================
# HITUNG PRIOR
# =========================
prior = {}
for c in classes:
    prior[c] = labels.count(c) / len(labels)

# =========================
# HITUNG FREKUENSI KATA
# =========================
word_freq = {c: {} for c in classes}
class_word_count = {c: 0 for c in classes}

for tokens, label in zip(texts, labels):
    for w in tokens:
        word_freq[label][w] = word_freq[label].get(w, 0) + 1
        class_word_count[label] += 1

# =========================
# VOCABULARY
# =========================
vocab = set()
for c in word_freq:
    vocab.update(word_freq[c].keys())

vocab_size = len(vocab)

# =========================
# FUNGSI PREDIKSI
# =========================
def predict(text):
    tokens = preprocess(text)
    scores = {}

    for c in classes:
        score = math.log(prior[c])
        for w in tokens:
            count = word_freq[c].get(w, 0) + 1
            score += math.log(count / (class_word_count[c] + vocab_size))
        scores[c] = score

    return max(scores, key=scores.get)

# =========================
# INTERFACE
# =========================
print("Model Naive Bayes siap digunakan")

while True:
    kalimat = input("\nMasukkan kalimat (exit untuk keluar): ")
    if kalimat.lower() == "exit":
        break

    hasil = predict(kalimat)

    if hasil == 0:
        print("Sentimen: NEGATIF")
    elif hasil == 1:
        print("Sentimen: NETRAL")
    else:
        print("Sentimen: POSITIF")
