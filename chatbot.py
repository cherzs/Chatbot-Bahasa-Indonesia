import os
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download("punkt")

# Membuat vektorisasi dan klasifikasi
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)
intents = [
    {
        "tag": "greeting",
        "patterns": [
            "Hai",
            "Hello",
            "Halo",
            "Apa kabar?",
            "Gimana kabarnya?",
            "Hai juga",
        ],
        "responses": [
            "Hai!",
            "Halo!",
            "Hai juga!",
            "Aku baik. Kamu gimana?",
            "Aku baik. Terima kasih sudah bertanya.",
        ],
    },
    {
        "tag": "goodbye",
        "patterns": [
            "Bye",
            "Sampai jumpa",
            "Sampai nanti",
            "Balik lagi ya",
            "Hati-hati",
        ],
        "responses": [
            "Sampai jumpa!",
            "Sampai nanti!",
            "Balik lagi ya!",
            "Sampai ketemu lagi!",
            "Hati-hati juga!",
        ],
    },
    {
        "tag": "thanks",
        "patterns": [
            "Terima kasih",
            "Makasih",
            "Makasih banyak",
            "Thx",
            "Trimakasih",
        ],
        "responses": [
            "Sama-sama!",
            "Tidak masalah.",
            "Senang bisa membantu.",
            "Tidak ada apa-apa.",
            "Terserah, aku di sini untuk membantu!",
        ],
    },
    {
        "tag": "about",
        "patterns": [
            "Apa yang bisa kamu lakukan",
            "Kamu siapa",
            "Kamu tuh apa sih",
            "Apa tujuanmu",
            "Ceritain dirimu dong",
        ],
        "responses": [
            "Aku adalah chatbot.",
            "Tujuanku adalah membantu kamu.",
            "Aku bisa menjawab pertanyaan dan membantu tugas-tugas.",
            "Aku di sini untuk membantu kamu dengan informasi dan tugas.",
            "Aku adalah asisten virtual yang dirancang untuk membantu kamu.",
        ],
    },
    {
        "tag": "help",
        "patterns": [
            "Bantu dong",
            "Aku perlu bantuan",
            "Bisa bantu aku",
            "Gimana caranya",
            "Tolong dong",
        ],
        "responses": [
            "Tentu, apa yang perlu kamu bantuan?",
            "Aku di sini untuk membantu. Ada masalah apa?",
            "Bagaimana aku bisa membantu kamu?",
            "Tentu, bagaimana aku bisa membantumu?",
            "Aku siap membantu. Ada yang bisa aku bantu?",
        ],
    },
    {
        "tag": "age",
        "patterns": [
            "Berapa usiamu",
            "Umur berapa",
            "Kamu berapa tahun",
            "Kamu lahir kapan",
            "Kamu tuh berapa tahun sih",
        ],
        "responses": [
            "Aku nggak punya usia. Aku cuma chatbot.",
            "Aku baru lahir di dunia digital.",
            "Umur cuma angka buatku.",
            "Aku nggak punya usia. Aku cuma chatbot.",
        ],
    },
    {
        "tag": "weather",
        "patterns": [
            "Gimana cuacanya",
            "Cuaca hari ini kayak gimana",
            "Info cuaca dong",
            "Ceritain cuacanya",
        ],
        "responses": [
            "Maaf, aku nggak bisa kasih info cuaca real-time.",
            "Kamu bisa cek cuaca di aplikasi cuaca atau situs web.",
            "Aku nggak bisa kasih info cuaca real-time, maaf ya.",
            "Mungkin coba cek aplikasi cuaca atau situs cuaca ya.",
        ],
    },
    {
        "tag": "relationship",
        "patterns": [
            "Apa saranmu tentang cinta-cintaan remaja",
            "Gimana cara hadapi perasaan sama seseorang",
            "Apa itu hubungan yang sehat",
            "Ceritain tentang asmara remaja",
        ],
        "responses": [
            "Cinta remaja bisa indah dan seru. Penting buat komunikasi terbuka sama pasangan dan saling menghormati perasaan.",
            "Hadapi perasaan buat seseorang bisa susah. Santai aja, kenal dia lebih dalam dan sampaikan perasaan kamu dengan jujur.",
            "Hubungan yang sehat didasari oleh kepercayaan, komunikasi, dan saling menghormati. Penting juga dukung pertumbuhan dan kebahagiaan pasangan.",
            "Asmara remaja itu bagian alami dari tumbuh dewasa. Ingat buat tetep fokus pada perkembangan pribadi kamu dan buat pilihan dengan bijak.",
        ],
    },
]

# Pra-pemrosesan data
tags = []
patterns = []
for intent in intents:
    for pattern in intent["patterns"]:
        tags.append(intent["tag"])
        patterns.append(pattern)

# Melatih model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)


def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent["tag"] == tag:
            response = random.choice(intent["responses"])
            return response


counter = 0


def main():
    global counter
    st.title("Chatbot")
    st.write(
        "Selamat datang di chatbot. Silakan ketik pesan dan tekan Enter untuk memulai percakapan."
    )

    counter += 1
    user_input = st.text_input("Anda:", key=f"user_input_{counter}")

    if user_input:
        response = chatbot(user_input)
        st.text_area(
            "Chatbot:",
            value=response,
            height=150,
            max_chars=None,
            key=f"chatbot_response_{counter}",
        )

        if response.lower() in ["selamat tinggal", "sampai jumpa"]:
            st.write(
                "Terima kasih telah berbincang dengan saya. Semoga harimu menyenangkan!"
            )
            st.stop()


if __name__ == "__main__":
    main()
