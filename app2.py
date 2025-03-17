import streamlit as st
import tensorflow as tf
import numpy as np
import nltk
import re
import pandas as pd
from PIL import Image
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dropout, GlobalMaxPooling1D, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import gensim.downloader as api

# تحميل الموديل
@st.cache_resource()
def load_model():
    model = tf.keras.models.load_model("cnn_mode_headline.h5")  # تأكد من وجود الملف بنفس المسار
    return model

model = load_model()

# تحميل بيانات NLP
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# تحميل البيانات
url = "https://raw.githubusercontent.com/ironhack-labs/project-nlp-challenge/refs/heads/main/dataset/data.csv"
df = pd.read_csv(url)

# تنظيف البيانات
df = df.drop(columns=['date', 'subject', 'text'], axis=1)

# اختيار 20% من البيانات
sample = df.sample(frac=0.2, random_state=42)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenization
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
    return ' '.join(tokens)

# تحميل Word2Vec
w2v_model = api.load("word2vec-google-news-300")
embedding_dim = 300

def text_to_sequence(text, model):
    words = word_tokenize(text.lower())
    return [model[word] for word in words if word in model]

df['vectors'] = df['title'].apply(lambda x: text_to_sequence(str(x), w2v_model))
max_length = 30
X = pad_sequences(df['vectors'], maxlen=max_length, dtype='float32', padding='post', truncating='post', value=0.0)
y = df['label'].values

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# بناء الموديل
cnn_model = Sequential([
    Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=(max_length, embedding_dim)),
    Dropout(0.5),
    GlobalMaxPooling1D(),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# تدريب الموديل
early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
cnn_model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_test, y_test), callbacks=[early_stop])

# تقييم الموديل
loss, accuracy = cnn_model.evaluate(X_test, y_test)
st.write(f"Improved CNN Model Accuracy: {accuracy:.4f}")

# واجهة Streamlit
st.title("CNN Text Classification App")
st.write("قم بإدخال نص ليتم تحليله بواسطة نموذج CNN.")

user_input = st.text_area("أدخل النص هنا:")

if st.button("تحليل النص"):
    if user_input:
        processed_text = preprocess_text(user_input)
        text_vector = text_to_sequence(processed_text, w2v_model)
        text_vector_padded = pad_sequences([text_vector], maxlen=max_length, dtype='float32', padding='post', truncating='post', value=0.0)
        prediction = cnn_model.predict(text_vector_padded)
        class_idx = np.argmax(prediction)
        st.write(f"التصنيف المتوقع: {class_idx}")
