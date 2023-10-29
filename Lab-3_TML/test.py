import requests
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import textwrap
import string

# Завантаження тексту книги
url = "http://www.gutenberg.org/files/11/11-0.txt"
response = requests.get(url)
response.encoding = 'utf-8'
book_text = response.text

# Ініціалізація токенізатора, списку стоп-слів і лематизатора
tokenizer = nltk.WordPunctTokenizer()
stop_words = set(nltk.corpus.stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

#Створення функції для опрацювання тексту згідно заданих умов
def preprocess_document(text):
    # Видалення цифр та неалфавітних символів
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Приведення тексту до нижнього регістру і видалення зайвих пробілів на початку і в кінці
    text = text.lower().strip()

    tokens = tokenizer.tokenize(text)

    # Видалення стоп-слів
    filtered_tokens = [token for token in tokens if token not in stop_words]

    # Лематизація
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # Повертаємо оброблений текст
    return ' '.join(lemmatized_tokens)

# Попередня обробка тексту
processed_text = preprocess_document(book_text)

# Виведення опрацьованого тексту
print("Опрацьований текст :")
console_width = 80 #обмеження ширини консолі, для зручного читання тексту
text_lines = textwrap.wrap(processed_text, width=console_width)
for line in text_lines:
    print(line)

print("\n" + "-"*50 + "\n")

# Повернення тексту до початкового формату
text = book_text

# Знаходження всіх позицій, де зустрічається "CHAPTER I"
positions = [match.start() for match in re.finditer(r'\bCHAPTER I\b', text)]

if len(positions) >= 2:
    # Відсікаємо текст до другого згадування "CHAPTER I"
    text = text[positions[1]:]
else:
    print(" Не виявлено другого згадування 'CHAPTER I'")

# Розділення тексту на глави з використанням римських чисел
chapters = re.split(r'\bCHAPTER [IVXLCDM]+\b', text)

# Видалення порожніх розділів та зайвих символів після розділу
chapters = [re.sub(r'^\s*\.+\s*', '', chapter) for chapter in chapters if chapter.strip()]


# Виведення розділеного на глави тексту
print("Текст розділений на глави:")
for idx, chapter in enumerate(chapters):
    print(f"Chapter {idx + 1}:\n")
    chapter_lines = textwrap.wrap(chapter, width=console_width)
    for line in chapter_lines:
        print(line)
    print("\n" + "-" * 50)

# Обробка кожної глави
processed_chapters = [preprocess_document(chapter) for chapter in chapters]

# Видалення знаків пунктуації для кожного абзацу
for idx in range(len(processed_chapters)):
    processed_chapters[idx] = processed_chapters[idx].translate(str.maketrans('', '', string.punctuation))

# Застосування TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_df=0.85, max_features=20)
tfidf_matrix = tfidf_vectorizer.fit_transform(processed_chapters)

# Вибір Топ-20 слів для кожної глави
top_words = {}

feature_names = tfidf_vectorizer.get_feature_names_out()
for idx, row in enumerate(tfidf_matrix):
    top_indices = row.toarray()[0].argsort()[-20:][::-1]
    top_features = [feature_names[i] for i in top_indices]
    top_words[f"Chapter {idx+1}"] = top_features

print("Топ-20 слів для кожної глави (TF-IDF):")
for chapter, top_20_words in top_words.items():
    print(f"{chapter}: {', '.join(top_20_words)}")

# Використання CountVectorizer
vectorizer = CountVectorizer(max_df=0.90, min_df=2)
dtm = vectorizer.fit_transform(chapters)

# LDA
lda_model = LatentDirichletAllocation(n_components=len(chapters), random_state=42)
lda_model.fit(dtm)

# Функція для виведення тем з LDA
def print_lda_topics(model, vectorizer, n_words=20):
    words = vectorizer.get_feature_names_out()
    for idx, topic in enumerate(model.components_):
        print(f"Chapter {idx+1} (LDA):", [words[i] for i in topic.argsort()[-n_words:][::-1]])

#  Виведення топ-20 слів використовуючи алгоритм LDA
print("\nLDA:")
for idx in range(len(chapters)):
    lda_topic_words = [vectorizer.get_feature_names_out()[i] for i in lda_model.components_[idx].argsort()[-20:][::-1]]
    lda_topic_words_str = ', '.join(lda_topic_words)
    print(f"Chapter {idx+1} (LDA): {lda_topic_words_str}")

#  Порівняння з TF-IDF
print("\nПорівняння результатів LDA і TF-IDF:")
for idx in range(len(chapters)):
    print(f"Chapter {idx + 1} TF-IDF: {', '.join(top_words[f'Chapter {idx + 1}'])}")
    print(f"Chapter {idx + 1} LDA: {', '.join([vectorizer.get_feature_names_out()[i] for i in lda_model.components_[idx].argsort()[-20:][::-1]])}")
    print("-" * 50)
