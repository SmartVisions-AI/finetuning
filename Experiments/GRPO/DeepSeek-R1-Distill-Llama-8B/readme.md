# Experiment: GRPO-Feinabstimmung


## Beschreibung

Llama 8B Distill-Version von deepseek trainiert mit GRPO

Modelle können auch hier gefunden werden: [Hugginface Collection](https://huggingface.co/collections/moslehGen/grpo-681ccc9a22e612185370e900)

## Inhalt

| Notizbuch | Beschreibung | Modell Repo |
|----------|-------------|-------------|
| `Tourismus_Deepseek_R1_Llama3_1__GRPO_TFIDF+SentReward.ipynb` | 10K Datensatz Teilmenge der 166K erhaltenen | moslehGen/DeepSeek-R1-Distill-Llama-8B5KSteps_TFIDF_SentReward
| Tourismus_Deepseek_R1_Llama3_1__GRPO_Training-500 Steps-Generic-Reward.ipynb" | 5k Datenproben, Teilmenge der 166K erhaltenen | moslehGen/DeepSeek-R1-Distill-Llama-8B-Steps-5000
| Touristik_Deepseek_R1_Llama3_1__GRPO_Training-Round-2.ipynb` | 500 Datenproben aus 26k COT-Datensatz | moslehGen/DeepSeek-R1-Distill-Llama-8B-500Steps

## Wie man ausführt

Die Inferenz mit Colab kann hier gefunden werden: [Colab](https://colab.research.google.com/drive/1y7ecU3swRg98_qW-EIL_AAq816Qrk5qD?usp=sharing)


## Belohnungsfunktionen

Für die `Tourism_Deepseek_R1_Llama3_1__GRPO_TFIDF+SentReward.ipynb` wurden zwei neue Reward-Funktionen eingeführt. Die eine nutzt TFIDF, um die Ähnlichkeit zwischen der Vervollständigung (modellgenerierte Antworten) und den Referenzantworten zu erkennen. Eine andere nutzt Sentiments, um ähnliche Stimmungen in der Vervollständigung und den Antworten zu finden, wenn sie übereinstimmen.


#### Ähnlichkeits-Belohnungsfunktion

```python

german_stopwords = stopwords.words('deutsch')
def tfidf_similarity_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    # Extrahiere Antworten und die Zielantwort
    answers = [completion[0]['content'] for completion in completions]
    target_answer = answer[0]

    # Extrahieren Sie die Frage (optional: Sie können die Aufforderung bei Bedarf in die Ähnlichkeitsberechnung einbeziehen)
    q = prompts[0][-1]['content']

    # Initialisieren Sie den TF-IDF-Vektorisierer
    vectorizer = TfidfVectorizer(stop_words=deutsche_stopwords)

    # Kombinieren Sie Zielantwort und Vervollständigungen für die Vektorisierung
    all_texts = [target_answer] + responses

    # Anpassung und Transformation der Texte in TF-IDF-Vektoren
    tfidf_matrix = vectorizer.fit_transform(alle_texte)

    # Berechnen der Cosinus-Ähnlichkeiten zwischen der Zielantwort (erster Vektor) und den Ergänzungen (Rest)
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])

    # Debugging-Druck (kann entfernt werden, wenn nicht benötigt)
    print('-'*20, f "Frage:\n{q}", f"\nAntwort:\n{Ziel-Antwort}", f"\nAntwort:\n{Antworten[0]}", f"\nÄhnlichkeitswert: {cosine_similarities[0][0]}")

    # Kosinusähnlichkeit auf Belohnung abbilden (z. B. wenn die Ähnlichkeit über einem Schwellenwert liegt, Belohnung)
    #Belohnungen = [1.0 if similarity[0] > SIMILARITY_THRESHOLD else 0.0 for similarity in cosine_similarities]
    rewards = [float(sim[0]) for sim in cosine_similarities] # Rückgabe der tatsächlichen Punktzahl als Belohnung

    Belohnungen zurückgeben

```

#### Sentiment-Belohnungsfunktion

Das Modell "oliverguhr/german-sentiment-bert" wurde als Sentiment-Analysator gewählt, da es bei unserer Referenzantwort 81% auf F1 erzielte.

```python

# Sentiment-Modell einmal laden
sentiment_pipeline = pipeline("sentiment-analysis", model="oliverguhr/german-sentiment-bert",truncation=True,max_length=512)

def sentiment_match_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    # Angenommen, Antwort ist eine Liste mit einem String (wie [answer])

    antwort_sentiment = sentiment_pipeline(antwort[0])[0]['label']

    answers = [completion[0]['content'] for completion in completions]
    abschluss_sentiment = sentiment_pipeline(antworten)

    Belohnungen = [
        1.0 if completion_sent['label'] == answer_sentiment else 0.0
        for completion_sent in completion_sentiments
    ]
    print('-'*20, f "Antwort-Sentiment:\n{Antwort-Sentiment}", f"\nAntwort-Sentiment:\n{Abschluss-Sentiment[0]['label']}", f"\nBelohnung:\n{Belohnung[0]}")

    Belohnungen zurückgeben

```


