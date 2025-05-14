# Experiment: GRPO-Fine tuning 


## Description

Llama 8B Distill version of deepseek trained with GRPO

Models can also be found here: [Hugginface Collection](https://huggingface.co/collections/moslehGen/grpo-681ccc9a22e612185370e900)

## Contents

| Notebook | Description | Model Repo | 
|----------|-------------|-------------|
| `Tourism_Deepseek_R1_Llama3_1__GRPO_TFIDF+SentReward.ipynb` | 10K Dataset subset of the 166K recieved | moslehGen/DeepSeek-R1-Distill-Llama-8B5KSteps_TFIDF_SentReward
| `Tourism_Deepseek_R1_Llama3_1__GRPO_Training-500 Steps-Generic-Reward.ipynb` | 5k data samples, subset of the 166K recieved | moslehGen/DeepSeek-R1-Distill-Llama-8B-Steps-5000
| `Tourism_Deepseek_R1_Llama3_1__GRPO_Training-Round-2.ipynb` | 500 Data samples from 26k COT Dataset | moslehGen/DeepSeek-R1-Distill-Llama-8B-500Steps

## How to Run

Inference with Colab can be found here: [Colab](https://colab.research.google.com/drive/1y7ecU3swRg98_qW-EIL_AAq816Qrk5qD?usp=sharing)


## Reward Functions

For the  `Tourism_Deepseek_R1_Llama3_1__GRPO_TFIDF+SentReward.ipynb` two new reward function were introduced. One using TFIDF to detect similarity between completion (model generated answers) and reference answers. Another using Sentiments to find similar sentiment from both completion and answers if they match.


#### Simarity Reward Function

```python

german_stopwords = stopwords.words('german')
def tfidf_similarity_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    # Extract responses and the target answer
    responses = [completion[0]['content'] for completion in completions]
    target_answer = answer[0]

    # Extract the question (optional: you can include the prompt if needed in the similarity calculation)
    q = prompts[0][-1]['content']

    # Initialize the TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words=german_stopwords)

    # Combine target answer and completions for vectorization
    all_texts = [target_answer] + responses

    # Fit and transform the texts into TF-IDF vectors
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    # Calculate cosine similarities between the target answer (first vector) and completions (rest)
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])

    # Debugging print (can be removed if not needed)
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{target_answer}", f"\nResponse:\n{responses[0]}", f"\nSimilarity Score: {cosine_similarities[0][0]}")

    # Map cosine similarity to reward (e.g., if similarity is above a threshold, reward)
    #rewards = [1.0 if similarity[0] > SIMILARITY_THRESHOLD else 0.0 for similarity in cosine_similarities]
    rewards = [float(sim[0]) for sim in cosine_similarities]  # return actual score as reward

    return rewards

```

#### Sentiment Reward Function

`"oliverguhr/german-sentiment-bert"` model was chosen as the senitment analyzer as it scored 81% on F1 on our reference answer.

```python

# Load sentiment model once
sentiment_pipeline = pipeline("sentiment-analysis", model="oliverguhr/german-sentiment-bert",truncation=True,max_length=512)

def sentiment_match_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    # Assume answer is a list of one string (like [answer])

    answer_sentiment = sentiment_pipeline(answer[0])[0]['label']

    responses = [completion[0]['content'] for completion in completions]
    completion_sentiments = sentiment_pipeline(responses)

    rewards = [
        1.0 if completion_sent['label'] == answer_sentiment else 0.0
        for completion_sent in completion_sentiments
    ]
    print('-'*20, f"Answer Sentiment:\n{answer_sentiment}", f"\nResponse Sentiment:\n{completion_sentiments[0]['label']}", f"\nReward:\n{rewards[0]}")

    return rewards

```


