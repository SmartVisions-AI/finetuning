# Experiment: GRPO-Feinabstimmung

## Beschreibung

Qwen 14B Distill-Version von deepseek trainiert mit GRPO
Modelle können auch hier gefunden werden: [Hugginface Collection](https://huggingface.co/collections/moslehGen/grpo-681ccc9a22e612185370e900)

## Inhalt

| Notizbuch | Beschreibung | Modell Repo |
|----------|-------------|-------------|
| `Tourismus_Deepseek_R1_Qwen14B__GRPO_Training-500Steps.ipynb` | 500 Beispiele, Teilmenge des 26k Cot Dataset erhalten | moslehGen/DeepSeek-R1-Distill-Qwen-14B-500Steps
|

## Wie wird ausgeführt?

Die Inferenz mit Colab kann hier gefunden werden: [Colab](https://colab.research.google.com/drive/1y7ecU3swRg98_qW-EIL_AAq816Qrk5qD?usp=sharing)


## Die Belohnungsfunktion ist die grundlegende Belohnungsfunktion aus dem gsm8k-Datensatz

```python

## Reward-Funktionen
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    Antworten = [completion[0]['content'] for completion in completions]
    q = Eingabeaufforderungen[0][-1]['Inhalt']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f "Frage:\n{q}", f"\nAntwort:\n{Antwort[0]}", f"\nAntwort:\n{Antworten[0]}", f"\nExtrahiert:\n{extrahierte_Antworten[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(vervollständigungen, **kwargs) -> list[float]:
    Antworten = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Belohnungsfunktion, die prüft, ob die Vervollständigung ein bestimmtes Format hat."""
    pattern = r"^<Begründung>\n.*?\n</Begründung>\n<Antwort>\n.*?\n</Antwort>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Belohnungsfunktion, die prüft, ob die Vervollständigung ein bestimmtes Format hat."""
    pattern = r"<Begründung>.*?</Begründung>\s*<Antwort>.*?</Antwort>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def count_xml(text) -> float:
    count = 0.0
    if text.count("<begründung>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<Antwort>\n") == 1:
        count += 0.125
        Anzahl -= len(text.split("\n</Antwort>\n")[-1])*0,001
    wenn text.count("\n</Antwort>") == 1:
        count += 0.125
        Anzahl -= (len(text.split("\n</Antwort>")[-1]) - 1)*0,001
    return count

def xmlcount_reward_func(vervollständigungen, **kwargs) -> list[float]:
    content = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

```
