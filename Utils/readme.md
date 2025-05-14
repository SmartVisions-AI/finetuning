# Speichern von Modellen in GGUF mit Unsloth

In dieser Anleitung wird erklärt, wie Sie Ihre fein abgestimmten Modelle mit [Unsloth](https://docs.unsloth.ai/basics/running-and-saving-models/saving-to-gguf) im GGUF-Format speichern, das für Tools wie LM Studio geeignet ist.

## Lokales Speichern

So speichern Sie Ihr Modell lokal im GGUF-Format:

Sie können es auch kostenlos in Hugginface Hub speichern und dann mit LM Studio laden
```python
model.push_to_hub_gguf(chosen_model+"-GGUF-q4_k_m", tokenizer, quantization_method = "q4_k_m",token = "YOUR_HF_TOKEN")

```


```python
model.save_pretrained_gguf("output_directory", tokenizer, quantization_method="q4_k_m")

```


