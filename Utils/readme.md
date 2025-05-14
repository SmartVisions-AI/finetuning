# Saving Models to GGUF with Unsloth

This guide explains how to save your fine-tuned models in GGUF format using [Unsloth](https://docs.unsloth.ai/basics/running-and-saving-models/saving-to-gguf), suitable for tools like LM studio.

## Saving Locally

To save your model in GGUF format locally:

You can also save it in hugginface hub for free and then load it through LM studio
```python
model.push_to_hub_gguf(chosen_model+"-GGUF-q4_k_m", tokenizer, quantization_method = "q4_k_m",token = "YOUR_HF_TOKEN")

```


```python
model.save_pretrained_gguf("output_directory", tokenizer, quantization_method="q4_k_m")

```


