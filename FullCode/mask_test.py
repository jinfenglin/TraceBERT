from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="./model/full_code",
    tokenizer="bert-base-uncased"
)
