from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="./model/check_point-192000",
    tokenizer="bert-base-uncased"
)

result = fill_mask(
    "public [MASK] static void")
print(result)