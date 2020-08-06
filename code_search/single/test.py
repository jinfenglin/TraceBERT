from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
sequence = tokenizer.encode_plus(text='Very severe pain in hands',
                            text_pair='Numbness of upper limb',
                            add_special_tokens=True,
                            return_token_type_ids=True)
print(sequence)