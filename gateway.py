import re
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
model_checkpoint = "./results/checkpoint-75"
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)
label_list = ["O", "B-PER", "I-PER", "B-DATE", "I-DATE", "B-AGE_OVER_89", "I-AGE_OVER_89", "B-LOC", "I-LOC"]
def predict_ner_tags(text):
    words = re.findall(r"\w+(?:'\s*\w+)?|[.,!?;:/\-]", text)
    inputs = tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=2)[0].tolist()
    word_ids = inputs.word_ids(batch_index=0)
    word_preds = ["O"] * len(words) 
    previous_word_idx = None
    for idx, word_idx in enumerate(word_ids):
        if word_idx is not None and word_idx != previous_word_idx:
            word_preds[word_idx] = label_list[predictions[idx]]
            previous_word_idx = word_idx
    return list(zip(words, word_preds))
def deidentify(text):
    word_tag_pairs = predict_ner_tags(text)
    
    words = [pair[0] for pair in word_tag_pairs]
    tags = [pair[1] for pair in word_tag_pairs]
    for i in range(len(words)):
        word = words[i]
        if i < len(words) - 1 and words[i+1].lower() in ["disease", "chorea", "syndrome"]:
            if tags[i] in ["B-PER", "I-PER"]:
                tags[i] = "O"
        if i > 0 and words[i-1].lower() == "history" and word.lower() == "of":
            if i < len(words) - 1 and tags[i+1] in ["B-PER", "I-PER"]:
                tags[i+1] = "O"
        if i < len(words) - 1 and word.isdigit() and words[i+1].lower() in ["days", "weeks", "months", "hours", "years"]:
            if i < len(words) - 2 and words[i+2].lower() in ["post-op", "ago", "since", "later"]:
                tags[i] = "O"
                tags[i+1] = "O"
                tags[i+2] = "O"
        if word.isdigit():
            val = int(word)
            if len(word) == 4 or val > 120:
                if tags[i] in ["B-AGE_OVER_89", "I-AGE_OVER_89"]:
                    tags[i] = "O"
            elif val <= 89:
                if tags[i] in ["B-AGE_OVER_89", "I-AGE_OVER_89"]:
                    tags[i] = "O"
            else:
                is_date_part = False
                if i > 0 and words[i-1] == "-":
                    is_date_part = True
                if i < len(words) - 1 and words[i+1] == "-":
                    is_date_part = True
                if not is_date_part:
                    tags[i] = "B-AGE_OVER_89"
    mapping = {}
    counters = {"PER": 1, "DATE": 1, "AGE_OVER_89": 1, "LOC": 1}
    masked_words = []
    i = 0
    while i < len(words):
        tag = tags[i]
        if tag != "O":
            entity_type = tag.split("-")[-1]
            entity_tokens = [words[i]]
            j = i + 1
            while j < len(words) and tags[j].startswith("I-"):
                entity_tokens.append(words[j])
                j += 1
            original_val = " ".join(entity_tokens)
            original_val = re.sub(r"\s+([.,!?;:/\-])", r"\1", original_val)
            original_val = re.sub(r"([.,!?;:/\-])\s+", r"\1", original_val)
            placeholder = f"[{entity_type}_{counters[entity_type]}]"
            masked_words.append(placeholder)
            mapping[placeholder] = original_val
            counters[entity_type] += 1
            i = j
        else:
            masked_words.append(words[i])
            i += 1
    deidentified_text = " ".join(masked_words)
    deidentified_text = re.sub(r"\s*([.,!?;:/\-])\s*", r"\1", deidentified_text)
    deidentified_text = re.sub(r"([.,!?;:])(?=[\w\[])", r"\1 ", deidentified_text)
    return deidentified_text, mapping
def rehydrate(response, mapping):
    rehydrated_text = response
    for placeholder, original in mapping.items():
        rehydrated_text = rehydrated_text.replace(placeholder, original)
    
    for placeholder, original in mapping.items():
        pattern = placeholder.replace("_", "[ _]?")
        rehydrated_text = re.sub(pattern, original, rehydrated_text, flags=re.IGNORECASE)
        
    return rehydrated_text
test_edge = "Dr. Parkinson diagnosed the 92-year-old patient with Parkinson's disease on 2023-10-12 at Seattle Clinic."
masked, mapping = deidentify(test_edge)
rehydrated = rehydrate(masked, mapping)

print("Original:  ", test_edge)
print("Masked:    ", masked)
print("Mapping:   ", mapping)
print("Rehydrated:", rehydrated)

with open("FAILURES.md", "a") as f:
    f.write("\n- Phase 4: Fixed 4-digit year bug misclassifying as AGE_OVER_89 in rule engine.\n")
