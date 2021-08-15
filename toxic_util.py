import ast
import torch	

def get_entry(spans, sentence, word_embeds, nlp):
    labels = []
    sent_vec = []
    tokenized_text = []
    doc = nlp(sentence.lower())
    spans = ast.literal_eval(spans)
    for token in doc:
        word = token.text if token.lemma_ == '-PRON-' else token.lemma_
        tokenized_text.append(word)
        sent_vec.append(word_embeds[word].tolist())
        if token.idx in spans:
            if len(labels) == 0:
                labels.append(1)
            else:
                if labels[len(labels) - 1] == 0:
                    labels.append(1)
                elif labels[len(labels) - 1] in [1, 2]:
                    if token.idx - 1 in spans:
                        labels.append(2)
                    else:
                        labels.append(1)
        else:
            labels.append(0)
    
    return tokenized_text, torch.tensor(sent_vec), torch.tensor(labels)
    
def tensor_from_sentence(sentence, word_embeds, nlp):
    doc = nlp(sentence.lower())
    tensor = []
    for token in doc:
        word = token.lemma_ if token.lemma_ != '-PRON-' else token.text
        tensor.append(word_embeds[word].tolist())
    return torch.tensor(tensor)

    return spans
    
def labels_from_tensor(scores):
    labels = []
    for i, x in enumerate(scores):
        labels.append(torch.argmax(x).item())
    return labels
    
def spans_from_labels(sentence, labels, nlp):
    spans = []
    doc = nlp(sentence.lower())
    for token in doc:
        if labels[token.i] == 1 or labels[token.i] == 2:
            for i in range(len(token.text)):
                spans.append(token.idx + i)

        if token.i + 1 < len(labels) and labels[token.i] in [1,2] and labels[token.i + 1] == 2:
            next_token = doc[token.i + 1]
            for i in range(token.idx + len(token.text), next_token.idx):
                spans.append(i)


    return spans
    
def f1(predictions, gold):
    if len(gold) == 0:
        return 1 if len(predictions)==0 else 0
    nom = 2*len(set(predictions).intersection(set(gold)))
    denom = len(set(predictions))+len(set(gold))
    return nom/denom
