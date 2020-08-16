import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import gc


def extract_features(text, bert_model, bert_tokenizer, encode_config, device=device, return_tensors='pt'):
    encoded_item = bert_tokenizer.encode_plus(text, **encode_config)
    input_ids = encoded_item['input_ids'].to(device)
    attention_mask = encoded_item['attention_mask'].to(device)
    with torch.no_grad():
        features = bert_model(input_ids=input_ids, attention_mask=attention_mask)
    last_hidden_state, pooled_output, hidden_states = features
    hidden_states = torch.stack(hidden_states).detach().cpu()
    last_hidden_state = last_hidden_state.detach().cpu()
    pooled_output = pooled_output.detach().cpu()
    if return_tensors != 'pt':
        last_hidden_state = last_hidden_state.detach().cpu().numpy()
        pooled_output = pooled_output.numpy()
        hidden_states = hidden_states.numpy()
    del input_ids, attention_mask, bert_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return last_hidden_state, pooled_output, hidden_states


def last_layer_features(text, bert_model, bert_tokenizer, encode_config, return_tensors='pt'):
    last_hidden_state = extract_features(text=text,
                                         bert_model=bert_model,
                                         bert_tokenizer=bert_tokenizer,
                                         encode_config=encode_config,
                                         return_tensors=return_tensors)[0].squeeze()
    return last_hidden_state


def text_vector(text, bert_model, bert_tokenizer, encode_config, return_tensors='pt'):
    hidden_states = extract_features(text=text,
                                     bert_model=bert_model,
                                     bert_tokenizer=bert_tokenizer,
                                     encode_config=encode_config,
                                     return_tensors=return_tensors)[2]
    token_embeddings = torch.squeeze(hidden_states, dim=1)
    token_embeddings = token_embeddings.permute(1, 0, 2)
    token_vecs_sum = []
    for token in token_embeddings:
        sum_vec = torch.sum(token[-4:], dim=0)
        token_vecs_sum.append(sum_vec)
    token_vecs = hidden_states[-2][0]
    sentence_embedding = torch.mean(token_vecs, dim=0)
    if return_tensors != 'pt':
        sentence_embedding = sentence_embedding.detach().cpu().numpy()
    return sentence_embedding
