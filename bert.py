from pytorch_pretrained_bert import BertModel, BertTokenizer
from math import ceil
import torch
import os.path
import os
import pickle


_models, _tokenizers = {}, {}


def encode(texts, targets=None, method=None, maximum_length=None, verbose=False):
    
    """ Encodes the given inputs with the chosen method. Encoding is done
    in batches to prevent out of memory errors.
    
    :param texts: a list of strings, representing the stance texts to encode
    :param targets: an optional list of strings, representing the stance targets;
        the i-th entry in the list is the stance target of the i-th stance text
    :param method: a string denoting the encoding method for the encoder, one of
        ["conditional-target", "conditional-text", None]
    :param maximum_length: the maximum length that the tokenised encoded inputs should have;
        if it is not specified, the length of longest input example is used;
    :return: a tensor representing the encoding """
    
    outputs = []
    batch_size = 200
    batches = ceil(len(texts) / batch_size)
    maximum_length = maximum_length or get_maximum_length(texts, targets, method)
    for batch in range(batches):
        if verbose and batch % 50 == 0:
            print("{} of {}".format(batch, batches))
        start = batch * batch_size
        end = min(start + batch_size, len(texts))
        _texts = texts[start:end]
        _targets = None if not targets else targets[start:end]
        o = _encode(_texts, targets=_targets, method=method, maximum_length=maximum_length)
        outputs.append(o.cpu())
    output = torch.cat(outputs, 0)
    return output


def _encode(texts, targets=None, method=None, maximum_length=None):
    model, tokenizer = get_bert_model_and_tokenizer()
    return bert_encode(model, tokenizer, texts, targets, maximum_length, method)


def get_bert_model_and_tokenizer(identifier="bert-base-uncased"):
    if identifier not in _models:
        _models[identifier] = BertModel.from_pretrained(identifier).eval().cuda()
        _tokenizers[identifier] = BertTokenizer.from_pretrained(identifier)
    return _models[identifier], _tokenizers[identifier]


def get_bert_sentence(text, target=None, method=None):
    method = method or "conditional-target"
    if not target or method == "text-only":
        return "[CLS] " + text + " [SEP]"
    elif method == "conditional-target":
        return "[CLS] " + target + " [SEP] " + text + " [SEP]"
    elif method == "conditional-text":
        return "[CLS] " + text + " [SEP] " + target + " [SEP]"
    
    
def get_bert_sentences(texts, targets=None, method=None):
    return [get_bert_sentence(text, targets[i] if targets else None, method=method) for i, text in enumerate(texts)]


def bert_encode(model, tokenizer, texts, targets=None, maximum_length=None, method=None):
    
    # Forms the sentences with BERT's specific encoding, that is, it preprend
    # the [CLS] token, appends the [SEP] token and separates first and second
    # sentences with the [SEP] token in case targets are given.
    
    sentences = get_bert_sentences(texts, targets, method=method)
    
    # Tokenizes the formed sentences with BERT's own tokenizer.
    
    tokenized_sentences = [tokenizer.tokenize(sentence) for sentence in sentences]
    
    # It determines the maximum length that tokenized sentences should have;
    # if a maximum length is specified, then all tokenized sentences longer
    # than that will be truncated to match the length once the closing token
    # is included. If it is not specified, the length of the longest tokenized
    # sentence is considered.
    
    maximum_length = maximum_length or max([len(tokenized_sentence) for tokenized_sentence in tokenized_sentences])   
    tokenized_sentences = ([tokenized_sentence[:maximum_length - 1] + ["[SEP]"] 
                            if len(tokenized_sentence) > maximum_length 
                            else tokenized_sentence for tokenized_sentence in tokenized_sentences])
        
    indexed_tokenized_sentences = [tokenizer.convert_tokens_to_ids(tokenized_sentence) for tokenized_sentence in tokenized_sentences]
    
    # Segment masks are created to denote distinction between first (target)
    # and second (text) sentence. Attention masks are also formed to distinguish
    # real tokens from padding tokens.

    segments = [[0] * (z.index("[SEP]") + 1) + [1] * (len(z) - z.index("[SEP]") - 1) + [0] * (maximum_length - len(z)) 
                for z in tokenized_sentences]
    masks = [[1] * len(z) + [0] * (maximum_length - len(z)) 
             for z in indexed_tokenized_sentences]
    
    # Finally, tokenized sentences are padded to have fixed length.
    
    padded_indexed_tokenized_sentences = [z + [0] * (maximum_length - len(z)) for z in indexed_tokenized_sentences]
    
    # Three key inputs are tensorized, parallelized and fed into the language model.
    
    inputs = torch.tensor(padded_indexed_tokenized_sentences)
    masks = torch.tensor(masks)
    segments = torch.tensor(segments)

    with torch.no_grad():
        output = model(input_ids=inputs.cuda(), token_type_ids=segments.cuda(), attention_mask=masks.cuda(), output_all_encoded_layers=False)
        return output[0][:,0,:]
    
    
def get_maximum_length(texts, targets=None, method=None):
    sentences = get_bert_sentences(texts, targets, method=method)
    _, tokenizer = get_bert_model_and_tokenizer()
    tokenized_sentences = [tokenizer.tokenize(sentence) for sentence in sentences]
    return max([len(z) for z in tokenized_sentences])