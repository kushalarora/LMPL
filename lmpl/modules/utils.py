import torch 
import torch.nn.functional as F 
import numpy 

from allennlp.data.vocabulary import Vocabulary

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-1e30):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    #assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        filter_tensor = torch.ones_like(sorted_logits) * filter_value
        sorted_logits = torch.where(sorted_indices_to_remove, filter_tensor, sorted_logits)
        logits.scatter_(-1, sorted_indices, sorted_logits)
    return logits

def expand_tensor(tensor: torch.Tensor, num_tokens_to_rollout):
    """ Reshape/expand tensor to effective batch size of
         batch_size * num_tokens_to_rollout.
    """
    tensor_shape = tensor.shape
    batch_size = tensor_shape[0]; non_batch_dims = tuple(tensor_shape[1:])
    return tensor.unsqueeze(1)\
            .expand(batch_size, num_tokens_to_rollout, *non_batch_dims)\
            .reshape(batch_size * num_tokens_to_rollout, *non_batch_dims)

def decode_tokens(batch_predicted_indices: torch.Tensor,
                  vocab: Vocabulary,
                  end_index: int, 
                  start_index: int = None,
                  vocab_namespace:str ='tokens',
                  truncate: bool = False):
    if not isinstance(batch_predicted_indices, numpy.ndarray):
        batch_predicted_indices = batch_predicted_indices.detach().cpu().numpy()

    all_predicted_tokens = []
    for predicted_indices in batch_predicted_indices:
        # Beam search gives us the top k results for each source sentence in the batch
        # but we just want the single best.
        
        if len(predicted_indices.shape) == 1:
            predicted_indices = numpy.expand_dims(predicted_indices, axis=0)

        instance_predicted_tokens = []    
        for indices in predicted_indices:
            # We add start token to the predictions.
            # In case it is present at position 0, remove it.
            if start_index is not None and start_index == indices[0]:
                indices = indices[1:]

            indices = list(indices)
            # Collect indices till the first end_symbol
            if truncate and end_index in indices:
                indices = indices[:indices.index(end_index)]
            predicted_tokens = [vocab.get_token_from_index(x, namespace=vocab_namespace)
                                for x in indices]

            instance_predicted_tokens.append(predicted_tokens)
        all_predicted_tokens.append(instance_predicted_tokens)
    return all_predicted_tokens
