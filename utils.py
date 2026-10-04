import torch


class LMCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, outputs, tokens, tokens_lens, loss_mask=None):
        """
        :param torch.Tensor outputs: Output from LM.forward. Shape: [B, T, V]
        :param torch.Tensor tokens: Batch of tokens. Shape: [B, T]
        :param torch.Tensor tokens_lens: Length of each sequence in batch
        :param torch.Tensor loss_mask: Valid next-token transitions. Shape: [B, T - 1]
        :return torch.Tensor: CrossEntropyLoss between corresponding logits and tokens
        """
        logits = outputs[:, :-1, :]
        targets = tokens[:, 1:]

        _, T = tokens.shape

        pos = torch.arange(T - 1, device=outputs.device).unsqueeze(0)

        length_mask = pos < (tokens_lens.to(outputs.device).unsqueeze(1) - 1)

        if loss_mask is not None:
            length_mask = length_mask & loss_mask.to(outputs.device)

        logits = logits[length_mask]
        targets = targets[length_mask]

        return super().forward(logits, targets)

class LMAccuracy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, outputs, tokens, tokens_lens, loss_mask=None):
        """
        :param torch.Tensor outputs: Output from LM.forward. Shape: [B, T, V]
        :param torch.Tensor tokens: Batch of tokens. Shape: [B, T]
        :param torch.Tensor tokens_lens: Length of each sequence in batch
        :param torch.Tensor loss_mask: Valid next-token transitions. Shape: [B, T - 1]
        :return torch.Tensor: Accuracy for given logits and tokens
        """
        logits = outputs[:, :-1, :]
        targets = tokens[:, 1:]
        predictions = logits.argmax(dim=-1)
        
        _, T = tokens.shape
        
        pos = torch.arange(T - 1, device=outputs.device).unsqueeze(0)
        
        length_mask = pos < (tokens_lens.to(outputs.device).unsqueeze(1) - 1)
        
        if loss_mask is not None:
            length_mask = length_mask & loss_mask.to(outputs.device)
        
        correct = (predictions == targets) & length_mask
        total = length_mask.sum()

        if total == 0:
            return torch.tensor(0.0, device=outputs.device)

        accuracy = correct.sum().float() / total.float()

        return accuracy
