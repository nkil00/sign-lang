from torch import nn
import torch
# conversion functions
def label_to_int_index(label: str, class_index_dict: dict):
    """
    Takes as input a label and a dictionary which maps the classes of the dataset to a specific
    index. Returns the index associated to that label.
        _param: label:
        _param: class_index_dict:
        return: int
    """
    return class_index_dict[label]

# train functions
def train_batch_classification(model: nn.Module, batch, optimizer: torch.optim.Optimizer, loss_function,
                class_index):
    # enable training

    model.train()
    img, tar = batch
    index_of_label = torch.tensor([label_to_int_index(t, class_index) for t in tar], dtype=torch.long)

    # begin training
    optimizer.zero_grad()
    out = model(img)
    batch_loss = loss_function(out, index_of_label)
    batch_loss.backward()
    optimizer.step()
    
    return batch_loss.item()
