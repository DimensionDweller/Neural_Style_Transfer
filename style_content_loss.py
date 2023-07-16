import torch

def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

def get_style_loss(gen, target):
    _, c, h, w = gen.size()
    gen = gen.view(c, h * w)
    target = target.view(c, h * w)
    G = torch.mm(gen, gen.t())
    A = torch.mm(target, target.t())
    style_loss = torch.mean((G - A)**2) / (c * h * w)
    return style_loss

def get_content_loss(gen, target):
    loss = torch.mean((gen - target)**2)
    return loss
