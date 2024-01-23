import torch

def get_gen_loss(crit_fake_pred):

    gen_loss = -crit_fake_pred.mean()
    return gen_loss

assert torch.isclose(
    get_gen_loss(torch.tensor(1.)), torch.tensor(-1.0)
)

assert torch.isclose(
    get_gen_loss(torch.rand(10000)), torch.tensor(-0.5), 0.05
)

print("Success!")

def get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda):

    crit_loss = crit_fake_pred.mean() - crit_real_pred + c_lambda * gp

    return crit_loss