import torch


decay_prop = {
    "Bs_Jpsiphi": [
        torch.tensor([-13, 13, -321, 321]), torch.tensor([1, 1, 1, 1, 1, 1])
    ],
    "Bd_JpsiKst": [
        torch.tensor([-13, 13, 211, -321]), torch.tensor([1, 1, 1, 1, 1, 1])
    ],
    "Bu_JpsiK": [
        torch.tensor([-13, 13, 321]), torch.tensor([1, 1, 1])
    ]

}


pid_dict = {
    "Prob_k": 321,
    "Prob_e": 11,
    "Prob_mu": 13,
    "Prob_p": 2212,
    "Prob_pi": 211,
    "Prob_ghost": -1
}
