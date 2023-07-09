import torch
import torch.nn as nn


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# define network
class ODEFunc(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ODEFunc, self).__init__()
        self.time_idx       = 0
        self.species        = None
        self.columns        = None
        self.batch_y        = None
        self.time_direction = None
        self.flag_print     = False
        self.batch_dydt     = None
        self.pred_dydt      = []
        self.clamp          = None
        self.clamp_max      = None
        self.clamp_min      = None

        latent_dim = 32

        self.net = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, output_dim),
        )

    def forward(self, t, y):
        species_idx = self.columns.index(self.species)
        if(self.clamp):
            output = torch.clamp(
                self.net(y.float()),
                max=float(self.clamp_max[species_idx]),
                min=float(self.clamp_min[species_idx])
            )
        else:
            output = self.net(y.float())
        self.pred_dydt.append(output)
        output_all = self.batch_dydt[self.time_idx].float()
        output_all[:,:,[species_idx]] = output
        if(self.flag_print):
            print('output all', output_all)
            print('batch_y', self.batch_dydt[self.time_idx])
        self.time_idx += self.time_direction
        return output_all
