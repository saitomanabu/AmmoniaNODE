import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint_adjoint as odeint
import matplotlib.pyplot as plt
from contextlib import suppress

device = 'cuda' if torch.cuda.is_available() else 'cpu')

def visualize(columns, t, true_y, pred_y, true_dydt, pred_dydt, odefunc, itr, selected_species, training_species):
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_dydt_trajectory = fig.add_subplot(133, frameon=False)
    ax_dydt_scatter = fig.add_subplot(132, frameon=False)

    ax_traj.cla()
    ax_traj.set_title('Trajectories')
    ax_traj.set_xlabel('t')
    ax_traj.set_ylabel('y')

    for i, species in enumerate(selected_species[1:]):
        # ax_traj.plot(t.cpu().detach().numpy(), true_y.cpu().detach().numpy()[:,0,0,columns.index(species)], label=species)
        if(species==training_species): ax_traj.plot(t.cpu().detach().numpy(), true_y.cpu().detach().numpy()[:,0,0,columns.index(species)], 'k.', label=species)
        if(species==training_species): ax_traj.plot(t.cpu().detach().numpy(), pred_y.cpu().detach().numpy()[:,0,0,columns.index(species)], 'b')

    ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
    ax_traj.legend()

    ax_dydt_trajectory.cla()
    ax_dydt_trajectory.set_title('Trajectory plot dydt')
    ax_dydt_trajectory.set_xlabel('Cantera')
    ax_dydt_trajectory.set_ylabel('ChemNODE')
    ax_dydt_trajectory.scatter(true_dydt[:-1,0,0,columns.index(training_species)], pred_dydt, s=0.3, color='b')
    lims = [
        np.min([ax_dydt_trajectory.get_xlim(), ax_dydt_trajectory.get_ylim()]),  # min of both axes
        np.max([ax_dydt_trajectory.get_xlim(), ax_dydt_trajectory.get_ylim()]),  # max of both axes
    ]
    ax_dydt_trajectory.plot(lims, lims, 'k-', alpha=0.75, zorder=0)

    ax_dydt_scatter.cla()
    ax_dydt_scatter.set_title('Scatter plot dydt')
    ax_dydt_scatter.set_xlabel('t')
    ax_dydt_scatter.set_ylabel('dydt')
    ax_dydt_scatter.scatter(t.cpu().detach().numpy()[:-1], true_dydt[:-1,0,0,columns.index(training_species)], s=0.3, color='k')
    ax_dydt_scatter.scatter(t.cpu().detach().numpy()[:-1], pred_dydt, s=0.3, color='b')

    fig.tight_layout()
    plt.savefig('./png/{:03d}'.format(itr))
    plt.close()
