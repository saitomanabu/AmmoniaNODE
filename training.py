from tqdm import tqdm
from os import listdir
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint_adjoint as odeint
from modules.visualization import visualize
from modules.batch_generation import Database
from modules.dataframe_generation import generate_dataframe_list, un_power_transformation, alpha
from modules.species_selection import select_species_by_threshold
from modules.model import ODEFunc
import csv

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('[+]  Device: ', device)

# ------------------ Settings ------------------
batch_time          = 10
batch_size          = 80
niters              = 2000000
test_freq           = 100
model_path          = 'model.pth'
dirpath             = '(training data directory)'
training_species    = 'Y_NH3'
training_species_pdf= 'PDF_NH3'
load_network        = True
set_clamp           = False
species_threshold   = 1e-2
unselect_species    = [
                      'pressure',
		              'density',
                      ]
train_filename_list = [
                      'phi1.0_temperature2300.0.csv',
                      #...
                      ]
test_filename_list = train_filename_list
# ------------------- end ----------------------

if __name__ == '__main__':
    # Generate database and store it in 'database' variable.
    select_species, max_val_dict = select_species_by_threshold(dirpath, species_threshold, train_filename_list)
    select_species               = [species for species in select_species if species not in unselect_species]
    print('\n[+]  Species selected as: ', select_species, ' ( total of ', len(select_species),' ) ')
    train_dataframe_list = generate_dataframe_list(dirpath, select_species, training_species_pdf, train_filename_list)
    train_database       = Database(train_dataframe_list, training_species_pdf, batch_time=batch_time, batch_size=batch_size)
    test_dataframe_list  = generate_dataframe_list(dirpath, select_species, training_species_pdf, test_filename_list)
    test_database        = Database(test_dataframe_list, training_species_pdf, batch_time=batch_time, batch_size=batch_size)
    
    batch_y0, _, _, _ = train_database.get_batch()
    input_dim = batch_y0.shape[2]
    output_dim = 1

    # define network
    func = ODEFunc(input_dim, output_dim).to(device)
    func.columns = train_database.columns
    func.species = training_species

    # load network
    if(load_network):
        try:
            func.load_state_dict(torch.load(model_path))
            print('[+]  Model loaded')
        except:
            pass

    # define loss function
    loss_func = nn.L1Loss()

    # for visualization
    ii  = 0

    losses      = []
    test_losses = []

    optimizer = optim.AdamW(func.parameters(), lr=1e-3)

    itr = 0
    while itr<niters:
        itr+=1

        # reset gradients
        optimizer.zero_grad()

        # get batch
        batch_y0, batch_t, batch_y, batch_dydt = train_database.get_batch()
        _, _, _, _ = test_database.get_batch()

        # prediction
        true_dydt           = train_database.true_dydt.to(device)
        func.clamp          = set_clamp
        func.clamp_max      = torch.amax(true_dydt, dim=(0,1,2))
        func.clamp_min      = torch.amin(true_dydt, dim=(0,1,2))
        func.time_idx       = 0
        func.time_direction = 1
        func.flag_print     = False
        func.batch_dydt     = batch_dydt

        pred_y = odeint(func, batch_y0, batch_t, method='euler').to(device)

        # calculate loss
        species_idx = func.columns.index(func.species)
        loss = loss_func(pred_y[:,:,:,species_idx], batch_y[:,:,:,species_idx])

        # back propagation & update
        func.time_direction = -1
        loss.backward()
        optimizer.step()
        losses.append(loss.tolist()/batch_time/batch_size)


        # evaluate
        if itr % test_freq == 0:
            true_y0             = test_database.true_y0.to(device)
            true_y              = test_database.true_y.to(device)
            true_t              = test_database.true_t.to(device)
            true_dydt           = test_database.true_dydt.to(device)
            func.time_idx       = 0
            func.batch_dydt     = test_database.true_dydt.to(device)
            func.time_direction = 1
            func.flag_print     = False
            func.pred_dydt      = []

            with torch.no_grad():
                pred_y = odeint(func, true_y0, true_t, method='euler')
                test_loss = loss_func(pred_y[species_idx], true_y[species_idx])
                test_losses.append(test_loss.tolist()/true_t.shape[0])
                print('[+]  Iter {} | Total Loss {}'.format(itr, test_loss.item()))
                func.time_direction = 0
                visualize(
                    test_database.columns,
                    true_t,
                    un_power_transformation(true_y, alpha),
                    un_power_transformation(pred_y, alpha),
                    true_dydt*(un_power_transformation(true_y, alpha))**(1-alpha),
                    torch.tensor(func.pred_dydt).to(device)*(un_power_transformation(pred_y, alpha)[:-1,0,0,test_database.columns.index(training_species)])**(1-alpha),
                    func,
                    ii,
                    select_species,
                    training_species,
                )
                ii += 1

            # save model
            torch.save(func.state_dict(), './model_seq/model_{:04d}.pth'.format(itr))