import numpy as np
import torch
from .dataframe_generation import power_transformation, alpha


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Database:
  def __init__(self, dataframe_list, training_species_pdf, batch_time, batch_size):
    self.dataframe_list        = dataframe_list
    self.dataframe_list_length = len(self.dataframe_list)
    self.batch_time            = batch_time
    self.batch_size            = batch_size
    self.columns               = list(self.dataframe_list[0].columns)
    self.idx                   = None
    self.true_t                = None
    self.true_y0               = None
    self.true_y                = None
    self.true_dydt             = None
    self.y0_norm               = None
    self.data_size             = 0
    self.training_species_pdf  = training_species_pdf

    self.columns.remove('t')
    print('[+]  Columns: ', self.columns)
    print('[+]  No. of columns: ', len(self.columns))

  def get_batch(self):
    # Choose temp/equiv condition
    self.idx = np.random.randint(0, self.dataframe_list_length)

    # Select t0s for time-batch
    self.data_size = self.dataframe_list[self.idx].shape[0]

    # Set true_y and true_y0 for evaluation
    self.set_true_ys()

    # sampling via PDF
    p = self.dataframe_list[self.idx][self.training_species_pdf].iloc[10:]
    p /= sum(p)
    
    initial_steps = np.random.choice(np.arange(self.data_size-self.batch_time, dtype=np.int64), self.batch_size, replace=False, p=p)
    
    # Stack data for each t0
    for i, initial_step in enumerate(initial_steps):
      # Copy dataframe
      df = self.dataframe_list[self.idx].copy()
      # Get t0
      # print(initial_step)
      t0 = df.iloc[initial_step]['t']
      # Get sequential data for t
      batch_t = torch.tensor(df.iloc[initial_step:initial_step+self.batch_time+1]['t'].values - t0)
      # Get sequential data for Ys and T
      df = df.drop(columns=['t', self.training_species_pdf])
      # Stack data for each t0
      if(i==0):
        batch_y0 = torch.tensor(df.iloc[initial_step].values).unsqueeze(0)
        batch_y  = torch.tensor(df.iloc[initial_step:initial_step+self.batch_time+1].values).unsqueeze(1).unsqueeze(2)
      else:
        batch_y0_temp = torch.tensor(df.iloc[initial_step].values).unsqueeze(0)
        batch_y_temp  = torch.tensor(df.iloc[initial_step:initial_step+self.batch_time+1].values).unsqueeze(1).unsqueeze(2)
        # concatenate batch
        batch_y0 = torch.cat([batch_y0, batch_y0_temp], dim=0)
        batch_y  = torch.cat([batch_y,  batch_y_temp], dim=1)

    # Calculate dydt for filling true values for output of ChemNODE
    dy             = batch_y[1:,:,:,:] - batch_y[:batch_y.shape[0]-1,:,:,:]
    #                  [.ooooo]
    #                      -
    #                   [ooooo.]
    dt             = batch_t[1:] - batch_t[:batch_t.shape[0]-1]
    dt             = np.expand_dims(dt,1)
    dt             = np.expand_dims(dt,1)
    dt             = np.expand_dims(dt,1)
    batch_dydt     = dy.clone().detach()/torch.tensor(dt)
    # remove '+1' part for batch_t and batch_y
    # '+1' was used for calculating dt and dy
    batch_t        = batch_t[:batch_t.shape[0]-1]
    batch_y        = batch_y[:batch_y.shape[0]-1,:,:,:]

    # Output shape:
    #    batch_y0  : (            batch_size, 1, No. of columns)
    #    batch_t   : (batch_time                               )
    #    batch_y   : (batch_time, batch_size, 1, No. of columns)
    #    batch_dydt: (batch_time, batch_size, 1, No. of columns)
    return batch_y0.unsqueeze(1).to(device), batch_t.to(device), batch_y.to(device), batch_dydt.to(device)

  def set_true_ys(self):
    df             = self.dataframe_list[self.idx].copy()
    self.true_t    = torch.tensor(df['t'].values)
    t0             = self.true_t[0]
    self.true_t    = (self.true_t - t0)
    df             = df.drop(columns=['t', self.training_species_pdf])
    self.true_y0   = torch.reshape(torch.tensor(df.iloc[0].values), (1,1,-1))
    self.true_y    = df.values
    self.true_t    = self.true_t[:self.data_size]
    dt             = self.true_t[1:] - self.true_t[:self.true_t.shape[0]-1]
    dt             = np.expand_dims(dt,1)
    dt             = np.expand_dims(dt,1)
    dt             = np.expand_dims(dt,1)
    self.true_y    = self.true_y[:self.data_size]
    self.y0_norm   = self.true_y[0]
    self.true_y    = np.expand_dims(self.true_y,1)
    self.true_y    = np.expand_dims(self.true_y,1)
    dy             = self.true_y[1:,:,:,:] - self.true_y[:self.true_y.shape[0]-1,:,:,:]
    self.true_y    = torch.tensor(self.true_y)
    self.true_dydt = torch.tensor(dy/dt)
    self.true_t    = self.true_t[:self.true_t.shape[0]-1]
    self.true_y    = self.true_y[:self.true_y.shape[0]-1,:,:,:]

  def get_dT_max_idx(self):
    dT_seq = self.true_dydt[:,0,0,self.columns.index('T')]
    return np.argmax(dT_seq)

  def normalize_by_max_val(self, max_val_dict):
    for dataframe in self.dataframe_list:
      for column in dataframe:
        if(column!='t'):
          dataframe[column] /= abs(power_transformation(max_val_dict[column], alpha, column))
