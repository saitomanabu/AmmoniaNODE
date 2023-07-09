from tqdm import tqdm
from os import listdir
import pandas as pd
from contextlib import suppress

alpha = 0.1

# 1. Collect cantera data for each temp/equiv ratio in csv in the form of dataframe (pandas).
# 2. Store each collected data in the form of list (dataframe_list).
def generate_dataframe_list(dirpath, select_species, training_species_pdf, train_filename_list=None):
  dataframe_list = []
  show_columns = False
  for filename in tqdm(listdir(dirpath)):
      if(train_filename_list==None or filename in train_filename_list):
        print("Currently processing: " + filename, end='\r')
        filepath = dirpath + filename
        df = pd.read_csv(filepath)
        if(show_columns):
            show_columns = False
        df = df[select_species+[training_species_pdf]]
        for species in select_species[1:]:
          df[species] = power_transformation(df[species] , alpha, species)
        df[training_species_pdf]=df[training_species_pdf][1:].min()/df[training_species_pdf]

        dataframe_list.append(df)

  print('\n[+]  Data loaded')
  return dataframe_list

def power_transformation(x, alpha, species):
  if(species!='T'):
    if(species=="Y_NH3" or species=="Y_O2"):
      return (x**alpha-1)/alpha
    else:
      return ((1-x)**alpha-1)/alpha
  else:
    return -(x-1000) / (-2000)

def un_power_transformation(x, alpha):
  return (x*alpha+1)**(1/alpha)
