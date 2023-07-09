from os import listdir
import pandas as pd

def select_species_by_threshold(dirpath, species_threshold, train_filename_list=None):
    select_species = []
    species_mask_list = []
    for filename in listdir(dirpath):
        if(train_filename_list==None or filename in train_filename_list):
            filepath = dirpath + filename
            df = pd.read_csv(filepath)
            df_max_list = df.max()
            species_mask = df_max_list>species_threshold
            species_mask_list.append(species_mask)

    for species_mask in species_mask_list:
        select_species_ = [species for species in species_mask.index if species_mask[species]]
        select_species.extend(select_species_)
    select_species = list(dict.fromkeys(select_species))
    return select_species
