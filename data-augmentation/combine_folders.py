import os
import shutil

base_path = '/share/acoustic_species_id/pretraining_data_combined'
combine_path = '/share/acoustic_species_id/BirdCLEF2021_audio'

def combine_folders():
    base_subfolders = [f.path.split('/')[-1] for f in os.scandir(base_path) if f.is_dir()]
    combine_subfolders = [f.path.split('/')[-1] for f in os.scandir(combine_path) if f.is_dir()]

    for species in combine_subfolders:
        # diff species, can move directly
        if species not in base_subfolders:
            shutil.copytree(os.path.join(combine_path, species),
                        os.path.join(base_path, species))
        # repeat species, assume files with same names are the same
        else:
            files = [f.path for f in os.scandir(os.path.join(combine_path, species))]
            count = 0
            for f in files:
                if not os.path.exists(os.path.join(base_path, species, f.split('/')[-1])):
                    shutil.copyfile(f, os.path.join(base_path,species, f.split('/')[-1]))
                    count += 1
            print(f"moved {count} files for {species}")

if __name__ == '__main__':
    combine_folders()


