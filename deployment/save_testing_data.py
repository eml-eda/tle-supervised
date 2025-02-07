import torch
import datetime

# shared  params
window_size = 1190
dir1 = '/space/benfenati/data_folder/SHM/AnomalyDetection_SS335/'
dir2 = '/space/benfenati/data_folder/SHM/Vehicles_Roccaprebalza/'
dir3 = '/space/benfenati/data_folder/SHM/Vehicles_Sacertis/'

###################### UC1 ######################
from data.AnomalyDetection_SS335.get_dataset import get_dataset as get_dataset_uc1
### Creating Testing Dataset for Normal Data
starting_date = datetime.date(2019,5,10) 
num_days = 4
print("Creating Testing Dataset -- Normal")
dataset = get_dataset_uc1(dir1, starting_date, num_days, sensor = 'S6.1.3', time_frequency = "frequency", windowLength = window_size)
data_loader_test_normal = torch.utils.data.DataLoader(
    dataset, shuffle=False,
    batch_size=1,
    num_workers=1,
    pin_memory='store_true',
    drop_last=True,
)
single_element = next(iter(data_loader_test_normal))
torch.save(single_element, 'deployment/data/uc1_data.pth')


###################### UC2 ######################
from data.Vehicles_Roccaprebalza.get_dataset import get_dataset as get_dataset_uc2
car = "y_car"
# training data (for both pre-training and fine-tuning)
_, dataset_test_car = get_dataset_uc2(dir2, window_sec_size = 60, shift_sec_size = 2, time_frequency = "frequency", car = car)
# testing data
data_loader_test_car = torch.utils.data.DataLoader(
    dataset_test_car, shuffle=False,
    batch_size=1,
    num_workers=1,
    pin_memory='store_true',
    drop_last=True,
)
single_element = next(iter(data_loader_test_car))
torch.save(single_element, 'deployment/data/uc2_data_car.pth')

car = "y_camion"
# training data (for both pre-training and fine-tuning)
_, dataset_test_camion = get_dataset_uc2(dir2, window_sec_size = 60, shift_sec_size = 2, time_frequency = "frequency", car = car)
# testing data
data_loader_test_camion = torch.utils.data.DataLoader(
    dataset_test_camion, shuffle=False,
    batch_size=1,
    num_workers=1,
    pin_memory='store_true',
    drop_last=True,
)
single_element = next(iter(data_loader_test_camion))
torch.save(single_element, 'deployment/data/uc2_data_camion.pth')


###################### UC3 ######################
from data.Vehicles_Sacertis.get_dataset import get_dataset as get_dataset_uc3
dataset = get_dataset_uc3(dir3, False, False, True,  sensor = "None", time_frequency = "frequency")
data_loader_test = torch.utils.data.DataLoader(
    dataset, shuffle=False,
    batch_size=1,
    num_workers=1,
    pin_memory='store_true',
    drop_last=True,
)
single_element = next(iter(data_loader_test))
torch.save(single_element, 'deployment/data/uc3_data.pth')