from Datasets.AnomalyDetection_SS335.get_dataset import get_dataset, get_data
import datetime
'''
Maintenance intervention day: 9th
Train: 24/05, 25/05, 26/05, 27/05
Validation: 28/05, 29/05, 30/05
Test post intervention: 01/06, 02/06, 03/06, 04/06 Test pre intervention: 01/05, 02/05, 03/05, 04/05

Data used from Amir:
# Week of Anomaly for test - 17 to 23 of April 2019
# Week of Normal for test - 10 to 14 of May 2019 
# Week of Normal for training - 20 or 22 to 29 of May 2019
'''
def main(directory):
    # starting_date = datetime.date(2019,5,24)
    # num_days = 1
    # dataset = get_dataset(directory, starting_date, num_days, sensor = 'D6.1.1', time_frequency = "time")
    dates = [
        # datetime.date(2019,6,1),
        # datetime.date(2019,5,1),
        # datetime.date(2019,5,10),
        # datetime.date(2019,5,20),
        datetime.date(2019,5,24),
        datetime.date(2019,5,28),
        datetime.date(2019,4,24),
        datetime.date(2019,4,26),
        datetime.date(2019,4,18),
    ]
    for date in dates:
        starting_date = date 
        num_days = 1
        dataset = get_data(directory, starting_date, num_days, sensor = 'S6.1.3', time_frequency = "time")
        import pdb;pdb.set_trace()
    


if __name__ == "__main__":
    dir = "./Datasets/AnomalyDetection_SS335/"
    main(dir)