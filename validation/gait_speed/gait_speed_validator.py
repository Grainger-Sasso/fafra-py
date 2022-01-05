import os


def read_UIUC_gaitspeed_dataset(path):
    """
    In the GaitSpeedValidation folder, you will find a python script for
    reading and plotting within subject 101's data folder (e.g. B1_T1),
    and the full wearable data for young and older adults in our experiment.
    All subjects but 102 and 223 should have full data, consisting of two
    blocks (B1 and B2) and two trials (T1 and T2) all with gait data performed
    at the subject's comfortable walking pace. In addition, we have a script
    used to convert the .wav binary file into a .csv for further analysis.

    Within a data folder, you should find an info.json file, data folder,
    and a statistics.csv summary file.

    Within main folder, there is rawdataanalysis folder with a few scripts
    needed for raw data reading and conversion.
    :param path:
    :return:
    """
    path = r'C:\Users\gsass\Desktop\Fall Project Master\fafra_testing\validation\gait_speed\test_data\UIUC\GaitSpeedValidation'




