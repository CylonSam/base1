# imports
import pyedflib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import wfdb
import datetime
import time
import warnings

from pathlib import Path
from scipy import signal
from pyts.image import RecurrencePlot
warnings.filterwarnings("ignore")

FRAME_LENGTH = 10
FRAME_SHIFT = 10

def read_eeg_signals(database, signal):
    '''
    Reads EEG signals from specified database and returns list of signals for each patient.

    Keyword arguments:

    database -- name of the database to be read

    signal -- name of the eeg signal to be read (c3 or c4)
    '''
    UCDDB_PATH = './physionet.org/files/ucddb/1.0.0/'
    CHALLENGE18_PATH = './physionet.org/files/challenge-2018/1.0.0/training'
    SLPDB_PATH = './physionet.org/files/slpdb/1.0.0'

    subjects_eeg_signals = None
    if (database == "ucddb"):
        print("reading ucddb database...")

        subjects_info_db1 = pd.read_excel(UCDDB_PATH + 'SubjectDetails.xls')

        subjects = subjects_info_db1['Study Number']
        number_of_subjects = len(subjects)

        signal = 3 if signal == 'c3' else 4

        subjects_eeg_signals = (pyedflib.EdfReader(
            UCDDB_PATH + subjects[p].lower() + '.rec').readSignal(signal) for p in range(number_of_subjects))
  

    elif (database == 'challenge18'):
        print("reading challenge18 database...")

        subjects_records = open(
            './physionet.org/files/challenge-2018/1.0.0/training/RECORDS', 'r').read().splitlines()

        number_of_subjects = len(subjects_records)

        signal = 2 if signal == 'c3' else 3

        subjects_eeg_signals = (wfdb.rdrecord(
                f"{CHALLENGE18_PATH}/{subjects_records[p]}/{subjects_records[p][:-1]}").p_signal[:, signal] for p in range(number_of_subjects))
       

    elif (database == 'slpdb'):
        print("reading slpdb database...")

        subjects_records = open(
            './physionet.org/files/slpdb/1.0.0/RECORDS', 'r').read().splitlines()

        number_of_subjects = len(subjects_records)

        subjects_eeg_signals = (wfdb.rdrecord(
            f"{SLPDB_PATH}/{subjects_records[p]}").p_signal[:, 2] for p in range(number_of_subjects))

    return subjects_eeg_signals

def create_frames(db_eeg_signals, sample_frequency, frame_length, frame_shift, resample=False):
    '''
    Create frames of specified length given an array of EEG signals

    Keyword arguments:

    db_eeg_signals -- array of EEG signals

    sample_frequency -- sample frequency of database

    frame_length -- desired frame length

    frame_shift -- desired frame shift
    '''
    
    resample_frequency = 250

    frame_samples = sample_frequency * frame_length

    for subject_eeg_signal in db_eeg_signals:
        step = frame_shift * sample_frequency

        eeg_signal_length = len(subject_eeg_signal)
        
        if resample:
            yield np.array([signal.resample(subject_eeg_signal[i:frame_samples+i], resample_frequency * frame_length) for i in range(0, eeg_signal_length - frame_samples, step)])
        else:
            yield np.array([subject_eeg_signal[i:frame_samples+i] for i in range(0, eeg_signal_length - frame_samples, step)])

def extract_seconds_from_time(time_str):
    x = time.strptime(time_str, "%H:%M:%S")
    return datetime.timedelta(hours=x.tm_hour, minutes=x.tm_min, seconds=x.tm_sec).total_seconds()

def compare_datetimes(datetime_to_compare, datetime_ref):
    res = (datetime.datetime.strptime(datetime_to_compare, '%H:%M:%S') - datetime_ref).total_seconds()
    if res < 0:
        return abs(res + (24*60*60))
    return res

def read_annotations(database):
    '''
    Read annotations from specified database

    Keyword arguments:

    database -- name of the database to be read
    '''
    UCDDB_PATH = './physionet.org/files/ucddb/1.0.0/'
    CHALLENGE18_PATH = './physionet.org/files/challenge-2018/1.0.0/training'
    SLPDB_PATH = './physionet.org/files/slpdb/1.0.0'

    subjects_annotations = []
    if (database == "ucddb"):
        subjects_info_db1 = pd.read_excel(UCDDB_PATH + 'SubjectDetails.xls')

        subjects_start_rec = subjects_info_db1['PSG Start Time']

        subjects = subjects_info_db1['Study Number']
        number_of_subjects = len(subjects)

        for s in range(number_of_subjects):
            subject_annotation_df_raw = pd.read_fwf(f"{UCDDB_PATH}{subjects[s].lower()}_respevt.txt", skiprows=2)[['Time', 'Type', 'Duration']]
            subject_rec_annotations = subject_annotation_df_raw.dropna()
            subject_rec_annotations.drop(['Type'], axis=1, inplace=True)

            # subjects_start_rec[s]
            subjects_start_rec_datetime = datetime.datetime.strptime(subjects_start_rec[s], '%H:%M:%S')

            subjects_start_rec_datetime = datetime.datetime.strptime(subjects_start_rec[s], '%H:%M:%S')
            subject_rec_annotations["Time"] = subject_rec_annotations['Time'] .apply(lambda time : compare_datetimes(time, subjects_start_rec_datetime))

            subjects_annotations.append(subject_rec_annotations)
        

    elif (database == 'challenge18'):
        subjects_records = open(
            './physionet.org/files/challenge-2018/1.0.0/training/RECORDS', 'r').read().splitlines()

        number_of_subjects = len(subjects_records)

        
        for s in range(number_of_subjects):
            subject_rec_path = f"{CHALLENGE18_PATH}/{subjects_records[s]}{subjects_records[s]}"[:-1]

            subject_rec_header = wfdb.rdheader(subject_rec_path)
            subject_rec_annotations = wfdb.rdann(subject_rec_path, "arousal")

            
            subject_rec_annotations_df = pd.DataFrame()

            subject_rec_annotations_samples = subject_rec_annotations.sample

            apnea_annotations_time = []
            apnea_annotations_duration = []

            annotations = subject_rec_annotations.aux_note

            start_time_event = 0
            for i in range(len(annotations)):
                if "(resp" in annotations[i]:
                    start_time_event = subject_rec_header.get_elapsed_time(time_value=subject_rec_annotations_samples[i]).seconds
                    apnea_annotations_time.append(start_time_event)


                elif "resp" in annotations[i]:
                    apnea_annotations_duration.append(subject_rec_header.get_elapsed_time(time_value=subject_rec_annotations_samples[i]).seconds - start_time_event)
                
            subject_rec_annotations_df.insert(0, "Time", apnea_annotations_time)
            subject_rec_annotations_df.insert(1, "Duration", apnea_annotations_duration)


            subjects_annotations.append(subject_rec_annotations_df)    

    elif (database == 'slpdb'):
        subjects_records = open(
            './physionet.org/files/slpdb/1.0.0/RECORDS', 'r').read().splitlines()

        number_of_subjects = len(subjects_records)

        for s in range(number_of_subjects):
            subject_rec_path = f"{SLPDB_PATH}/{subjects_records[s]}"

            subject_rec_header = wfdb.rdheader(subject_rec_path)
            subject_rec_annotations = wfdb.rdann(subject_rec_path, "st")

            subject_rec_annotations_df = pd.DataFrame()

            annotations_samples = subject_rec_annotations.sample
            annotations = subject_rec_annotations.aux_note

            apnea_annotations_time = []
            apnea_annotations_duration = []

            start_time_event = 0
            for i in range(len(annotations)):
                has_apnea = False

                for char in annotations[i]:
                    char = char.lower()
                    if char == "c" or char == "o" or char == "x" or char == "h":
                        has_apnea = True
                        break
                
                if has_apnea:
                    start_time_event = subject_rec_header.get_elapsed_time(time_value=annotations_samples[i]).seconds
                    apnea_annotations_time.append(start_time_event)
                    apnea_annotations_duration.append(30)

            subject_rec_annotations_df.insert(0, "Time", apnea_annotations_time)
            subject_rec_annotations_df.insert(1, "Duration", apnea_annotations_duration)


            subjects_annotations.append(subject_rec_annotations_df) 


    return subjects_annotations

def plot_frame_rp(frame, save=False, type=None, name=None):

    transformer = RecurrencePlot(threshold=np.pi/18)
    normal_rp = transformer.transform(np.array([frame]))

    plt.imshow(normal_rp[0], cmap='binary', origin='lower',
                extent=[0, 10, 0, 10])
    plt.xticks([])
    plt.yticks([])

    cwd = Path.cwd()

    if type == 1:
        plt.savefig(Path(cwd, "apnea", f"{name}.png"))
    else:
        plt.savefig(Path(cwd, "normal", f"{name}.png"))

    if not save:
        plt.show()

def create_frames_labels(frames, frame_length, annotations):
    '''
    Label subject frames given specified annotations

    Keyword arguments:

    frames -- array of frames

    frame_length -- size of frame

    annotations -- subject annotations dataframe
    '''
    event_current_second = 0

    for frame in frames:
        frame_start_second = event_current_second
        frame_end_second = event_current_second + frame_length

        annotations_array = annotations.to_numpy()

        has_apnea = False

        # if frame_start_second > 1080:
        #     print(2 + 2)
        #     print("ok")

        for annotation in annotations_array:
            annotation_end = annotation[0] + annotation[1]
            if annotation[0] > frame_end_second:
                break

            elif (frame_start_second >= annotation[0] and frame_start_second < annotation_end) or (frame_end_second >= annotation[0] and frame_end_second < annotation_end):
                has_apnea = True
                break
            # elif (annotation[0] >= frame_start_second and annotation[0] < frame_end_second) or (annotation_end > frame_start_second and annotation_end <= frame_end_second):
            #     has_apnea = True
            #     break
            
        
        if has_apnea:
            yield np.array([frame, 1])
        else:
            yield np.array([frame, 0])
        event_current_second += frame_length
    
def generate_frame_rp_figs(database):
    if (database == "ucddb"):
        ucddb_annotations = read_annotations('ucddb')
        ucddb_eeg_signals = read_eeg_signals('ucddb', 'c4')
        ucddb_frames = create_frames(ucddb_eeg_signals,128, FRAME_LENGTH, FRAME_SHIFT, resample=True)

        subject_count = 0
        frame_count = 0
        # subjects_apneia = 0

        for subject_frames in ucddb_frames:
            subject_frames_with_label = create_frames_labels(subject_frames, FRAME_LENGTH, ucddb_annotations[subject_count])
            if subject_count == 1:
                for frame_with_label in subject_frames_with_label:
                    if frame_with_label[1] == 1:
                        # subjects_apneia += 1
                        plot_frame_rp(frame_with_label[0], save=True, type=1, name=f"{subject_count}_{frame_count}_apnea")
                    elif frame_with_label[1] == 0:
                        plot_frame_rp(frame_with_label[0], save=True, type=0, name=f"{subject_count}_{frame_count}_normal")
                    frame_count += 1
            subject_count+=1

