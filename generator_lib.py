# UCDDB_PATH = './physionet.org/files/ucddb/1.0.0/'
import math
import os
import shutil

import pyedflib
import datetime
import random
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Generator
from matplotlib.image import imsave
from pyts.image import RecurrencePlot


def compare_datetime(datetime_to_compare, datetime_ref):
    res = (datetime.datetime.strptime(datetime_to_compare, '%H:%M:%S') - datetime_ref).total_seconds()
    if res < 0:
        return abs(res + (24 * 60 * 60))
    return res


def read_annotations(database_path):
    """
    Read subjects signal annotations from specified database and returns a tuple of a list of subjects with apnea and a
    list of those subjects apnea annotations

    :param database_path: The absolute database path
    :type database_path: str
    """

    subjects_details = pd.read_excel(database_path + 'SubjectDetails.xls')

    subjects_recording_start = subjects_details['PSG Start Time']

    subjects = subjects_details['Study Number']

    subjects_length = len(subjects)

    eligible_subjects = []
    subjects_annotations = []
    for s in range(subjects_length):
        subject_annotation_df_raw = pd.read_fwf(f"{database_path}{subjects[s].lower()}_respevt.txt", skiprows=2)[
            ['Time', 'Type', 'Duration']]
        subject_rec_annotations = subject_annotation_df_raw.dropna()

        has_apnea = len(subject_rec_annotations[subject_rec_annotations['Type'].str.contains('APN')]) > 0

        if has_apnea:
            subject_rec_annotations = subject_rec_annotations[subject_rec_annotations['Type'].str.contains('APN')]
            subject_rec_annotations.drop(['Type'], axis=1, inplace=True)

            subjects_start_rec_datetime = datetime.datetime.strptime(subjects_recording_start[s], '%H:%M:%S')
            subject_rec_annotations["Time"] = subject_rec_annotations['Time'].apply(
                lambda time: compare_datetime(time, subjects_start_rec_datetime))

            eligible_subjects.append(s)
            subjects_annotations.append(subject_rec_annotations)

    return eligible_subjects, subjects_annotations


def read_eeg_signals(database_path: str, channel: str, eligible_subjects):
    """
    Reads EEG signals from specified database and returns a generator of signals for each patient.

    :param database_path: The absolute database path
    :type database_path: str
    :param channel: Name of the EEG channel to be read, c3 or c4
    :type channel: str
    :param eligible_subjects: indicates which subjects should have their data read
    :type eligible_subjects: list
    :returns: Generator that yields each subject EEG signal
    """

    print("reading subject info...")
    subjects_details = pd.read_excel(database_path + 'SubjectDetails.xls')

    subjects = subjects_details['Study Number']

    channel = 3 if channel == 'c3' else 4

    print("reading database...")
    subjects_eeg_signals = (pyedflib.EdfReader(
        database_path + subjects[p].lower() + '.rec').readSignal(channel) for p in eligible_subjects)

    return subjects_eeg_signals


def create_frames(subjects_eeg_signals: Generator, sample_frequency: int, frame_length: int, frame_shift: int):
    """
    Create frames of specified length given an array of EEG signals

    :param subjects_eeg_signals: Generator of subjects EEG signals
    :type subjects_eeg_signals: Generator
    :param sample_frequency: Sample frequency of database
    :type sample_frequency: int
    :param frame_length: Desired frame length in seconds for signal partitioning
    :type frame_length: int
    :param frame_shift: Desired frame shift in seconds for signal partitioning
    :type frame_shift: int
    :returns: Generator that yields each subject collection of their EEG signal frames
    """

    frame_samples = sample_frequency * frame_length

    for subject_eeg_signal in subjects_eeg_signals:
        step = frame_shift * sample_frequency

        signal_length = len(subject_eeg_signal)

        yield np.array(
            [subject_eeg_signal[i:frame_samples + i] for i in range(0, signal_length - frame_samples, step)])


def create_frames_labels(frames: np.ndarray, frame_length: int, frame_shift: int, annotations: pd.DataFrame):
    """
    Label subject frames given specified annotations and returns a generator of subject labeled frames

    :param frames: Numpy array of subject frames'
    :type frames: np.ndarray
    :param frame_length: Frame length in seconds used for signal partitioning
    :type frame_length: int
    :param frame_shift: Frame shift in seconds used for signal partitioning
    :type frame_shift: int
    :param annotations: Subject annotations dataframe
    :type annotations: pd.Dataframe
    :returns: Generator that yields the subject labeled frames
    """
    event_current_second = 0

    for frame in frames:
        frame_start = event_current_second
        frame_end = event_current_second + frame_length

        annotations_array = annotations.to_numpy()

        has_apnea = False
        apnea_duration = 0

        for annotation in annotations_array:
            annotation_start = annotation[0]
            annotation_end = annotation_start + annotation[1]

            if annotation_start > frame_end:
                break

            elif frame_start <= annotation_start < frame_end:
                has_apnea = True
                apnea_duration = frame_end - annotation_start
                break
            elif annotation_start <= frame_start and annotation_end >= frame_end:
                has_apnea = True
                apnea_duration = frame_length
                break
            elif frame_start < annotation_end <= frame_end:
                has_apnea = True
                apnea_duration = annotation_end - frame_start
                break

        yield frame, has_apnea, apnea_duration
        # if has_apnea:
        #     yield frame, 1, apnea_duration
        # else:
        #     yield np.array([frame, 0, apnea_duration])
        event_current_second += frame_shift


def save_frame_recurrence_plot_image(frame: np.ndarray, name: str, threshold: float, data_path: Path):
    """
    Creates recurrence plot image for a signal frame and saves it locally

    :param frame: A single frame from an EEG signal
    :param label: Frame label (apnea or non apnea)
    :param name: Name to be used when saving the file
    :param threshold: Threshold to be used when creating the recurrence plot
    :param data_path: The absolute path of folder to be used when saving the images
    """
    transformer = RecurrencePlot(threshold=threshold)
    rp = transformer.transform(np.array([frame]))
    rp_matrix = rp[0]
    imsave(Path(data_path, f"{name}.png"), rp_matrix, cmap='binary')


def log(data_path: str, generation_events: str, subjects: list, sample_frequency: int, frame_length: int,
        frame_shift: int, rp_threshold: float, total: int):
    generation_timestamp = datetime.datetime.now()
    image_size = sample_frequency * frame_length

    with open(Path(data_path, f"generation_info.txt"), "w") as file:
        file.write(f"generation_time: {generation_timestamp}{os.linesep}")
        file.write(f"generation_events: {generation_events}{os.linesep}")
        file.write(f"subjects: {subjects}{os.linesep}")
        file.write(f"sample_frequency: {sample_frequency}{os.linesep}")
        file.write(f"frame_length: {frame_length}{os.linesep}")
        file.write(f"frame_shift: {frame_shift}{os.linesep}")
        file.write(f"image_size: {image_size}x{image_size}{os.linesep}")
        file.write(f"rp_threshold: {str(rp_threshold)}{os.linesep}")
        file.write(f"total_sample: {total}{os.linesep}")


def balance_dataset(data_pool_path: Path):
    random.seed(42)

    apnea_images = []
    non_apnea_images = []

    for (dir_path, dir_names, filenames) in os.walk(data_pool_path):
        for filename in filenames:
            if filename[0] == "a":
                apnea_images.append(filename)
            elif filename[0] == "n":
                non_apnea_images.append(filename)
        break

    normal_images_to_delete = []
    for patient in range(25):
        print("flagging images for deletion - patient #" + str(patient))
        patient_apnea_images = list(filter(lambda x: x.split("_")[1] == str(patient), apnea_images))
        patient_normal_images = list(filter(lambda y: y.split("_")[1] == str(patient), non_apnea_images))

        apnea_images_length = len(patient_apnea_images)
        normal_images_length = len(patient_normal_images)

        normal_images_to_delete_length = max(normal_images_length - apnea_images_length, 0)

        normal_images_to_delete.extend(random.sample(patient_normal_images, k=normal_images_to_delete_length))

    print("starting deletion..")
    for n in normal_images_to_delete:
        path_to_file = Path(data_pool_path, n)
        os.unlink(path_to_file)

    return len(apnea_images)


def divide_dataset(mode="default", eligible_subjects=None, data_pool_path: Path = ""):
    random.seed(42)
    random.shuffle(eligible_subjects)

    images = []
    for (dir_path, dir_names, filenames) in os.walk(data_pool_path):
        images.extend(filenames)
        break

    subjects_length = len(eligible_subjects)

    data_directory = data_pool_path.parent

    datasets = [Path(data_directory, "data")]

    test_size = 0.25

    if mode == "wrong":
        test_size = 0
    elif mode == "subject_spec":
        test_size = 1 / subjects_length
        datasets = []
        for subject in eligible_subjects:
            subject_data_path = Path(data_directory, f"subject_{subject}")
            datasets.append(subject_data_path)

    # for each dataset, should create divisions
    for dataset in datasets:
        dataset_train_path = Path(dataset, "train")
        dataset_test_path = Path(dataset, "test")

        # creating required directories
        os.mkdir(dataset)
        os.mkdir(dataset_train_path)
        os.mkdir(dataset_test_path)
        os.mkdir(Path(dataset_train_path, "apnea"))
        os.mkdir(Path(dataset_train_path, "non_apnea"))
        os.mkdir(Path(dataset_test_path, "apnea"))
        os.mkdir(Path(dataset_test_path, "non_apnea"))

        # randomly chooses test subjects for this dataset
        test_subjects = random.sample(eligible_subjects, math.floor(test_size * subjects_length))

        # iterates over subjects and assign a place (train or test) for each one
        for subject in eligible_subjects:
            subject_apnea_images = list(filter(lambda x: x.split("_")[1] == str(subject) and x[0] == "a", images))
            subject_non_apnea_images = list(filter(lambda y: y.split("_")[1] == str(subject) and y[0] == "n", images))

            subset = "test" if subject in test_subjects else "train"

            subset_apnea_path = Path(dataset, subset, "apnea")
            subset_non_apnea_path = Path(dataset, subset, "non_apnea")

            for i in subject_apnea_images:
                shutil.move(Path(data_pool_path, i), Path(subset_apnea_path, i))
            for j in subject_non_apnea_images:
                shutil.move(Path(data_pool_path, j), Path(subset_non_apnea_path, j))


def generate_rp_images_dataset(mode: str, database_path: str, storage_path: str, sample_frequency: int,
                               frame_length: int, frame_shift: int, threshold: float):
    eligible_subjects, subjects_annotations = read_annotations(database_path)
    subjects_eeg_signals = read_eeg_signals(database_path, 'c3', eligible_subjects)
    subjects_frames = create_frames(subjects_eeg_signals, sample_frequency, frame_length, frame_shift)

    # creates temporary pool directory for images
    pool_path = Path(storage_path, "pool")
    os.mkdir(pool_path)

    subject_count = 0
    for subject_frames in subjects_frames:

        frame_count = 0
        subject_frames_with_label = create_frames_labels(subject_frames, frame_length, frame_shift,
                                                         subjects_annotations[subject_count])

        print(f"{datetime.datetime.now()} generating rp images for patient #{eligible_subjects[subject_count]}")
        for frame_with_label in subject_frames_with_label:
            apnea_time_slice = int(frame_with_label[2])
            has_apnea = frame_with_label[1]
            label_name = "apnea" if has_apnea else "non-apnea"

            save_frame_recurrence_plot_image(frame=frame_with_label[0],
                                             name=f"{label_name}_{eligible_subjects[subject_count]}_{frame_count}_{apnea_time_slice}",
                                             threshold=threshold,
                                             data_path=pool_path)
            frame_count += 1
        subject_count += 1

    total_img = balance_dataset(pool_path)
    divide_dataset(mode, eligible_subjects, pool_path)
    log(storage_path,
        "APNEA",
        eligible_subjects,
        sample_frequency,
        frame_length,
        frame_shift,
        threshold,
        total_img)


class RPImageGenerator:
    def __init__(self, mode, database_path: str, storage_path: str, frame_length: int, frame_shift: int,
                 threshold: float, sample_frequency: int):
        self.mode = mode
        self.database_path = database_path
        self.storage_path = storage_path
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.threshold = threshold
        self.sample_frequency = sample_frequency

    def generate(self):
        generate_rp_images_dataset(self.mode, self.database_path, self.storage_path, self.sample_frequency,
                                   self.frame_length, self.frame_shift, self.threshold)
