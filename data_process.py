import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm


def data_split(file_path, save_path):
    """
    将train.csv按照每次earthquake切分
    :param file_path: train.csv路径
    :param save_path: 输出文件路径（仅文件夹）
    :return: none
    """
    print(f"Loading data from {file_path} file:", end="")
    train_df = pd.read_csv(filepath_or_buffer=file_path,
                           dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
    print("Done")

    chunk_size = 150000
    segments = int(np.floor(train_df.shape[0] / chunk_size))

    earthquake_id = 1
    start_index = 0

    print("splitting")

    for segment_num in tqdm(range(segments)):
        begin = segment_num * chunk_size
        end = segment_num * chunk_size + chunk_size

        segment_df = train_df.iloc[begin: end]
        time_to_failure = segment_df['time_to_failure'].values

        # segments with an earthquake in it.
        if np.abs(time_to_failure[0] - time_to_failure[-1]) > 1:
            current_index = 0
            for index in range(len(segment_df) - 1):
                if np.abs(time_to_failure[index + 1] - time_to_failure[index]) > 1:
                    current_index = begin + index
                    # last_time_to_failure = time_to_failure[index+1]

            acoustic_data = train_df['acoustic_data'][start_index: current_index]
            time_to_failure_series = train_df['time_to_failure'][start_index: current_index]

            earthquake_segment = pd.concat([acoustic_data, time_to_failure_series], axis=1)
            print(f" earthquake_{earthquake_id}: {earthquake_segment.shape[0]}")
            earthquake_segment.columns = ['acoustic_data', 'time_to_failure']
            earthquake_segment.to_csv(save_path + "earthquake_" + str(earthquake_id) + ".csv", index=0)

            earthquake_id = earthquake_id + 1
            start_index = current_index + 1


def feature_generate(segment_signal):
    pass


if __name__ == '__main__':
    file_path = '../input.csv/train.csv'
    save_path = '../output/'
    data_split(file_path, save_path)
