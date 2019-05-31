import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
from matplotlib import pyplot as plt
import os


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


def load_origin_data(file_path):
    signal_data = pd.read_csv(filepath_or_buffer=file_path,
                              dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
    return signal_data


def plot_origin_data(signal_data, step, name, output_path):
    train_ad = signal_data['acoustic_data'][::step]
    train_ttf = signal_data['time_to_failure'][::step]
    title = f"{name} Acoustic data and time to failure: 1% sampled data"
    fig, ax1 = plt.subplots(figsize=(12, 8))
    plt.title(title)
    plt.plot(np.arange(train_ad.size), train_ad, color='r')
    ax1.set_ylabel('acoustic data', color='r')
    plt.legend(['acoustic data'], loc=(0.01, 0.95))
    ax2 = ax1.twinx()
    plt.plot(np.arange(train_ttf.size), train_ttf, color='b')
    ax2.set_ylabel('time to failure', color='b')
    plt.legend(['time to failure'], loc=(0.01, 0.9))
    plt.grid(True)

    # plt.savefig(output_path + name)
    plt.show()


if __name__ == '__main__':
    path = "../earthquakes/"
    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    s = []
    for file in tqdm(files):  # 遍历文件夹
        if not os.path.isdir(file) and file.split(".")[1] == "csv":  # 判断是否是文件夹，不是文件夹才打开
            signal_data = load_origin_data(path + file)
            plot_origin_data(signal_data, 150000, file.split(".")[0], path)
