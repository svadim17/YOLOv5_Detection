import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import signal
import cv2
import datetime


signals_folder = r"D:\YOLOv5 DATASET\STEP 3\IMAGES\autel_evo_pro_v3_sig"
images_folder = r"D:\YOLOv5 DATASET\STEP 3\IMAGES\autel_evo_max_shift"


def open_all_bin_files(filepath):
    for signal_file in os.listdir(filepath):
        signal_file_path = os.path.join(filepath, signal_file)        # get full path for signal
        head_path, tail_path = os.path.split(signal_file_path)
        name, extension = os.path.splitext(tail_path)

        if extension == '.i16bin':
            print(f'Opening and processing file {signal_file_path}...')
            iq_signal = open_i16bin_file(signal_file_path)
            Sxx_db = calc_spectrogram_i16bin(data=iq_signal)
            img = convert_to_image(Sxx_db)
            save_image(img, images_folder)
            # plot_and_save_spectrogram(t, f, Sxx_db, name=r'\\' + name + '.jpg')

        elif extension == '.sglbin':
            print(f'Opening and processing file {signal_file_path}...')
            spectrum_signal = open_sglbin_file(signal_file_path)
            img = convert_to_image(spectrum_signal)
            save_image(img, images_folder)

        elif extension == '.magi8':
            print(f'Opening and processing file {signal_file_path}...')
            spectrum_signal = open_magi8_file(signal_file_path)
            img = convert_to_image(spectrum_signal)
            save_image(img, images_folder)

        else:
            print('Unknown file extension!')


def open_i16bin_file(path):
    """ This function reads binary file with format int16 big-endian and the convert it to I/Q signal """
    data = np.fromfile(path, dtype=np.dtype('>i2'), count=2097152*2)
    iq_sig = data[0::2] + 1j * data[1::2]
    return iq_sig


def calc_spectrogram_i16bin(data, n_fft=1024, fs=100000000):
    """ This function calculates spectrogram and normalizes data """
    f, t, Sxx = signal.spectrogram(x=data,
                                   fs=fs,
                                   return_onesided=False,
                                   nfft=n_fft,
                                   nperseg=n_fft,
                                   noverlap=0)

    Sxx_db = np.log10(np.fft.fftshift(Sxx, axes=0)) * 10
    spectrum_sig = np.transpose(Sxx_db)
    return spectrum_sig


def open_sglbin_file(path):
    data = np.fromfile(path, dtype=np.dtype('>f4'))
    sig_2D = data.reshape(2048, 1024)
    return sig_2D


def open_magi8_file(path):
    data = np.fromfile(path, dtype=np.dtype('>i1'))
    sig_2D = data.reshape(2048, 1024)
    return sig_2D


def convert_to_image(data):
    """ This function normalizes data to image format and converts it to image """
    # data = np.transpose(data + 122)
    data = np.transpose(data + 122)

    z_min = -45
    z_max = 35
    norm_data = 255 * (data - z_min) / (z_max - z_min)
    norm_data = norm_data.astype(np.uint8)

    color_image = cv2.applyColorMap(norm_data, cv2.COLORMAP_RAINBOW)
    screen = cv2.resize(color_image, (640, 640))
    return screen


def save_image(image, save_path):
    """ This function saves image to folder """
    filename = datetime.datetime.now().strftime('%m-%d-%H-%M-%S-%f')
    cv2.imwrite(filename=save_path + '\\' + filename + '.jpg', img=image)


def plot_and_save_spectrogram(t, f, Sxx_db, name):
    f = np.fft.fftshift(f)
    a = plt.pcolormesh(t, f, Sxx_db, cmap='rainbow')       # BuGn
    plt.tight_layout(pad=0)        # improve layout for better image quality
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Set margins to zero

    fig = a.get_figure()
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncol, nrow = fig.canvas.get_width_height()
    image = np.frombuffer(buf, dtype=np.uint8).reshape(nrow, ncol, 3)
    print(image.shape)
    # image = Image.fromarray(image).show()


# def plot_and_save_spectrogram(t, f, Sxx_db, folder_to_save, name):
#     f = np.fft.fftshift(f)
#     plt.pcolormesh(t, f, Sxx_db, cmap='rainbow')
#     plt.tight_layout(pad=0)
#     plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
#     plt.savefig(folder_to_save + name)
#     plt.close()                                             # Close the plot to free up memory


if __name__ == '__main__':
    open_all_bin_files(filepath=signals_folder)

    # iq_sig = open_bin_file(path=filepath)
    # f, t, Sxx_db = calc_spectrogram(iq_sig, n_fft=1024, fs=100000000)
    # plot_and_save_spectrogram(t, f, Sxx_db)




