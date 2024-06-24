# import os
# import numpy as np
# import cv2
# import win32gui
# import time
# from mss import mss
# os.getcwd()
#
#
# os.system('calc')
# sct = mss()
# xx = 1
# tstart = time.time()
# while xx < 10000:
#     hwnd = win32gui.FindWindow(None, 'Calculator')
#     left_x, top_y, right_x, bottom_y = win32gui.GetWindowRect(hwnd)
#     # screen = np.array(ImageGrab.grab( bbox = (left_x, top_y, right_x, bottom_y ) ) )
#     bbox = {'top': top_y, 'left': left_x, 'width': right_x - left_x, 'height': bottom_y - top_y}
#     screen = sct.grab(bbox)
#     scr = np.array(screen)
#
#     cv2.imshow('window', scr)
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         cv2.destroyAllWindows()
#         break
#     xx += 1
# cv2.destroyAllWindows()
# tend = time.time()
# print(xx / (tend - tstart))
# print((tend - tstart))
# os.system('taskkill /f /im calculator.exe')




# import cv2
# import torch
# from mss import mss
# import numpy as np
# from PIL import Image
#
#
# model = torch.hub.load(r"C:\Users\v.stecko\Desktop\YOLOv5 Project\yolov5", 'custom', path=r"C:\Users\v.stecko\Desktop\YOLOv5 Project\yolov5\runs\train\yolov5_STEP_1\weights\best.pt", source='local')
#
# sct = mss()
#
# while 1:
#     top, left = 100, 100
#     w, h = 640, 640
#     monitor = {'top': top, 'left': left, 'width': w, 'height': h}
#     img = Image.frombytes('RGB', (w, h), sct.grab(monitor).rgb)
#     imr_arr = np.array(img)
#
#     screen = cv2.cvtColor(imr_arr, cv2.COLOR_RGB2BGR)
#     # set the model use the screen
#     result = model(screen, size=640)
#     cv2.imshow('Screen', result.render()[0])
#
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         cv2.destroyAllWindows()
#         break



import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import cv2


signals_folder = r"D:\temp_folder"
images_folder = r"D:\temp_folder_2"


def open_all_bin_files(filepath, folder_to_save):
    for signal_file in os.listdir(filepath):
        signal_file_path = os.path.join(filepath, signal_file)        # get full path for signal
        head_path, tail_path = os.path.split(signal_file_path)
        name, extension = os.path.splitext(tail_path)
        if extension == '.i16bin':
            print(f'Opening and processing file {signal_file_path}...')
            iq_signal = open_bin_file(signal_file_path)

            f, t, Sxx_db = calc_spectrogram(data=iq_signal)
            check_plot(Sxx_db)
            # plot_and_save_spectrogram(t, f, Sxx_db, folder_to_save, name=r'\\' + name + '.jpg')
        else:
            print('Unknown file extension!')


def open_bin_file(path):
    """ This function reads binary file with format int16 big-endian and the convert it to I/Q signal """
    data = np.fromfile(path, dtype=np.dtype('>i2'), count=2097152*2)

    iq_sig = data[0::2] + 1j * data[1::2]
    return iq_sig


def calc_spectrogram(data, n_fft=1024, fs=100000000):

    """ This function calculates spectrogram and normalizes data """
    # iq_sig = np.array(data[0::2] + 1j * data[1::2])
    # iq_sig = [re + im * 1j for re, im in zip(data[0::2], data[1::2])]
    # iq_sig = np.array(iq_sig)

    f, t, Sxx = signal.spectrogram(x=data,
                                   fs=fs,
                                   return_onesided=False,
                                   nfft=n_fft,
                                   nperseg=n_fft,
                                   noverlap=0)

    Sxx_db = np.log10(np.fft.fftshift(Sxx, axes=0)) * 10
    return f, t, Sxx_db


def plot_and_save_spectrogram(t, f, Sxx_db, folder_to_save, name):
    f = np.fft.fftshift(f)

    plt.pcolormesh(t, f, Sxx_db, cmap='gist_rainbow')       # gist_rainbow
    # plt.colorbar()
    # plt.title('Spectrogram of I/Q signal')
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')

    plt.tight_layout(pad=0)        # Remove padding to maximize image area
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Set margins to zero
    # plt.imsave(images_folder + name, arr=Sxx_db)
    plt.savefig(folder_to_save + name)
    plt.close()


def check_plot(data):
    # Normalize data to range [0, 255] for image representation
    norm_data = 255 * (data - np.min(data)) / (np.max(data) - np.min(data))
    norm_data = norm_data.astype(np.uint8)
    print(norm_data)

    # Use OpenCV to create a color image from the normalized data
    color_image = cv2.applyColorMap(norm_data, cv2.COLORMAP_RAINBOW)

    # Resize the image to the desired size for model input
    screen = cv2.resize(color_image, (1024, 2048))

    cv2.imshow('img', screen)
    input()


if __name__ == '__main__':
    start_time = time.time()
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)
    open_all_bin_files(filepath=signals_folder, folder_to_save=images_folder)
    print(f'Program working time: {time.time() - start_time} sec.')




    # iq_sig = open_bin_file(path=r"D:\20ms SORTED DATASET\mavic3\mavic3_2G4_SNR16_05sec_WiFi_part12_20ms.i16bin")
    # f, t, Sxx_db = calc_spectrogram(iq_sig, n_fft=1024, fs=100000000)
    # plot_and_save_spectrogram(t, f, Sxx_db, 'gasf')





