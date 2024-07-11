import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
from pathlib import Path


def plot_results(file, save_dir):
    """ Plots training results from a 'results.csv' file; accepts file path and directory as arguments. """
    fig, ax = plt.subplots(2, 5, figsize=(12, 6), tight_layout=True)
    ax = ax.ravel()
    try:
        file_path = Path(file)
        data = pd.read_csv(file_path)
        s = [x.strip() for x in data.columns]
        x = data.values[:, 0]
        for i, j in enumerate([1, 2, 3, 4, 5, 8, 9, 10, 6, 7]):
            y = data.values[:, j].astype("float")
            # y[y == 0] = np.nan  # don't show zero values
            ax[i].plot(x, y, marker=".", label=file_path.stem, linewidth=2, markersize=8)  # actual results
            ax[i].plot(x, gaussian_filter1d(y, sigma=3), ":", label="smooth", linewidth=2)  # smoothing line
            ax[i].set_title(s[j], fontsize=12)
            # if j in [8, 9, 10]:  # share train and val loss y axes
            #     ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])
    except Exception as e:
        print(f"Warning: Plotting error for {file}: {e}")
    if savepath is not None:
        ax[1].legend()
        fig.savefig(save_dir + "/results.png", dpi=200)
    plt.show()
    plt.close()


if __name__ == '__main__':
    results_csv_path = r"C:\Users\v.stecko\Desktop\YOLOv5 Project\yolov5\runs\train\yolov5m_6classes_aftertrain\results.csv"
    savepath = r"C:\Users\v.stecko\Desktop\YOLOv5 Project\yolov5\runs\train\yolov5m_6classes_aftertrain"    # or None
    plot_results(file=results_csv_path, save_dir=savepath)
