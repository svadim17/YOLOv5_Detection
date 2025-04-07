import os
import unidecode


def rename_symbols_in_files(filepath):
    for file in os.listdir(filepath):
        file_filepath = os.path.join(filepath, file)  # get full path for signal
        head_path, tail_path = os.path.split(file_filepath)
        name, extension = os.path.splitext(tail_path)

        if ',' in name or ' ' in name or '.' in name:
            new_filename = unidecode.unidecode(name).replace(",", "_").replace(" ", "_").replace(".", "_")
            os.rename(os.path.join(filepath, file), os.path.join(filepath, new_filename + extension))
            print(f"File '{file}' renamed to '{new_filename + extension}'")


if __name__ == "__main__":
    filepath_folder = r"C:\Users\v.stecko\Desktop\YOLOv5 Project\yolov5\data\obj\labels"
    rename_symbols_in_files(filepath_folder)
