import os


filepath_file = r"C:\Users\v.stecko\Desktop\file_2024-10-10_10-20-39_291424.log"
new_filepath_file = r"C:\Users\v.stecko\Desktop\edited_file_2024-10-10_10-20-39_291424.log"


if __name__ == '__main__':
    with open(filepath_file, 'r') as f:
        new_correct_lines = []
        text = f.readlines()
        for string in text:
            if 'Unknown connection name' not in string:
                new_correct_lines.append(string)
        f.close()
        with open(new_filepath_file, 'x') as f:
            f.writelines(new_correct_lines)
