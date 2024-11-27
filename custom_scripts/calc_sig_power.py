import pandas as pd


# Данные для DataFrame
data = [
    [90.992401, 474.26120, 605.102539, 525.417969, 0.784711, 2, 'autel_lite'],
]

# Названия столбцов
columns = ['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name']

# Создаем DataFrame
df = pd.DataFrame(data, columns=columns)

# Выводим результат
print(df)

xmin_sig = 2048 / 640 * df['xmin']
ymin_sig = 1024 / 640 * df['ymin']
xmax_sig = 2048 / 640 * df['xmax']
ymax_sig = 1024 / 640 * df['ymax']


print('xmin_sig = ', xmin_sig)
print('ymin_sig = ', ymin_sig)
print('xmax_sig = ', xmax_sig)
print('ymax_sig = ', ymax_sig)
