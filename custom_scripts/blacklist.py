import pandas as pd


map_list = ['noise', 'autel', 'fpv', 'dji', 'wifi']

df = pd.read_csv('yolov5/return_example.csv')

data = df.groupby(['name'])['confidence'].max()

labels_to_combine = ['autel_lite', 'autel_max', 'autel_pro_v3', 'autel_tag']

# Получаем значения этих меток, если они существуют, иначе None
values = [data.get(label) for label in labels_to_combine]

values = [value for value in values if value is not None]

if values:
    # Выбираем максимальное значение среди доступных
    max_value = max(values)
else:
    max_value = None  # Или какое-то другое значение по умолчанию

# Создаем новый Series с учетом объединения
new_data = data.drop(labels_to_combine, errors='ignore')
new_data['autel'] = max_value

print(new_data.index)
print(new_data.values)

print(data.index)
print(data.values)

result_list = []

for name in map_list:
    try:
        result_list.append(data[name])
    except KeyError:
        result_list.append(0)

# print(result_list)




