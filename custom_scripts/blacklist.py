import torch
from models.common import DetectMultiBackend  # если используешь YOLOv5 >=v6.0, иначе см. ниже

# Путь к обученной модели YOLOv5 (полный чекпоинт)
weights = r"C:\Users\user\Downloads\last.pt"


# Загружаем модель из файла (автоматически определит архитектуру и веса)
model = DetectMultiBackend(weights)

# Сохраняем только state_dict (веса)
torch.save(model.model.state_dict(), 'model_weights_only.pt')

print("Только веса модели успешно сохранены в model_weights_only.pt")