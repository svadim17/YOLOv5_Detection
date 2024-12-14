class YourClass:
    def __init__(self):
        self.enabled_channels_counter = {}  # Счетчики каналов
        self.last_received = {}  # Время последнего получения данных от канала
        self.max_inactivity_time = 5  # Максимальное время без активности (в секундах)
        self.max_retries = 3  # Максимальное количество повторных попыток

    def restartProcess(self, name):
        print(f"Перезапуск канала {name}...")

    def monitor_channels(self):
        current_time = time.time()  # Текущее время
        for band_name, last_time in list(self.last_received.items()):
            if current_time - last_time > self.max_inactivity_time:
                if band_name in self.enabled_channels_counter:
                    self.enabled_channels_counter[band_name] += 1
                    if self.enabled_channels_counter[band_name] > self.max_retries:
                        self.restartProcess(band_name)
                        self.enabled_channels_counter[band_name] = 0  # Сбросить счетчик после перезапуска
                else:
                    # Канал не был активен, но появляется в словаре
                    self.enabled_channels_counter[band_name] = 1

    def process_responses(self, responses):
        for response in responses:
            band_name = response.band_name
            self.last_received[band_name] = time.time()  # Обновить время последнего получения данных
            if band_name in self.enabled_channels_counter:
                self.enabled_channels_counter[band_name] = 0  # Сбросить счетчик, если данные от канала пришли
            else:
                self.enabled_channels_counter[band_name] = 0  # Канал появляется в словаре

    def main_loop(self, stub):
        while True:
            responses = stub.ProceedDataStream(API_pb2.ProceedDataStreamRequest(detected_img=self.show_img_status,
                                                                                clear_img=self.clear_img_status,
                                                                                spectrum=self.show_spectrum_status))

            self.process_responses(responses)  # Обработать пришедшие данные
            self.monitor_channels()  # Проверить каналы на неактивность

            time.sleep(1)  # Пауза для проверки каналов и задержки в обработке