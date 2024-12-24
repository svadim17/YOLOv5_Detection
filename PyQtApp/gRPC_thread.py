import time

import grpc
from grpc import StatusCode
import neuro_pb2_grpc as API_pb2_grpc
import neuro_pb2 as API_pb2
import custom_utils
from PyQt6 import QtCore
from PyQt6.QtCore import pyqtSignal


gRPC_channel_options = [
        ('grpc.keepalive_time_ms', 60000),      # Интервал между пингами - 60 секунд
        ('grpc.keepalive_timeout_ms', 20000),   # Таймаут ответа на пинг - 20 секунд
        ('grpc.keepalive_permit_without_calls', True),  # Разрешить пинги без активных вызовов
        ('grpc.http2.max_pings_without_data', 0),  # Без ограничения количества пингов без данных
        ('grpc.http2.min_time_between_pings_ms', 60000)  # Минимальный интервал между пингами - 60 секунд
    ]


def connect_to_gRPC_server(ip: str, port: str):
    return grpc.insecure_channel(target=f'{ip}:{port}', options=gRPC_channel_options)
    # try:
    #     gRPC_channel = grpc.insecure_channel(target=f'{ip}:{port}', options=gRPC_channel_options)
    #     logger_.success(f'Successfully connected to {ip}:{port}!')
    # except Exception as e:
    #     self.logger_.error(f'Error with connecting to {ip}:{port}! \n{e}')


class gRPCServerErrorThread(QtCore.QThread):
    signal_dataStream_response = pyqtSignal(dict)

    def __init__(self, channel, logger_):
        QtCore.QThread.__init__(self)
        self.logger_ = logger_
        self.gRPC_channel = channel
        self.max_gRPC_retries = 55555

    def run(self):
        stub = API_pb2_grpc.DataProcessingServiceStub(self.gRPC_channel)
        retry_count = 0

        while retry_count < self.max_gRPC_retries:
            if self.isInterruptionRequested():                  # stop thread
                self.logger_.info('gRPCThread is interrupted.')
                break
            try:
                error_responses = stub.ServerErrorStream(API_pb2.VoidRequest())
                for err in error_responses:
                    if self.isInterruptionRequested():
                        self.logger_.info('gRPCThread is interrupted.')
                        break
                    self.logger_.critical(f'SERVER ERROR: {err}')
                    self.msleep(5)
            except grpc.RpcError as rpc_error:
                if rpc_error.code() == grpc.StatusCode.UNAVAILABLE:
                    self.logger_.warning(f"Сервер недоступен. Попытка переподключения... {rpc_error}")
                    retry_count += 1
                    self.msleep(5)
                else:
                    self.logger_.error(rpc_error)
            else:
                break


class gRPCThread(QtCore.QThread):
    signal_dataStream_response = pyqtSignal(dict)
    signal_process_status = pyqtSignal(bool)

    def __init__(self, channel: int,
                 map_list: list,
                 detected_img_status: bool,
                 clear_img_status: bool,
                 spectrum_status: bool,
                 watchdog: bool,
                 logger_):
        QtCore.QThread.__init__(self)
        self.map_list = map_list
        self.gRPC_channel = channel
        self.max_gRPC_retries = 55555
        self.available_channels = None
        self.enabled_channels_counter = {}
        self.show_img_status = detected_img_status
        self.clear_img_status = clear_img_status
        self.show_spectrum_status = spectrum_status
        self.watchdog = watchdog
        self.logger_ = logger_

    def getAvailableChannelsRequest(self):
        try:
            stub = API_pb2_grpc.DataProcessingServiceStub(self.gRPC_channel)
            response = stub.GetAvailableChannels(API_pb2.ChannelsRequest())
            self.logger_.info(f'Available channels: {response.channels}')
            self.available_channels = list(response.channels)
            return tuple(response.channels)
        except Exception as e:
            self.logger_.error(f'Error with getting available channels! \n{e}')
            self.logger_.debug(f'Response = {response}')

    def startChannelRequest(self, channel_name: str):
        try:
            stub = API_pb2_grpc.DataProcessingServiceStub(self.gRPC_channel)
            response = stub.StartChannel(API_pb2.StartChannelRequest(connection_name=channel_name))
            self.logger_.info(response.connection_status)
            return response
        except Exception as e:
            self.logger_.error(f'Error with getting response from StartChannelRequest! \n{e}')

    def gerCurrentRecognitionSettings(self):
        try:
            stub = API_pb2_grpc.DataProcessingServiceStub(self.gRPC_channel)
            response = stub.GetRecognitionSettings(API_pb2.GetRecognitionSettingsRequest())
            chan_names = response.band_name
            accum_size = response.accumulation_size
            threshold = response.threshold
            exceedance = response.exceedance

            # Convert lists to dictionary
            current_recogn_settings_dict = {}
            for i in range(len(chan_names)):
                current_recogn_settings_dict[chan_names[i]] = {'accum_size': accum_size[i],
                                                               'threshold': threshold[i],
                                                               'exceedance': exceedance[i]}

            self.logger_.info(f'Current Recognition Settings: {current_recogn_settings_dict}')
            return current_recogn_settings_dict
        except Exception as e:
            self.logger_.error(f'Error with getting current Recognition Settings! \n{e}')

    def getCurrentZScaleRequest(self):
        try:
            stub = API_pb2_grpc.DataProcessingServiceStub(self.gRPC_channel)
            response = stub.GetCurrentZScale(API_pb2.CurrentZScaleRequest())
            chan_names = response.band_names
            z_min = response.z_min
            z_max = response.z_max

            # Convert lists to dictionary
            current_zscale_dict = {}
            for i in range(len(chan_names)):
                current_zscale_dict[chan_names[i]] = [z_min[i], z_max[i]]

            self.logger_.info(f'Current ZScale: {current_zscale_dict}')
            return current_zscale_dict
        except Exception as e:
            self.logger_.error(f'Error with getting current ZScale! \n{e}')

    def sendRecognitionSettings(self, channel_name: str, accum_size: int, threshold: float, exceedance: float):
        self.logger_.info('SEND RECOGN SETTINGS ', channel_name)
        try:
            stub = API_pb2_grpc.DataProcessingServiceStub(self.gRPC_channel)
            response = stub.RecognitionSettings(API_pb2.RecognitionSettingsRequest(band_name=channel_name,
                                                                                   accumulation_size=accum_size,
                                                                                   threshold=threshold,
                                                                                   exceedance=exceedance))
            self.logger_.info(response.status)
            return response
        except Exception as e:
            self.logger_.error(f'Error with changing recognition settings in channel {channel_name} \n{e}')

    def changeZScaleRequest(self, channel_name: str, z_min: int, z_max: int):
        try:
            stub = API_pb2_grpc.DataProcessingServiceStub(self.gRPC_channel)
            response = stub.ZScaleChanging(API_pb2.ZScaleRequest(band_name=channel_name, z_min=z_min, z_max=z_max))
            self.logger_.info(response.status)
            return response
        except Exception as e:
            self.logger_.error(f'Error with changing Z scale in channel {channel_name} \n{e}')

    def saveConfigRequest(self, password: str):
        try:
            stub = API_pb2_grpc.DataProcessingServiceStub(self.gRPC_channel)
            response = stub.SaveConfig(
                API_pb2.SaveConfigRequest(password_hash=custom_utils.create_password_hash(password=password)))
            self.logger_.info(response.status)
            return response
        except Exception as e:
            self.logger_.error(f'Error with saving config! \n{e}')

    def onOffAccumulationRequest(self, state: int):
        try:
            stub = API_pb2_grpc.DataProcessingServiceStub(self.gRPC_channel)
            response = stub.OnOffAccumulation(API_pb2.OnOffAccumulationRequest(accum_status=bool(state)))
            self.logger_.info(response.accum_status)
            return response
        except Exception as e:
            self.logger_.error(f'Error with changing accumulation status! \n{e}')

    def getProcessStatusRequest(self, name: str):
        try:
            stub = API_pb2_grpc.DataProcessingServiceStub(self.gRPC_channel)
            response = stub.GetProcessStatus(API_pb2.GetProcessStatusRequest(channel_name=name))
            self.signal_process_status.emit(response.status)
            self.logger_.info(f'Process "{name}" status is {response.status}.')
            return response
        except Exception as e:
            self.logger_.error(f'Error with getting process "{name}" status! \n{e}')

    def restartProcess(self, name: str):
        try:
            stub = API_pb2_grpc.DataProcessingServiceStub(self.gRPC_channel)
            response = stub.RestartProcess(API_pb2.RestartProcessRequest(channel_name=name))
            self.signal_process_status.emit(response.status)
            self.logger_.info(f'Restarting process "{name}" !')
            return response
        except Exception as e:
            self.logger_.error(f'Error with restarting process "{name}"! \n{e}')

    def init_enabled_channels(self, enabled_channels: list):
        for channel in enabled_channels:
            self.enabled_channels_counter[channel] = time.time()

    def check_channels_time(self):
        current_time = time.time()
        for key, value in self.enabled_channels_counter.items():
            if current_time - value > 30:
                self.logger_.warning(f'Data from the {key} has not arrived for {(current_time - value):.1f} seconds. '
                                     f'Restarting {key}...')
                self.restartProcess(name=key)
                self.enabled_channels_counter[key] = time.time()

    def change_watchdog_status(self, status: int):
        self.watchdog = bool(status)

    def run(self):
        stub = API_pb2_grpc.DataProcessingServiceStub(self.gRPC_channel)
        retry_count = 0

        while retry_count < self.max_gRPC_retries:
            if self.isInterruptionRequested():                  # stop thread
                self.logger_.info('gRPCThread is interrupted.')
                break
            try:
                # print('refresh channels time')
                # for channel in self.enabled_channels_counter.keys():
                #     self.enabled_channels_counter[channel] = time.time()
                responses = stub.ProceedDataStream(API_pb2.ProceedDataStreamRequest(detected_img=self.show_img_status,
                                                                                    clear_img=self.clear_img_status,
                                                                                    spectrum=self.show_spectrum_status))

                for response in responses:                  # Обработка каждой порции данных
                    # print('start receiveng responses')
                    if self.isInterruptionRequested():
                        self.logger_.info('gRPCThread is interrupted.')
                        break
                    band_name = response.band_name
                    if band_name in self.available_channels:
                        if self.watchdog:
                            if band_name in self.enabled_channels_counter:
                                self.enabled_channels_counter[band_name] = time.time()
                                self.check_channels_time()

                        response_dict = {'band_name': band_name}
                        drones_list = []
                        for uav in response.uavs:
                            drone_name = self.map_list[uav.type]
                            drone_state = uav.state
                            drone_freq = uav.freq
                            drones_list.append({'name': drone_name, 'state': drone_state, 'freq': drone_freq})
                        response_dict['drones'] = drones_list
                        if response.HasField('detected_img'):
                            response_dict['detected_img'] = response.detected_img
                        if response.HasField('clear_img'):
                            response_dict['clear_img'] = response.clear_img
                        if response.HasField('spectrum'):
                            response_dict['spectrum'] = response.spectrum

                        self.signal_dataStream_response.emit(response_dict)
                        self.msleep(5)

            except grpc.RpcError as rpc_error:
                if rpc_error.code() == grpc.StatusCode.UNAVAILABLE:
                    self.logger_.warning(f"Сервер недоступен. Попытка переподключения... {rpc_error}")
                    retry_count += 1
                    self.msleep(5)
                else:
                    self.logger_.error(rpc_error)
            else:
                break


