import grpc
from grpc import StatusCode
import neuro_pb2_grpc as API_pb2_grpc
import neuro_pb2 as API_pb2
import asyncio
from loguru import logger
import sys
import yaml
from PyQt6 import QtCore
from PyQt6.QtCore import pyqtSignal


logger.remove(0)
log_level = "TRACE"
log_format = ("<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> |"
              " <level>{level: <8}</level> |"
              " {extra} |"
              " <yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>")
logger.add(sys.stderr, format=log_format, colorize=True, backtrace=True, diagnose=True, enqueue=True)

gRPC_channel_options = [
        ('grpc.keepalive_time_ms', 60000),      # Интервал между пингами - 60 секунд
        ('grpc.keepalive_timeout_ms', 20000),   # Таймаут ответа на пинг - 20 секунд
        ('grpc.keepalive_permit_without_calls', True),  # Разрешить пинги без активных вызовов
        ('grpc.http2.max_pings_without_data', 0),  # Без ограничения количества пингов без данных
        ('grpc.http2.min_time_between_pings_ms', 60000)  # Минимальный интервал между пингами - 60 секунд
    ]


class gRPCThread(QtCore.QThread):
    signal_dataStream_response = pyqtSignal(dict)

    def __init__(self, map_list, img_status: bool):
        QtCore.QThread.__init__(self)
        self.map_list = map_list
        self.gRPC_channel = None
        self.max_gRPC_retries = 55555
        self.available_channels = None
        self.show_img_status = img_status

    def connect_to_gRPC_server(self, ip: str, port: str):
        try:
            self.gRPC_channel = grpc.insecure_channel(target=f'{ip}:{port}', options=gRPC_channel_options)
            logger.success(f'Successfully connected to {ip}:{port}!')
        except Exception as e:
            logger.error(f'Error with connecting to {ip}:{port}! \n{e}')

    def getAvailableChannelsRequest(self):
        try:
            stub = API_pb2_grpc.DataProcessingServiceStub(self.gRPC_channel)
            response = stub.GetAvailableChannels(API_pb2.ChannelsRequest())
            logger.info(f'Available channels: {response.channels}')
            self.available_channels = list(response.channels)
            return tuple(response.channels)
        except Exception as e:
            logger.error(f'Error with getting available channels! \n{e}')
            logger.debug(f'Response = {response}')

    def startChannelRequest(self, channel_name: str):
        try:
            stub = API_pb2_grpc.DataProcessingServiceStub(self.gRPC_channel)
            response = stub.StartChannel(API_pb2.StartChannelRequest(connection_name=channel_name))
            logger.info(response.connection_status)
            return response
        except Exception as e:
            logger.error(f'Error with getting response from StartChannelRequest! \n{e}')

    def run(self):
        stub = API_pb2_grpc.DataProcessingServiceStub(self.gRPC_channel)
        retry_count = 0
        while retry_count < self.max_gRPC_retries:
            if self.isInterruptionRequested():                  # stop thread
                logger.info('gRPCThread is interrupted.')
                break
            try:
                responses = stub.ProceedDataStream(API_pb2.ProceedDataStreamRequest(img=self.show_img_status))

                for response in responses:                  # Обработка каждой порции данных
                    if self.isInterruptionRequested():
                        logger.info('gRPCThread is interrupted.')
                        break
                    band_name = response.band_name
                    if band_name in self.available_channels:
                        response_dict = {'band_name': band_name}
                        drones_list = []
                        for uav in response.uavs:
                            drone_name = self.map_list[uav.type]
                            drone_state = uav.state
                            drone_freq = uav.freq
                            drones_list.append({'name': drone_name, 'state': drone_state, 'freq': drone_freq})
                        response_dict['drones'] = drones_list
                        if response.HasField('img'):
                            response_dict['img'] = response.img
                        self.signal_dataStream_response.emit(response_dict)
                        self.msleep(5)

            except grpc.RpcError as rpc_error:
                if rpc_error.code() == grpc.StatusCode.UNAVAILABLE:
                    logger.warning(f"Сервер недоступен. Попытка переподключения... {rpc_error}")
                    retry_count += 1
                    self.msleep(5)
                else:
                    logger.error(rpc_error)
            else:
                break






































