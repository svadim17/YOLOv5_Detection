import time
import jetson_pb2_grpc as API_pb2_grpc
import jetson_pb2 as API_pb2
from loguru import logger
import grpc

IP = '192.168.1.241'
gRPC_PORT = 51235


class GetTemperatureClient():
    def __init__(self):
        self.gRPC_channel = None
        self.connect_to_server()

        while True:
            self.getTemperatureRequest()
            time.sleep(1)

    def connect_to_server(self):
        try:
            self.gRPC_channel = grpc.insecure_channel(target=f'{IP}:{gRPC_PORT}')
            logger.success(f'Successfully connected to {IP}:{gRPC_PORT}!')
        except (ConnectionError, Exception) as e:
            logger.error(f'Error with connecting to {IP}:{gRPC_PORT}! \n{e}')

    def getTemperatureRequest(self):
        try:
            stub = API_pb2_grpc.JetsonProcessingServiceStub(self.gRPC_channel)
            response = stub.JetsonTemperature(API_pb2.TemperatureRequest())
            logger.info(response.parameters)
        except Exception as e:
            logger.error(f'Error with getting response from StartChannelRequest! \n{e}')


if __name__ == '__main__':
    GetTemperatureClient()


