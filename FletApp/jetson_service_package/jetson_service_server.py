import jetson_pb2_grpc as API_pb2_grpc
import jetson_pb2 as API_pb2
from jetson_get_params import JetsonParams
import grpc
from concurrent import futures
import time
from loguru import logger
import json


class JetsonProcessingServiceServicer(API_pb2_grpc.JetsonProcessingServiceServicer):
    def __init__(self):
        self.jetson = JetsonParams()

    def JetsonTemperature(self, request, context):
        logger.info(f'Client requested temperature {context.peer()}')

        temperature_json = json.dumps(self.jetson.getTemp())
        return API_pb2.TemperatureResponse(parameters=temperature_json)


def serve():
    gRPC_PORT = 51235

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    API_pb2_grpc.add_JetsonProcessingServiceServicer_to_server(JetsonProcessingServiceServicer(), server)
    server.add_insecure_port(f'[::]:{gRPC_PORT}')
    server.start()

    logger.success(f"gRPC Server is running on port {gRPC_PORT}...")
    try:
        while True:
            time.sleep(86400)  # Удерживаем сервер в рабочем состоянии
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    serve()
