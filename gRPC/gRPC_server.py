import grpc
from concurrent import futures
import time
import define_API_pb2_grpc
import define_API_pb2
import gRPC_client


class DataProcessingService(define_API_pb2_grpc.DataProcessingServiceServicer):
    def __init__(self):
        self.data_store = {}

    def SendData(self, request, context):
        band = request.band
        results = request.results

        print(f"Received data from port {band}:")

        for res in results:
            print(f'{res.type} = {res.state}')

        print(results)

        print('-----------------------------------\n')

        # формируем ответ
        return define_API_pb2.DataResponse(message="Data received successfully.")


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    define_API_pb2_grpc.add_DataProcessingServiceServicer_to_server(DataProcessingService(), server)
    server.add_insecure_port(f'[::]:{gRPC_client.gRPC_PORT}')
    server.start()
    print(f"gRPC Server is running on port {gRPC_client.gRPC_PORT}...")
    try:
        while True:
            time.sleep(86400)  # Удерживаем сервер в рабочем состоянии
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    serve()
