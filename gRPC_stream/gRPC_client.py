import grpc
import neuro_pb2_grpc as API_pb2_grpc
import neuro_pb2 as API_pb2

gRPC_PORT = '50051'


def make_server_streaming_request(stub):
    request = API_pb2.VoidRequest()
    for r in stub.ProceedDataStream(request):
        print(f"Response - {r}")


def main():
    with grpc.insecure_channel(f'localhost:{gRPC_PORT}') as grpc_channel:
        stub = API_pb2_grpc.DataProcessingServiceStub(grpc_channel)
        responses = stub.ProceedDataStream(API_pb2.VoidRequest())
        for response in responses:
            print(f"Band: {response.band}")
            for uav in response.uavs:
                print(f"UAV Type: {uav.type}, State: {uav.state}")


if __name__ == '__main__':
    main()
