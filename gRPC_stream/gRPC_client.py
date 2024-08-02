import grpc
import NDD_API_pb2_grpc as API_pb2_grpc
import NDD_API_pb2 as API_pb2

gRPC_PORT = '50051'


def make_server_streaming_request(client):
    res = client.ProceedDataStream(API_pb2.VoidRequest())
    for r in res:
        print(f"Response - {r}")

def main():
    grpc_channel = grpc.insecure_channel(f'localhost:{gRPC_PORT}')
    client = API_pb2_grpc.DataProcessingService(grpc_channel)
    make_server_streaming_request(client)


if __name__ == '__main__':
    main()
