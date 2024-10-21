import grpc
import neuro_pb2_grpc as API_pb2_grpc
import neuro_pb2 as API_pb2
import time
import numpy as np
import cv2
import base64
import utils
import asyncio
from loguru import logger


async def connect_to_server(ip: str, port: str):
    try:
        grpc_channel = grpc.insecure_channel(f'{ip}:{port}')
        logger.success(f'Successfully connected to {ip}:{port}!')
        return grpc_channel
    except Exception as e:
        logger.error(f'Error with connecting to {ip}:{port}! \n{e}')


async def startChannelRequest(channel, channel_name: str):
    try:
        stub = API_pb2_grpc.DataProcessingServiceStub(channel)
        response = stub.StartChannel(API_pb2.StartChannelRequest(connection_name=channel_name))
        logger.info(response.connection_status)
        return response
    except Exception as e:
        logger.error(f'Error with getting response from StartChannelRequest! \n{e}')


async def imageStream(channel, gallery: dict):
    stub = API_pb2_grpc.DataProcessingServiceStub(channel)
    img_responses = stub.SpectrogramImageStream(API_pb2.ImageRequest(band_name='Start'))

    for img_response in img_responses:
        band_name = img_response.band_name
        size = (img_response.height, img_response.width, 3)  # (640, 640, 3)
        img_base64 = utils.get_image_from_bytes(arr=img_response.data, size=size)
        if band_name in gallery:
            await gallery[band_name].update_image(img_base64=img_base64)
            await asyncio.sleep(0.005)


async def dataStream(channel, map_list: list, gallery: dict):
    stub = API_pb2_grpc.DataProcessingServiceStub(channel)
    responses = stub.ProceedDataStream(API_pb2.VoidRequest())
    for response in responses:
        band_name = response.band_name
        if band_name in gallery:
            for uav in response.uavs:
                drone_name = map_list[uav.type]
                drone_state = uav.state
                drone_freq = uav.freq
                await gallery[band_name].update_buttons(drone_name, drone_state, drone_freq)
            await asyncio.sleep(0.005)


async def start_gRPC_streams(grpc_channel, map_list: list, gallery: dict, image_stream_status: bool):
    # grpc_options = [('grpc.max_receive_message_length', 2 * 1024 * 1024)]  # 2 MB
    try:
        data_stream_task = asyncio.create_task(dataStream(channel=grpc_channel, map_list=map_list, gallery=gallery))
        logger.info('Data stream started successfully!')
    except Exception as e:
        logger.error(f'Error with starting data stream! \n{e}')

    if image_stream_status:
        try:
            image_stream_task = asyncio.create_task(imageStream(channel=grpc_channel, gallery=gallery))
            logger.info('Image stream started successfully!')
        except Exception as e:
            logger.error(f'Error with starting image stream! \n{e}')
        return image_stream_task, data_stream_task
    else:
        return data_stream_task


async def start_data_stream(grpc_channel, map_list: list, gallery: dict):
    try:
        data_stream_task = asyncio.create_task(dataStream(channel=grpc_channel, map_list=map_list, gallery=gallery))
        logger.info('Data stream started successfully!')
    except Exception as e:
        logger.error(f'Error with starting data stream! \n{e}')
    return data_stream_task


async def stop_gRPC_streams(grpc_task, grpc_channel):
    if grpc_task:
        grpc_task.cancel()  # Отмена задачи
        try:
            return grpc_task  # Дождитесь отмены задачи
        except asyncio.CancelledError:
            logger.info('gRPC stream task cancelled successfully.')


async def changeZScaleRequest(grpc_channel, channel_name: str, z_min: int, z_max: int):
    try:
        stub = API_pb2_grpc.DataProcessingServiceStub(grpc_channel)
        response = stub.ZScaleChanging(API_pb2.ZScaleRequest(band_name=channel_name, z_min=z_min, z_max=z_max))
        logger.info(response.status)
        return response
    except Exception as e:
        logger.error(f'Error with changing Z scale in channel {channel_name} \n{e}')


async def LoadConfigRequest(grpc_channel, password: str, config: str):
    try:
        stub = API_pb2_grpc.DataProcessingServiceStub(grpc_channel)
        response = stub.LoadConfig(API_pb2.LoadConfigRequest(config=config,
                                                             password_hash=utils.create_password_hash(password=password)))
        logger.info(response.status)
        return response
    except Exception as e:
        logger.error(f'Error with loading config! \n{e}')


async def SaveConfigRequest(grpc_channel, password: str):
    try:
        stub = API_pb2_grpc.DataProcessingServiceStub(grpc_channel)
        response = stub.SaveConfig(API_pb2.SaveConfigRequest(password_hash=utils.create_password_hash(password=password)))
        logger.info(response.status)
        return response
    except Exception as e:
        logger.error(f'Error with saving config! \n{e}')


async def RecognitionSettingsRequest(grpc_channel, channel_name: str, accum_size: int, threshold: float, exceedance: float):
    try:
        stub = API_pb2_grpc.DataProcessingServiceStub(grpc_channel)
        response = stub.RecognitionSettings(API_pb2.RecognitionSettingsRequest(band_name=channel_name,
                                                                               accumulation_size=accum_size,
                                                                               threshold=threshold,
                                                                               exceedance=exceedance))
        logger.info(response.status)
        return response
    except Exception as e:
        logger.error(f'Error with changing recognition settings in channel {channel_name} \n{e}')


def getAvailableChannelsRequest(grpc_channel):
    try:
        stub = API_pb2_grpc.DataProcessingServiceStub(grpc_channel)
        response = stub.GetAvailableChannels(API_pb2.ChannelsRequest())
        logger.info(f'Available channels: {response.channels}')
        return tuple(response.channels)
    except Exception as e:
        logger.error(f'Error with getting available channels! \n{e}')


def getCurrentZScaleRequest(grpc_channel):
    try:
        stub = API_pb2_grpc.DataProcessingServiceStub(grpc_channel)
        response = stub.GetCurrentZScale(API_pb2.CurrentZScaleRequest())
        chan_names = response.band_names
        z_min = response.z_min
        z_max = response.z_max

        # Convert lists to dictionary
        current_zscale_dict = {}
        for i in range(len(chan_names)):
            current_zscale_dict[chan_names[i]] = [z_min[i], z_max[i]]

        logger.info(f'Current ZScale: {current_zscale_dict}')
        return current_zscale_dict
    except Exception as e:
        logger.error(f'Error with getting current ZScale! \n{e}')


def getRecognitionSettings(grpc_channel):
    try:
        stub = API_pb2_grpc.DataProcessingServiceStub(grpc_channel)
        response = stub.GetRecognitionSettings(API_pb2.GetRecognitionSettingsRequest())
        chan_names = response.band_name
        accum_size = response.accumulation_size
        threshold = response.threshold
        exceedance = response.exceedance

        # Convert lists to dictionary
        current_recogn_settings_dict = {}
        for i in range(len(chan_names)):
            current_recogn_settings_dict[chan_names[i]] = [accum_size[i], threshold[i], exceedance[i]]
        logger.info(f'Current Recognition Settings: {current_recogn_settings_dict}')
        return current_recogn_settings_dict
    except Exception as e:
        logger.error(f'Error with getting current Recognition Settings! \n{e}')










