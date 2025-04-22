import grpc
from concurrent import futures
import neuro_pb2_grpc as API_pb2_grpc
import neuro_pb2 as API_pb2
import time
from multiprocessing import Process, Queue, Pipe, Event
from loguru import logger
import yaml
import custom_utils
import os
from tcp_control_alinx import FCM_Alinx, Task_
from client_process import Client
from monitoring_process import start_monitoring


logger.remove(0)
log_level = "TRACE"
log_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | {extra} | <yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>")
# logger.add(sys.stderr, format=log_format, colorize=True, backtrace=True, diagnose=True)
logger.add("server_logs/file_{time}.log",
           level=log_level,
           format=log_format,
           colorize=False,
           backtrace=True,
           diagnose=True,
           rotation='10 MB',
           # retention='12 days',
           retention=600,
           enqueue=True)
ORIGINAL_HASH_PASSWORD = b'\xac\x00\xeb\xf5+2\xd2\xa4\x90\x0e&\x84rz-O=b\xee\xf0:\xa0g\x01w\x8b\x9aD\x1e<\x94\xbb'


def load_conf(config_path: str):
    try:
        with open(config_path, encoding='utf-8') as f:
            config = dict(yaml.load(f, Loader=yaml.SafeLoader))
            logger.success(f'Config loaded successfully!')
            return config
    except Exception as e:
        logger.error(f'Error with loading config: {e}')


def dump_conf(config_path: str, config: dict):
    with open(config_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)


class DataProcessingService(API_pb2_grpc.DataProcessingServiceServicer):
    def __init__(self, config_path):
        self.custom_logger = logger.bind(process='gRPC')
        self.config_path = config_path
        self.config = load_conf(config_path=self.config_path)
        self.processes = {}
        self.data_queues = {}
        self.error_queues = {'gRPC': Queue(maxsize=40)}
        self.connections = self.config['connections']
        self.last_update_times = {}

        self.task_queue = None              # needed for control FCM
        # Feedback for FCM task done
        self.events = {'gRPC': Event()}
        for name in self.connections.keys():
            self.events[name] = Event()
        if self.config['freq_conversion_module']['is_used']:
            self.init_Alinx_FCM_control()

    def ServerErrorStream(self, request, context):
        self.custom_logger.debug(f'Start server error stream ')
        start_monitoring(self.error_queues['gRPC'])
        while True:
            for proc_name, q in self.error_queues.items():
                while not q.empty():
                    error = q.get()
                    #if error['status']:
                    yield API_pb2.ServerErrorResponse(status=error['status'], msg=error['msg'])
                else:
                    time.sleep(0.005)

    def ProceedDataStream(self, request, context):
        self.custom_logger.debug(f'Start data stream with detected img status = {request.detected_img} and '
                                 f'clear img status = {request.clear_img}')
        while True:
            for channel_name, queue in self.data_queues.items():
                if not queue.empty():
                    res = queue.get()
                    if res is not None:
                        band_name = res['name']
                        channel_freq = res['channel_freq']
                        Uavs = [API_pb2.UavObject(type=API_pb2.DroneType.Autel, state=res['results'][0],
                                                  freq=res['frequencies'][0]),
                                API_pb2.UavObject(type=API_pb2.DroneType.Fpv, state=res['results'][1],
                                                  freq=res['frequencies'][1]),
                                API_pb2.UavObject(type=API_pb2.DroneType.Dji, state=res['results'][2],
                                                  freq=res['frequencies'][2]),
                                API_pb2.UavObject(type=API_pb2.DroneType.Wifi, state=res['results'][3],
                                                  freq=res['frequencies'][3])
                                ]

                        # error_q = self.error_queues[channel_name].get()
                        response_data = {'band_name': band_name, 'uavs': Uavs, 'channel_central_freq': channel_freq}
                        if request.clear_img:
                            response_data['clear_img'] = res['clear_img'].tobytes()
                        if request.detected_img:
                            response_data['detected_img'] = res['detected_img'].tobytes()
                        if request.spectrum:
                            response_data['spectrum'] = res['spectrum'].tobytes()

                        yield API_pb2.DataResponse(**response_data)
                        self.custom_logger.trace(f'Data was send from band {band_name}')
                    self.last_update_times[channel_name] = time.time()
                if time.time() - self.last_update_times[channel_name] > 32:
                    yield API_pb2.ServerErrorResponse(status=True, msg=f'Big ping from {channel_name}. '
                                                                       f'Trying to restart process!')
                    self.restart_process(channel_name=channel_name)
                    for name in self.last_update_times.keys():
                        self.last_update_times[name] = time.time()

    def GetAvailableChannels(self, request, context):
        available_channels = tuple(self.connections.keys())
        channels = []
        for name, fields in self.connections.items():
            channels.append(API_pb2.ChannelObject(name=name,
                                                  hardware_type=fields['hardware']['type'],
                                                  central_freq=fields['hardware']['central_freq']))
        self.custom_logger.debug(f'Client connected {context.peer()}. '
                                 f'Sent avaliable channels: {available_channels} ')
        return API_pb2.ChannelsResponse(channels=available_channels, info=channels)

    def GetRecognitionSettings(self, request, context):
        chan_names = tuple(self.connections.keys())
        accum_size, threshold, exceedance = [], [], []
        for conn_name in chan_names:
            accum_size.append(self.connections[conn_name]['detection_settings']['accumulation_size'])
            threshold.append(self.connections[conn_name]['detection_settings']['threshold'])
            exceedance.append(self.connections[conn_name]['detection_settings']['exceedance'])
        return API_pb2.GetRecognitionSettingsResponse(band_name=chan_names,
                                                      accumulation_size=accum_size,
                                                      threshold=threshold,
                                                      exceedance=exceedance)

    def GetCurrentZScale(self, request, context):
        chan_names = tuple(self.connections.keys())
        z_min, z_max = [], []
        for conn_name in chan_names:
            z_min.append(self.connections[conn_name]['neural_network_settings']['z_min'])
            z_max.append(self.connections[conn_name]['neural_network_settings']['z_max'])
        return API_pb2.CurrentZScaleResponse(band_names=chan_names, z_min=z_min, z_max=z_max)

    def StartChannel(self, request, context):
        if request.connection_name in self.connections:
            if request.connection_name not in self.processes:
                conn_name = request.connection_name
                connection = self.connections[conn_name]
                self.last_update_times[conn_name] = time.time()
                hardware_type = str(connection['hardware']['type'])
                try:
                    # if self.alinxControlThread:
                    #     if hardware_type == 'alinx' or hardware_type == 'Alinx' or hardware_type == 'ALINX':
                    #         self.init_alinxTCPControl()

                    self.init_nn_client(conn_name=conn_name)
                    return API_pb2.StartChannelResponse(channelConnectionState=API_pb2.ConnectionState.Connected,
                                                        description=f'Connected successfully'
                                                                    f' to {str(connection["ip"])}:'
                                                                    f'{str(connection["port"])}')
                except Exception as e:
                    self.custom_logger.error(f'Connection error: {e}')
                    return API_pb2.StartChannelResponse(channelConnectionState=API_pb2.ConnectionState.Disconnected,
                                                        description=f'Error with connecting'
                                                                    f' to {str(connection["ip"])}:'
                                                                    f'{str(connection["port"])}')
            else:
                self.custom_logger.warning(f'Channel {request.connection_name} is already exists!')
                return API_pb2.StartChannelResponse(channelConnectionState=API_pb2.ConnectionState.Connected,
                                                    description=f'Channel {request.connection_name} is already exists!')
        else:
            self.custom_logger.warning(f'Unknown name {request.connection_name} !')
            return API_pb2.StartChannelResponse(channelConnectionState=API_pb2.ConnectionState.Disconnected,
                                                description=f'Unknown channel: {request.connection_name} !')
    @logger.catch
    def init_nn_client(self, conn_name: str):
        connection = self.connections[conn_name]
        data_queue = Queue(maxsize=5)
        error_queue = Queue(maxsize=40)

        cl = Client(name=conn_name,
                    address=(str(connection['ip']), int(connection['port'])),
                    hardware_type=str(connection['hardware']['type']),
                    receive_freq_status=bool(connection['hardware']['receive_freq']),
                    central_freq=int(connection['hardware']['central_freq'][0]),
                    freq_list=list(connection['hardware']['central_freq']),
                    autoscan=bool(connection['hardware']['autoscan']),
                    model_version=str(connection['neural_network_settings']['version']),
                    weights_path=connection['neural_network_settings']['weights_path'],
                    project_path=connection['neural_network_settings']['project_path'],
                    sample_rate=int(connection['signal_settings']['sample_rate']),
                    signal_width=int(connection['signal_settings']['signal_width']),
                    signal_height=int(connection['signal_settings']['signal_height']),
                    img_size=tuple(connection['neural_network_settings']['img_size']),
                    all_classes=tuple(connection['detection_settings']['all_classes']),
                    map_list=tuple(connection['detection_settings']['map_list']),
                    z_min=int(connection['neural_network_settings']['z_min']),
                    z_max=int(connection['neural_network_settings']['z_max']),
                    threshold=float(connection['detection_settings']['threshold']),
                    accumulation_size=int(connection['detection_settings']['accumulation_size']),
                    exceedance=float(connection['detection_settings']['exceedance']),
                    data_queue=data_queue,
                    error_queue=error_queue,
                    FCM_control_queue=self.task_queue,
                    task_done_event=self.events[conn_name],
                    logger_=self.custom_logger.bind(process=str(conn_name)))
        cl.start()
        self.custom_logger.debug(f'Client {conn_name} started!')
        self.processes[conn_name] = cl
        self.data_queues[conn_name] = data_queue
        self.error_queues[conn_name] = error_queue
        self.custom_logger.success(f'Connected successfully to {str(connection["ip"])}:{str(connection["port"])}')

    def init_Alinx_FCM_control(self):
        self.task_queue = Queue()  # Очередь для задач
        error_queue = Queue(maxsize=40)
        self.error_queues['FCM'] = error_queue


        self.alinxControlThread = FCM_Alinx(address=(self.config['freq_conversion_module']['ip'],
                                                     self.config['freq_conversion_module']['port']),
                                            freq_codes=self.config['freq_conversion_module']['freq_codes'],
                                            logger_=self.custom_logger.bind(process='FCM'),
                                            task_queue=self.task_queue,
                                            error_queue=error_queue,
                                            events=self.events)
        self.alinxControlThread.start()
        self.custom_logger.info('FCM module was started.')

    def ZScaleChanging(self, request, context):
        name = request.band_name
        if name in self.processes:
            self.processes[name].control_q.put({'func': 'change_zscale', 'args': (request.z_min, request.z_max)})
            # self.processes[name].change_zscale(z_min=request.z_min, z_max=request.z_max)
            self.custom_logger.debug(f'Z scale in channel {name} was changed on [{request.z_min}, {request.z_max}]')
            return API_pb2.ZScaleResponse(
                status=f'Z scale in channel {name} was changed on [{request.z_min}, {request.z_max}]')
        else:
            self.custom_logger.error(f'Unknown channel {name}')
            return API_pb2.ZScaleResponse(status=f'Unknown channel: {name}!')

    def LoadConfig(self, request, context):
        if request.password_hash == ORIGINAL_HASH_PASSWORD:
            try:
                new_config = dict(yaml.load(request.config, Loader=yaml.SafeLoader))
                dump_conf(config_path=self.config_path, config=new_config)
            except Exception as e:
                self.custom_logger.error(f'Config did not load! Error: {e}')
                return API_pb2.LoadConfigResponse(status=f'Config did not load! Error: {e}')
            self.custom_logger.success('Config loaded successfully!')
            return API_pb2.LoadConfigResponse(status='Config loaded successfully!')
        else:
            self.custom_logger.warning('Incorrect password!')
            return API_pb2.LoadConfigResponse(status='Incorrect password!')

    def SaveConfig(self, request, context):
        if request.password_hash == ORIGINAL_HASH_PASSWORD:
            try:
                for name, process in self.processes.items():
                    process.control_q.put({'func': 'get_current_settings', 'args': []})
                    try:
                        info = process.config_q.get()
                        self.custom_logger.debug(f'Data for saving config is: {info}')
                    except Exception as e:
                        self.custom_logger.error(f'Error with extracting data from queue for save config! \n {e}')
                    self.config['connections'] = custom_utils.deep_update(self.config['connections'], info)

                dump_conf(self.config_path, self.config)
                return API_pb2.SaveConfigResponse(status='Config saved successfully!')
            except Exception as e:
                self.custom_logger.error(f'Error with saving config! \n{e}')
                return API_pb2.SaveConfigResponse(status='Error with saving config!')

    def RecognitionSettings(self, request, context):
        name = request.band_name
        accum_size = request.accumulation_size
        threshold = request.threshold
        exceedance = request.exceedance
        if name in self.processes:
            self.processes[name].control_q.put({'func': 'change_recognition_settings',
                                                'args': (accum_size, threshold, exceedance)})
            self.custom_logger.debug(f'Accumulation was changed on {accum_size}, '
                                     f'Threshold was changed on {threshold}, '
                                     f'Exceedance was changed on {exceedance} in channel {name}')
            return API_pb2.RecognitionSettingsResponse(
                status=f'Accumulation was changed on {accum_size}, '
                       f'Threshold was changed on {threshold}, '
                       f'Exceedance was changed on {exceedance} in channel {name}')
        else:
            self.custom_logger.error(f'Unknown channel {name}')
            return API_pb2.RecognitionSettingsResponse(status=f'Unknown channel: {name}!')

    def RecordImages(self, request, context):
        name = request.band_name
        status = request.record_status
        if name in self.processes:
            self.processes[name].control_q.put({'func': 'change_record_images_status',
                                                'args': (status,)})
            self.custom_logger.debug(f'Record images status was changed on {status} in {name}')
            return API_pb2.RecordImagesResponse(
                numb_of_files=custom_utils.count_files(directory=self.processes[name].img_save_path))

        else:
            self.custom_logger.error(f'Unknown channel {name}')
            return API_pb2.RecognitionSettingsResponse(numb_of_files=-1)

    def OnOffAccumulation(self, request, context):
        accum_status = request.accum_status
        for name, process in self.processes.items():
            process.control_q.put({'func': 'change_accum_status', 'args': (accum_status,)})
        self.custom_logger.debug(f'Accumulation status was changed on {accum_status}')
        return API_pb2.OnOffAccumulationResponse(accum_status=f'Accumulation status was changed on {accum_status}')

    def GetProcessStatus(self, request, context):
        channel_name = request.channel_name
        process_status = self.processes[channel_name].is_alive()
        return API_pb2.GetProcessStatusResponse(status=process_status)

    def RestartProcess(self, request, context):
        channel_name = request.channel_name
        if channel_name in self.processes:
            self.restart_process(channel_name=channel_name)
            process_status = self.processes[channel_name].is_alive()
            return API_pb2.RestartProcessResponse(status=process_status)
        else:
            self.custom_logger.error(f'Unknown process {channel_name}!')
            return API_pb2.RestartProcessResponse(status=False)

    def restart_process(self, channel_name: str):
        try:
            self.processes[channel_name].terminate()
            del self.processes[channel_name]
            del self.data_queues[channel_name]
            self.custom_logger.info(f'Process {channel_name} was terminated.')
        except Exception as e:
            self.custom_logger.error(f'Error with terminate process {channel_name}\n{e}')
        try:
            self.init_nn_client(conn_name=channel_name)
            self.custom_logger.info(f'Process {channel_name} was started.')
        except Exception as e:
            self.custom_logger.error(f'Connection error: {e}')
            self.custom_logger.error(f'Error with start process {channel_name}\n{e}')

    def AlinxSetFrequency(self, request, context):
        freq = request.value
        for name, process in self.processes.items():
            process.control_q.put({'func': 'set_FCM_frequency', 'args': (freq,)})
        msg = f'Central frequency was changed on {freq}'
        self.custom_logger.debug(msg)
        return API_pb2.AlinxSetFrequencyResponse(status=f'Central frequency was changed on {freq}.')

    def AlinxSetAttenuation(self, request, context):
        channel_name = request.channel_name
        gain = request.value
        self.task_queue.put(Task_(channel='gRPC', cmd=f'set_gain_{channel_name}', value=gain))
        if self.events['gRPC'].wait(5):
            return API_pb2.AlinxSetAttenuationResponse(status=f'Send gain {gain} for {channel_name}.')
        else:
            return API_pb2.AlinxSetAttenuationResponse(status=f'Something went wrong')

    def AutoscanFrequency(self, request, context):
        for name, process in self.processes.items():
            process.control_q.put({'func': 'set_autoscan_state', 'args': (request.status,)})
        msg = f'Autoscan status was changed on {request.status}'
        self.custom_logger.debug(msg)
        return API_pb2.SetAutoscanFreqResponse(status=msg)

    def AlinxSoftVer(self, request, context):
        # # # тут должен быть запрос о версии ПО Alinx # # #
        return API_pb2.AlinxSoftVerResponse(version='We are working on it...')

    def AlinxLoadDetectState(self, request, context):
        # # # тут должен быть запрос о состоянии Load Detect Alinx # # #
        return API_pb2.AlinxLoadDetectResponse(state='We are working on it...')

    def USRPSetFrequency(self, request, context):
        try:
            channel_name = request.channel_name
            freq = request.value
            if channel_name in self.processes:
                self.processes[channel_name].control_q.put({'func': 'set_USRP_central_freq', 'args': (freq,)})
            return API_pb2.USRPSetFrequencyResponse(status=f'USRP central frequency was changed on {freq} in {channel_name}')
        except Exception as e:
            self.custom_logger.error(f'Error with setting USRP central freq! \n{e}')

    def NNInfo(self, request, context):
        chan_names = request.channels
        versions, names = [], []
        for conn_name in chan_names:
            versions.append(self.connections[conn_name]['neural_network_settings']['version'])
            model_name = os.path.basename(self.connections[conn_name]['neural_network_settings']['weights_path'])
            names.append(model_name)
        return API_pb2.NNInfoResponse(band_name=chan_names, versions=versions, names=names)

    def SignalSettings(self, request, context):
        chan_names = request.channels
        width, height, fs = [], [], []
        for conn_name in chan_names:
            width.append(self.connections[conn_name]['signal_settings']['signal_width'])
            height.append(self.connections[conn_name]['signal_settings']['signal_height'])
            fs.append(self.connections[conn_name]['signal_settings']['sample_rate'])

        return API_pb2.SignalSettingsResponse(band_name=chan_names, width=width, height=height, fs=fs)


def serve():
    gRPC_PORT = 51234
    script_path = os.path.dirname(__file__)
    CONFIG_PATH = os.path.join(script_path, 'server_conf.yaml')

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), )
    # interceptors=[ConnectionInterceptor()])  # Добавляем наш interceptor
    API_pb2_grpc.add_DataProcessingServiceServicer_to_server(DataProcessingService(config_path=CONFIG_PATH), server)
    server.add_insecure_port(f'[::]:{gRPC_PORT}')
    server.start()

    print(f"gRPC Server is running on port {gRPC_PORT}...")
    try:
        while True:
            time.sleep(86400)  # Удерживаем сервер в рабочем состоянии
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    serve()