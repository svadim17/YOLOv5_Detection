import socket
import time
from multiprocessing import Process, Queue, Pipe, Lock
from collections import namedtuple

Task_ = namedtuple('Task_', ['channel', 'cmd', 'value'])


class FCM_Alinx(Process):
    """ FCM - frequency control module"""
    def __init__(self, address: tuple,
                       freq_codes: dict,
                       task_queue: Queue,
                       error_queue: Queue,
                       events: dict,
                       logger_):
        super().__init__()
        self.address = address
        self.freq_codes = freq_codes
        self.task_queue = task_queue
        self.events = events
        self.error_queue = error_queue
        self.logger = logger_

    def generate_command_to_send(self, command):
        pass

    def get_software_command(self):
        addr = b'\x01'
        id_command = b'\x61'
        param_command = b'\x00'
        checksum = (sum(addr + id_command + param_command) % 255).to_bytes(length=1, byteorder='big')
        return addr + id_command + param_command + checksum

    def get_loadDetect_command(self):
        addr = b'\x01'
        id_command = b'\x41'
        param_command = b'\x00'
        checksum = (sum(addr + id_command + param_command) % 255).to_bytes(length=1, byteorder='big')
        return addr + id_command + param_command + checksum

    def set_attenuator_2G4_coeff_command(self, value: int):
        addr = b'\x01'
        id_command = b'\x52'
        param_command = value.to_bytes(length=1, byteorder='big')
        checksum = (sum(addr + id_command + param_command) % 255).to_bytes(length=1, byteorder='big')
        return addr + id_command + param_command + checksum

    def set_attenuator_5G8_coeff_command(self, value: int):
        addr = b'\x01'
        id_command = b'\x51'
        param_command = value.to_bytes(length=1, byteorder='big')
        checksum = (sum(addr + id_command + param_command) % 255).to_bytes(length=1, byteorder='big')
        return addr + id_command + param_command + checksum

    def set_frequency_command(self, value: int):
        """ value = frequency  in Hz """
        addr = b'\x01'
        id_command = b'\x21'
        freq_code = int(self.freq_codes[value]).to_bytes(1, 'big')
        checksum = (sum(addr + id_command + freq_code) % 255).to_bytes(length=1, byteorder='big')
        return addr + id_command + freq_code + checksum

    def receive_response(self):
        try:
            response = self.sock.recv(4)
            self.logger.trace(f'Response from FCM: {" ".join([hex(i)[2:] for i in response])}')
            response_int = list(response)
            cmd_code = response_int[1]
            status = response_int[2]
            if status == 1:
                response_dict = {'status': False, 'msg': 'Load Detect set.'}
                self.logger.info('Load Detect set.')
            elif status == 0:
                if cmd_code == 65:
                    response_dict = {'status': False, 'msg': 'Load Detect not set.'}
                    self.logger.info('Load Detect not set.')
                elif cmd_code == 33:
                    response_dict = {'status': False, 'msg': 'Frequency successfully set!'}
                    self.logger.success('Frequency successfully set!')
                else:
                    response_dict = {'status': True, 'msg': f'Unknown status code: {status}'}
                    self.logger.warning(f'Unknown status code: {status}')
            elif status == 20:
                response_dict = {'status': False, 'msg': 'LockDetect went to 1 early'}
                self.logger.warning('Attenuation 2G4 successfully set!')
            elif status == 205:
                response_dict = {'status': False, 'msg': 'Attenuation 2G4 successfully set!'}
                self.logger.success('Attenuation 2G4 successfully set!')
            elif status == 13:
                response_dict = {'status': False, 'msg': 'Attenuation 2G4 setting error!'}
                self.logger.warning('Attenuation 2G4 setting error!')
            elif status == 252:
                response_dict = {'status': False, 'msg': 'Attenuation 5G8 successfully set!'}
                self.logger.success('Attenuation 5G8 successfully set!')
            elif status == 12:
                response_dict = {'status': False, 'msg': 'Attenuation 5G8 setting error!'}
                self.logger.warning('Attenuation 5G8 setting error!')
            elif status == 150:
                response_dict = {'status': False, 'msg': 'Unknown command!'}
                self.logger.warning('Unknown command!')
            elif status == 102:
                response_dict = {'status': True, 'msg': 'Transmission error!'}
                self.logger.warning('Transmission error!')
            elif status == 17:
                response_dict = {'status': False, 'msg': 'Frequency setting command is inaccurate'}
                self.logger.warning('Frequency setting command is inaccurate')
            else:
                msg = f'FCM: Unknown response status {status}!'
                response_dict = {'status': True, 'msg': msg}
                self.logger.warning(msg)

            if not response:
                response_dict = {'status': True, 'msg': 'No response form TCPControl server!'}
                self.logger.warning('No response form TCPControl server!')
            return response_dict

        except socket.timeout:
            response_dict = {'status': True, 'msg': 'Timeout while receiving response from FCM server'}
            self.logger.warning('Timeout while receiving response from TCPControl server')
            return response_dict

        except Exception as e:
            response_dict = {'status': True, 'msg': f'Error receiving from FCM server: {e}'}
            self.logger.error(f'Error receiving from FCM server: {e}')
            return response_dict

    def response_request(self, cmd_name: str, value):
        if cmd_name == 'set_frequency':
            cmd = self.set_frequency_command(value)
        elif cmd_name == 'get_software':
            cmd = self.get_software_command()
        elif cmd_name == 'get_loadDetect':
            cmd = self.get_loadDetect_command()
        elif cmd_name == 'set_attenuator_2G4_coeff':
            cmd = self.set_attenuator_2G4_coeff_command(value)
        elif cmd_name == 'set_attenuator_5G8_coeff':
            cmd = self.set_attenuator_5G8_coeff_command(value)
        elif cmd_name == 'set_gain_2G4':
            cmd = self.set_attenuator_2G4_coeff_command(63 - value*2)
        elif cmd_name == 'set_gain_5G8':
            cmd = self.set_attenuator_5G8_coeff_command(63 - value * 2)
        else:
            self.logger.error(f'Unknown FCM command: {cmd_name} {value}')
            return
        self.sock.send(cmd)
        gRPC_msg = self.receive_response()
        self.error_queue.put(gRPC_msg)

    def run(self):
        while True:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as self.sock:
                self.sock.settimeout(4)
                try:
                    self.sock.connect(self.address)
                    self.logger.success('FCM has been connected.')
                    self.error_queue.put({'status': False, 'msg': 'FCM has been connected.'})
#                    with self.logger.catch():
                    while True:
                        task_ = self.task_queue.get()  # Берем task_id из очереди
                        self.logger.info(f'FCM get task: {task_}')
                        if task_ is None:  # Признак завершения работы
                            break
                        self.response_request(task_.cmd, task_.value)
                        self.logger.success(f'FCM finish task: {task_}')
                        self.events[task_.channel].set()

                except socket.error as e:
                    self.logger.error(f'FCM connection error! {e}')
                    self.error_queue.put({'status': True, 'msg': f'FCM connection error! {e}'})
                    time.sleep(2)
                except Exception as e:
                    self.logger.error(f'FCM Unknown error! Type:{type(e)} Text:{e}')
                    self.error_queue.put({'status': True, 'msg': f'FCM Unknown error! {e}'})

        end_msg = 'Connection channel with FCM was closed.'
        self.logger.warning(end_msg)
        self.error_queue.put({'status': False, 'msg': end_msg})
