import socket
import time
from multiprocessing import Process, Queue, Pipe, Lock


class TCPControlAlinx(Process):
    def __init__(self, address: tuple, freq_codes: dict, response_queue, control_queue, logger_):
        super().__init__()
        self.address = address
        self.freq_codes = freq_codes
        self.response_queue = response_queue
        self.control_queue = control_queue
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
        freq_code = bytes.fromhex(self.freq_codes[str(value)])
        checksum = (sum(addr + id_command + freq_code) % 255).to_bytes(length=1, byteorder='big')
        return addr + id_command + freq_code + checksum

    def receive_response(self, sock):
        try:
            response = sock.recv(4)
            self.logger.debug(f'Response from TCPAlinxThread: {response.hex()}')
            response_int = list(response)
            status = response_int[2]
            match status:
                case 1:
                    response_dict = {'status': True, 'answer': 'Load Detect set.'}
                    self.logger.info('Load Detect set.')
                case 0:
                    response_dict = {'status': True, 'answer': 'Load Detect not set.'}
                    self.logger.info('Load Detect not set.')
                case 205:
                    response_dict = {'status': True, 'answer': 'Attenuation 2G4 successfully set!'}
                    self.logger.success('Attenuation 2G4 successfully set!')
                case 13:
                    response_dict = {'status': True, 'answer': 'Attenuation 2G4 setting error!'}
                    self.logger.warning('Attenuation 2G4 setting error!')
                case 252:
                    response_dict = {'status': True, 'answer': 'Attenuation 5G8 successfully set!'}
                    self.logger.success('Attenuation 5G8 successfully set!')
                case 12:
                    response_dict = {'status': True, 'answer': 'Attenuation 5G8 setting error!'}
                    self.logger.warning('Attenuation 5G8 setting error!')
                case 150:
                    response_dict = {'status': True, 'answer': 'Unknown command!'}
                    self.logger.warning('Unknown command!')
                case 102:
                    response_dict = {'status': True, 'answer': 'Transmission error!'}
                    self.logger.warning('Transmission error!')
                case 0:
                    response_dict = {'status': True, 'answer': 'Frequency successfully set!'}
                    self.logger.success('Frequency successfully set!')
                case 17:
                    response_dict = {'status': True, 'answer': 'Frequency setting command is inaccurate'}
                    self.logger.warning('Frequency setting command is inaccurate')
                case _:
                    response_dict = {'status': True, 'answer': 'Unknown response status!'}
                    self.logger.warning('Unknown response status!')

            if not response:
                response_dict = {'status': False, 'answer': 'No response form TCPControl server!'}
                self.logger.warning('No response form TCPControl server!')
            return response_dict

        except sock.timeout:
            response_dict = {'status': False, 'answer': 'Timeout while receiving response from TCPControl server'}
            self.logger.warning('Timeout while receiving response from TCPControl server')
            return response_dict

        except Exception as e:
            response_dict = {'status': False, 'answer': f'Error receiving from TCPControl server: {e}'}
            self.logger.error(f'Error receiving from TCPControl server: {e}')
            return response_dict

    def run(self):
        while True:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                try:
                    sock.settimeout(10)
                    sock.connect(self.address)
                    self.logger.success(f'Connected to TCPControl server {self.address}!')

                    while True:
                        if self.control_queue:
                            command = self.control_queue.get()

                            if command is None:
                                self.control_queue.task_done()
                                break

                            command_to_send = self.generate_command_to_send(command=command)
                            sock.send(command_to_send)

                            response_dict = self.receive_response(sock=sock)

                            if self.response_queue:
                                self.response_queue.put(response_dict)

                            self.control_queue.task_done()

                except socket.error as e:
                    sock.close()
                    self.logger.error(f'Connection error! {e}')
                    time.sleep(2)
                except Exception as e:
                    sock.close()
                    self.logger.error(f'Unknown error! {e}')
                    break

        self.logger.debug(f'TCPControlAlinx ({self.address}) finished work!')
