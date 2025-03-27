from PyQt6 import QtCore
import pygame

pygame.mixer.init()     # initialize pygame


class SoundThread(QtCore.QThread):

    def __init__(self, sound_name: str, logger_):
        QtCore.QThread.__init__(self)
        self.sound_path = 'assets/sounds/'
        self.sound_file = self.sound_path + sound_name
        self.logger = logger_
        try:
            pygame.mixer.music.load(self.sound_file)
        except Exception as e:
            self.logger.error(f'Error with loading sound {self.sound_file} \n {e}')

    def play_sound(self):
        try:
            pygame.mixer.music.play()
        except Exception as e:
            self.logger.error(f'Error with play {self.sound_file} \n{e}')

    def sound_file_changed(self, new_sound_name: str):
        self.sound_file = self.sound_path + new_sound_name
        pygame.mixer.music.load(self.sound_file)

    def start_stop_sound_thread(self, status: bool):
        if status:
            if not pygame.mixer.music.get_busy():
                self.play_sound()
        else:
            pygame.mixer.music.stop()

    def run(self):
        pass
