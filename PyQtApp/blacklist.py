self.box_type_of_sound = QGroupBox('Type of sound')
self.cb_sound = QComboBox()
all_sounds = self.get_all_sounds()
for sound_name in all_sounds:
    self.cb_sound.addItem(sound_name)
if self.input_sound_name in all_sounds:
    self.cb_sound.setCurrentText(self.input_sound_name)
else:
    self.logger.warning(f'Unknown sound {self.input_sound_name}. {self.cb_sound.currentText()} is set to default')