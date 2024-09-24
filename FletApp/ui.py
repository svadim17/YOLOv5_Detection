import flet as ft
import gRPC_interface
import time


class StartDialog(ft.UserControl):
    def __init__(self, grpc_channel, callback_func):
        super().__init__()
        self.grpc_channel = grpc_channel
        self.callback = callback_func

        self.available_channels = gRPC_interface.getAvailableChannelsRequest(grpc_channel=self.grpc_channel)
        self.checkboxes = {}
        self.enabled_channels = []

        self.dialog_start = ft.AlertDialog(modal=False, title=ft.Text('Start channel settings'))

        for name in self.available_channels:
            checkbox = ft.Checkbox(label=name, value=True)
            self.dialog_start.actions.append(checkbox)
            self.checkboxes[name] = checkbox

        self.btn_start_channel = ft.ElevatedButton(text='Start channel', on_click=self.btn_start)
        self.dialog_start.actions.append(self.btn_start_channel)

    async def btn_start(self, event):
        for checkbox in self.checkboxes.values():
            if checkbox.value:
                self.enabled_channels.append(checkbox.label)
                await gRPC_interface.startChannelRequest(channel=self.grpc_channel, channel_name=checkbox.label)
        self.close_dialog()
        await self.callback(self.enabled_channels)

    def close_dialog(self):
        self.dialog_start.open = False
        self.dialog_start.update()

    def update(self):
        self.dialog_start.open = True
        self.dialog_start.update()

    def build(self):
        """ Метод, который создает и возвращает интерфейс приложения """
        return self.dialog_start


class RecognitionContainer(ft.UserControl):
    def __init__(self, grpc_channel, image64_background, channel_name: str, map_list: list):
        super().__init__()
        self.grpc_channel = grpc_channel
        self.channel_name = channel_name
        self.map_list = map_list

        self.image = ft.Image(src_base64=image64_background, expand=True, fit=ft.ImageFit.CONTAIN)
        self.fps_view = ft.Text('FPS: 0', size=22)
        self.image_stack = ft.Stack([self.image, self.fps_view])

        self.label = ft.Text(value=self.channel_name, size=22)

        self.slider_zscale = ft.RangeSlider(min=-120, max=120, divisions=240, start_value=-40, end_value=40,
                                            label="{value}", width=500, on_change=self.slider_zscale_changed)

        self.drones_row = ft.Row()
        self.drons_dict_btns = {}
        self.drons_dict_texts = {}
        for dron in self.map_list:
            drone_btn_obj = ft.FilledButton(text=dron, disabled=True)
            drone_text_obj = ft.Text('None', size=17)
            self.drons_dict_btns[dron] = drone_btn_obj
            self.drons_dict_texts[dron] = drone_text_obj
            self.drones_row.controls.append(ft.Column(controls=[drone_btn_obj, drone_text_obj],
                                                      horizontal_alignment=ft.CrossAxisAlignment.CENTER))

        self.last_time = 0

    async def slider_zscale_changed(self, e):
        await gRPC_interface.changeZScaleRequest(grpc_channel=self.grpc_channel,
                                                 channel_name=self.channel_name,
                                                 z_min=int(self.slider_zscale.start_value),
                                                 z_max=int(self.slider_zscale.end_value))

    async def update_image(self, img_base64):
        self.image.src_base64 = img_base64
        await self.image.update_async()
        if self.fps_view:
            self.fps_view.value = f'FPS: {1 / (time.time() - self.last_time):.2f}'
            self.last_time = time.time()
            self.fps_view.update()

    async def update_buttons(self, name, state, freq):
        if state:
            self.drons_dict_btns[name].style.bgcolor = ft.colors.RED_500
            self.drons_dict_texts[name].value = str(freq)
        else:
            self.drons_dict_btns[name].style.bgcolor = ft.colors.GREY
            self.drons_dict_texts[name].value = 'None'
        self.drons_dict_btns[name].update()
        self.drons_dict_texts[name].update()


    def build(self):
        container = ft.Container(
            content=ft.Column(
                controls=[
                    self.label,
                    self.slider_zscale,
                    ft.Container(
                        content=self.image_stack,
                        margin=10,
                        padding=10,
                        alignment=ft.alignment.center,
                        bgcolor=ft.colors.BLUE_GREY_900,
                        border_radius=5,
                        expand=True,
                    ),
                    self.drones_row,
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,  # Центрирование всех элементов по горизонтали
            )
        )
        return container




