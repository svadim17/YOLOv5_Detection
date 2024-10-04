import flet as ft
import gRPC_interface
import time


class StartDialog(ft.UserControl):
    def __init__(self, grpc_channel, callback_func):
        super().__init__()
        self.grpc_channel = grpc_channel
        self.callback = callback_func

        self.available_channels = gRPC_interface.getAvailableChannelsRequest(grpc_channel=self.grpc_channel)
        self.current_zscale_dict = gRPC_interface.getCurrentZScaleRequest(grpc_channel=self.grpc_channel)
        self.checkboxes = {}
        self.enabled_channels = []

        self.dialog_start = ft.AlertDialog(modal=False, title=ft.Text('Available channels'))

        for name in self.available_channels:
            checkbox = ft.Checkbox(label=name, value=True)
            self.dialog_start.actions.append(checkbox)
            self.checkboxes[name] = checkbox

        self.btn_start_channel = ft.ElevatedButton(text='Start channel', on_click=self.btn_start)
        self.dialog_start.actions.append(self.btn_start_channel)

    async def btn_start(self, event):
        # self.dialog_start.clean()
        # # self.dialog_start.actions.clear()
        # self.dialog_start.update()
        # self.dialog_start.actions.append(ft.Column([ft.ProgressRing(), ft.Text('Connecting to channels...')],
        #                                            horizontal_alignment=ft.CrossAxisAlignment.CENTER))
        # self.dialog_start.update()

        for checkbox in self.checkboxes.values():
            if checkbox.value:
                self.enabled_channels.append(checkbox.label)
                await gRPC_interface.startChannelRequest(channel=self.grpc_channel, channel_name=checkbox.label)
        self.close_dialog()
        await self.callback(self.enabled_channels, self.current_zscale_dict)

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
    def __init__(self, grpc_channel, image64_background, channel_name: str, map_list: list, z_min: int, z_max: int):
        super().__init__()
        self.grpc_channel = grpc_channel
        self.channel_name = channel_name
        self.map_list = map_list
        self.z_min = z_min
        self.z_max = z_max
        print(f'Starting recognition container {channel_name}')
        self.image = ft.Image(src_base64=image64_background, expand=True, fit=ft.ImageFit.CONTAIN)
        self.fps_view = ft.Text('FPS: 0', size=22)
        self.image_stack = ft.Stack([self.image, self.fps_view])

        self.label = ft.Text(value=self.channel_name, size=22)

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

        self.settings_container = self.create_settings_controls()

    def create_settings_controls(self):
        self.label_settings = ft.Text(f'{self.channel_name}', theme_style=ft.TextThemeStyle.TITLE_MEDIUM)

        self.label_slider_zscale = ft.Text('Z Scale range', theme_style=ft.TextThemeStyle.TITLE_SMALL)
        self.slider_zscale = ft.RangeSlider(min=-120,
                                            max=120,
                                            divisions=240,
                                            start_value=self.z_min,
                                            end_value=self.z_max,
                                            label="{value}",
                                            on_change=self.slider_zscale_changed,
                                            expand=True)
        self.label_value_slider_zscale = ft.Text(f'[{self.slider_zscale.start_value}, {self.slider_zscale.end_value}]')
        column_zscale = ft.Column(controls=[self.label_slider_zscale,
                                            ft.Row(controls=[self.slider_zscale, self.label_value_slider_zscale])],
                                  spacing=0)

        self.label_slider_accumulation = ft.Text('Accumulation size', theme_style=ft.TextThemeStyle.TITLE_SMALL)
        self.slider_accumulation = ft.Slider(min=1,
                                             max=50,
                                             divisions=49,
                                             value=25,
                                             label='{value}',
                                             on_change=self.slider_accumulation_changed,
                                             expand=True)
        self.label_value_slider_accumulation = ft.Text(f'{int(self.slider_accumulation.value)}')
        column_accumulation = ft.Column(controls=[self.label_slider_accumulation,
                                        ft.Row(controls=[self.slider_accumulation, self.label_value_slider_accumulation])],
                                        spacing=0)

        self.label_slider_threshold = ft.Text('Threshold', theme_style=ft.TextThemeStyle.TITLE_SMALL)
        self.slider_threshold = ft.Slider(min=0.05,
                                          max=0.95,
                                          divisions=18,
                                          value=0.6,
                                          on_change=self.slider_threshold_changed,
                                          expand=True)
        self.label_value_slider_threshold = ft.Text(f'{self.slider_threshold.value:.2f}')
        column_threshold = ft.Column(controls=[self.label_slider_threshold,
                                     ft.Row(controls=[self.slider_threshold, self.label_value_slider_threshold])],
                                     spacing=0)

        self.label_slider_exceedance = ft.Text('Exceedance', theme_style=ft.TextThemeStyle.TITLE_SMALL)
        self.slider_exceedance = ft.Slider(min=0.05,
                                           max=0.95,
                                           divisions=18,
                                           value=0.6,
                                           on_change=self.slider_exceedance_changed,
                                           expand=True)
        self.label_value_slider_exceedance = ft.Text(f'{self.slider_exceedance.value:.2f}')
        column_exceedance = ft.Column(controls=[self.label_slider_exceedance,
                                      ft.Row(controls=[self.slider_exceedance, self.label_value_slider_exceedance])],
                                      spacing=0)

        return ft.Container(
            content=ft.Column(
                controls=[self.label_settings, column_zscale, column_accumulation, column_threshold, column_exceedance],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            padding=ft.Padding(left=10, top=30, right=10, bottom=10),
            expand=True
        )

    async def slider_zscale_changed(self, e):
        await gRPC_interface.changeZScaleRequest(grpc_channel=self.grpc_channel,
                                                 channel_name=self.channel_name,
                                                 z_min=int(self.slider_zscale.start_value),
                                                 z_max=int(self.slider_zscale.end_value))
        self.label_value_slider_zscale.value = f'[{int(self.slider_zscale.start_value)}, {int(self.slider_zscale.end_value)}]'
        self.label_value_slider_zscale.update()

    async def slider_accumulation_changed(self, e):
        self.label_value_slider_accumulation.value = f'{int(self.slider_accumulation.value)}'
        self.label_value_slider_accumulation.update()
        await self.recognitions_settings_changed()

    async def slider_threshold_changed(self, e):
        self.label_value_slider_threshold.value = f'{self.slider_threshold.value:.2f}'
        self.label_value_slider_threshold.update()
        await self.recognitions_settings_changed()

    async def slider_exceedance_changed(self, e):
        self.label_value_slider_exceedance.value = f'{self.slider_exceedance.value:.2f}'
        self.label_value_slider_exceedance.update()
        await self.recognitions_settings_changed()

    async def recognitions_settings_changed(self):
        threshold = self.slider_accumulation.value * self.slider_threshold.value * self.slider_exceedance.value
        await gRPC_interface.RecognitionSettingsRequest(grpc_channel=self.grpc_channel,
                                                        channel_name=self.channel_name,
                                                        accum_size=int(self.slider_accumulation.value),
                                                        threshold=threshold)

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


class BottomBarContent(ft.UserControl):
    def __init__(self, parent_page):
        super().__init__()
        self.page = parent_page

        self.page.theme_mode = ft.ThemeMode.DARK
        self.menu_button = ft.IconButton(icon=ft.icons.MENU_ROUNDED, icon_size=25)
        self.theme_button = ft.IconButton(icon=ft.icons.DARK_MODE_OUTLINED, icon_size=25, on_click=self.change_theme)

    def change_theme(self, e):
        if self.page.theme_mode == ft.ThemeMode.LIGHT:
            self.theme_button.icon = ft.icons.DARK_MODE_OUTLINED
            self.page.theme_mode = ft.ThemeMode.DARK
        else:
            self.theme_button.icon = ft.icons.WB_SUNNY_OUTLINED
            self.page.theme_mode = ft.ThemeMode.LIGHT
        self.page.update()
        self.theme_button.update()

    def build(self):
        return ft.Row(controls=[
            self.menu_button,
            ft.Container(expand=True),
            self.theme_button
        ])


class MenuContent():
    def __init__(self, parent_page, grpc_channel):
        super().__init__()
        self.page = parent_page
        self.grpc_channel = grpc_channel
        self.btn_save_config = ft.TextButton(text='Save config', on_click=self.save_config)
        column = ft.Column(controls=[ft.Text('Channels Settings', theme_style=ft.TextThemeStyle.TITLE_LARGE),],
                           alignment=ft.MainAxisAlignment.CENTER,
                           horizontal_alignment=ft.CrossAxisAlignment.CENTER
                           )

        self.menu = ft.NavigationDrawer(controls=[column])

    def add_channel_settings(self, container):
        self.menu.controls.append(container)

    def add_btn_save_config(self):
        self.menu.controls.append(self.btn_save_config)

    async def save_config(self, event):
        await gRPC_interface.SaveConfigRequest(grpc_channel=self.grpc_channel, password='kgbradar')

    def open_menu(self, e):
        self.menu.open = True
        self.page.update()



