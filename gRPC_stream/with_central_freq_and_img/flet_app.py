import time
import flet as ft
import grpc
import neuro_pb2_grpc as API_pb2_grpc
import neuro_pb2 as API_pb2
import cv2
import numpy as np
import base64
import asyncio
import yaml
import hashlib


gRPC_PORT = '51234'
MAP_LIST = ['Autel', 'FPV', 'DJI', 'WiFi']


def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def create_password_hash(password: str):
    salt = 'KURVA_bober'.encode()
    dk = hashlib.pbkdf2_hmac(hash_name='sha256', password=password.encode(), salt=salt, iterations=100000)
    return dk


def main(page: ft.Page):
    global grpc_task, grpc_channel, last_time

    grpc_channel = grpc.insecure_channel(f'localhost:{gRPC_PORT}')


    img_base64 = image_to_base64("spectrum-analysis.png")
    fps_view = ft.Text('FPS: 0', size=22)
    grpc_counter_view = ft.Text('Packets: 0', size=18)
    text_2G4 = ft.Text('2G4', size=22)
    text_5G8 = ft.Text('5G8', size=22)

    # Создать элемент Image
    img0 = ft.Image(
        src_base64=img_base64,
        expand=True,
        fit=ft.ImageFit.CONTAIN  # Как изображение будет вписываться в заданные размеры
    )

    img0_stack = ft.Stack([img0, fps_view])

    # Создать элемент Image
    img1 = ft.Image(
        src_base64=img_base64,  # URL или путь к изображению
        expand=True,
        fit=ft.ImageFit.CONTAIN  # Как изображение будет вписываться в заданные размеры
    )

    drones_row_0 = ft.Row()
    drons_dict_btns_0 = {}
    drons_dict_texts_0 = {}
    for dron in MAP_LIST:
        drone_btn_obj = ft.FilledButton(text=dron, disabled=True)
        drone_text_obj = ft.Text('None', size=17)
        drons_dict_btns_0[dron] = drone_btn_obj
        drons_dict_texts_0[dron] = drone_text_obj

        drones_row_0.controls.append(ft.Column(controls=[drone_btn_obj, drone_text_obj],
                                               horizontal_alignment=ft.CrossAxisAlignment.CENTER))
        # drones_row_0.controls.append(drone_btn_obj)

    drones_row_1 = ft.Row()
    drons_dict_btns_1 = {}
    drons_dict_texts_1 = {}
    for dron in MAP_LIST:
        drone_btn_obj = ft.FilledButton(text=dron, disabled=True)
        drone_text_obj = ft.Text('None', size=17)
        drons_dict_btns_1[dron] = drone_btn_obj
        drons_dict_texts_1[dron] = drone_text_obj
        drones_row_1.controls.append(ft.Column(controls=[drone_btn_obj, drone_text_obj],
                                               horizontal_alignment=ft.CrossAxisAlignment.CENTER))
        # drones_row_1.controls.append(drone_btn_obj)

    async def open_dialog_start_channel_settings(e):
        dialog_start_channel_settings.open = True
        dialog_start_channel_settings.update()

    async def btn_start_channel_grpc_clicked(event):
        global grpc_task
        start_channel_task = asyncio.create_task(startChannelRequest(channel=grpc_channel,
                                                                     channel_name=channels_dropdown.value))
        await asyncio.gather(start_channel_task)

    async def btn_start_client_grpc_clicked(event):
        global grpc_task
        if btn_start_client_grpc.icon == ft.icons.PLAY_CIRCLE_FILL_ROUNDED:
            btn_start_client_grpc.icon = ft.icons.PAUSE_CIRCLE_FILLED_ROUNDED
            grpc_task = asyncio.create_task(start_gRPC_streams())  # Запуск gRPC в фоновом режиме
        elif btn_start_client_grpc.icon == ft.icons.PAUSE_CIRCLE_FILLED_ROUNDED:
            btn_start_client_grpc.icon = ft.icons.PLAY_CIRCLE_FILL_ROUNDED
            await stop_gRPC_streams()  # Отключение gRPC
        btn_start_client_grpc.update()

    async def startChannelRequest(channel, channel_name: str):
        stub = API_pb2_grpc.DataProcessingServiceStub(channel)
        if channel_name == 'All':
            response = stub.StartChannel(API_pb2.StartChannelRequest(connection_name='2G4'))
            print(response)
            response = stub.StartChannel(API_pb2.StartChannelRequest(connection_name='5G8'))
            print(response)

    async def imageStream(channel):
        global last_time

        stub = API_pb2_grpc.DataProcessingServiceStub(channel)
        img_responses = stub.SpectrogramImageStream(API_pb2.ImageRequest(band_name='Start'))
        last_time = time.time()
        img_counter_0 = 0
        img_counter_1 = 0
        img_counter = 0

        for img_response in img_responses:
            band_name = img_response.band_name
            size = (img_response.height, img_response.width, 3)  # (640, 640, 3)
            img_arr = np.frombuffer(img_response.data, dtype=np.uint8).reshape(size)
            color_image = cv2.applyColorMap(img_arr, cv2.COLORMAP_RAINBOW)

            # Convert image to base64
            _, buffer = cv2.imencode('.png', color_image)
            img_base64 = base64.b64encode(buffer).decode()
            grpc_counter_view.value = f'Packets: {img_counter}'
            grpc_counter_view.update()
            img_counter += 1
            if band_name == '2G4':
                if img_counter_0 == 1:
                    await update_image(img0, img_base64, fps_view)
                    img_counter_0 = 0
                else:
                    img_counter_0 += 1
            elif band_name == '5G8':
                if img_counter_1 == 1:
                    await update_image(img1, img_base64)
                    img_counter_1 = 0
                else:
                    img_counter_1 += 1
            else:
                print(f'Unknown band {band_name}')

            await asyncio.sleep(0.01)

    async def update_image(img, img_base64, fps_view=None):
        global last_time
        img.src_base64 = img_base64
        img.update()
        if fps_view:
            fps_view.value = f'FPS: {1 / (time.time() - last_time):.2f}'
            last_time = time.time()
            fps_view.update()

    async def dataStream(channel):
        stub = API_pb2_grpc.DataProcessingServiceStub(channel)
        responses = stub.ProceedDataStream(API_pb2.VoidRequest())

        for response in responses:
            band_name = response.band_name
            # print('------------------')
            # print(f"Band: {band_name}")
            for uav in response.uavs:
                drone_name = MAP_LIST[uav.type]
                # print(f"UAV Type: {drone_name}, State: {uav.state}, Frequency: {uav.freq}")
                await update_buttons(band_name, drone_name, uav.state, uav.freq)
            await asyncio.sleep(0.01)

    async def update_buttons(band_name, name, state, freq):
        if band_name == '2G4':
            if state:
                drons_dict_btns_0[name].style.bgcolor = ft.colors.RED_500
                drons_dict_texts_0[name].value = str(freq)
            else:
                drons_dict_btns_0[name].style.bgcolor = ft.colors.GREY
                drons_dict_texts_0[name].value = 'None'
            drons_dict_btns_0[name].update()
            drons_dict_texts_0[name].update()
        elif band_name == '5G8':
            if state:
                drons_dict_btns_1[name].style.bgcolor = ft.colors.RED_500
                drons_dict_texts_1[name].value = str(freq)
            else:
                drons_dict_btns_1[name].style.bgcolor = ft.colors.GREY
                drons_dict_texts_1[name].value = 'None'
            drons_dict_btns_1[name].update()
            drons_dict_texts_1[name].update()
        else:
            print(f'Unknown band: {band_name}')

    async def start_gRPC_streams():
        global grpc_channel
        # grpc_options = [('grpc.max_receive_message_length', 2 * 1024 * 1024)]  # 2 MB
        image_stream_task = asyncio.create_task(imageStream(channel=grpc_channel))
        data_stream_task = asyncio.create_task(dataStream(channel=grpc_channel))
        print('Start streams!!!!')
        await asyncio.gather(image_stream_task, data_stream_task)

    async def stop_gRPC_streams():
        global grpc_task, grpc_channel
        if grpc_task:
            grpc_task.cancel()  # Отмена задачи
            try:
                await grpc_task  # Дождитесь отмены задачи
            except asyncio.CancelledError:
                print("gRPC stream task cancelled successfully.")
            grpc_task = None

    def change_page_theme(event, theme):
        if theme == 'system':
            event.page.theme_mode = ft.ThemeMode.SYSTEM
        elif theme == 'dark':
            event.page.theme_mode = ft.ThemeMode.DARK
        elif theme == 'light':
            event.page.theme_mode = ft.ThemeMode.LIGHT
        event.page.update()

    async def slider_zscale0_changed(event):
        print(event.control)
        global grpc_channel
        change_zscale0_task = asyncio.create_task(changeZScaleRequest(channel=grpc_channel, channel_name='2G4'))
        await asyncio.gather(change_zscale0_task)

    async def slider_zscale1_changed(event):
        global grpc_channel
        change_zscale1_task = asyncio.create_task(changeZScaleRequest(channel=grpc_channel, channel_name='5G8'))
        await asyncio.gather(change_zscale1_task)

    async def changeZScaleRequest(channel, channel_name):
        global grpc_channel
        stub = API_pb2_grpc.DataProcessingServiceStub(channel)
        if channel_name == '2G4':
            response = stub.ZScaleChanging(API_pb2.ZScaleRequest(band_name=channel_name,
                                                                 z_min=int(slider_zscale0.start_value),
                                                                 z_max=int(slider_zscale0.end_value)))
        elif channel_name == '5G8':
            response = stub.ZScaleChanging(API_pb2.ZScaleRequest(band_name=channel_name,
                                                                 z_min=int(slider_zscale1.start_value),
                                                                 z_max=int(slider_zscale1.end_value)))
        print(response)

    async def LoadConfigRequest(config: str):
        global grpc_channel
        stub = API_pb2_grpc.DataProcessingServiceStub(grpc_channel)
        response = stub.LoadConfig(API_pb2.LoadConfigRequest(config=config,
                                                             password_hash=create_password_hash(password_field.value)))
        print(response)

    async def btn_get_available_channels_clicked(event):
        global grpc_channel
        get_available_channels_task = asyncio.create_task(getAvailableChannelsRequest(channel=grpc_channel))
        await asyncio.gather(get_available_channels_task)

    async def getAvailableChannelsRequest(channel):
        global grpc_channel
        stub = API_pb2_grpc.DataProcessingServiceStub(channel)
        response = stub.GetAvailableChannels(API_pb2.ChannelsRequest())
        print(f'Available channels: {response.channels}')

    async def authorization(e):
        dialog_authorization.open = True
        dialog_authorization.update()

    def open_file_picker(e):
        page.add(file_picker)
        file_picker.pick_files(allow_multiple=False)

    async def pick_files_result(e: ft.FilePickerResultEvent):
        global grpc_channel
        if e.files and len(e.files) > 0:
            selected_file = e.files[0].path     # Получаем путь к первому выбранному файлу
            print(selected_file)
            with open(selected_file, encoding='utf-8',) as f:
                config_str = f.read()
                # config_str = str(yaml.load(f, Loader=yaml.SafeLoader))
                print(f'Loaded file {selected_file}')
                config_task = asyncio.create_task(LoadConfigRequest(config=config_str))
                dialog_authorization.open = False
                dialog_authorization.update()
                await asyncio.gather(config_task)
        else:
            print('Error with loading config!')


    page.title = "NN Interface"     # Установить заголовок страницы
    page.padding = 0
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.horizontal_alignment = ft.MainAxisAlignment.CENTER

    password_field = ft.TextField(label='Password', icon=ft.icons.PASSWORD)
    dialog_authorization = ft.AlertDialog(modal=False,
                                                   title=ft.Text('Authorize to continue!'),
                                                   actions=[password_field, ft.ElevatedButton(text='Submit',
                                                                                              on_click=open_file_picker)])

    channels_dropdown = ft.Dropdown(label='Channel name',
                                    options=[ft.dropdown.Option('2G4'),
                                             ft.dropdown.Option('5G8'),
                                             ft.dropdown.Option('All')],
                                    value='2G4')
    btn_start_channel = ft.ElevatedButton(text='Start channel',
                                          on_click=btn_start_channel_grpc_clicked)
    dialog_start_channel_settings = ft.AlertDialog(modal=False,
                                                   title=ft.Text('Start channel settings'),
                                                   actions=[channels_dropdown, btn_start_channel])
    btn_start_channel_grpc = ft.ElevatedButton(text='Start Channels',
                                               on_click=open_dialog_start_channel_settings)
    btn_get_available_channels = ft.ElevatedButton(text='Get available channels',
                                                   on_click=btn_get_available_channels_clicked)

    slider_zscale0 = ft.RangeSlider(min=-150,
                                    max=150,
                                    divisions=300,
                                    start_value=-40,
                                    end_value=40,
                                    label="{value}",
                                    on_change=slider_zscale0_changed,
                                    width=500)

    slider_zscale1 = ft.RangeSlider(min=-150,
                                    max=150,
                                    divisions=300,
                                    start_value=-40,
                                    end_value=40,
                                    label="{value}",
                                    on_change=slider_zscale1_changed,
                                    width=500)

    page.appbar = ft.AppBar(title=btn_start_channel_grpc,
                            actions=[btn_get_available_channels],)

    btn_start_client_grpc = ft.IconButton(icon=ft.icons.PLAY_CIRCLE_FILL_ROUNDED,
                                          icon_color="blue700",
                                          icon_size=35,
                                          tooltip="Pause record",
                                          on_click=btn_start_client_grpc_clicked)

    container_img0 = ft.Container(
        content=ft.Column(
            controls=[
                text_2G4,
                slider_zscale0,
                ft.Container(
                    content=img0_stack,
                    margin=10,
                    padding=10,
                    alignment=ft.alignment.center,
                    bgcolor=ft.colors.BLUE_GREY_900,
                    border_radius=5,
                    expand=True,
                ),
                drones_row_0,
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,  # Центрирование всех элементов по горизонтали
        )
    )

    container_img1 = ft.Container(
        content=ft.Column(
            controls=[
                text_5G8,
                slider_zscale1,
                ft.Container(
                    content=img1,
                    margin=10,
                    padding=10,
                    alignment=ft.alignment.center,
                    bgcolor=ft.colors.BLUE_GREY_900,
                    border_radius=5,
                    expand=True,
                ),
                drones_row_1
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,  # Центрирование всех элементов по горизонтали
        )
    )

    main_container = ft.Row(
        controls=[
            container_img0,
            container_img1
        ],
        scroll='always'
    )
    page.add(main_container)

    btn_settings = ft.PopupMenuButton(
        items=[
            ft.PopupMenuItem(text='System Theme', on_click=lambda e: change_page_theme(e, theme='system')),
            ft.PopupMenuItem(text='Dark Theme', on_click=lambda e: change_page_theme(e, theme='dark')),
            ft.PopupMenuItem(text='Light Theme', on_click=lambda e: change_page_theme(e, theme='light'))
        ]
    )



    file_picker = ft.FilePicker(on_result=pick_files_result)
    # page.overlay.append(file_picker)  # Добавляем FilePicker в page.overlay

    page.bottom_appbar = ft.BottomAppBar(
        # bgcolor=ft.colors.GREY_700,
        shape=ft.NotchShape.CIRCULAR,
        padding=ft.Padding(left=10, top=0, right=10, bottom=0),  # Добавляем отступы
        content=ft.Row(
            controls=[
                btn_start_client_grpc,
                grpc_counter_view,
                ft.Container(expand=True),
                btn_settings,
                ft.ElevatedButton(
                    "Load Config",
                    icon=ft.icons.UPLOAD_FILE,
                    on_click=authorization,
                )
            ],
        ),
    )

    page.overlay.append(dialog_start_channel_settings)
    page.overlay.append(dialog_authorization)
    page.update()       # Обновить страницу, чтобы отобразить изменения
    page.theme_mode = ft.ThemeMode.SYSTEM


ft.app(target=main,
       port=54421,
      # view=ft.AppView.WEB_BROWSER,
       )     # Запустить приложение Flet

