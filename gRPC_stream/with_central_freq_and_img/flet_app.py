import time

import flet as ft
import grpc
import neuro_pb2_grpc as API_pb2_grpc
import neuro_pb2 as API_pb2
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import base64
import asyncio


gRPC_PORT = '50051'
MAP_LIST = ['Autel', 'FPV', 'DJI', 'WiFi']


def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def main(page: ft.Page):
    global grpc_task, grpc_channel, last_time

    img_base64 = image_to_base64("spectrum-analysis.png")
    fps_view = ft.Text("FPS: 0", size=22)
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
    drons_dict_0 = {}
    for dron in MAP_LIST:
        drone_text_obj = ft.FilledButton(text=dron, disabled=True)
        drons_dict_0[dron] = drone_text_obj
        drones_row_0.controls.append(drone_text_obj)

    drones_row_1 = ft.Row()
    drons_dict_1 = {}
    for dron in MAP_LIST:
        drone_text_obj = ft.FilledButton(text=dron, disabled=True)
        drons_dict_1[dron] = drone_text_obj
        drones_row_1.controls.append(drone_text_obj)

    async def btn_start_grpc_clicked(event):
        global grpc_task
        if btn_start_grpc.icon == ft.icons.PLAY_CIRCLE_FILL_ROUNDED:
            btn_start_grpc.icon = ft.icons.PAUSE_CIRCLE_FILLED_ROUNDED
            grpc_task = asyncio.create_task(connect_to_gRPC())  # Запуск gRPC в фоновом режиме
        elif btn_start_grpc.icon == ft.icons.PAUSE_CIRCLE_FILLED_ROUNDED:
            btn_start_grpc.icon = ft.icons.PLAY_CIRCLE_FILL_ROUNDED
            await disconnect_from_gRPC()  # Отключение gRPC
        btn_start_grpc.update()

    async def imageStream(channel):
        global last_time

        stub = API_pb2_grpc.DataProcessingServiceStub(channel)
        img_responses = stub.SpectrogramImageStream(API_pb2.ImageRequest())
        last_time = time.time()
        for img_response in img_responses:
            band = img_response.band
            size = (img_response.height, img_response.width, 3)  # (640, 640, 3)
            img_arr = np.frombuffer(img_response.data, dtype=np.uint8).reshape(size)
            color_image = cv2.applyColorMap(img_arr, cv2.COLORMAP_RAINBOW)

            # Convert image to base64
            _, buffer = cv2.imencode('.png', color_image)
            img_base64 = base64.b64encode(buffer).decode()

            if band == 0:
                await update_image(img0, img_base64, fps_view)
            elif band == 1:
                await update_image(img1, img_base64)
            else:
                print(f'Unknown band {band}')
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
            band = response.band
            print('------------------')
            print(f"Band: {band}")
            for uav in response.uavs:
                drone_name = MAP_LIST[uav.type]
                print(f"UAV Type: {drone_name}, State: {uav.state}, Frequency: {uav.freq}")
                await update_buttons(band, drone_name, uav.state, uav.freq)
            await asyncio.sleep(0.01)

    async def update_buttons(band, name, state, freq):
        if band == 0:
            if state:
                drons_dict_0[name].style.bgcolor = ft.colors.RED_500
            else:
                drons_dict_0[name].style.bgcolor = ft.colors.GREY
            drons_dict_0[name].update()
        elif band == 1:
            if state:
                drons_dict_1[name].style.bgcolor = ft.colors.RED_500
            else:
                drons_dict_1[name].style.bgcolor = ft.colors.GREY
            drons_dict_1[name].update()

    async def connect_to_gRPC():
        global grpc_channel
        grpc_channel = grpc.insecure_channel(f'localhost:{gRPC_PORT}')

        image_stream_task = asyncio.create_task(imageStream(channel=grpc_channel))
        data_stream_task = asyncio.create_task(dataStream(channel=grpc_channel))
        await asyncio.gather(image_stream_task, data_stream_task)

    async def disconnect_from_gRPC():
        global grpc_task, grpc_channel
        if grpc_task:
            grpc_task.cancel()  # Отмена задачи
            try:
                await grpc_task  # Дождитесь отмены задачи
            except asyncio.CancelledError:
                print("gRPC stream task cancelled successfully.")
            grpc_task = None

        # Закрываем канал gRPC
        if grpc_channel:
            grpc_channel.close()
            grpc_channel = None

    page.title = "NN Interface"     # Установить заголовок страницы
    page.padding = 0
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.horizontal_alignment = ft.MainAxisAlignment.CENTER

    btn_start_grpc = ft.IconButton(
                    icon=ft.icons.PLAY_CIRCLE_FILL_ROUNDED,
                    icon_color="blue700",
                    icon_size=50,
                    tooltip="Pause record",
                    on_click=btn_start_grpc_clicked
    )

    container_img0 = ft.Container(
        content=ft.Column(
            controls=[
                text_2G4,
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

    page.bottom_appbar = ft.BottomAppBar(
        bgcolor=ft.colors.GREY_700,
        shape=ft.NotchShape.CIRCULAR,
        content=ft.Row(
            controls=[
                btn_start_grpc,
                ft.Container(expand=False),
            ],

        ),
    )

    page.update()       # Обновить страницу, чтобы отобразить изменения
    page.theme_mode = ft.ThemeMode.DARK


ft.app(target=main, port=54421, view=ft.AppView.WEB_BROWSER)     # Запустить приложение Flet
