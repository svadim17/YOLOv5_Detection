import flet as ft
import gRPC_interface
import ui
import custom_utils


async def main(page: ft.Page):
    global gallery
    MAP_LIST = ['Autel', 'FPV', 'DJI', 'WiFi']
    gRPC_PORT = '51234'
    page.title = "Neural Detection"
    # page.window.title_bar_hidden = True
    page.window.height, page.window.min_height = 400, 400
    page.window.width, page.window.min_width = 400, 400
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.scroll = ft.ScrollMode.AUTO
    page.update()

    gRPC_channel = await gRPC_interface.connect_to_server(ip='192.168.3.111', port=gRPC_PORT)

    async def create_gallery(channels_names: list, zscale: dict):
        global gallery
        gallery = {}

        gallery_column = ft.Column(scroll=ft.ScrollMode.AUTO, spacing=20, alignment=ft.alignment.center)
        page.add(gallery_column)

        for name in channels_names:
            container = ui.SimpleRecognitionContainer(channel_name=name, map_list=MAP_LIST)
            gallery[name] = container
            gallery_column.controls.append(container)
            gallery_column.update()

        await gRPC_interface.start_data_stream(grpc_channel=gRPC_channel,
                                               map_list=MAP_LIST,
                                               gallery=gallery)

    # Создаем диалоговое окно и добавляем его на страницу
    startDialog = ui.StartDialog(grpc_channel=gRPC_channel, callback_func=create_gallery)
    page.overlay.append(startDialog)
    page.update()
    startDialog.update()

ft.app(target=main)
