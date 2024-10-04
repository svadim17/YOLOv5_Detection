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
import utils
import gRPC_interface
import ui


async def main(page: ft.Page):
    global gallery
    MAP_LIST = ['Autel', 'FPV', 'DJI', 'WiFi']
    gRPC_PORT = '51234'
    page.title = "Neural Detection"
    # page.window.title_bar_hidden = True
    page.window.height, page.window.min_height = 920, 920
    page.window.width, page.window.min_width = 1410, 1410
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.update()
    bottomBarContent = ui.BottomBarContent(parent_page=page)
    page.bottom_appbar = ft.BottomAppBar(shape=ft.NotchShape.CIRCULAR, content=bottomBarContent)
    page.bottom_appbar.height = 60
    page.scroll = ft.ScrollMode.AUTO

    gRPC_channel = await gRPC_interface.connect_to_server(port=gRPC_PORT)

    menuContent = ui.MenuContent(parent_page=page, grpc_channel=gRPC_channel)
    page.drawer = menuContent.menu
    bottomBarContent.menu_button.on_click = menuContent.open_menu

    async def create_gallery(channels_names: list, zscale: dict):
        global gallery
        gallery = {}

        if 2 < len(channels_names) <= 4:
            gallery_rows = [ft.Row(scroll=ft.ScrollMode.AUTO), ft.Row(scroll=ft.ScrollMode.AUTO)]
            page.add(gallery_rows[0]), page.add(gallery_rows[1])
        else:
            gallery_rows = [ft.Row(scroll=ft.ScrollMode.AUTO)]
            page.add(gallery_rows[0])

        i = 0
        for name in channels_names:
            container = ui.RecognitionContainer(grpc_channel=gRPC_channel,
                                                image64_background=utils.image_to_base64("spectrum-analysis.png"),
                                                channel_name=name,
                                                map_list=MAP_LIST,
                                                z_min=zscale[name][0],
                                                z_max=zscale[name][1])
            gallery[name] = container
            if len(gallery_rows) == 2:
                gallery_rows[i % 2].controls.append(container)
                i += 1
            else:
                gallery_rows[0].controls.append(container)
            menuContent.add_channel_settings(gallery[name].settings_container)
        menuContent.add_btn_save_config()
        for gallery_row in gallery_rows:
            gallery_row.update()
            page.add(gallery_row)

        await gRPC_interface.start_gRPC_streams(grpc_channel=gRPC_channel,
                                                map_list=MAP_LIST,
                                                gallery=gallery)

    # Создаем диалоговое окно и добавляем его на страницу
    startDialog = ui.StartDialog(grpc_channel=gRPC_channel, callback_func=create_gallery)
    page.overlay.append(startDialog)
    page.update()
    startDialog.update()


ft.app(target=main)
