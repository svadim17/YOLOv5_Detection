import time
import flet as ft
import yaml
import custom_utils
import gRPC_interface
import ui
import asyncio
import os


project_path: str = os.path.abspath(os.path.dirname(__file__))
CONFIG_PATH = os.path.join(project_path, 'client_conf.yaml')


async def main(page: ft.Page):
    global gallery
    conf = gRPC_interface.load_conf(CONFIG_PATH)
    map_list = conf['map_list']
    server_port = conf['server_port']
    jetson_service_port = conf['jetson_service_port']
    server_addr = conf['server_addr']
    show_images_status = conf['show_images']
    show_freq_status = conf['show_frequinces']

    # START_IMAGE = "spectrum-analysis.png"
    START_IMAGE = None
    page.title = "Neural Detection"
    # page.window.title_bar_hidden = True
    page.window.height, page.window.min_height = 920, 200
    page.window.width, page.window.min_width = 1410, 380
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.update()
    bottomBarContent = ui.BottomBarContent(parent_page=page)
    page.bottom_appbar = ft.BottomAppBar(shape=ft.NotchShape.CIRCULAR, content=bottomBarContent)
    page.bottom_appbar.height = 60
    page.scroll = ft.ScrollMode.AUTO

    gRPC_channel = await gRPC_interface.connect_to_server(ip=server_addr, port=server_port)


    menuContent = ui.MenuContent(parent_page=page, grpc_channel=gRPC_channel)
    page.drawer = menuContent.menu
    bottomBarContent.menu_button.on_click = menuContent.open_menu

    async def create_gallery(channels_names: list, zscale: dict, recogn_settings: dict):
        global gallery
        gallery = {}

        if 2 < len(channels_names) <= 4:
            gallery_rows = [ft.Row(scroll=ft.ScrollMode.AUTO, spacing=30), ft.Row(scroll=ft.ScrollMode.AUTO, spacing=30)]
            page.add(gallery_rows[0]), page.add(gallery_rows[1])
        else:
            gallery_rows = [ft.Row(scroll=ft.ScrollMode.AUTO, spacing=30)]
            page.add(gallery_rows[0])

        i = 0
        for name in channels_names:
            container = ui.RecognitionContainer(grpc_channel=gRPC_channel,
                                                image64_background=custom_utils.image_to_base64(START_IMAGE),
                                                channel_name=name,
                                                map_list=list(map_list),
                                                z_min=zscale[name][0],
                                                z_max=zscale[name][1],
                                                accumulation_size=recogn_settings[name][0],
                                                threshold=recogn_settings[name][1],
                                                exceedance=recogn_settings[name][2],
                                                show_images_status=bool(show_images_status),
                                                show_freq_status=bool(show_freq_status))
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
        await asyncio.sleep(5)
        await gRPC_interface.start_gRPC_streams(grpc_channel=gRPC_channel,
                                                map_list=list(map_list),
                                                gallery=gallery,
                                                image_stream_status=bool(show_images_status))
        bottomBarContent.pb.visible = False
        bottomBarContent.pb.update()

    # РЎРѕР·РґР°РµРј РґРёР°Р»РѕРіРѕРІРѕРµ РѕРєРЅРѕ Рё РґРѕР±Р°РІР»СЏРµРј РµРіРѕ РЅР° СЃС‚СЂР°РЅРёС†Сѓ
    startDialog = ui.StartDialog(grpc_channel=gRPC_channel, callback_func=create_gallery)
    page.overlay.append(startDialog)
    page.update()
    startDialog.update()

ft.app(target=main)
