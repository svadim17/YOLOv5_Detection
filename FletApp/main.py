import flet as ft
import yaml
import utils
import gRPC_interface
import ui
from loguru import logger
import sys

logger.remove(0)
log_level = "TRACE"
log_format = ("<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> |"
              " <level>{level: <8}</level> |"
              " {extra} |"
              " <yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>")
logger.add(sys.stderr, format=log_format, colorize=True, backtrace=True, diagnose=True)


def load_conf(config_path: str):
    try:
        with open(config_path, encoding='utf-8') as f:
            config = dict(yaml.load(f, Loader=yaml.SafeLoader))
            logger.success(f'Config loaded successfully!')
            return config
    except Exception as e:
        logger.error(f'Error with loading config: {e}')


async def main(page: ft.Page):
    global gallery
    conf = load_conf('client_conf.yaml')
    map_list = conf['map_list']
    server_port = conf['server_port']
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
                                                image64_background=utils.image_to_base64(START_IMAGE),
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

        await gRPC_interface.start_gRPC_streams(grpc_channel=gRPC_channel,
                                                map_list=list(map_list),
                                                gallery=gallery,
                                                image_stream_status=bool(show_images_status))
        bottomBarContent.pb.visible = False
        bottomBarContent.pb.update()

    # Создаем диалоговое окно и добавляем его на страницу
    startDialog = ui.StartDialog(grpc_channel=gRPC_channel, callback_func=create_gallery)
    page.overlay.append(startDialog)
    page.update()
    startDialog.update()

ft.app(target=main)
