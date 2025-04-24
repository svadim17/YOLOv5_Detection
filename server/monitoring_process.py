import psutil
import time
from loguru import logger
from threading import Thread
import subprocess
import json


def detect_platform():
    try:
        # Jetson имеет специфичный файл в /proc
        with open("/proc/device-tree/model") as f:
            model = f.read().strip()
            if "NVIDIA" in model or 'Jetson' in model or 'jetson' in model or 'Nvidia' in model:
                return "jetson"
    except Exception:
        pass

    # Попробуем найти nvidia-smi → значит, это ПК/сервер с GPU
    try:
        subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL)
        return "nvidia"
    except FileNotFoundError:
        pass
    return "cpu"


def monitor_system(interval=1.0, queue_=None):
    net_old = psutil.net_io_counters()
    disk_old = psutil.disk_io_counters()
    platform = detect_platform()
    print(f'Detected platform: {platform}')

    while True:
        time.sleep(interval)

        # --- Network ---
        net_new = psutil.net_io_counters()
        up_speed = (net_new.bytes_sent - net_old.bytes_sent) / interval / 1024
        down_speed = (net_new.bytes_recv - net_old.bytes_recv) / interval / 1024
        net_old = net_new

        # --- CPU ---
        cpu_percent = psutil.cpu_percent(interval=None)

        # --- RAM ---
        mem = psutil.virtual_memory()
        ram_used = mem.used / (1024 * 1024)
        ram_total = mem.total / (1024 * 1024)
        ram_percent = mem.percent

        # --- Disk ---
        disk_new = psutil.disk_io_counters()
        read_speed = (disk_new.read_bytes - disk_old.read_bytes) / interval / 1024
        write_speed = (disk_new.write_bytes - disk_old.write_bytes) / interval / 1024
        disk_old = disk_new

        try:
            # if platform == "jetson":
            #     # Jetson: через jtop или jetson_stats
            #     output = subprocess.check_output(["jtop", "-s", "-f", "json"])
            #     gpu_msg = f"[GPU] Jetson stats: {output.decode().strip()}"
            if platform == "nvidia":
                output = subprocess.check_output([
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits"
                ])
                gpu_msg = f"[GPU] NVIDIA stats: {output.decode().strip()}"
            else:
                gpu_msg = ""
        except Exception as e:
            gpu_msg = "[GPU] Monitoring failed."

        if queue_ is not None:
            msg = {"CPU load": cpu_percent,
                   "RAM used": ram_used,
                   "RAM total": ram_total,
                   "NET up": up_speed,
                   "NET down": down_speed}
            try:
                usage, temp, mem_used, mem_total = output.decode().strip().split(", ")
                msg['GPU load'] = float(usage)
                msg['GPU temp'] = float(temp)
                msg['GPU mem_used'] = float(mem_used)
                msg['GPU mem_total'] = float(mem_total)
            except Exception as e:
                pass

            queue_.put({'status': 3, 'msg': json.dumps(msg)})

        else:
            msg_log = (f"[PSUTIL] ~ "
                       f"CPU: {cpu_percent:.1f}% | "
                       f"RAM: {ram_used:.1f}/{ram_total:.1f}MB ({ram_percent:.1f}%) | "
                       # f"[Disk] R/W: {read_speed:.1f}/{write_speed:.1f} KB/s | "
                       f"Net Up/Down: {up_speed:.1f}/{down_speed:.1f} KB/s | "
                       f"{gpu_msg}")
            logger.info(msg_log)


def read_temp(path="/sys/class/thermal/thermal_zone1/temp"):
    """ If don't want to use jtop """
    try:
        with open(path) as f:
            return int(f.read()) / 1000.0  # в °C
    except:
        return None


def monitor_jetson(interval=2.0, queue_=None):
    with jtop.jtop() as jetson:
        while jetson.ok():
            stats = jetson.stats

            cpu_temp = stats.get("Temp CPU", "N/A")
            gpu_temp = stats.get("Temp GPU", "N/A")
            # cpu_usage = stats.get("CPU1", "N/A")  # Можно и CPU2, CPU3...
            gpu_load = stats.get("GPU", "N/A")

            if queue_ is not None:
                msg = {"GPU load": float(gpu_load),
                        "CPU temp": float(cpu_temp),
                        "GPU temp": float(gpu_temp), }
                queue_.put({'status': 3, 'msg': json.dumps(msg)})
            else:
                msg = (f"[JETSON] ~ "
                       f"GPU load: {gpu_load}% | "
                       f"CPU temp: {cpu_temp}°C | "
                       f"GPU temp: {gpu_temp}°C")
                # f"CPU1 Load: {cpu_usage}% | "
                logger.info(msg)
            time.sleep(interval)


def start_monitoring(q):
    platform = detect_platform()
    t = Thread(target=monitor_system, args=(1.0, q), daemon=True)
    t.start()
    if platform == "jetson":
        import jtop
        jt = Thread(target=monitor_jetson, args=(1.0, q), daemon=True)
        jt.start()


if __name__ == "__main__":
    platform = detect_platform()
    if platform == "jetson":
        import jtop
        t1 = Thread(target=monitor_jetson, args=(1.0,), daemon=False)
        t1.start()

    t = Thread(target=monitor_system, args=(1.0,), daemon=False)
    t.start()

