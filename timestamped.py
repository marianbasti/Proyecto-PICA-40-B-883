import picamera
import datetime as dt
import os, platform
import argparse

# Checkea OS para determinar librería a usar
# Raspberry Pi: linux
if platform.machine=='armv71':
    import picamera
else 'aarch64':
    # Libreria para Jetson?
else:
    raise Exception(f"OS no reconocido:{platform.machine}")

# Parsea los argumentos de la línea de comandos
parser = argparse.ArgumentParser(description='Configura la cámara para grabar.')
parser.add_argument('--resolution', type=str, default='1600x1080', help='Resolución de la cámara como WxH')
parser.add_argument('--framerate', type=int, default=10, help='Framerate de la cámara')
parser.add_argument('--shutter_speed', type=int, default=800, help='Velocidad del obturador de la cámara')
parser.add_argument('--start_hour', type=int, default=7, help='Hora de inicio de la filmación')
parser.add_argument('--end_hour', type=int, default=19, help='Hora de fin de la filmación')
args = parser.parse_args()

# Extrae la resolución
resolution = tuple(map(int, args.resolution.split('x')))

def ensure_dir(directory):
    os.makedirs(directory, exist_ok=True)

with picamera.PiCamera() as camera:
    camera.resolution = resolution
    camera.framerate = args.framerate
    camera.annotate_background = picamera.Color('black')
    camera.shutter_speed = args.shutter_speed

    while True:
        now = dt.datetime.now()
        camera.annotate_text = now.strftime('%Y-%m-%d %H:%M:%S')

        if args.start_hour <= now.hour <= args.end_hour:
            print(now)
            date_str = now.strftime('%Y-%m-%d')
            time_str = now.strftime('%H_%M_%S')
            base_path = f"/home/pi/output/{date_str}"
            video_path = f"{base_path}/video"
            
            ensure_dir(video_path)

            video_filename = f"{video_path}/{date_str}_{time_str}.h264"
            camera.start_recording(video_filename)

            start = dt.datetime.now()
            while (dt.datetime.now() - start).seconds < 900:
                camera.annotate_text = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                camera.wait_recording(0.2)

            camera.stop_recording()
            print("Video terminado")
