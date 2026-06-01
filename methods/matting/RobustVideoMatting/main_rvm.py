import os
import torch
from model import MattingNetwork
from inference import convert_video

# 1. Detectar automàticament la carpeta exacta on està aquest script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---> AQUÍ ÉS ON VA LA LÍNIA <---
# Assegura't que el nom coincideix EXACTAMENT amb el del teu fitxer
video_entrada = os.path.join(BASE_DIR, 'fortnite_test3.mp4') 
modelo_pesos = os.path.join(BASE_DIR, 'rvm_mobilenetv3.pth')

print(f"El script està buscant el vídeo exactament a: {video_entrada}")

# Comprovació de seguretat
if not os.path.isfile(video_entrada):
    print("❌ ERROR! Python continua sense veure el fitxer en aquesta ruta.")
    print("Si us plau, revisa si el fitxer es diu en realitat 'fortinte_test.mp4' o si té alguna altra lletra diferent.")
    exit()

print("✅ El vídeo s'ha trobat amb èxit.")
print("Carregant el model a la CPU...")

# 2. Carregar el model (fem servir .cpu() per evitar problemes amb la RTX 5060)
model = MattingNetwork('mobilenetv3').eval().cpu()
model.load_state_dict(torch.load(modelo_pesos, map_location=torch.device('cpu')))

print("Processant el vídeo amb RVM (com que va per CPU pot trigar una mica més per frame)...")

# 3. Executar la inferència amb rutes absolutes de sortida
convert_video(
    model,
    input_source=video_entrada,
    output_type='video',
    output_composition=os.path.join(BASE_DIR, 'resultado_fondo_verde.mp4'),
    output_alpha=os.path.join(BASE_DIR, 'resultado_mascara.mp4'),
    output_foreground=os.path.join(BASE_DIR, 'resultado_recorte.mp4'),
    output_video_mbps=4,
    downsample_ratio=None
)

print("🎉 Processament completat amb èxit! Revisa la teva carpeta.")