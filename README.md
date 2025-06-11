# 🚀 Projeto: Reconhecimento de Objetos em Tempo Real com ESP32-CAM + Python + YOLO

Este é um projeto pessoal que desenvolvi como prática de integração entre **hardware IoT** e **Visão Computacional**.

O projeto consiste em utilizar uma **ESP32-CAM** para capturar imagens em tempo real, transmitir essas imagens via Wi-Fi, e processá-las com um cliente Python que aplica um modelo de inteligência artificial (**YOLOv3-tiny**) para realizar o reconhecimento de objetos.

A motivação foi explorar como pequenos dispositivos embarcados podem ser combinados com modelos de IA leves para aplicações de monitoramento inteligente, automação e IoT.

---

## 🧐 O que o projeto faz?

- A **ESP32-CAM** atua como uma câmera conectada via Wi-Fi.
- Um script Python (**ReconheceObj.py**) captura as imagens fornecidas pela ESP32-CAM.
- Essas imagens são processadas com um modelo **YOLOv3-tiny** (via biblioteca CVLib), realizando a **detecção de objetos em tempo real**.
- O script exibe na tela a imagem com as caixas delimitadoras (bounding boxes) e as classes dos objetos reconhecidos.

---

## 🎯 Tecnologias e ferramentas utilizadas

### Hardware:

- [ESP32-CAM AI-Thinker](https://randomnerdtutorials.com/getting-started-with-esp32-cam/) — microcontrolador com câmera integrada.

### Software:

- **Arduino IDE** — utilizado para programar a ESP32-CAM.
- **Python 3.10** — para o cliente de captura e reconhecimento.
- **OpenCV** — biblioteca de visão computacional em Python.
- **CVLib** — wrapper para uso simplificado de YOLO em Python.
- **YOLOv3-tiny** — modelo leve de detecção de objetos.

### Outras bibliotecas Python:

```bash
pip install opencv-python cvlib numpy
