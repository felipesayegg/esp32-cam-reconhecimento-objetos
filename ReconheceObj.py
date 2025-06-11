# ReconheceObj.py — YOLOv3-Tiny com cv2.dnn e ESP32-CAM — versão final e robusta

import cv2
import numpy as np
import urllib.request

# Parâmetros
CONFIDENCE_THRESHOLD = 0.1  # mais baixo para ESP32-CAM
NMS_THRESHOLD = 0.4

# Caminhos dos arquivos
config_path = 'yolov3-tiny.cfg'
weights_path = 'yolov3-tiny.weights'
classes_file = 'coco.names'

# Carrega as classes
with open(classes_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Cria a rede neural
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Obtém nomes das camadas de saída
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# URL da ESP32-CAM
url = 'http://192.168.0.90/cam-hi.jpg'

while True:
    try:
        # Captura imagem da ESP32-CAM
        img_resp = urllib.request.urlopen(url, timeout=5)
        img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        height, width = frame.shape[:2]

        # Cria blob e passa pela rede
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (320, 320), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)

        boxes = []
        confidences = []
        class_ids = []

        # Processa as saídas da rede
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > CONFIDENCE_THRESHOLD:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

        # Desenha as caixas
        if len(indices) > 0:
            for i in indices:
                # Compatível com todas as versões
                if isinstance(i, (list, tuple, np.ndarray)):
                    i = i[0]
                box = boxes[i]
                x, y, w, h = box
                label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
                color = (0, 255, 0)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Mostra a imagem
        cv2.imshow("YOLOv3-Tiny Detection", frame)

        # Pressiona 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"Erro ao capturar ou processar imagem: {e}")

# Fecha as janelas
cv2.destroyAllWindows()
