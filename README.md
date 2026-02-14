# Сравнительный анализ алгоритмов многообъектного отслеживания

## Работа в виртуальном окружении

```
python3.9 -m venv .venv             # проверена работа на версии 3.9
source ./.venv/bin/activate         # активация окружения
pip3 install -r requirements.txt    # установка зависимостей
```

## Снятие метрик на датасете COCO (валидационная выборка)

Необходимо подготовить [валидационную выборку и разметку](./scripts/coco.yaml).

Для модели `yolo_nas`:
```
cd ./scripts
python3 yolo_nas_eval.py validate s > yolo_nas_s.txt
```

Для модели `yolo`:
```
cd ./scripts

# yolo26n
python3 yolo_eval.py validate 26 n > yolo26n.txt

# yolo11n
python3 yolo_eval.py validate 1 n > yolo11n.txt

# yolov8n
python3 yolo_eval.py validate v8 n > yolov8n.txt
```

Результат работы скрипта будет перенаправлен в .txt файла, графики и примеры работы моделей находятся в папке [run](./scripts/runs)

## Конвертация в onnx 

Для модели `yolo_nas`:
```
cd ./scripts

# yolo_nas (остальные по аналогии с предыдущим пунктом)
python3 yolo_nas_eval.py export_to_onnx s
```

Для модели `yolo`:
```
cd ./scripts

# yolo11n (остальные по аналогии с предыдущим пунктом)
python3 yolo_eval.py export_to_onnx 11 n
```
