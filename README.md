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

## Конвертация в engine
```
./to_engine.sh {path_to_trtexec} {path_to_onnx_model}
```
Модель в формате engine будет создана в директории onnx модели.

## Замер скорости инференса:
### Сборка исполняемых файлов для инференса
Для `TensorRT` предлагается совместимость с JetPack 4.6, поэтмоу при конфигурации можно указать `-DJETSON=ON`, иначе сборка будет происходить под  TensorRTv10.

Также при сборке на ПК необходимо указать пути до include(`-DTENSORRT_INCLUDE_DIR={path}`), lib(`-DTENSORRT_LIB_DIR={path}`) директорий библиотеки TensorRT, на Jetson данные библиотеки лежат в системных путях (при корректной установке JetPack).
```
mkdir build &&
cd build && 
cmake ../native -DBUILD_TEST_PROGRAM=ON
make -j6
```
### OpenCV_DNN:
```
{path_to_measure_onnx} --image={path_to_img} --model={path_to_onnx_model} --times={amount_of_measures}
```
### TensorRT: 
```
{path_to_measure_onnx} --image={path_to_img} --model={path_to_engine_model} --times={amount_of_measures}
```
По окончании выполнения в директории запуска будет представлен файл c названием запускаемого детектора и форматом csv с результатами каждого этапа пайплайна детектирования.

