# Cashier Copilot Audio/Video Processor

Локальное приложение для аналитики кассовой зоны по RTSP. Система обнаруживает и отслеживает объекты, определяет клиента и кассира, оценивает положение рук, распознаёт речь, показывает субтитры и собирает изображения для последующей разметки Florence-2.

## Возможности

- RTSP-видео с автоматическим переподключением;
- отдельный RTSP-поток микрофона;
- поиск объектов через YOLO-World;
- единый внешний класс `object` без продуктовой классификации;
- трекинг ByteTrack и устойчивый подсчёт;
- отдельные окна с crop найденных объектов;
- YOLO Pose для клиента, кассира и положения рук;
- состояния кассы и контроль отсутствия кассира;
- русская речь через GigaAM-v3;
- субтитры с историей за последние 10 секунд;
- WAV и JSON-транскрипт каждого визита;
- сбор лучшего crop каждого трека;
- отдельная пакетная разметка Florence-2.

## Архитектура

```text
RTSP video
  -> RTSPReader
  -> YOLO-World -> ByteTrack -> counter / object windows / pending dataset
  -> YOLO Pose  -> customer/cashier / hand position
  -> OpenCV overlay

RTSP microphone
  -> FFmpeg PCM 16 kHz mono
  -> visit audio buffer
  -> GigaAM-v3 -> subtitles / WAV / JSON

dataset_output/pending
  -> separate Florence-2 script
  -> dataset_output/images/<class>
  -> dataset_output/metadata.jsonl
```

## Требования и установка

- Python 3.10 или 3.11;
- FFmpeg в `PATH`;
- доступ к RTSP-потокам;
- GUI-сессия для `cv2.imshow`;
- macOS Apple Silicon для MLX/MPS-конфигурации по умолчанию.

```bash
brew install ffmpeg
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Проверка:

```bash
ffmpeg -version
```

## Модели

Локальные веса:

```text
weights/yolov8s-worldv2.pt
yolo11n-pose.pt
```

Речь по умолчанию:

```text
GigaAM-v3 e2e RNN-T
```

Разметка датасета:

```text
microsoft/Florence-2-base-ft
```

Загрузить Florence заранее:

```bash
.venv/bin/hf download microsoft/Florence-2-base-ft
```

## Конфигурация камеры

Камеры задаются в `STREAMS` файла `config.py`:

```python
STREAMS = {
    "cam10": {
        "url": "rtsp://user:password@host/video",
        "audio_url": "rtsp://host:8554/microphone",
        "scan_roi": (1100, 100, 1750, 1100),
        "customer_roi": (1800, 0, 3200, 900),
        "cashier_roi": (0, 0, 1200, 1300),
    },
}
```

- `url` — видео;
- `audio_url` — отдельный микрофон; если отсутствует, используется `url`;
- `scan_roi` — область объектов;
- `customer_roi` — зона клиента;
- `cashier_roi` — зона кассира.

ROI: `(x1, y1, x2, y2)` в координатах исходного кадра.

## Детекция

```python
TARGET_FPS = 15
SCAN_CONFIDENCE = 0.2
PERSON_CONFIDENCE = 0.4
POSE_KEYPOINT_CONFIDENCE = 0.3
SCAN_IMAGE_SIZE = (1088, 1920)
POSE_IMAGE_SIZE = (1088, 1920)
```

YOLO-World требует prompts даже без продуктовой классификации:

```python
SCAN_WORLD_PROMPTS = [
    "retail product", "product package", "bottle", "can",
    "box", "bag", "fruit", "vegetable",
]
```

Prompts применяются внутри модели. Все результаты наружу выходят как `object`.

## ByteTrack и счётчик

Конфигурация: `trackers/bytetrack_retail.yaml`.

```yaml
tracker_type: bytetrack
track_high_thresh: 0.20
track_low_thresh: 0.05
new_track_thresh: 0.25
track_buffer: 90
match_thresh: 0.90
fuse_score: true
```

Объект получает ID, например `ID 7 object 0.82`. Счётчик сначала сопоставляет по ID. После пропуска и смены ByteTrack ID выполняется повторное связывание по IoU и расстоянию. Внутренние треки хранятся до 45 обработанных кадров.

Для каждого объекта открывается окно с увеличенным crop. Оно остаётся две секунды после потери трека, чтобы не мерцать.

## YOLO Pose и руки

Роль человека определяется по нижней центральной точке bounding box относительно ROI. Используются COCO keypoints:

- `5/6` — плечи;
- `7/8` — локти;
- `9/10` — запястья.

Состояния:

- `raised` — запястье выше плеча;
- `extended` — рука вытянута в сторону;
- `bent` — рука согнута;
- `down` — рука опущена;
- `unknown` — keypoints недостаточно уверенные.

На кадре рисуются суставы, линии и подписи.

## Состояние кассы

```python
CUSTOMER_TIMEOUT = 3.0
CASHIER_TIMEOUT = 2.0
```

Статусы:

- `IDLE`;
- `CUSTOMER_WAITING`;
- `SERVICE_STARTED`;
- `NO_CASHIER`.

События:

```text
CUSTOMER_PRESENT
CUSTOMER_LEFT
CASHIER_PRESENT
CASHIER_LEFT
PRODUCT_COUNTED:4
```

## Распознавание речи

FFmpeg преобразует RTSP-аудио в PCM signed 16-bit mono 16 kHz. Фильтры `aresample` и `asetpts` нормализуют нестабильные RTSP timestamps.

STT привязан к визиту клиента. Prebuffer сохраняет звук с первого появления, включая период подтверждения клиента.

```bash
export SPEECH_RECOGNITION_ENABLED=1
export WHISPER_BACKEND=gigaam
export GIGAAM_MODEL=v3_e2e_rnnt
export GIGAAM_DEVICE=auto
export WHISPER_LANGUAGE=ru
export TRANSCRIPTS_DIR=transcripts
```

Backend:

- `gigaam` — специализированная русская модель по умолчанию;
- `mlx` — Apple Silicon;
- `faster-whisper` — CPU/CUDA;
- `sensevoice` — экспериментальный.

Для Faster Whisper:

```bash
export WHISPER_COMPUTE_TYPE=int8
```

### Субтитры

Речь распознаётся перекрывающимися окнами. Справа показываются фразы за последние 10 секунд. Тихие фрагменты отбрасываются по RMS. Для GigaAM записи длиннее 25 секунд автоматически режутся на 24-секундные части, а word-level timestamps объединяются в общую шкалу визита.

### Файлы визита

```text
transcripts/
  cam10_20260712_120000_ab12cd34.wav
  cam10_20260712_120000_ab12cd34.json
```

```json
{
  "camera": "cam10",
  "visit_id": "...",
  "started_at": "2026-07-12T12:00:00+05:00",
  "ended_at": "2026-07-12T12:01:15+05:00",
  "duration": 75.0,
  "timestamps_relative_to": "customer_arrived",
  "segments": [
    {"start": 3.24, "end": 5.81, "text": "Здравствуйте"}
  ],
  "audio_file": "cam10_20260712_120000_ab12cd34.wav"
}
```

Таймкоды считаются от начала визита.

## Сбор датасета без Florence

Основное приложение только сохраняет лучший crop трека. Florence в видеопроцессе не загружается.

```bash
export DATASET_COLLECTION_ENABLED=1
export DATASET_DIR=dataset_output
export DATASET_TRACK_TIMEOUT=2.0
```

Crop выбирается по confidence, площади и резкости:

```text
dataset_output/pending/
  cam10_track_7_20260712_120000_123456.jpg
  cam10_track_7_20260712_120000_123456.json
```

Sidecar JSON содержит камеру, ID, confidence, время и статус `pending`.

## Отдельная разметка Florence-2

```bash
.venv/bin/python -m dataset.label_with_florence
```

С параметрами:

```bash
.venv/bin/python -m dataset.label_with_florence \
  --dataset-dir dataset_output \
  --model microsoft/Florence-2-base-ft
```

Скрипт загружает Florence один раз, выполняет `<CAPTION>`, нормализует описание в имя класса, перемещает изображение и добавляет запись в `metadata.jsonl`.

```text
dataset_output/
  pending/
  images/
    bottle_of_cola/
    banana/
  metadata.jsonl
```

Это псевдоразметка. Перед обучением нужно вручную объединить синонимы, проверить классы и удалить ошибки.

## Запуск

```bash
source .venv/bin/activate
python main.py
```

Клавиши:

- `q` или `Esc` — выход;
- `r` — сброс счётчика.

## Audio test

По умолчанию распознаёт фрагменты по 60 секунд:

```bash
.venv/bin/python audio_test/main.py
CHUNK_SECONDS=30 .venv/bin/python audio_test/main.py
```

## Тесты

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q .
```

## Структура

```text
audio/rtsp_transcriber.py        RTSP audio, STT, субтитры
audio_test/main.py               тест аудио
camera/rtsp_reader.py            RTSP video reader
dataset/object_collector.py      pending crops
dataset/label_with_florence.py   отдельная разметка
dataset/phi4_collector.py        Florence runtime (legacy filename)
logic/checkout_state.py          состояние кассы
logic/product_counter.py         подсчёт и re-association
models/detection.py              объекты
models/person.py                 люди и руки
trackers/bytetrack_retail.yaml   ByteTrack
ui/overlay.py                    оверлей
vision/person_detector.py        YOLO Pose
vision/scan_detector.py          YOLO-World + tracking
config.py                        настройки
main.py                          основной цикл
```

## Диагностика

### Детекция мигает

- уменьшайте `SCAN_CONFIDENCE` небольшими шагами;
- проверьте prompts и `scan_roi`;
- улучшите освещение;
- увеличьте `track_buffer`.

Слишком низкий confidence создаёт ложные объекты.

### Объект считается повторно

Проверьте `trackers/bytetrack_retail.yaml`. После очень долгого исчезновения объект намеренно считается новым.

### Нет аудио

```bash
ffprobe rtsp://host:8554/microphone
ffmpeg -rtsp_transport tcp -i rtsp://host:8554/microphone -t 10 test.wav
```

### `non monotonically increasing dts`

Не удаляйте фильтры `aresample=async=1:first_pts=0` и `asetpts=N/SR/TB`.

### `load_npz` в MLX Whisper

Используйте `mlx-community/whisper-large-v3-turbo-q4`. Несовместимое имя `...-8bit` автоматически заменяется на `q4`.

### Конфликт `libavdevice` на macOS

PyAV и OpenCV могут загрузить разные FFmpeg dylib. Используйте `WHISPER_BACKEND=mlx`.

### Florence не запускается

```bash
pip install -e .
.venv/bin/hf download microsoft/Florence-2-base-ft
```

Florence запускается только отдельным скриптом и не влияет на FPS основного приложения.

## Ограничения

- YOLO-World зависит от prompts и не является полностью class-agnostic detector;
- Florence создаёт описание, но не гарантирует точный SKU;
- для точного SKU нужен каталог, OCR/штрихкод или дообученная модель;
- роль человека определяется геометрически по ROI;
- 2D pose ошибается при перекрытиях;
- `cv2.imshow` не работает на headless-сервере без дисплея.

## Безопасность

Не публикуйте RTSP URL с логинами и паролями. Для production вынесите адреса камер из `config.py` в переменные окружения или секрет-хранилище.
