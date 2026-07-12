# Запуск на macOS и Windows

Краткая инструкция для локального запуска приложения и проброса TCP-доступа через Cloudflare Access.

## Общие требования

- Python 3.10 или 3.11;
- FFmpeg в `PATH`;
- `cloudflared` в `PATH`;
- доступ к RTSP-потокам;
- GUI-сессия, так как приложение использует `cv2.imshow`.

## Cloudflare Access TCP

Перед запуском приложения откройте отдельный терминал и поднимите локальный TCP-туннель:

```bash
cloudflared access tcp \
    --hostname tcc1.naliv.kz \
    --url localhost:8554
```

Оставьте этот терминал открытым на всё время работы приложения.

## macOS

Установите зависимости:

```bash
brew install ffmpeg cloudflared
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Проверьте доступность FFmpeg:

```bash
ffmpeg -version
```

Запустите приложение:

```bash
source .venv/bin/activate
python main.py
```

## Windows

Установите Python 3.10 или 3.11, FFmpeg и Cloudflared. Убедитесь, что `python`, `ffmpeg` и `cloudflared` доступны из PowerShell:

```powershell
python --version
ffmpeg -version
cloudflared --version
```

Создайте виртуальное окружение и установите проект:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Если PowerShell блокирует активацию окружения, разрешите запуск скриптов для текущего пользователя:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

После этого повторите активацию:

```powershell
.\.venv\Scripts\Activate.ps1
```

Запустите Cloudflare Access TCP в отдельном окне PowerShell:

```powershell
cloudflared access tcp `
    --hostname tcc1.naliv.kz `
    --url localhost:8554
```

Запустите приложение:

```powershell
.\.venv\Scripts\Activate.ps1
python main.py
```

## Управление

- `q` или `Esc` - выход;
- `r` - сброс счётчика.

## Быстрая диагностика

Если приложение не видит RTSP или аудио, сначала проверьте, что окно с `cloudflared access tcp` продолжает работать.

Проверка локального RTSP-порта:

```bash
ffprobe rtsp://localhost:8554/microphone
```

На Windows та же команда выполняется из PowerShell:

```powershell
ffprobe rtsp://localhost:8554/microphone
```
