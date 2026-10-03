# FRIDA-Decisions — OpenAI-совместимый образ

[ai-forever/FRIDA-Decisions](https://huggingface.co/ai-forever/FRIDA-Decisions) — энкодер на 823M параметров. Он принимает решения по русскому тексту за один проход: выбор варианта, оценка по шкале, да/нет, ранжирование. Генерировать текст модель не умеет. В образе она обёрнута в контракт OpenAI `chat/completions`: на вход подаётся JSON-запрос в сообщении пользователя, на выход возвращается JSON-ответ в `message.content`.

## Сборка и запуск

```bash
docker build --platform linux/amd64 -t <registry>/frida-decisions:0.2.0 .
docker push <registry>/frida-decisions:0.2.0
docker run --gpus all -p 8000:8000 <registry>/frida-decisions:0.2.0
```

Характеристики образа:

- Основа: `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime`, сжатый размер около 4 ГБ.
- Веса модели (~1,9 ГБ, HF-коммит `0096b538…`) уже лежат в образе. На старте образ не обращается к Hugging Face (`HF_HUB_OFFLINE=1`).
- Нужен GPU с CUDA 12.8 и драйвер NVIDIA ≥ 570. Ожидается, что подойдут видеокарты от Turing и новее: RTX 20xx–50xx, A10, A100, L4, H100. На GPU это не проверялось. Фактический список архитектур выводится в лог сборки, в строке `arch [...]`. На vast выставьте фильтр `cuda_max_good >= 12.8`.
- Пиковое потребление видеопамяти — около 2 ГБ, поэтому подойдёт любая карта с 8 ГБ и больше.
- Если GPU нет, сервер работает на CPU в fp32. Это медленно, но для отладки годится.

## Эндпоинты (порт 8000)

| Метод | Путь | Что делает |
|---|---|---|
| GET | `/health` | Отвечает `200 {"status":"ok"}`, когда модель загружена и прогрета. Пока модель грузится, отвечает `503`. |
| GET | `/v1/models` | Возвращает одну модель: `frida-decisions`. |
| POST | `/v1/chat/completions` | Принимает запрос в контракте OpenAI. Подробности ниже. |
| POST | `/judge` | Нативный формат FRIDA-Decisions без обёртки. Нужен на случай, если сервис ставят в обход ЛЛМ-прокси. |

### `/v1/chat/completions`

Сервер берёт последнее сообщение с `role: "user"`. Его `content` должен быть JSON-запросом FRIDA. Допускается строка, список частей `[{"type":"text","text":...}]` и обёртка в ```` ```json ````.

Поля `temperature`, `max_tokens`, `response_format` и system-сообщения сервер принимает и игнорирует. На `stream: true` он отвечает одним SSE-чанком и затем отправляет `[DONE]`.

Пример запроса:

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "frida-decisions",
  "messages": [{"role": "user", "content": "{\"state\": \"Не приходит код подтверждения уже час\", \"questions\": {\"topic\": {\"type\": \"choice\", \"instructions\": \"К какой теме относится обращение?\", \"criteria\": {\"login\": \"вход, коды, пароли\", \"payment\": \"оплата, возвраты\"}}}}"}]
}'
```

Ответ (поле `content` — это JSON-строка):

```json
{"id": "chatcmpl-…", "object": "chat.completion", "model": "frida-decisions",
 "choices": [{"index": 0, "finish_reason": "stop",
   "message": {"role": "assistant",
     "content": "{\"answers\": {\"topic\": {\"type\": \"choice\", \"choice\": \"login\", \"probabilities\": {…}, \"confidence\": …}}}"}}],
 "usage": {"prompt_tokens": 61, "completion_tokens": 0, "total_tokens": 61}}
```

Тот же запрос через OpenAI SDK:

```python
from openai import OpenAI
import json

client = OpenAI(base_url="http://<host>:8000/v1", api_key="none")
req = {"state": "...", "questions": {...}}
resp = client.chat.completions.create(model="frida-decisions",
                                      messages=[{"role": "user", "content": json.dumps(req, ensure_ascii=False)}])
answers = json.loads(resp.choices[0].message.content)["answers"]
```

Если запрос некорректен, сервер возвращает `400` с ошибкой в формате OpenAI: `{"error": {"message", "type", "param", "code"}}`.

### Формат запроса FRIDA

```json
{"state": "текст (или любой JSON)",
 "questions": {
   "topic":   {"type": "choice",  "instructions": "...", "criteria": {"id": "описание", "...": "..."}},
   "urgency": {"type": "score",   "instructions": "...", "criteria": ["уровень 0", "уровень 1", "..."]},
   "angry":   {"type": "noul",    "instructions": "..."},
   "best":    {"type": "ranking", "instructions": "...", "criteria": {"doc-1": "текст", "...": "..."}}}}
```

Полное описание формата — в [README пакета](https://github.com/ai-forever/FRIDA-Decisions).

## Переменные окружения

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `PORT` | `8000` | Порт сервера. |
| `MODEL_ID` | `frida-decisions` | Имя модели в `/v1/models`. |
| `DEVICE` | авто | `cuda` или `cpu`. |
| `STATE_MAX` | `384` | Сколько токенов текста оставлять: остальное обрезается. Модель обучали на длинах до 512. |
| `STATE_CACHE_MB` | `512` | Размер кеша K/V для повторяющегося текста. `0` отключает кеш. |
| `CONTENT_MODE` | `answers` | Значение `full` добавляет в `content` поля `margins` и `usage`. |

## Проверка

На машине, где поднят контейнер, выполните:

```bash
pip install openai
```

```bash
python smoke_test.py http://<host>:8000
```

Скрипт проверяет `/health`, `/v1/models`, работу через OpenAI SDK в обычном и потоковом режиме и сверяет ответы `/v1/chat/completions` с ответами `/judge`. Ещё он проверяет, что некорректный запрос получает `400`.

## Ограничения

- Внутри процесса сервер обрабатывает запросы по одному. Модель отвечает за ~30 мс (по данным авторов модели), то есть один контейнер выдерживает порядка 30 запросов в секунду. Чтобы обработать больше, запускайте несколько реплик.
- `usage.prompt_tokens` — это число токенов, которые прошли через энкодер. `completion_tokens` всегда равен 0.
