"""Smoke test of a running container: python smoke_test.py [http://host:8000]

Checks /health, /v1/models, the OpenAI SDK path (plain and stream), the native /judge
path, that both paths give the same answers, and that a bad request gets a 400.
"""
import json
import sys
import urllib.error
import urllib.request

from openai import BadRequestError, OpenAI

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")

REQUEST = {
    "state": "Здравствуйте, у меня не приходит код подтверждения уже час. Срочно нужно оплатить билет.",
    "questions": {
        "topic": {"type": "choice", "instructions": "К какой теме относится обращение?",
                  "criteria": {"login": "вход в аккаунт, коды подтверждения, пароли",
                               "payment": "оплата, списания, возвраты",
                               "delivery": "доставка заказа"}},
        "urgency": {"type": "score", "instructions": "Насколько срочное обращение?",
                    "criteria": ["не срочно", "средне", "очень срочно"]},
        "angry": {"type": "noul", "instructions": "Клиент раздражён?"},
        "reply": {"type": "ranking", "instructions": "Какой ответ лучше подходит?",
                  "criteria": {"a": "Проверьте папку «Спам» и запросите код повторно.",
                               "b": "Ваш заказ уже в пути.",
                               "c": "Возврат средств занимает до 10 дней."}},
    },
}


def decisions(a):
    """What must match between paths; probabilities may differ in low bits on GPU."""
    return {q: (v.get("choice"), v.get("ranking"), round(v.get("score", 0), 2), round(v.get("noul", 0), 2))
            for q, v in a.items()}


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=10) as r:
        return r.status, json.loads(r.read())


def post(path, body):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


status, health = get("/health")
assert status == 200, health
print("health:", health)
print("models:", get("/v1/models")[1])

client = OpenAI(base_url=BASE + "/v1", api_key="none")
model = client.models.list().data[0].id

resp = client.chat.completions.create(
    model=model, temperature=0, max_tokens=256,
    messages=[{"role": "system", "content": "ignored"},
              {"role": "user", "content": json.dumps(REQUEST, ensure_ascii=False)}])
answers = json.loads(resp.choices[0].message.content)["answers"]
print("chat.completions:", json.dumps(answers, ensure_ascii=False, indent=1))
print("usage:", resp.usage)
assert resp.choices[0].finish_reason == "stop"

# content as a list of parts + ```json fence
resp_parts = client.chat.completions.create(model=model, messages=[{"role": "user", "content": [
    {"type": "text", "text": "```json\n" + json.dumps(REQUEST, ensure_ascii=False) + "\n```"}]}])
assert decisions(json.loads(resp_parts.choices[0].message.content)["answers"]) == decisions(answers)

stream = client.chat.completions.create(
    model=model, stream=True,
    messages=[{"role": "user", "content": json.dumps(REQUEST, ensure_ascii=False)}])
streamed = "".join(c.choices[0].delta.content or "" for c in stream if c.choices)
assert decisions(json.loads(streamed)["answers"]) == decisions(answers)

native = post("/judge", REQUEST)
assert decisions(native["answers"]) == decisions(answers), (native["answers"], answers)
print("native usage:", native["usage"])

try:
    client.chat.completions.create(model=model, messages=[{"role": "user", "content": "привет"}])
    raise AssertionError("expected 400")
except BadRequestError as e:
    print("bad request ->", e.status_code, e.body)

print("OK")
