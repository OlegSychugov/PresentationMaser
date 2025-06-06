import os
import json
from langchain_ollama import OllamaLLM

# === Настройки клиента ===
CLIENT_ID = "arenadata"
BASE_DIR = f"./clients/{CLIENT_ID}"

# === Загрузка контекста и инструкций ===
with open(os.path.join(BASE_DIR, "context.json"), "r", encoding="utf-8") as f:
    client_context = json.load(f)
with open(os.path.join(BASE_DIR, "goldratt.txt"), "r", encoding="utf-8") as f:
    goldratt_text = f.read()
with open(os.path.join(BASE_DIR, "spin.txt"), "r", encoding="utf-8") as f:
    spin_text = f.read()

# === Подключение Ollama ===
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

llm = OllamaLLM(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL
)

# === Этап 1: Анализ Голдратта ===
goldratt_prompt = (
    f"{goldratt_text}\n\n"
    f"Проанализируй цель встречи: {client_context['meeting_goal']}.\n"
    "Найди ключевую цель, конфликт, ложные предпосылки. "
    "Ответь JSON: 'цель', 'боль', 'предпосылки', 'гипотеза', 'ключевая задача встречи'."
)
goldratt_output = llm.invoke(goldratt_prompt)
print("=== Голдратт-эксперт ===")
print(goldratt_output)

# === Этап 2: SPIN-анализ ===
spin_prompt = (
    f"{spin_text}\n\n"
    "Используй результат анализа:\n"
    f"{goldratt_output}\n\n"
    "Определи SPIN-стадию, собери аргументы и блоки для презентации. "
    "Ответь JSON: 'situation', 'problem', 'implication', 'need_payoff', 'рекомендованные слайды'."
)
spin_output = llm.invoke(spin_prompt)
print("\n=== SPIN-эксперт ===")
print(spin_output)

# === Этап 3: Storyteller — финальная структура презентации ===
story_prompt = (
    "Ты — Storyteller. На основе вывода SPIN:\n"
    f"{spin_output}\n\n"
    "Собери 5–7 слайдов на основе этого вывода. Каждый слайд: заголовок, описание, аналогия. "
    "Оформи результат в виде Markdown-плана с блоками презентации."
)
story_output = llm.invoke(story_prompt)
print("\n=== Storyteller ===")
print(story_output)
