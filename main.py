import os
import json
from crewai import Agent, Task, Crew
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

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

# === Подключение локальной модели Mistral ===
MISTRAL_PATH = os.environ.get("MISTRAL_PATH", "./mistral-model")

tokenizer = AutoTokenizer.from_pretrained(MISTRAL_PATH)
model = AutoModelForCausalLM.from_pretrained(MISTRAL_PATH)

generation_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    temperature=0.2,
    do_sample=True,
)

llm = HuggingFacePipeline(pipeline=generation_pipe)

# === Агенты ===
goldratt_agent = Agent(
    role="Голдратт — мышление через посылки",
    goal="Выявить корневую цель встречи, конфликт и предпосылки",
    backstory=goldratt_text,
    verbose=True,
    llm=llm
)

spin_agent = Agent(
    role="SPIN-продажник",
    goal="Определить стадию сделки и блоки для презентации",
    backstory=spin_text,
    verbose=True,
    llm=llm
)

story_agent = Agent(
    role="Storyteller",
    goal="Упаковать всё в структуру презентации",
    backstory="Ты создаешь мощные презентации на основе аналитики, структур и аналогий.",
    verbose=True,
    llm=llm
)

# === Задания ===
goldratt_task = Task(
    description=f"Проанализируй цель встречи: {client_context['meeting_goal']}. Найди ключевую цель, конфликт, ложные предпосылки.",
    expected_output="JSON: 'цель', 'боль', 'предпосылки', 'гипотеза', 'ключевая задача встречи'",
    agent=goldratt_agent
)

spin_task = Task(
    description="Определи SPIN-стадию, собери аргументы и блоки для презентации.",
    expected_output="JSON: 'situation', 'problem', 'implication', 'need_payoff', 'рекомендованные слайды'",
    agent=spin_agent,
    context=[goldratt_task]
)

story_task = Task(
    description="Собери 5–7 слайдов на основе вывода SPIN. Каждый слайд: заголовок, описание, аналогия.",
    expected_output="Markdown-план с блоками презентации",
    agent=story_agent,
    context=[spin_task]
)

# === Запуск пайплайна ===
crew = Crew(
    agents=[goldratt_agent, spin_agent, story_agent],
    tasks=[goldratt_task, spin_task, story_task],
    verbose=True
)

result = crew.kickoff()
print("=== Результат презентации ===")
print(result)
