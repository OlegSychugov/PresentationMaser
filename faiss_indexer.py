# faiss_indexer.py

import os
import faiss
import json
from sentence_transformers import SentenceTransformer

INPUT_FILE = "Clients/ArenaData/company_input"
FAISS_INDEX_FILE = "Clients/ArenaData/company_context.index"
CHUNKS_JSON = "company_chunks.json"

# 1. Загружаем текстовые блоки
def load_chunks(file_path):
    with open(file_path, encoding="utf-8") as f:
        # Можно сплитить по строкам или другому разделителю — тут по пустой строке между абзацами
        text = f.read()
    # Разделяем по двум или более переводам строки (пустая строка = новый блок)
    chunks = [chunk.strip() for chunk in text.split('\n\n') if chunk.strip()]
    return chunks

# 2. Получаем эмбеддинги для каждого блока
def embed_chunks(chunks, model):
    return model.encode(chunks, show_progress_bar=True)

# 3. Индексируем в FAISS
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def main():
    # Модель эмбеддингов — можно заменить на любую свою
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 1. Чтение и разбиение на куски
    chunks = load_chunks(INPUT_FILE)
    print(f"Загружено {len(chunks)} блоков контекста.")
    
    # 2. Эмбеддинг
    embeddings = embed_chunks(chunks, model).astype('float32')
    print("Эмбеддинги рассчитаны.")
    
    # 3. Индексация
    index = build_faiss_index(embeddings)
    faiss.write_index(index, FAISS_INDEX_FILE)
    print(f"FAISS индекс сохранён: {FAISS_INDEX_FILE}")
    
    # 4. Сохраняем список исходных кусков для последующего поиска по индексу
    with open(CHUNKS_JSON, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Исходные блоки сохранены: {CHUNKS_JSON}")

if __name__ == "__main__":
    main()
