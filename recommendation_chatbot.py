# ============================================================
#  AI Product Recommendation Chatbot — Google Colab + Gemini
#  + Hybrid Memory RAG (Sliding Window + Summary + Vector)
# ============================================================

# ---- INSTALL (ilk dəfə işlət) ----
!pip install -q sentence-transformers faiss-cpu google-generativeai

# ---- IMPORTS ----
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai
from google.colab import userdata

# ---- API KEY ----
# Colab Secrets-ə "GEMINI_API_KEY" adı ilə əlavə et (sol paneldəki 🔑 ikonu)
API_KEY = userdata.get("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)
model_ai = genai.GenerativeModel("gemini-2.5-flash-lite")

# ---- EMBEDDING MODEL ----
print("Embedding modeli yüklənir...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Hazır!")

# ---- PRODUCTS ----
products = [
    {"name": "Lenovo Legion 5",      "desc": "Gaming laptop, RTX 3060, 16GB RAM, 144Hz ekran",        "price": "1299 AZN", "tags": "gaming laptop oyun"},
    {"name": "Asus ROG Strix G16",   "desc": "Gaming laptop, RTX 4060, 16GB RAM, 165Hz, premium GPU", "price": "1699 AZN", "tags": "gaming laptop oyun premium"},
    {"name": "MacBook Air M3",       "desc": "Yüngül iş noutbuku, 13 saat batareya, MacOS, sakit",    "price": "1899 AZN", "tags": "laptop iş yüngül apple macbook"},
    {"name": "iPhone 15 Pro",        "desc": "Apple smartfon, 48MP kamera, titanium, A17 Pro",        "price": "1499 AZN", "tags": "telefon smartfon kamera apple iphone"},
    {"name": "Samsung Galaxy S24",   "desc": "Android flagship, 50MP kamera, AI, AMOLED ekran",       "price": "1199 AZN", "tags": "telefon smartfon kamera android samsung"},
    {"name": "Xiaomi Redmi Note 13", "desc": "Budget smartfon, 108MP kamera, 5000mAh batareya",       "price": "449 AZN",  "tags": "telefon smartfon kamera budget ucuz"},
    {"name": "HP Pavilion 15",       "desc": "Gündəlik iş noutbuku, Intel i5, 8GB RAM, ofis üçün",   "price": "749 AZN",  "tags": "laptop iş ofis budget"},
]

# ============================================================
#  1. SLIDING WINDOW MEMORY
# ============================================================
WINDOW_SIZE = 5

# ============================================================
#  2. SUMMARY MEMORY
# ============================================================
def update_summary(summary, new_turn):
    prompt = f"""
Mövcud xülasə: {summary}
Yeni söhbət: {new_turn}
Yenilənmiş xülasə yaz — qısa saxla. Azərbaycan dilində.
"""
    return model_ai.generate_content(prompt).text

# ============================================================
#  3. VECTOR MEMORY
# ============================================================
memory_index = faiss.IndexFlatL2(384)
memory_texts = []

def add_to_vector_memory(text):
    vec = embed_model.encode([text]).astype("float32")
    memory_index.add(vec)
    memory_texts.append(text)

def search_vector_memory(query, k=2):
    if len(memory_texts) == 0:
        return []
    q_vec = embed_model.encode([query]).astype("float32")
    k = min(k, len(memory_texts))
    _, indices = memory_index.search(q_vec, k=k)
    return [memory_texts[i] for i in indices[0] if i < len(memory_texts)]

# ============================================================
#  INTENT FILTER
# ============================================================
def filter_by_intent(query, products):
    q = query.lower()
    if any(w in q for w in ["telefon", "smartfon", "phone", "iphone", "samsung", "xiaomi"]):
        return [p for p in products if "telefon" in p["tags"]]
    if any(w in q for w in ["gaming", "oyun", "game", "rtx", "gpu"]):
        return [p for p in products if "gaming" in p["tags"]]
    if any(w in q for w in ["laptop", "noutbuk", "macbook", "iş", "ofis", "work"]):
        return [p for p in products if "laptop" in p["tags"]]
    if any(w in q for w in ["ucuz", "budget", "aşağı qiymət", "cheap", "en ucuz"]):
        return sorted(products, key=lambda p: int(p["price"].replace(" AZN", "")))
    return products

# ============================================================
#  SEMANTIC SEARCH (FAISS)
# ============================================================
def semantic_search(query, product_pool, top_k=3):
    if not product_pool:
        return []
    texts = [f"{p['name']} {p['desc']} {p['tags']}" for p in product_pool]
    embeddings = embed_model.encode(texts).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    q_vec = embed_model.encode([query]).astype("float32")
    k = min(top_k, len(product_pool))
    _, indices = index.search(q_vec, k=k)
    return [product_pool[i] for i in indices[0]]

# ============================================================
#  4. PROMPT BUILDER — hamısını birləşdir
# ============================================================
def build_prompt(query, top_products, history, summary, vector_mem):
    window      = "\n".join(history[-WINDOW_SIZE:])
    vec_context = "\n".join(vector_mem) if vector_mem else "Yoxdur"
    product_context = "\n".join([
        f"- {p['name']} ({p['price']}): {p['desc']}"
        for p in top_products
    ]) if top_products else "Uyğun məhsul tapılmadı"

    return f"""Sən bir e-ticarət platformasının AI məhsul assistantısan. Azərbaycan dilində cavab ver.
Heç vaxt salam vermə, birbaşa cavaba keç.
Əgər istifadəçi "ucuz" və ya "budget" deyibsə, ən ucuz məhsulu tövsiyə et.

Söhbət xülasəsi:
{summary if summary else "Hələ yoxdur"}

Son söhbət:
{window if window else "Hələ yoxdur"}

Oxşar keçmiş mesajlar:
{vec_context}

Mövcud məhsullar:
{product_context}

İstifadəçinin sualı: "{query}"

Ən uyğun məhsulu tövsiyə et. Niyə uyğun olduğunu qısa izah et (2-3 cümlə).
Qiymət məlumatını da qeyd et. Dostcasına və faydalı ol."""

# ============================================================
#  MAIN CHAT LOOP — Hybrid Memory RAG
# ============================================================
def chat():
    print("=" * 55)
    print("  AI Məhsul Tövsiyə Chatbotu")
    print("  Hybrid Memory RAG — Sliding Window + Summary + Vector")
    print("  Çıxmaq üçün: 'q' yazın")
    print("=" * 55)

    history = []   # Sliding Window
    summary = ""   # Summary Memory

    while True:
        print()
        query = input("Siz: ").strip()

        if not query:
            continue
        if query.lower() in ["q", "quit", "çıx", "exit"]:
            print("Görüşənədək! 👋")
            break

        print("Axtarılır...", end="", flush=True)

        # 1. Vector memory-dən oxşar keçmiş mesajları tap
        vec_memory = search_vector_memory(query)

        # 2. Məhsulları tap
        filtered     = filter_by_intent(query, products)
        top_products = semantic_search(query, filtered, top_k=3)

        # 3. Prompt qur
        prompt = build_prompt(query, top_products, history, summary, vec_memory)

        # 4. Cavab al
        answer = model_ai.generate_content(prompt).text

        # 5. Yaddaşları yenilə
        new_turn = f"İstifadəçi: {query}\nAI: {answer}"
        history.append(f"İstifadəçi: {query}")
        history.append(f"AI: {answer}")
        add_to_vector_memory(new_turn)
        summary = update_summary(summary, new_turn)

        # 6. Output
        print(f"\r{' ':30}\r")
        print("-" * 45)
        print(f"AI: {answer}")
        print("-" * 45)
        print("\nTövsiyə edilən məhsullar:")
        for i, p in enumerate(top_products, 1):
            print(f"  {i}. {p['name']} — {p['price']}")

# ---- RUN ----
chat()
