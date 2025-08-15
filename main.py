import os
import io
import uuid
import json
import base64
import numpy as np
from PIL import Image
import imageio.v3 as iio
from flask import Flask, request
import telebot
from openai import OpenAI
from pinecone import Pinecone

# =========================
# ENV VARS (Render)
# =========================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL       = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
PINECONE_API_KEY   = os.getenv("PINECONE_API_KEY")
PINECONE_HOST      = os.getenv("PINECONE_HOST")  # es: https://<index-id>.svc.<region>.pinecone.io

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN mancante")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY mancante")
if not PINECONE_API_KEY or not PINECONE_HOST:
    raise ValueError("PINECONE_API_KEY o PINECONE_HOST mancanti")

# Config extra
SCORE_MIN   = float(os.getenv("SEARCH_SCORE_MIN", "0.60"))          # soglia /cerca
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")    # 1536 dim

# =========================
# CLIENTS
# =========================
app = Flask(__name__)
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)
oai = OpenAI(api_key=OPENAI_API_KEY)
pc  = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_HOST)   # SDK v5: connessione via host

# Memoria “fase” per chat (IT/EN)
PHASE = {}  # {chat_id: "IT"|"EN"}


# =========================
# HELPER
# =========================
def _ns(message):
    return str(message.chat.id)

def embed_text(text: str) -> list:
    resp = oai.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding

def friendly_image_error(e: Exception) -> str:
    s = str(e)
    if "organization must be verified" in s or "verify" in s.lower():
        return ("⚠️ Immagini bloccate: l'organizzazione OpenAI dev'essere verificata "
                "(modello gpt-image-1). Dopo la verifica riprova.")
    return f"⚠️ Errore generazione immagine: {s}"

def gen_image_png(prompt: str) -> bytes:
    result = oai.images.generate(model="gpt-image-1", prompt=prompt, size="1024x1024")
    b64 = result.data[0].b64_json
    return base64.b64decode(b64)

def gen_glitch_gif(prompt: str, frames: int = 6, dur_ms: int = 140) -> bytes:
    imgs_np = []
    for i in range(frames):
        p = f"{prompt}. Glitch, chromatic aberration, scanlines, noise, frame {i+1}/{frames}"
        r = oai.images.generate(model="gpt-image-1", prompt=p, size="512x512")
        b = base64.b64decode(r.data[0].b64_json)
        img = Image.open(io.BytesIO(b)).convert("RGB")
        imgs_np.append(np.array(img))
    out = io.BytesIO()
    iio.imwrite(out, imgs_np, extension=".gif", fps=int(1000/dur_ms))
    out.seek(0)
    return out.read()

def pinecone_upsert(namespace: str, text: str):
    vec = embed_text(text)
    index.upsert(vectors=[{"id": str(uuid.uuid4()), "values": vec, "metadata": {"text": text}}],
                 namespace=namespace)

def pinecone_query(namespace: str, query: str, top_k: int = 3):
    vec = embed_text(query)
    res = index.query(namespace=namespace, vector=vec, top_k=top_k,
                      include_values=False, include_metadata=True)
    items = []
    for m in res.get("matches", []):
        items.append((m["metadata"].get("text", ""), float(m.get("score", 0.0))))
    return items

def rag_text(namespace: str, user_prompt: str, phase: str) -> str:
    results = pinecone_query(namespace, user_prompt, top_k=5)
    context = "\n".join([f"- {t}" for t, _ in results]) if results else "(nessun contesto memorizzato)"
    sys_it = ("Sei un copywriter del progetto 2Peak. Tono minimal, ascetico, potente. "
              "Frasi brevi. Evita cliché. Integra il contesto se utile.")
    sys_en = ("You are a 2Peak copywriter. Minimal, ascetic, powerful tone. "
              "Short lines. Avoid clichés. Weave in context when useful.")
    system = sys_it if phase == "IT" else sys_en
    msg = oai.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Brief: {user_prompt}\n\nContext:\n{context}"},
        ],
    )
    return msg.output_text


# =========================
# WEBHOOK (Auto-set con HTTPS)
# =========================
@app.route("/", methods=["GET"])
def home():
    """
    Visita questa pagina per impostare/aggiornare il webhook.
    Forziamo https perché Telegram accetta solo HTTPS.
    """
    base = request.url_root or ""
    if not base.endswith("/"):
        base += "/"
    base = base.replace("http://", "https://")  # forza HTTPS su Render
    try:
        bot.remove_webhook()
    except Exception:
        pass
    webhook_url = base + TELEGRAM_BOT_TOKEN
    bot.set_webhook(url=webhook_url)
    return f"Webhook set → {webhook_url}", 200

@app.route(f"/{TELEGRAM_BOT_TOKEN}", methods=["POST"])
def telegram_webhook():
    update_json = request.get_data().decode("utf-8")
    upd = telebot.types.Update.de_json(update_json)
    try:
        bot.process_new_updates([upd])
    except Exception as e:
        print("Errore update:", e)
    return "OK", 200


# =========================
# COMANDI
# =========================
@bot.message_handler(commands=["start"])
def cmd_start(message):
    PHASE[_ns(message)] = PHASE.get(_ns(message), "IT")
    bot.reply_to(
        message,
        "👋 Benvenuto in 2Peak.\n"
        "Comandi: /fase it|en · /ricorda <testo> · /cerca <query> · /svuota ·\n"
        "/bozza <brief> · /gif <prompt> · /glitch <testo>\n"
        "Il secondo picco non si spiega. Si scala."
    )

@bot.message_handler(commands=["help"])
def cmd_help(message):
    bot.reply_to(
        message,
        "📚 Aiuto:\n"
        "/fase it|en – imposta lingua dello stile creativo\n"
        "/ricorda <testo> – salva nel vettore\n"
        "/cerca <query> – cerca tra ciò che hai memorizzato\n"
        "/svuota – cancella memoria di questa chat\n"
        "/bozza <brief> – bozza creativa con RAG (2Peak style)\n"
        "/gif <prompt> – genera un’immagine tipo poster\n"
        "/glitch <testo> – genera una GIF glitch animata"
    )

@bot.message_handler(commands=["fase"])
def cmd_fase(message):
    args = message.text.split(maxsplit=1)
    if len(args) == 1:
        bot.reply_to(message, f"Fase corrente: {PHASE.get(_ns(message), 'IT')}")
        return
    val = args[1].strip().lower()
    if val in ("it", "en"):
        PHASE[_ns(message)] = val.upper()
        bot.reply_to(message, f"Fase impostata: {val.upper()}")
    else:
        bot.reply_to(message, "Usa: /fase it oppure /fase en")

@bot.message_handler(commands=["ricorda", "memorizza"])
def cmd_ricorda(message):
    text = message.text.split(" ", 1)
    if len(text) < 2 or not text[1].strip():
        bot.reply_to(message, "Scrivi qualcosa dopo /ricorda")
        return
    content = text[1].strip()
    try:
        pinecone_upsert(_ns(message), content)
        bot.reply_to(message, "Memorizzato ✅")
    except Exception as e:
        bot.reply_to(message, f"⚠️ Errore memorizzazione: {e}")

@bot.message_handler(commands=["cerca"])
def cmd_cerca(message):
    text = message.text.split(" ", 1)
    if len(text) < 2 or not text[1].strip():
        bot.reply_to(message, "Scrivi una query dopo /cerca")
        return
    query = text[1].strip()
    try:
        results = pinecone_query(_ns(message), query, top_k=3)
    except Exception as e:
        bot.reply_to(message, f"⚠️ Errore ricerca: {e}")
        return

    if not results:
        bot.reply_to(message, "Nessun risultato.")
        return

    above = [(t, s) for t, s in results if s >= SCORE_MIN]
    show = above if above else results[:1]
    lines = [f"• {t}\n(score: {s:.3f})" for t, s in show]
    if not above:
        lines.append("\n_(no match ≥ threshold; showing best available)_")
    bot.reply_to(message, "\n\n".join(lines), parse_mode="Markdown")

@bot.message_handler(commands=["svuota"])
def cmd_svuota(message):
    try:
        index.delete(delete_all=True, namespace=_ns(message))
        bot.reply_to(message, "Memoria di questa chat svuotata ✅")
    except Exception as e:
        bot.reply_to(message, f"⚠️ Errore svuota: {e}")

@bot.message_handler(commands=["bozza"])
def cmd_bozza(message):
    text = message.text.split(" ", 1)
    if len(text) < 2 or not text[1].strip():
        bot.reply_to(message, "Scrivi il brief dopo /bozza")
        return
    brief = text[1].strip()
    phase = PHASE.get(_ns(message), "IT")
    try:
        out = rag_text(_ns(message), brief, phase)
        bot.reply_to(message, out[:4000])  # Telegram ~4096 char
    except Exception as e:
        bot.reply_to(message, f"⚠️ Errore bozza: {e}")

@bot.message_handler(commands=["gif"])
def cmd_gif(message):
    text = message.text.split(" ", 1)
    if len(text) < 2 or not text[1].strip():
        bot.reply_to(message, "Prompt mancante. Usa: /gif <prompt>")
        return
    prompt = text[1].strip()
    bot.send_chat_action(message.chat.id, "upload_photo")
    try:
        png_bytes = gen_image_png(prompt + ". Stile manifesto 2Peak, minimal, high contrast, bold typography.")
        bot.send_photo(message.chat.id, io.BytesIO(png_bytes))
        bot.reply_to(message, "Immagine pronta.")
    except Exception as e:
        bot.reply_to(message, friendly_image_error(e))

@bot.message_handler(commands=["glitch"])
def cmd_glitch(message):
    text = message.text.split(" ", 1)
    if len(text) < 2 or not text[1].strip():
        bot.reply_to(message, "Testo mancante. Usa: /glitch <testo>")
        return
    prompt = text[1].strip()
    bot.send_chat_action(message.chat.id, "upload_document")
    try:
        gif_bytes = gen_glitch_gif(prompt)
        bot.send_document(message.chat.id, ("glitch.gif", io.BytesIO(gif_bytes)))
        bot.reply_to(message, "Glitch pronto.")
    except Exception as e:
        bot.reply_to(message, friendly_image_error(e))


# =========================
# AVVIO LOCALE
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
