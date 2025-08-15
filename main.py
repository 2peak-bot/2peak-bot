import os
import time
import json
import logging
from datetime import datetime
from uuid import uuid4

import telebot
from telebot import types
from flask import Flask, request

# LLM & embeddings
from openai import OpenAI

# Pinecone 5.x (opzionale ma supportato)
try:
    from pinecone import Pinecone
    HAVE_PC = True
except Exception:
    HAVE_PC = False

# Scheduler giornaliero
from apscheduler.schedulers.background import BackgroundScheduler
import pytz


# ------------- Config di base -------------
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small")

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
PINECONE_HOST = os.environ.get("PINECONE_HOST", "")  # es. "your-index-yourproject.svc.aped-xxx.pinecone.io"
SCORE_MIN = float(os.environ.get("SEARCH_SCORE_MIN", "0.60"))

# Timezone per lo scheduler
TZ_NAME = os.environ.get("TZ", "Europe/Rome")
tz = pytz.timezone(TZ_NAME)

# URL base per il webhook (Render lo espone in RENDER_EXTERNAL_URL)
WEBHOOK_BASE = os.environ.get("WEBHOOK_BASE") or os.environ.get("RENDER_EXTERNAL_URL")
AUTO_SET_WEBHOOK = os.environ.get("AUTO_SET_WEBHOOK", "true").lower() == "true"

# GIF/asset – puoi sovrascrivere via ENV su Render
GIF_MANIFESTO_URL = os.environ.get("GIF_MANIFESTO_URL", "")
GIF_OLTRE_URL     = os.environ.get("GIF_OLTRE_URL", "")
GIF_ONDA_URL      = os.environ.get("GIF_ONDA_URL", "")

GLITCH_MANIFESTO_URL = os.environ.get("GLITCH_MANIFESTO_URL", GIF_MANIFESTO_URL)
GLITCH_OLTRE_URL     = os.environ.get("GLITCH_OLTRE_URL", GIF_OLTRE_URL)
GLITCH_ONDA_URL      = os.environ.get("GLITCH_ONDA_URL", GIF_ONDA_URL)

# ------------- Client e oggetti globali -------------
bot = telebot.TeleBot(TOKEN, parse_mode="HTML")
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

llm = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

pc = None
if HAVE_PC and PINECONE_API_KEY and PINECONE_HOST:
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(host=PINECONE_HOST)
    except Exception as e:
        logging.error(f"Pinecone init error: {e}")
        pc = None
else:
    index = None

# Stato utente (fase + altro)
user_state = {}  # chat_id -> {"fase": "IT"|"EN"}

# Scheduler
scheduler = BackgroundScheduler(timezone=tz)
scheduler.start()
scheduled_jobs = {}   # chat_id -> job_id
last_sent = {}        # chat_id -> last bot message_id


# ------------- Helper UI -------------
def main_menu():
    kb = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=3)
    kb.add(
        types.KeyboardButton("/fase IT"),
        types.KeyboardButton("/fase EN"),
        types.KeyboardButton("/bozza"),
    )
    kb.add(
        types.KeyboardButton("/ricorda"),
        types.KeyboardButton("/cerca"),
        types.KeyboardButton("/svuota"),
    )
    kb.add(
        types.KeyboardButton("/post manifesto"),
        types.KeyboardButton("/post oltre"),
        types.KeyboardButton("/post onda"),
    )
    kb.add(
        types.KeyboardButton("/schedule 09:00 manifesto"),
        types.KeyboardButton("/pinlast"),
    )
    return kb


def get_fase(chat_id: int) -> str:
    return user_state.get(chat_id, {}).get("fase", "IT")


def set_fase(chat_id: int, fase: str):
    user_state.setdefault(chat_id, {})["fase"] = fase.upper()


def send_gif_with_caption(chat_id: int, url: str, caption: str):
    """Invia una GIF con didascalia e ricorda l’ultimo messaggio per /pinlast."""
    msg = bot.send_animation(chat_id=chat_id, animation=url, caption=caption)
    last_sent[chat_id] = msg.message_id
    return msg


def embed_text(text: str):
    if not llm:
        raise RuntimeError("OPENAI_API_KEY mancante")
    v = llm.embeddings.create(model=EMBED_MODEL, input=text)
    return v.data[0].embedding


# ------------- Comandi base -------------
@bot.message_handler(commands=['start'])
def cmd_start(message: types.Message):
    chat_id = message.chat.id
    set_fase(chat_id, get_fase(chat_id))  # default IT
    bot.reply_to(
        message,
        "👋 Benvenuto in <b>2Peak</b>.\n"
        "Comandi rapidi nel menu.\n\n"
        "<b/Memoria</b>\n"
        "• /ricorda <testo>\n"
        "• /cerca <query>\n\n"
        "<b>Media</b>\n"
        "• /post <manifesto|oltre|onda>\n"
        "• /schedule <HH:MM> <chiave>\n"
        "• /unschedule\n"
        "• /pinlast\n\n"
        "<b>Fase</b>\n"
        "• /fase IT  • /fase EN\n\n"
        "Il secondo picco non si spiega. Si scala.",
        reply_markup=main_menu()
    )


@bot.message_handler(commands=['fase'])
def cmd_fase(message: types.Message):
    chat_id = message.chat.id
    parts = message.text.strip().split()
    if len(parts) == 1:
        bot.reply_to(message, "Usa: /fase IT oppure /fase EN")
        return
    fase = parts[1].upper()
    if fase not in ("IT", "EN"):
        bot.reply_to(message, "Valori ammessi: IT, EN")
        return
    set_fase(chat_id, fase)
    bot.reply_to(message, f"Fase impostata: <b>{fase}</b>")


# ------------- Bozza (copy creativo) -------------
@bot.message_handler(commands=['bozza'])
def cmd_bozza(message: types.Message):
    if not llm:
        bot.reply_to(message, "OPENAI_API_KEY mancante.")
        return

    parts = message.text.split(maxsplit=1)
    prompt = parts[1] if len(parts) > 1 else "scrivi un tweet evocativo su 2Peak"
    fase = get_fase(message.chat.id)
    sys = "Sei un copywriter di 2Peak. Tono breve, evocativo, senza hashtag tranne #2Peak quando serve."
    if fase == "EN":
        sys = "You are 2Peak's copywriter. Short, evocative tone. No hashtags unless #2Peak fits."

    try:
        resp = llm.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=120
        )
        text = resp.choices[0].message.content.strip()
        bot.reply_to(message, text)
    except Exception as e:
        bot.reply_to(message, f"⚠️ Errore generazione: {e}")


# ------------- Memoria (Pinecone) -------------
def pinecone_ready():
    return (pc is not None) and (index is not None) and bool(PINECONE_HOST)


@bot.message_handler(commands=['ricorda'])
def cmd_ricorda(message: types.Message):
    parts = message.text.split(maxsplit=1)
    if len(parts) < 2:
        bot.reply_to(message, "Usa: /ricorda <testo>")
        return
    text = parts[1].strip()
    if not pinecone_ready():
        bot.reply_to(message, "Memoria disabilitata (Pinecone non configurato).")
        return

    try:
        vec = embed_text(text)
        ns = str(message.chat.id)
        vid = f"{int(time.time())}-{uuid4().hex[:6]}"
        index.upsert(
            vectors=[{"id": vid, "values": vec, "metadata": {"text": text}}],
            namespace=ns
        )
        bot.reply_to(message, "Memorizzato ✅")
    except Exception as e:
        bot.reply_to(message, f"Errore memorizzazione: {e}")


@bot.message_handler(commands=['cerca'])
def cmd_cerca(message: types.Message):
    parts = message.text.split(maxsplit=1)
    if len(parts) < 2:
        bot.reply_to(message, "Usa: /cerca <query>")
        return
    query = parts[1].strip()
    if not pinecone_ready():
        bot.reply_to(message, "Ricerca disabilitata (Pinecone non configurato).")
        return

    try:
        qv = embed_text(query)
        ns = str(message.chat.id)
        res = index.query(vector=qv, namespace=ns, top_k=3, include_values=False, include_metadata=True)
        out = []
        above = False
        for m in res["matches"]:
            score = m["score"]
            txt = (m.get("metadata") or {}).get("text", "")
            out.append(f"• {txt}\n(score: {round(score, 3)})")
            if score >= SCORE_MIN:
                above = True
        if not out:
            bot.reply_to(message, "Nessun risultato.")
            return
        if not above:
            out.append("\n_(no match ≥ threshold; showing best available)_")
        bot.reply_to(message, "\n\n".join(out))
    except Exception as e:
        bot.reply_to(message, f"Errore ricerca: {e}")


@bot.message_handler(commands=['svuota'])
def cmd_svuota(message: types.Message):
    if not pinecone_ready():
        bot.reply_to(message, "Pinecone non configurato.")
        return
    try:
        ns = str(message.chat.id)
        index.delete(namespace=ns, delete_all=True)
        bot.reply_to(message, "Memoria di questa chat svuotata ✅")
    except Exception as e:
        bot.reply_to(message, f"Errore svuota: {e}")


# ------------- Post: GIF + caption in un unico messaggio -------------
CAPTIONS_IT = {
    "manifesto": "Il secondo picco non si spiega. Si scala.",
    "oltre": "Oltre la soglia.",
    "onda": "Porta l’onda più in alto."
}
CAPTIONS_EN = {
    "manifesto": "The second peak can’t be explained. You climb it.",
    "oltre": "Beyond the threshold.",
    "onda": "Push the wave higher."
}

# mappe GIF
GIFS = {
    "manifesto": GIF_MANIFESTO_URL,
    "oltre": GIF_OLTRE_URL,
    "onda": GIF_ONDA_URL
}
GLITCHES = {
    "manifesto": GLITCH_MANIFESTO_URL,
    "oltre": GLITCH_OLTRE_URL,
    "onda": GLITCH_ONDA_URL
}


def caption_for(key: str, fase: str) -> str:
    return (CAPTIONS_EN if fase.upper() == "EN" else CAPTIONS_IT).get(key, "")


@bot.message_handler(commands=['post'])
def cmd_post(message: types.Message):
    parts = message.text.split(maxsplit=1)
    if len(parts) < 2:
        bot.reply_to(message, "Usa: /post <manifesto|oltre|onda>")
        return
    key = parts[1].strip().lower()
    url = GIFS.get(key)
    if not url:
        bot.reply_to(message, "Chiave non valida. Usa: manifesto, oltre, onda.")
        return
    fase = get_fase(message.chat.id)
    try:
        send_gif_with_caption(message.chat.id, url, caption_for(key, fase))
        bot.reply_to(message, f"✅ Post «{key}» inviato (fase {fase}).")
    except Exception as e:
        bot.reply_to(message, f"⚠️ Errore invio GIF: {e}")


# Compatibilità: /gif e /glitch (usano le stesse mappe)
@bot.message_handler(commands=['gif'])
def cmd_gif(message: types.Message):
    parts = message.text.split(maxsplit=1)
    if len(parts) < 2:
        bot.reply_to(message, "Usa: /gif <manifesto|oltre|onda>")
        return
    key = parts[1].strip().lower()
    url = GIFS.get(key)
    if not url:
        bot.reply_to(message, "Chiave non valida. Usa: manifesto, oltre, onda.")
        return
    fase = get_fase(message.chat.id)
    try:
        send_gif_with_caption(message.chat.id, url, caption_for(key, fase))
        bot.reply_to(message, f"✅ GIF «{key}» inviata (fase {fase}).")
    except Exception as e:
        bot.reply_to(message, f"⚠️ Errore invio GIF: {e}")


@bot.message_handler(commands=['glitch'])
def cmd_glitch(message: types.Message):
    parts = message.text.split(maxsplit=1)
    if len(parts) < 2:
        bot.reply_to(message, "Usa: /glitch <manifesto|oltre|onda>")
        return
    key = parts[1].strip().lower()
    url = GLITCHES.get(key)
    if not url:
        bot.reply_to(message, "Chiave non valida. Usa: manifesto, oltre, onda.")
        return
    fase = get_fase(message.chat.id)
    try:
        send_gif_with_caption(message.chat.id, url, caption_for(key, fase))
        bot.reply_to(message, f"✅ Glitch «{key}» inviato (fase {fase}).")
    except Exception as e:
        bot.reply_to(message, f"⚠️ Errore invio glitch: {e}")


# ------------- Scheduler giornaliero (/schedule, /unschedule) -------------
def scheduled_post(chat_id: int, key: str, fase: str):
    url = GIFS.get(key)
    if not url:
        bot.send_message(chat_id, f"⚠️ schedule: chiave non valida ({key})")
        return
    try:
        send_gif_with_caption(chat_id, url, caption_for(key, fase))
        bot.send_message(chat_id, f"🗓️ Post schedulato «{key}» inviato ({fase}).")
    except Exception as e:
        bot.send_message(chat_id, f"⚠️ Errore schedule: {e}")


@bot.message_handler(commands=['schedule'])
def cmd_schedule(message: types.Message):
    parts = message.text.split()
    if len(parts) != 3:
        bot.reply_to(message, "Usa: /schedule <HH:MM> <manifesto|oltre|onda>")
        return
    hhmm, key = parts[1], parts[2].lower()
    try:
        hh, mm = map(int, hhmm.split(":"))
        chat_id = message.chat.id

        # rimuovi job precedente, se presente
        if chat_id in scheduled_jobs:
            try:
                scheduler.remove_job(scheduled_jobs[chat_id])
            except Exception:
                pass

        fase = get_fase(chat_id)
        job = scheduler.add_job(
            scheduled_post,
            'cron',
            hour=hh, minute=mm,
            args=[chat_id, key, fase],
            id=f"job_{chat_id}"
        )
        scheduled_jobs[chat_id] = job.id
        bot.reply_to(message, f"🗓️ Schedulato ogni giorno {hhmm} → «{key}» (fase {fase}, TZ {TZ_NAME}).")
    except Exception as e:
        bot.reply_to(message, f"⚠️ Errore schedule: {e}")


@bot.message_handler(commands=['unschedule'])
def cmd_unschedule(message: types.Message):
    chat_id = message.chat.id
    if chat_id not in scheduled_jobs:
        bot.reply_to(message, "Nessuna schedulazione attiva.")
        return
    try:
        scheduler.remove_job(scheduled_jobs[chat_id])
        del scheduled_jobs[chat_id]
        bot.reply_to(message, "🗑️ Schedulazione rimossa.")
    except Exception as e:
        bot.reply_to(message, f"⚠️ Errore unschedule: {e}")


# ------------- Pin dell’ultimo post -------------
@bot.message_handler(commands=['pinlast'])
def cmd_pinlast(message: types.Message):
    chat_id = message.chat.id
    msg_id = last_sent.get(chat_id)
    if not msg_id:
        bot.reply_to(message, "Non ho un ultimo post da fissare in questa chat.")
        return
    try:
        bot.pin_chat_message(chat_id, msg_id, disable_notification=True)
        bot.reply_to(message, "📌 Post fissato in alto.")
    except Exception as e:
        bot.reply_to(message, f"⚠️ Errore pin: {e}\nAssicurati che il bot sia admin se è canale/gruppo.")


# ------------- Webhook Flask -------------
@app.route("/", methods=["GET"])
def home():
    return "Bot 2Peak attivo 🚀"

@app.route(f"/{TOKEN}", methods=["POST"])
def receive_update():
    try:
        json_str = request.stream.read().decode("utf-8")
        update = telebot.types.Update.de_json(json_str)
        bot.process_new_updates([update])
    except Exception as e:
        logging.exception(e)
    return "OK", 200


def set_webhook_if_needed():
    if not AUTO_SET_WEBHOOK or not WEBHOOK_BASE:
        return
    url = f"{WEBHOOK_BASE.rstrip('/')}/{TOKEN}"
    try:
        r = bot.set_webhook(url=url, drop_pending_updates=True)
        logging.info(f"Webhook set: {r} → {url}")
    except Exception as e:
        logging.error(f"Webhook error: {e}")


# ------------- Avvio -------------
if __name__ == "__main__":
    set_webhook_if_needed()
    # avvio server Flask (Render lo esegue con gunicorn in produzione)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "10000")))
