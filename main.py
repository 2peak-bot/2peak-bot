# 2PEAK AI — FULL BOT (Reasoning + Memory + Media)
# Flask + Telebot + OpenAI + Pinecone (RAG) + Supabase (DB)
# Post unificati (GIF+caption), pinlast, schedule/unschedule (RAM ack), status

import os
import re
import time
import json
import logging
from datetime import datetime
from uuid import uuid4

from flask import Flask, request
import telebot
from telebot import types as t

# --- OpenAI (LLM + Embeddings) ---
from openai import OpenAI

# --- Pinecone (vector store) ---
from pinecone import Pinecone

# --- Supabase (DB metadati, assets, jobs, logs) ---
from supabase import create_client, Client

# ================== ENV / CONFIG ==================
# Telegram
TELEGRAM_BOT_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN")

# Webhook
AUTO_SET_WEBHOOK = (os.getenv("AUTO_SET_WEBHOOK", "true").lower() == "true")
WEBHOOK_BASE = (os.getenv("WEBHOOK_BASE") or os.getenv("RENDER_EXTERNAL_URL") or "").strip()
if WEBHOOK_BASE and not WEBHOOK_BASE.startswith("https://"):
    WEBHOOK_BASE = ""  # Telegram richiede HTTPS

# Fase lingua & fuso
PHASE_DEFAULT = (os.getenv("PHASE", "IT") or "IT").upper()  # IT | EN
TZ = os.getenv("TZ", "Europe/Rome")

# OpenAI
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL    = os.getenv("EMBED_MODEL", "text-embedding-3-small")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY")

# Pinecone
PINECONE_API_KEY = (os.getenv("PINECONE_API_KEY") or "").strip()
PINECONE_HOST    = (os.getenv("PINECONE_HOST") or "").strip()
if not PINECONE_API_KEY or not PINECONE_HOST:
    raise RuntimeError("Missing Pinecone config (PINECONE_API_KEY / PINECONE_HOST)")

SEARCH_SCORE_MIN = float(os.getenv("SEARCH_SCORE_MIN", "0.60"))

# Supabase
SUPABASE_URL = (os.getenv("SUPABASE_URL") or "").strip()
SUPABASE_KEY = (os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_SERVICE_KEY") or "").strip()
# Supabase è consigliato (persistenza); se assente, il bot funziona lo stesso ma senza log/asset persistenti

# Media URLs (HTTPS pubblici)
WELCOME_GIF_URL   = (os.getenv("WELCOME_GIF_URL") or "").strip()
GIF_MANIFESTO_URL = (os.getenv("GIF_MANIFESTO_URL") or "").strip()
GIF_OLTRE_URL     = (os.getenv("GIF_OLTRE_URL") or "").strip()
GIF_ONDA_URL      = (os.getenv("GIF_ONDA_URL")  or "").strip()

WELCOME_TEXT_IT = os.getenv("WELCOME_TEXT_IT",
    "👋 Benvenuto in 2Peak.\nComandi: /post manifesto | /post oltre | /post onda · /bozza <brief> · /ricorda <testo> · /cerca <query> · /schedule HH:MM <chiave> · /pinlast · /fase IT|EN")
WELCOME_TEXT_EN = os.getenv("WELCOME_TEXT_EN",
    "👋 Welcome to 2Peak.\nCommands: /post manifesto | /post oltre | /post onda · /bozza <brief> · /ricorda <text> · /cerca <query> · /schedule HH:MM <key> · /pinlast · /fase IT|EN")

# ================== CLIENTS ==================
app = Flask(__name__)
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN, parse_mode="HTML", disable_web_page_preview=True)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("2peak")

oai = OpenAI(api_key=OPENAI_API_KEY)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_HOST)

supabase: Client | None = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        log.info("Supabase client ready.")
    except Exception as e:
        log.error(f"Supabase init error: {e}")
        supabase = None
else:
    log.warning("Supabase not configured (SUPABASE_URL/KEY missing). Running without DB persistence.")

# ================== STATO RAM ==================
CHAT_PHASE: dict[int, str] = {}      # chat_id -> "IT"/"EN"
LAST_BOT_MSG: dict[int, int] = {}    # chat_id -> message_id (per /pinlast)
SCHEDULES: dict[int, dict] = {}      # chat_id -> {"time":"HH:MM","key":"manifesto","phase":"IT"}

# ================== MEDIA/CAPTIONS ==================
MEDIA_URLS = {
    "manifesto": GIF_MANIFESTO_URL,
    "oltre":     GIF_OLTRE_URL,
    "onda":      GIF_ONDA_URL
}
VALID_KEYS = [k for k,v in MEDIA_URLS.items() if v] or ["manifesto","oltre","onda"]  # mostra anche se vuoti
CAPTIONS = {
    "IT": {
        "manifesto": "Il secondo picco non si spiega. Si scala.",
        "oltre":     "Oltre la soglia.",
        "onda":      "Sali l’onda finché non diventa un picco."
    },
    "EN": {
        "manifesto": "The second peak isn’t explained. It’s climbed.",
        "oltre":     "Beyond the threshold.",
        "onda":      "Ride the wave until it becomes a peak."
    }
}

# ================== HELPERS ==================
def phase_of(chat_id: int) -> str:
    return CHAT_PHASE.get(chat_id, PHASE_DEFAULT)

def set_last(chat_id: int, msg):
    if getattr(msg, "message_id", None):
        LAST_BOT_MSG[chat_id] = msg.message_id

def caption_for(key: str, phase: str) -> str:
    return (CAPTIONS.get(phase) or CAPTIONS["IT"]).get(key, "")

def embed_text(text: str) -> list[float]:
    r = oai.embeddings.create(model=EMBED_MODEL, input=text)
    return r.data[0].embedding

def pc_upsert_text(ns: str, text: str) -> str:
    vec = embed_text(text)
    vid = f"{int(time.time())}-{uuid4().hex[:6]}"
    index.upsert(vectors=[{"id": vid, "values": vec, "metadata": {"text": text}}], namespace=ns)
    return vid

def pc_query(ns: str, query: str, top_k: int = 3):
    qv = embed_text(query)
    res = index.query(vector=qv, namespace=ns, top_k=top_k, include_values=False, include_metadata=True)
    return res.get("matches", [])

def db_log(kind: str, chat_id: int, payload: dict):
    if not supabase: return
    try:
        supabase.table("logs").insert({"kind": kind, "chat_id": chat_id, "payload": payload}).execute()
    except Exception as e:
        log.error(f"DB log error: {e}")

def db_upsert_user(chat_id: int, username: str | None, phase: str):
    if not supabase: return
    try:
        supabase.table("users").upsert(
            {"chat_id": chat_id, "username": username or "", "phase": phase, "tz": TZ},
            on_conflict="chat_id"
        ).execute()
    except Exception as e:
        log.error(f"DB upsert user error: {e}")

# ================== CORE SENDER ==================
def send_gif_with_caption(chat_id: int, key: str, phase: str):
    url = (MEDIA_URLS.get(key) or "").strip()
    if not url:
        return bot.send_message(chat_id, f"⚠️ Media non configurato per «{key}». Aggiungi URL su Render.", disable_web_page_preview=True)
    cap = caption_for(key, phase)
    try:
        msg = bot.send_animation(chat_id, url, caption=cap, disable_notification=True)
        set_last(chat_id, msg)
        db_log("post", chat_id, {"key": key, "phase": phase, "url": url})
        return msg
    except Exception as e:
        return bot.send_message(chat_id, f"⚠️ Errore invio media «{key}»: <code>{e}</code>")

# ================== HANDLERS ==================
@bot.message_handler(commands=["start"])
def cmd_start(m: t.Message):
    chat_id = m.chat.id
    phase = phase_of(chat_id)
    db_upsert_user(chat_id, getattr(m.from_user, "username", None), phase)

    text = WELCOME_TEXT_IT if phase == "IT" else WELCOME_TEXT_EN
    sent = None
    if WELCOME_GIF_URL:
        try:
            sent = bot.send_animation(chat_id, WELCOME_GIF_URL, caption=text, disable_notification=True)
        except Exception as e:
            app.logger.warning(f"/start GIF failed: {e}")
    if not sent:
        sent = bot.send_message(chat_id, text, disable_web_page_preview=True)
    set_last(chat_id, sent)
    db_log("start", chat_id, {"phase": phase})

@bot.message_handler(commands=["fase"])
def cmd_fase(m: t.Message):
    chat_id = m.chat.id
    parts = m.text.strip().split()
    if len(parts) == 1:
        bot.reply_to(m, f"Fase corrente: {phase_of(chat_id)}")
        return
    val = parts[1].upper()
    if val not in ("IT","EN"):
        bot.reply_to(m, "Usa: /fase IT oppure /fase EN")
        return
    CHAT_PHASE[chat_id] = val
    db_upsert_user(chat_id, getattr(m.from_user, "username", None), val)
    bot.reply_to(m, f"❤️‍🔥 Fase impostata: {val}")

@bot.message_handler(commands=["post"])
def cmd_post(m: t.Message):
    chat_id = m.chat.id
    args = m.text.strip().split(maxsplit=1)
    if len(args) < 2:
        bot.reply_to(m, f"Chiave mancante. Usa: {', '.join(VALID_KEYS)}")
        return
    key = args[1].strip().lower()
    if key not in MEDIA_URLS:
        bot.reply_to(m, f"Chiave non valida. Usa: {', '.join(VALID_KEYS)}")
        return
    send_gif_with_caption(chat_id, key, phase_of(chat_id))

@bot.message_handler(commands=["pinlast"])
def cmd_pinlast(m: t.Message):
    chat_id = m.chat.id
    msg_id = LAST_BOT_MSG.get(chat_id)
    if not msg_id:
        bot.reply_to(m, "Nessun messaggio da fissare. Invia prima un /post.")
        return
    try:
        bot.pin_chat_message(chat_id, msg_id, disable_notification=True)
        bot.reply_to(m, "📌 Ultimo post fissato in alto.")
    except Exception as e:
        bot.reply_to(m, f"⚠️ Impossibile fissare: <code>{e}</code>\nVerifica permessi admin se canale/gruppo.")

@bot.message_handler(commands=["schedule"])
def cmd_schedule(m: t.Message):
    chat_id = m.chat.id
    mm = re.match(r"^/schedule\s+(\d{2}):(\d{2})\s+(\w+)$", m.text.strip(), re.I)
    if not mm:
        bot.reply_to(m, "Usa: /schedule HH:MM <chiave>  (es. /schedule 09:00 manifesto)")
        return
    hh, mn, key = mm.groups()
    key = key.lower()
    if key not in MEDIA_URLS:
        bot.reply_to(m, f"Chiave non valida. Usa: {', '.join(VALID_KEYS)}")
        return
    SCHEDULES[chat_id] = {"time": f"{hh}:{mn}", "key": key, "phase": phase_of(chat_id)}
    bot.reply_to(m, f"🗓️ Schedulato ogni giorno {hh}:{mn} → «{key}» (fase {phase_of(chat_id)}, TZ {TZ}).")
    db_log("schedule_set", chat_id, SCHEDULES[chat_id])

@bot.message_handler(commands=["unschedule"])
def cmd_unschedule(m: t.Message):
    chat_id = m.chat.id
    if chat_id not in SCHEDULES:
        bot.reply_to(m, "Nessuna schedulazione attiva.")
        return
    SCHEDULES.pop(chat_id, None)
    bot.reply_to(m, "🗑️ Schedulazione rimossa.")
    db_log("schedule_unset", chat_id, {})

# ======= Memory (RAG) =======
@bot.message_handler(commands=["ricorda"])
def cmd_ricorda(m: t.Message):
    args = m.text.split(maxsplit=1)
    if len(args) < 2:
        bot.reply_to(m, "Usa: /ricorda <testo>")
        return
    text = args[1].strip()
    ns = str(m.chat.id)
    try:
        vid = pc_upsert_text(ns, text)
        if supabase:
            supabase.table("notes_meta").insert({"chat_id": m.chat.id, "text": text, "pinecone_id": vid}).execute()
        bot.reply_to(m, "Memorizzato ✅")
        db_log("remember", m.chat.id, {"text": text, "pinecone_id": vid})
    except Exception as e:
        bot.reply_to(m, f"⚠️ Errore memorizzazione: <code>{e}</code>")

@bot.message_handler(commands=["cerca"])
def cmd_cerca(m: t.Message):
    args = m.text.split(maxsplit=1)
    if len(args) < 2:
        bot.reply_to(m, "Usa: /cerca <query>")
        return
    query = args[1].strip()
    ns = str(m.chat.id)
    try:
        matches = pc_query(ns, query, top_k=3)
        if not matches:
            bot.reply_to(m, "Nessun risultato.")
            return
        above = []
        for mt in matches:
            score = mt.get("score", 0.0)
            md = mt.get("metadata") or {}
            txt = md.get("text", "")
            if score >= SEARCH_SCORE_MIN:
                above.append((txt, score))
        show = above if above else [(matches[0].get("metadata",{}).get("text",""), matches[0].get("score",0.0))]
        lines = [f"• {t}\n(score: {s:.3f})" for t,s in show]
        if not above:
            lines.append("\n<i>(no match ≥ soglia; mostro il migliore disponibile)</i>")
        bot.reply_to(m, "\n\n".join(lines))
        db_log("search", m.chat.id, {"query": query, "results": show})
    except Exception as e:
        bot.reply_to(m, f"⚠️ Errore ricerca: <code>{e}</code>")

@bot.message_handler(commands=["bozza"])
def cmd_bozza(m: t.Message):
    args = m.text.split(maxsplit=1)
    brief = args[1].strip() if len(args) > 1 else "Scrivi un testo breve stile 2Peak."
    phase = phase_of(m.chat.id)
    sys_it = ("Sei l’editor di 2Peak. Tono: criptico, selettivo, anti-hype. "
              "Frasi brevi, pause. Mai spiegare il 'secondo picco'.")
    sys_en = ("You are 2Peak’s editor. Tone: cryptic, selective, anti-hype. "
              "Short lines. Never explain the 'second peak'.")
    system = sys_it if phase == "IT" else sys_en
    try:
        resp = oai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"system","content":system},{"role":"user","content":brief}],
            temperature=0.8,
            max_tokens=180,
        )
        out = (resp.choices[0].message.content or "").strip()
        bot.reply_to(m, out[:4000])
        db_log("bozza", m.chat.id, {"brief": brief, "out_len": len(out)})
    except Exception as e:
        bot.reply_to(m, f"⚠️ Errore bozza: <code>{e}</code>")

@bot.message_handler(commands=["status"])
def cmd_status(m: t.Message):
    chat_id = m.chat.id
    phase = phase_of(chat_id)
    checks = [
        f"Fase: <b>{phase}</b>",
        f"TZ: <b>{TZ}</b>",
        f"OpenAI: <b>{'ok' if OPENAI_API_KEY else 'no'}</b>",
        f"Pinecone: <b>{'ok' if PINECONE_HOST else 'no'}</b>",
        f"Supabase: <b>{'ok' if supabase else 'no'}</b>",
        "Media:",
        f"• manifesto: {'ok' if MEDIA_URLS.get('manifesto') else '—'}",
        f"• oltre: {'ok' if MEDIA_URLS.get('oltre') else '—'}",
        f"• onda: {'ok' if MEDIA_URLS.get('onda') else '—'}",
    ]
    if chat_id in SCHEDULES:
        s = SCHEDULES[chat_id]
        checks.append(f"Sched: {s['time']} → {s['key']} ({s['phase']})")
    bot.reply_to(m, "\n".join(checks))

# ================== WEBHOOK ==================
@app.route("/", methods=["GET"])
def root():
    return "2Peak AI bot up"

@app.route(f"/{TELEGRAM_BOT_TOKEN}", methods=["POST"])
def hook():
    try:
        update = telebot.types.Update.de_json(request.stream.read().decode("utf-8"))
        bot.process_new_updates([update])
    except Exception as e:
        log.exception(e)
    return "OK", 200

def set_webhook_if_needed():
    if not AUTO_SET_WEBHOOK or not WEBHOOK_BASE:
        log.info("Skipping auto webhook setup.")
        return
    url = f"{WEBHOOK_BASE.rstrip('/')}/{TELEGRAM_BOT_TOKEN}"
    try:
        bot.remove_webhook()
        bot.set_webhook(url=url, drop_pending_updates=True)
        log.info(f"Webhook set: {url}")
    except Exception as e:
        log.error(f"Webhook error: {e}")

# ================== MAIN ==================
if __name__ == "__main__":
    set_webhook_if_needed()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "10000")))
