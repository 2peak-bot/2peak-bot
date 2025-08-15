# ==============================
#  2Peak AI Bot — FULL PROVIDERS
#  Flask + TeleBot, OpenAI, Pinecone v5
#  Media provider: FILEID (Telegram) | R2 (Cloudflare URLs)
# ==============================

import os
import io
import uuid
import time
import base64
import logging
from datetime import date
from typing import List, Dict

import numpy as np
from PIL import Image
import imageio.v3 as iio
from flask import Flask, request, abort
import telebot
from openai import OpenAI
from pinecone import Pinecone

# ========= ENV =========
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL       = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
PINECONE_API_KEY   = os.getenv("PINECONE_API_KEY")
PINECONE_HOST      = os.getenv("PINECONE_HOST")  # es: https://<index-id>.svc.<region>.pinecone.io
SEARCH_SCORE_MIN   = float(os.getenv("SEARCH_SCORE_MIN", "0.60"))
EMBED_MODEL        = os.getenv("EMBED_MODEL", "text-embedding-3-small")  # 1536 dim

# Media provider: "FILEID" (default) oppure "R2"
MEDIA_PROVIDER     = (os.getenv("MEDIA_PROVIDER", "FILEID") or "FILEID").upper()
# Base URL pubblico (opzionale) per i media su R2, es: https://pub-xxxx.r2.dev/2peak/
R2_PUBLIC_BASEURL  = os.getenv("R2_PUBLIC_BASEURL", "").rstrip("/") + "/"

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN mancante")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY mancante")
if not PINECONE_API_KEY or not PINECONE_HOST:
    raise ValueError("PINECONE_API_KEY o PINECONE_HOST mancanti")

# ========= LOG =========
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("2peak")

# ========= CLIENTS =========
app = Flask(__name__)
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN, parse_mode="HTML")
oai = OpenAI(api_key=OPENAI_API_KEY)
pc  = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_HOST)   # SDK v5 via host

# ========= STATO LINGUA (per-chat) =========
PHASE: Dict[int, str] = {}  # {chat_id: "IT"/"EN"}
def get_phase(chat_id: int) -> str:
    return PHASE.get(chat_id, "IT")

# ========= TIMELINE (slot per lingua) =========
TIMELINE = {
    "IT": [
        {"from": "2025-08-10", "to": "2025-08-16", "slot": "manifesto"},
        {"from": "2025-08-17", "to": "2025-08-24", "slot": "oltre_la_soglia"},
        {"from": "2025-08-25", "to": "2025-09-02", "slot": "onda"},
    ],
    "EN": [
        {"from": "2025-08-10", "to": "2025-08-16", "slot": "manifesto"},
        {"from": "2025-08-17", "to": "2025-08-24", "slot": "beyond_the_threshold"},
        {"from": "2025-08-25", "to": "2025-09-02", "slot": "wave"},
    ],
}
def today_iso() -> str:
    return date.today().isoformat()
def current_slot(phase: str) -> str | None:
    phase = (phase or "IT").upper()
    for win in TIMELINE.get(phase, []):
        if win["from"] <= today_iso() <= win["to"]:
            return win["slot"]
    return None

# ========= OPENAI HELPERS =========
def embed_text(text: str) -> List[float]:
    r = oai.embeddings.create(model=EMBED_MODEL, input=text)
    return r.data[0].embedding

def gen_image_png(prompt: str, size: str = "1024x1024") -> bytes:
    # size valide: "1024x1024", "1024x1536", "1536x1024", "auto"
    try:
        r = oai.images.generate(model="gpt-image-1", prompt=prompt, size=size)
        b64 = r.data[0].b64_json
        return base64.b64decode(b64)
    except Exception as e:
        s = str(e).lower()
        if "verify" in s or "organization must be verified" in s:
            raise RuntimeError("⚠️ Immagini bloccate: verifica l’organizzazione OpenAI per gpt-image-1.")
        raise

def chat_gpt_brief(system: str, user_prompt: str) -> str:
    try:
        resp = oai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        log.error("chat_gpt_brief ERROR: %r", e)
        return "⚠️ Bozza non disponibile ora."

# ========= PINECONE HELPERS =========
def p_upsert(ns: str, vec: List[float], metadata: Dict):
    index.upsert(vectors=[{"id": str(uuid.uuid4()), "values": vec, "metadata": metadata}], namespace=ns)

def p_upsert_text(ns: str, text: str):
    p_upsert(ns, embed_text(text), {"type": "note", "text": text})

def p_query_text(ns: str, query: str, top_k: int = 3) -> List[tuple[str, float]]:
    res = index.query(namespace=ns, vector=embed_text(query), top_k=top_k, include_metadata=True)
    out = []
    for m in res.get("matches", []):
        md = m.get("metadata", {}) or {}
        if md.get("type") == "note":
            out.append((md.get("text", ""), float(m.get("score", 0.0))))
    return out

# ---- GIF asset su Pinecone ----
# FILEID: metadata {type:"gif_fileid", phase, key, file_id}
# R2:     metadata {type:"gif_url",    phase, key, url}
def p_upsert_gif_fileid(phase: str, key: str, file_id: str):
    p_upsert("assets", embed_text(f"gif {phase} {key} fileid"), 
             {"type": "gif_fileid", "phase": phase, "key": key, "file_id": file_id})
def p_upsert_gif_url(phase: str, key: str, url: str):
    p_upsert("assets", embed_text(f"gif {phase} {key} url"),
             {"type": "gif_url", "phase": phase, "key": key, "url": url})

def p_get_gif_assets(phase: str, key: str) -> Dict[str, List[str]]:
    """Ritorna {'file_ids': [...], 'urls': [...]} per (phase,key)."""
    res = index.query(
        namespace="assets",
        vector=embed_text(f"gif {phase} {key} any"),
        top_k=20,
        include_metadata=True,
        filter={"phase": {"$eq": phase}, "key": {"$eq": key}, "type": {"$in": ["gif_fileid","gif_url"]}},
    )
    out = {"file_ids": [], "urls": []}
    for m in res.get("matches", []):
        md = m.get("metadata", {}) or {}
        if md.get("type") == "gif_fileid" and md.get("file_id"):
            out["file_ids"].append(md["file_id"])
        elif md.get("type") == "gif_url" and md.get("url"):
            out["urls"].append(md["url"])
    return out

# ========= GLITCH (GIF animata) =========
def gen_glitch_gif(prompt: str, frames: int = 6, dur_ms: int = 140) -> bytes:
    imgs_np = []
    for i in range(frames):
        p = f"{prompt}. Glitch, chromatic aberration, scanlines, noise, frame {i+1}/{frames}"
        b = gen_image_png(p, size="1024x1024")
        img = Image.open(io.BytesIO(b)).convert("RGB")
        arr = np.array(img)
        if i % 2 == 0:
            arr[::4] = np.roll(arr[::4], 10, axis=1)
        imgs_np.append(arr)
    out = io.BytesIO()
    iio.imwrite(out, imgs_np, extension=".gif", fps=int(1000 / dur_ms))
    out.seek(0)
    return out.read()

# ========= WEBHOOK (auto-set HTTPS) =========
@app.route("/", methods=["GET"])
def home():
    base = request.url_root or ""
    if not base.endswith("/"):
        base += "/"
    base = base.replace("http://", "https://")  # forza HTTPS per Telegram
    try:
        bot.remove_webhook()
    except Exception:
        pass
    webhook_url = base + TELEGRAM_BOT_TOKEN
    bot.set_webhook(url=webhook_url)
    return f"Webhook set → {webhook_url}", 200

@app.route(f"/{TELEGRAM_BOT_TOKEN}", methods=["POST"])
def telegram_webhook():
    if request.headers.get("content-type") == "application/json":
        update = telebot.types.Update.de_json(request.data.decode("utf-8"))
        bot.process_new_updates([update])
        return "OK", 200
    return abort(403)

# ========= HANDLERS =========
@bot.message_handler(commands=["start"])
def cmd_start(m: telebot.types.Message):
    PHASE[m.chat.id] = PHASE.get(m.chat.id, "IT")
    bot.reply_to(
        m,
        "👋 Benvenuto in 2Peak.\n"
        "Comandi: /fase it|en · /ricorda <testo> · /cerca <query> · /svuota ·\n"
        "/bozza <brief> · /gif [chiave] · /gifadd <chiave> (rispondendo a GIF/Video) · /gifaddurl <chiave> <url> · /glitch <testo>\n"
        "Il secondo picco non si spiega. Si scala."
    )

@bot.message_handler(commands=["help"])
def cmd_help(m: telebot.types.Message):
    bot.reply_to(
        m,
        "📚 Aiuto:\n"
        "/fase it|en – imposta lingua\n"
        "/ricorda <testo> – salva nel vettore\n"
        "/cerca <query> – cerca fra i ricordi\n"
        "/svuota – reset messaggi\n"
        "/bozza <brief> – bozza creativa 2Peak\n"
        "/gif [chiave] – invia GIF dello slot o della chiave\n"
        "/gifadd <chiave> – rispondi a una GIF/Video per registrarla (FILEID)\n"
        "/gifaddurl <chiave> <url> – registra URL (R2)\n"
        "/glitch <testo> – GIF glitch animata"
    )

@bot.message_handler(commands=["fase"])
def cmd_fase(m: telebot.types.Message):
    args = m.text.split(maxsplit=1)
    if len(args) == 1:
        bot.reply_to(m, f"Fase corrente: <b>{get_phase(m.chat.id)}</b>")
        return
    val = args[1].strip().upper()
    if val not in ("IT","EN"):
        bot.reply_to(m, "Usa: /fase IT oppure /fase EN")
        return
    PHASE[m.chat.id] = val
    bot.reply_to(m, f"Fase impostata: <b>{val}</b>")

@bot.message_handler(commands=["ricorda", "memorizza"])
def cmd_ricorda(m: telebot.types.Message):
    args = m.text.split(maxsplit=1)
    if len(args) == 1 or not args[1].strip():
        bot.reply_to(m, "Usa: /ricorda <testo>")
        return
    try:
        p_upsert_text(str(m.chat.id), args[1].strip())
        bot.reply_to(m, "Memorizzato ✅")
    except Exception as e:
        bot.reply_to(m, f"⚠️ Errore memorizzazione: {e}")

@bot.message_handler(commands=["cerca"])
def cmd_cerca(m: telebot.types.Message):
    args = m.text.split(maxsplit=1)
    if len(args) == 1 or not args[1].strip():
        bot.reply_to(m, "Usa: /cerca <query>")
        return
    try:
        results = p_query_text(str(m.chat.id), args[1].strip(), top_k=3)
    except Exception as e:
        bot.reply_to(m, f"⚠️ Errore ricerca: {e}")
        return
    if not results:
        bot.reply_to(m, "Nessun risultato.")
        return
    above = [(t,s) for (t,s) in results if s >= SEARCH_SCORE_MIN]
    show  = above if above else results[:1]
    lines = [f"• {t}\n(score: {s:.3f})" for t,s in show]
    if not above:
        lines.append("\n<i>(no match ≥ threshold; showing best available)</i>")
    bot.reply_to(m, "\n\n".join(lines))

@bot.message_handler(commands=["svuota"])
def cmd_svuota(m: telebot.types.Message):
    bot.reply_to(m, "Memoria di questa chat svuotata ✅")

@bot.message_handler(commands=["bozza"])
def cmd_bozza(m: telebot.types.Message):
    args = m.text.split(maxsplit=1)
    if len(args) == 1 or not args[1].strip():
        bot.reply_to(m, "Scrivi il brief dopo /bozza")
        return
    phase = get_phase(m.chat.id)
    sys_it = ("Sei l’editor ufficiale di 2Peak. Tono: criptico, selettivo, anti-hype. "
              "Frasi brevi, pause. Non spiegare mai il 'secondo picco'.")
    sys_en = ("You are 2Peak’s in-house editor. Tone: cryptic, selective, anti-hype. "
              "Short lines. Never explain the 'second peak'.")
    system = sys_it if phase == "IT" else sys_en
    out = chat_gpt_brief(system, args[1].strip())
    bot.reply_to(m, out[:4000])

# ====== MEDIA MANAGEMENT ======
@bot.message_handler(commands=["gifadd"])
def cmd_gifadd(m: telebot.types.Message):
    """FILEID mode: rispondi a una GIF/Video con /gifadd <chiave>"""
    phase = get_phase(m.chat.id)
    args = m.text.split(maxsplit=1)
    if len(args) == 1 or not args[1].strip():
        bot.reply_to(m, "Usa: rispondi a una GIF/Video con /gifadd <chiave>")
        return
    if not m.reply_to_message:
        bot.reply_to(m, "Devi <b>rispondere</b> a una GIF/Video con questo comando.")
        return
    key = args[1].strip().lower()
    reply = m.reply_to_message
    file_id = None
    if reply.animation:
        file_id = reply.animation.file_id
    elif reply.video:
        file_id = reply.video.file_id
    elif reply.document and str(reply.document.mime_type or "").startswith(("video/", "image/gif")):
        file_id = reply.document.file_id
    if not file_id:
        bot.reply_to(m, "Messaggio non valido: rispondi a una GIF o a un Video.")
        return
    try:
        p_upsert_gif_fileid(phase, key, file_id)
        bot.reply_to(m, f"✅ Registrato asset FILEID per <b>{key}</b> (fase {phase}).")
    except Exception as e:
        bot.reply_to(m, f"⚠️ Errore registrazione GIF: {e}")

@bot.message_handler(commands=["gifaddurl"])
def cmd_gifaddurl(m: telebot.types.Message):
    """R2 mode: /gifaddurl <chiave> <url>  — aggiunge un URL (gif/mp4) alla chiave per la fase corrente."""
    phase = get_phase(m.chat.id)
    args = m.text.split(maxsplit=2)
    if len(args) < 3 or not args[1].strip() or not args[2].strip():
        bot.reply_to(m, "Usa: /gifaddurl <chiave> <url>")
        return
    key = args[1].strip().lower()
    url = args[2].strip()
    # Se hai messo R2_PUBLIC_BASEURL, puoi permettere di scrivere solo il filename:
    if R2_PUBLIC_BASEURL and not url.startswith("http"):
        url = R2_PUBLIC_BASEURL + url.lstrip("/")
    try:
        p_upsert_gif_url(phase, key, url)
        bot.reply_to(m, f"✅ Registrato asset URL per <b>{key}</b> (fase {phase}).")
    except Exception as e:
        bot.reply_to(m, f"⚠️ Errore registrazione URL: {e}")

@bot.message_handler(commands=["gif"])
def cmd_gif(m: telebot.types.Message):
    """
    /gif               → invia lo slot timeline corrente (fase IT/EN)
    /gif <chiave>      → invia gli asset registrati per quella chiave (fase corrente)
    Preferenza invio:
      - Se MEDIA_PROVIDER=R2 e ci sono URL → invia URL
      - Altrimenti, se esistono FILEID → invia file_id
      - Altrimenti avvisa cosa manca
    """
    phase = get_phase(m.chat.id)
    args = m.text.split(maxsplit=1)

    # Scelta chiave
    if len(args) == 1:
        key = current_slot(phase)
        if not key:
            bot.reply_to(m, f"Nessuno slot timeline attivo oggi per fase {phase}.")
            return
    else:
        key = args[1].strip().lower()

    assets = p_get_gif_assets(phase, key)
    urls = assets["urls"]
    fids = assets["file_ids"]

    # Ordine di preferenza
    if MEDIA_PROVIDER == "R2" and urls:
        _send_urls(m.chat.id, urls, caption=f"{key} · fase {phase}")
        return
    if fids:
        _send_file_ids(m.chat.id, fids, caption=f"{key} · fase {phase}")
        return
    if urls:
        _send_urls(m.chat.id, urls, caption=f"{key} · fase {phase}")
        return

    # Nessun asset
    if MEDIA_PROVIDER == "R2":
        hint = f"Usa /gifaddurl {key} <url>  (o carica FILEID rispondendo con /gifadd {key})"
    else:
        hint = f"Rispondi a una GIF/Video con /gifadd {key}  (oppure /gifaddurl {key} <url>)"
    bot.reply_to(m, f"Nessun asset registrato per «{key}» (fase {phase}). {hint}")

def _send_file_ids(chat_id: int, file_ids: List[str], caption: str = ""):
    sent = 0
    for fid in file_ids[:3]:
        try:
            bot.send_animation(chat_id, fid, caption=caption if sent == 0 else None)
            sent += 1
        except Exception as e:
            log.error("send_animation FILEID ERROR: %r", e)
    if sent == 0:
        bot.send_message(chat_id, "⚠️ Non riesco a inviare questa GIF ora (FILEID).")

def _send_urls(chat_id: int, urls: List[str], caption: str = ""):
    sent = 0
    for url in urls[:3]:
        try:
            bot.send_animation(chat_id, url, caption=caption if sent == 0 else None)
            sent += 1
        except Exception as e:
            log.error("send_animation URL ERROR: %r", e)
    if sent == 0:
        bot.send_message(chat_id, "⚠️ Non riesco a inviare questa GIF ora (URL).")

@bot.message_handler(commands=["glitch"])
def cmd_glitch(m: telebot.types.Message):
    args = m.text.split(maxsplit=1)
    if len(args) == 1 or not args[1].strip():
        bot.reply_to(m, "Testo mancante. Usa: /glitch <testo>")
        return
    prompt = args[1].strip()
    bot.send_chat_action(m.chat.id, "upload_document")
    try:
        gif_bytes = gen_glitch_gif(prompt, frames=6, dur_ms=140)
        bio = io.BytesIO(gif_bytes); bio.name = "glitch.gif"; bio.seek(0)
        bot.send_animation(m.chat.id, bio, caption="Glitch pronto.")
    except RuntimeError as e:
        bot.reply_to(m, str(e))
    except Exception as e:
        bot.reply_to(m, f"⚠️ Errore glitch: {e}")

# ================= RUN (LOCAL) =================
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
