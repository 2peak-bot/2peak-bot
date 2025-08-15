# ==============================
#  2Peak / 2Pick AI Bot — FULL STACK (Opzione C) + Step 2B
#  Flask + TeleBot, OpenAI (chat/embeddings), Pinecone v5
#  Media: R2 (URL) + Telegram FILEID, glitch locale, menu bottoni, test, batch
#  NEW: /post, /caption, /keys (Timeline Assistant minimo)
# ==============================

import os
import io
import re
import uuid
import time
import base64
import logging
from datetime import date
from typing import List, Dict, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, request, abort
import telebot
from telebot import types as t
from openai import OpenAI
from pinecone import Pinecone

# ========= ENV =========
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL       = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
PINECONE_API_KEY   = os.getenv("PINECONE_API_KEY")
PINECONE_HOST      = os.getenv("PINECONE_HOST")  # es: https://<index-id>.svc.<region>.pinecone.io
SEARCH_SCORE_MIN   = float(os.getenv("SEARCH_SCORE_MIN", "0.60"))
EMBED_MODEL        = os.getenv("EMBED_MODEL", "text-embedding-3-small")  # 1536-dim

# Provider media (mix). Se R2 è configurato, preferisce URL; altrimenti FILEID
MEDIA_PROVIDER     = (os.getenv("MEDIA_PROVIDER", "FILEID") or "FILEID").upper()
R2_PUBLIC_BASEURL  = (os.getenv("R2_PUBLIC_BASEURL", "") or "").rstrip("/")
if R2_PUBLIC_BASEURL:
    R2_PUBLIC_BASEURL += "/"

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN mancante")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY mancante")
if not PINECONE_API_KEY or not PINECONE_HOST:
    raise ValueError("PINECONE_API_KEY o PINECONE_HOST mancanti")

# ========= LOG =========
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("2peak_bot")

# ========= CLIENTS =========
app = Flask(__name__)
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN, parse_mode="HTML")
oai = OpenAI(api_key=OPENAI_API_KEY)
pc  = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_HOST)   # SDK v5 via host

# ========= STATO LINGUA (per chat) =========
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
def current_slot(phase: str) -> Optional[str]:
    ph = (phase or "IT").upper()
    for win in TIMELINE.get(ph, []):
        if win["from"] <= today_iso() <= win["to"]:
            return win["slot"]
    return None

# ========= OPENAI HELPERS =========
def embed_text(text: str) -> List[float]:
    r = oai.embeddings.create(model=EMBED_MODEL, input=text)
    return r.data[0].embedding

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

def p_query_text(ns: str, query: str, top_k: int = 3) -> List[Tuple[str,float]]:
    res = index.query(namespace=ns, vector=embed_text(query), top_k=top_k, include_metadata=True)
    out = []
    for m in res.get("matches", []):
        md = m.get("metadata", {}) or {}
        if md.get("type") == "note":
            out.append((md.get("text", ""), float(m.get("score", 0.0))))
    return out

# ---- Asset GIF/IMG su Pinecone ----
# FILEID: metadata {type:"asset_fileid", phase, key, file_id, media_kind}
# URL:    metadata {type:"asset_url",    phase, key, url,      media_kind}
def p_upsert_fileid(phase: str, key: str, file_id: str, media_kind: str):
    p_upsert("assets",
             embed_text(f"asset {phase} {key} {media_kind} fileid"),
             {"type": "asset_fileid", "phase": phase, "key": key, "file_id": file_id, "media_kind": media_kind})

def p_upsert_url(phase: str, key: str, url: str, media_kind: str):
    p_upsert("assets",
             embed_text(f"asset {phase} {key} {media_kind} url"),
             {"type": "asset_url", "phase": phase, "key": key, "url": url, "media_kind": media_kind})

def p_get_assets(phase: str, key: str) -> Dict[str, List[Dict]]:
    """Ritorna {'fileids': [{'file_id','media_kind'}], 'urls': [{'url','media_kind'}]}"""
    res = index.query(
        namespace="assets",
        vector=embed_text(f"asset {phase} {key} any"),
        top_k=200,
        include_metadata=True,
        filter={"phase": {"$eq": phase}, "key": {"$eq": key}, "type": {"$in": ["asset_fileid","asset_url"]}},
    )
    out = {"fileids": [], "urls": []}
    for m in res.get("matches", []):
        md = m.get("metadata", {}) or {}
        if md.get("type") == "asset_fileid" and md.get("file_id"):
            out["fileids"].append({"file_id": md["file_id"], "media_kind": md.get("media_kind", "animation")})
        elif md.get("type") == "asset_url" and md.get("url"):
            out["urls"].append({"url": md["url"], "media_kind": md.get("media_kind", "animation")})
    return out

def p_list_keys(phase: str) -> List[str]:
    """Lista chiavi disponibili per la fase (best-effort via query ampia)."""
    res = index.query(
        namespace="assets",
        vector=embed_text(f"list assets {phase}"),
        top_k=200,
        include_metadata=True,
        filter={"phase": {"$eq": phase}, "type": {"$in": ["asset_fileid","asset_url"]}},
    )
    keys = []
    for m in res.get("matches", []):
        md = m.get("metadata", {}) or {}
        k = md.get("key")
        if k and k not in keys:
            keys.append(k)
    return keys

# ---- Captions su Pinecone ----
# metadata {type:"caption", phase, key, text}
def set_caption(phase: str, key: str, text: str):
    p_upsert("captions", embed_text(f"caption {phase} {key} {text[:50]}"), {"type": "caption", "phase": phase, "key": key, "text": text})

def get_caption(phase: str, key: str) -> Optional[str]:
    res = index.query(
        namespace="captions",
        vector=embed_text(f"caption {phase} {key}"),
        top_k=5,
        include_metadata=True,
        filter={"phase": {"$eq": phase}, "key": {"$eq": key}, "type": {"$eq": "caption"}},
    )
    for m in res.get("matches", []):
        md = m.get("metadata", {}) or {}
        txt = md.get("text")
        if txt:
            return txt
    return None

# ========= GLITCH LOCALE (GIF animata, no servizi esterni) =========
def gen_glitch_gif_local(text: str, size: int = 512, frames: int = 12, fps: int = 10) -> bytes:
    W = H = size
    INDIGO = (16,22,55); BLUE = (28,40,120); WHITE = (245,247,255)
    def grad():
        base = Image.new("RGB", (W,H), INDIGO)
        top  = Image.new("RGB", (W,H), BLUE)
        mask = Image.linear_gradient("L").resize((W,H))
        return Image.composite(top, base, mask)
    def shift_rgb(img, dx=2):
        arr = np.array(img)
        r,g,b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
        r = np.roll(r, dx, axis=1); b = np.roll(b, -dx, axis=1)
        return Image.fromarray(np.stack([r,g,b], axis=2))
    def scanlines(img, strength=28):
        arr = np.array(img).astype(np.int16)
        arr[::4,:,:] = np.clip(arr[::4,:,:] - strength, 0, 255)
        return Image.fromarray(arr.astype(np.uint8))
    def center_text(img, txt, fill=WHITE, sz=28):
        d = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", sz)
        except:
            font = ImageFont.load_default()
        bbox = d.textbbox((0,0), txt, font=font)
        x = (img.width - (bbox[2]-bbox[0]))//2
        y = (img.height - (bbox[3]-bbox[1]))//2
        d.text((x,y), txt, font=font, fill=fill)
        return img
    frames_list = []
    for i in range(frames):
        bg = grad()
        layer = center_text(bg.copy(), text, sz=28)
        if i%3==0: layer = shift_rgb(layer, dx=3)
        if i%4==0:
            arr = np.array(layer)
            arr[::8,:,:] = np.roll(arr[::8,:,:], 6, axis=1)
            layer = Image.fromarray(arr)
        frames_list.append(scanlines(layer, 26))
    bio = io.BytesIO()
    frames_list[0].save(bio, format="GIF", save_all=True, append_images=frames_list[1:], duration=int(1000/fps), loop=0, disposal=2)
    bio.seek(0)
    return bio.read()

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

# ========= INVIO MEDIA =========
IMG_EXTS  = (".png",".jpg",".jpeg",".webp")
ANIM_EXTS = (".gif",)
VID_EXTS  = (".mp4",".mov",".webm",".mkv")

def _send_url(chat_id: int, url: str, kind_hint: str = "", caption: str = "") -> bool:
    u = url.lower()
    try:
        if kind_hint == "image" or u.endswith(IMG_EXTS):
            bot.send_photo(chat_id, url, caption=caption or None)
        elif kind_hint == "video" or u.endswith(VID_EXTS):
            bot.send_video(chat_id, url, caption=caption or None)
        else:
            bot.send_animation(chat_id, url, caption=caption or None)
        return True
    except Exception as e:
        log.error("send_url ERROR: %r", e)
        return False

def _send_fileid(chat_id: int, file_id: str, kind: str = "", caption: str = "") -> bool:
    try:
        k = (kind or "").lower()
        if k == "photo":
            bot.send_photo(chat_id, file_id, caption=caption or None)
        elif k == "video":
            bot.send_video(chat_id, file_id, caption=caption or None)
        elif k == "animation":
            bot.send_animation(chat_id, file_id, caption=caption or None)
        else:
            try:
                bot.send_animation(chat_id, file_id, caption=caption or None)
            except Exception:
                try:
                    bot.send_photo(chat_id, file_id, caption=caption or None)
                except Exception:
                    bot.send_document(chat_id, file_id, caption=caption or None)
        return True
    except Exception as e:
        log.error("send_fileid ERROR: %r", e)
        return False

def _send_assets(chat_id: int, assets: Dict[str, List[Dict]], caption: str = ""):
    # Preferenza al provider configurato
    if MEDIA_PROVIDER == "R2" and assets["urls"]:
        sent = 0
        for a in assets["urls"][:3]:
            if _send_url(chat_id, a["url"], a.get("media_kind",""), caption if sent == 0 else ""):
                sent += 1
        if sent: return True
    # FILEID
    if assets["fileids"]:
        sent = 0
        for a in assets["fileids"][:3]:
            if _send_fileid(chat_id, a["file_id"], a.get("media_kind",""), caption if sent == 0 else ""):
                sent += 1
        if sent: return True
    # fallback URL anche se provider non è R2
    if MEDIA_PROVIDER != "R2" and assets["urls"]:
        sent = 0
        for a in assets["urls"][:3]:
            if _send_url(chat_id, a["url"], a.get("media_kind",""), caption if sent == 0 else ""):
                sent += 1
        if sent: return True
    return False

# ========= MENÙ BOTTONI =========
def make_menu(chat_id: int) -> t.InlineKeyboardMarkup:
    kb = t.InlineKeyboardMarkup(row_width=3)
    kb.add(
        t.InlineKeyboardButton("🇮🇹 Fase IT", callback_data="phase:IT"),
        t.InlineKeyboardButton("🇬🇧 Fase EN", callback_data="phase:EN"),
    )
    kb.add(
        t.InlineKeyboardButton("🎞 manifesto", callback_data="gif:manifesto"),
        t.InlineKeyboardButton("⚡ oltre_la_soglia", callback_data="gif:oltre_la_soglia"),
        t.InlineKeyboardButton("🌊 onda", callback_data="gif:onda"),
    )
    kb.add(
        t.InlineKeyboardButton("🛰 beyond_the_threshold", callback_data="gif:beyond_the_threshold"),
        t.InlineKeyboardButton("📶 glitch_signal", callback_data="gif:glitch_signal"),
        t.InlineKeyboardButton("📣 Post slot", callback_data="post:slot"),
    )
    kb.add(t.InlineKeyboardButton("🗂 Elenco keys", callback_data="list:keys"))
    return kb

@bot.message_handler(commands=["menu"])
def cmd_menu(m: telebot.types.Message):
    bot.reply_to(m, f"Fase corrente: <b>{get_phase(m.chat.id)}</b>\nScegli un’azione:", reply_markup=make_menu(m.chat.id))

@bot.callback_query_handler(func=lambda c: True)
def on_callback(c: telebot.types.CallbackQuery):
    try:
        data = c.data or ""
        chat_id = c.message.chat.id
        if data.startswith("phase:"):
            val = data.split(":",1)[1].upper()
            if val in ("IT","EN"):
                PHASE[chat_id] = val
                bot.answer_callback_query(c.id, f"Fase → {val}")
                bot.edit_message_text(f"Fase impostata: <b>{val}</b>", chat_id, c.message.message_id, parse_mode="HTML", reply_markup=make_menu(chat_id))
            else:
                bot.answer_callback_query(c.id, "Valore fase non valido.")
        elif data.startswith("gif:"):
            key = data.split(":",1)[1]
            if key == "slot":
                key = current_slot(get_phase(chat_id))
                if not key:
                    bot.answer_callback_query(c.id, "Nessuno slot attivo oggi.")
                    return
            assets = p_get_assets(get_phase(chat_id), key)
            ok = _send_assets(chat_id, assets, caption=f"{key} · fase {get_phase(chat_id)}")
            bot.answer_callback_query(c.id, "Inviato." if ok else "Nessun asset registrato.")
        elif data == "post:slot":
            k = current_slot(get_phase(chat_id))
            if not k:
                bot.answer_callback_query(c.id, "Nessuno slot oggi.")
                return
            _post_with_caption(chat_id, k)
            bot.answer_callback_query(c.id, "Post inviato.")
        elif data == "list:keys":
            keys = p_list_keys(get_phase(chat_id))
            txt = "Keys disponibili:\n" + (" • " + "\n • ".join(keys) if keys else " (nessuna)")
            bot.answer_callback_query(c.id, "OK")
            bot.send_message(chat_id, txt)
        else:
            bot.answer_callback_query(c.id, "Ignorato.")
    except Exception as e:
        log.error("callback ERROR: %r", e)
        try:
            bot.answer_callback_query(c.id, "Errore.")
        except Exception:
            pass

# ========= HANDLERS BASE =========
@bot.message_handler(commands=["start"])
def cmd_start(m: telebot.types.Message):
    PHASE[m.chat.id] = PHASE.get(m.chat.id, "IT")
    bot.reply_to(
        m,
        "👋 Benvenuto in 2Peak/2Pick.\n"
        "Comandi: /fase it|en · /ricorda <testo> · /cerca <query> · /svuota ·\n"
        "/bozza <brief> · /gif [chiave] · /gifadd (reply) · /gifaddurl <chiave> <url> · /glitch <testo>\n"
        "/post [chiave] · /caption <chiave> <testo> · /keys · /test_it · /test_en · /batch <azioni> · /menu"
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
        "/gif [chiave] – invia asset registrati\n"
        "/gifadd <chiave> – rispondi a una GIF/Video/Foto per registrarla (FILEID)\n"
        "/gifaddurl <chiave> <url> – registra URL (R2 o altro)\n"
        "/glitch <testo> – GIF glitch animata (locale)\n"
        "/post [chiave] – invia asset + caption (se assente, genera)\n"
        "/caption <chiave> <testo> – imposta caption personalizzata\n"
        "/keys – elenca chiavi disponibili nella fase corrente\n"
        "/test_it · /test_en – test rapidi\n"
        "/batch – più comandi in un messaggio con ; o nuove linee\n"
        "/menu – bottoni rapidi"
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

@bot.message_handler(commands=["ricorda","memorizza"])
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
        lines.append("\n<i>(no match ≥ soglia; mostro il migliore disponibile)</i>")
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
    """
    FILEID mode: rispondi a una GIF/Video/Foto con /gifadd <chiave>
    Riconosce: animation, video, photo, document immagine
    """
    phase = get_phase(m.chat.id)
    args = m.text.split(maxsplit=1)
    if len(args) == 1 or not args[1].strip():
        bot.reply_to(m, "Usa: rispondi a una GIF/Video/Foto con /gifadd <chiave>")
        return
    if not m.reply_to_message:
        bot.reply_to(m, "Devi <b>rispondere</b> a un media con questo comando.")
        return
    key = args[1].strip().lower()

    reply = m.reply_to_message
    file_id = None
    media_kind = None
    if reply.animation:
        file_id = reply.animation.file_id
        media_kind = "animation"
    elif reply.video:
        file_id = reply.video.file_id
        media_kind = "video"
    elif reply.photo:
        p = sorted(reply.photo, key=lambda x: x.file_size or 0)[-1]
        file_id = p.file_id
        media_kind = "photo"
    elif reply.document:
        mt = (reply.document.mime_type or "").lower()
        file_id = reply.document.file_id
        media_kind = "photo" if mt.startswith("image/") else "document"
    else:
        bot.reply_to(m, "Messaggio non valido: rispondi a GIF/Video/Foto/Documento immagine.")
        return

    try:
        p_upsert_fileid(phase, key, file_id, media_kind)
        bot.reply_to(m, f"✅ Registrato asset ({media_kind}) per <b>{key}</b> (fase {phase}).")
    except Exception as e:
        bot.reply_to(m, f"⚠️ Errore registrazione asset: {e}")

@bot.message_handler(commands=["gifaddurl"])
def cmd_gifaddurl(m: telebot.types.Message):
    """R2 mode: /gifaddurl <chiave> <url|path>  — auto-detect immagine/animazione/video da estensione."""
    phase = get_phase(m.chat.id)
    args = m.text.split(maxsplit=2)
    if len(args) < 3 or not args[1].strip() or not args[2].strip():
        bot.reply_to(m, "Usa: /gifaddurl <chiave> <url|path>")
        return
    key = args[1].strip().lower()
    raw = args[2].strip()
    url = raw if raw.lower().startswith("http") else (R2_PUBLIC_BASEURL + raw.lstrip("/"))

    ul = url.lower()
    if ul.endswith((".png",".jpg",".jpeg",".webp")):
        media_kind = "image"
    elif ul.endswith((".mp4",".mov",".webm",".mkv")):
        media_kind = "video"
    else:
        media_kind = "animation"  # gif o altro

    try:
        p_upsert_url(phase, key, url, media_kind)
        bot.reply_to(m, f"✅ Registrato URL ({media_kind}) per <b>{key}</b> (fase {phase}).")
    except Exception as e:
        bot.reply_to(m, f"⚠️ Errore registrazione URL: {e}")

@bot.message_handler(commands=["gif"])
def cmd_gif(m: telebot.types.Message):
    """
    /gif               → invia lo slot timeline corrente (fase IT/EN)
    /gif <chiave>      → invia gli asset registrati per quella chiave (fase corrente)
    """
    phase = get_phase(m.chat.id)
    args = m.text.split(maxsplit=1)
    key = None
    if len(args) == 1:
        key = current_slot(phase)
        if not key:
            bot.reply_to(m, f"Nessuno slot timeline attivo oggi per fase {phase}.")
            return
    else:
        key = args[1].strip().lower()
    assets = p_get_assets(phase, key)
    if not _send_assets(m.chat.id, assets, caption=f"{key} · fase {phase}"):
        hint = f"Usa /gifadd {key} (rispondendo a un media) o /gifaddurl {key} <url|path>"
        bot.reply_to(m, f"Nessun asset disponibile per «{key}» (fase {phase}). {hint}")

@bot.message_handler(commands=["glitch"])
def cmd_glitch(m: telebot.types.Message):
    args = m.text.split(maxsplit=1)
    if len(args) == 1 or not args[1].strip():
        bot.reply_to(m, "Testo mancante. Usa: /glitch <testo>")
        return
    prompt = args[1].strip()
    bot.send_chat_action(m.chat.id, "upload_document")
    try:
        gif_bytes = gen_glitch_gif_local(prompt, size=512, frames=12, fps=10)
        bio = io.BytesIO(gif_bytes); bio.name = "glitch.gif"; bio.seek(0)
        bot.send_animation(m.chat.id, bio, caption="Glitch pronto.")
    except Exception as e:
        bot.reply_to(m, f"⚠️ Errore glitch: {e}")

# ======= TEST & BATCH =======
def _send_key(chat_id: int, key: str) -> bool:
    phase = get_phase(chat_id)
    assets = p_get_assets(phase, key)
    return _send_assets(chat_id, assets, caption=f"{key} · fase {phase}")

@bot.message_handler(commands=["test_it"])
def cmd_test_it(m: telebot.types.Message):
    PHASE[m.chat.id] = "IT"
    keys = ["manifesto", "oltre_la_soglia", "onda"]
    missing = []
    for k in keys:
        bot.send_chat_action(m.chat.id, "upload_photo")
        ok = _send_key(m.chat.id, k)
        if not ok:
            missing.append(k)
        time.sleep(0.4)
    if missing:
        bot.reply_to(m, "⚠️ Mancano asset per: " + ", ".join(missing))
    else:
        bot.reply_to(m, "✅ Test IT completato.")

@bot.message_handler(commands=["test_en"])
def cmd_test_en(m: telebot.types.Message):
    PHASE[m.chat.id] = "EN"
    key = "beyond_the_threshold"
    bot.send_chat_action(m.chat.id, "upload_photo")
    ok = _send_key(m.chat.id, key)
    if not ok:
        bot.reply_to(m, f"⚠️ Nessun asset per «{key}».")
    else:
        bot.reply_to(m, "✅ Test EN completato.")

@bot.message_handler(commands=["batch"])
def cmd_batch(m: telebot.types.Message):
    """
    Esegue più azioni in un solo messaggio.
    Sintassi: /batch <righe o ; separate>
    Azioni supportate:
      - fase it|en
      - gif <chiave>
      - gif          (usa slot timeline corrente)
    """
    args = m.text.split(maxsplit=1)
    if len(args) == 1 or not args[1].strip():
        bot.reply_to(m, "Usa /batch con azioni separate da ; o nuove linee.\nEsempio:\n/batch\nfase it; gif manifesto; gif oltre_la_soglia; gif onda")
        return

    script = args[1].strip()
    parts = [p.strip() for p in re.split(r"[;\n]+", script) if p.strip()]
    results = []
    for p in parts:
        pl = p.lower()
        # fase
        if pl.startswith("fase "):
            val = pl.split(maxsplit=1)[1].strip().upper()
            if val in ("IT","EN"):
                PHASE[m.chat.id] = val
                results.append(f"fase→{val}")
            else:
                results.append(f"fase⚠️({val})")
        # gif <key>
        elif pl.startswith("gif "):
            key = p.split(maxsplit=1)[1].strip()
            ok = _send_key(m.chat.id, key)
            results.append(f"gif {key}→{'ok' if ok else 'manca'}")
            time.sleep(0.3)
        # gif (slot timeline)
        elif pl == "gif":
            phase = get_phase(m.chat.id)
            key = current_slot(phase)
            if not key:
                results.append("gif(slot)⚠️ nessuno slot attivo")
            else:
                ok = _send_key(m.chat.id, key)
                results.append(f"gif(slot:{key})→{'ok' if ok else 'manca'}")
                time.sleep(0.3)
        else:
            results.append(f"ignora: {p}")

    bot.reply_to(m, "Batch:\n" + " · ".join(results))

# ======= STEP 2B: POST + CAPTION + KEYS =======
def _gen_caption_auto(phase: str, key: str) -> str:
    sys_it = ("Sei l’editor ufficiale di 2Peak. Tono: criptico, selettivo, anti-hype. "
              "Frasi brevi, pause. Non spiegare mai il 'secondo picco'. Usa hashtag minimi.")
    sys_en = ("You are 2Peak’s in-house editor. Tone: cryptic, selective, anti-hype. "
              "Short lines. Never explain the 'second peak'. Minimal hashtags.")
    system = sys_it if phase == "IT" else sys_en
    up = f"Scrivi una caption breve per la chiave «{key}». Niente spiegoni. 1–2 righe massimo."
    if phase == "EN":
        up = f"Write a short caption for key «{key}». No explanations. Max 1–2 lines."
    out = chat_gpt_brief(system, up)
    return (out or "").strip()

def _post_with_caption(chat_id: int, key: str):
    phase = get_phase(chat_id)
    cap = get_caption(phase, key)
    if not cap:
        cap = _gen_caption_auto(phase, key)
    assets = p_get_assets(phase, key)
    ok = _send_assets(chat_id, assets, caption=cap)
    if not ok:
        bot.send_message(chat_id, f"⚠️ Nessun asset per «{key}» (fase {phase}). Registra con /gifadd o /gifaddurl.")
    else:
        bot.send_message(chat_id, f"✅ Post «{key}» inviato (fase {phase}).")

@bot.message_handler(commands=["post"])
def cmd_post(m: telebot.types.Message):
    args = m.text.split(maxsplit=1)
    if len(args) == 1 or not args[1].strip():
        k = current_slot(get_phase(m.chat.id))
        if not k:
            bot.reply_to(m, "Nessuno slot attivo oggi. Usa: /post <chiave>")
            return
        _post_with_caption(m.chat.id, k)
        return
    key = args[1].strip().lower()
    _post_with_caption(m.chat.id, key)

@bot.message_handler(commands=["caption"])
def cmd_caption(m: telebot.types.Message):
    args = m.text.split(maxsplit=2)
    if len(args) < 3 or not args[1].strip() or not args[2].strip():
        bot.reply_to(m, "Usa: /caption <chiave> <testo>")
        return
    key = args[1].strip().lower()
    txt = args[2].strip()
    set_caption(get_phase(m.chat.id), key, txt)
    bot.reply_to(m, f"✅ Caption salvata per «{key}» (fase {get_phase(m.chat.id)}).")

@bot.message_handler(commands=["keys"])
def cmd_keys(m: telebot.types.Message):
    keys = p_list_keys(get_phase(m.chat.id))
    if not keys:
        bot.reply_to(m, "Nessuna chiave asset registrata per questa fase.")
        return
    bot.reply_to(m, "Keys disponibili:\n" + "\n".join(f"• {k}" for k in keys))

# ================= HEALTH =================
@app.route("/health", methods=["GET"])
def health():
    return "ok", 200

# ================= RUN (LOCAL) =================
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
