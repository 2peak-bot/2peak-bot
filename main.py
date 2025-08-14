import os, csv, base64, tempfile, time
import telebot
from flask import Flask, request
from openai import OpenAI
from pinecone import Pinecone

# ========= ENV =========
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_HOST = os.environ.get("PINECONE_HOST")
LANG_DEFAULT = (os.environ.get("LANG_MODE", "IT") or "IT").upper()  # "IT" | "EN"
SCORE_MIN = float(os.environ.get("SEARCH_SCORE_MIN", "0.60"))

bot = telebot.TeleBot(TOKEN)
app = Flask(__name__)
client = OpenAI(api_key=OPENAI_API_KEY)

pc = Pinecone(api_key=PINECONE_API_KEY) if (PINECONE_API_KEY and PINECONE_HOST) else None
pindex = pc.Index(host=PINECONE_HOST) if pc else None

# lingua runtime per chat (volatile)
LANG_RUNTIME = {}

# ========= UTILS =========
def current_lang(chat_id: int) -> str:
    return LANG_RUNTIME.get(chat_id, LANG_DEFAULT)

def embed_text(txt: str) -> list:
    resp = client.embeddings.create(model="text-embedding-3-small", input=[txt])
    return resp.data[0].embedding

def pinecone_query(text: str, namespace: str, top_k: int = 10):
    vec = embed_text(text)
    res = pindex.query(vector=vec, top_k=top_k, include_metadata=True, namespace=namespace)
    return res.get("matches", []) if isinstance(res, dict) else res.matches

def top_unique_matches(matches, k=3, score_min=SCORE_MIN):
    def _score(m): return m.get("score", 0.0) if isinstance(m, dict) else getattr(m, "score", 0.0)
    def _text(m):
        return (m["metadata"]["text"] if isinstance(m, dict)
                else (m.metadata.get("text", "") if getattr(m, "metadata", None) else "")).strip()
    matches = sorted(matches, key=_score, reverse=True)
    seen, out = set(), []
    for m in matches:
        t, s = _text(m), _score(m)
        if not t or t in seen: continue
        if s >= score_min:
            out.append((t, s)); seen.add(t)
            if len(out) == k: break
    if not out:
        for m in matches:
            t = _text(m)
            if t: out = [(t, _score(m))]; break
    return out

def save_temp_bytes(data: bytes, suffix: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data); return tmp.name

def gen_image_png(prompt: str, size="1024x1024") -> bytes:
    # gestione errori "pulita" (403/500 → RuntimeError con messaggio leggibile)
    try:
        r = client.images.generate(model="gpt-image-1", prompt=prompt, size=size, n=1)
        return base64.b64decode(r.data[0].b64_json)
    except Exception as e:
        raise RuntimeError(str(e)) from e

# ========= HANDLERS =========
@bot.message_handler(commands=['start'])
def start_cmd(message):
    lang = current_lang(message.chat.id)
    bot.reply_to(message, "Ciao! Il bot 2Peak è attivo 🚀" if lang=="IT" else "Hi! 2Peak bot is live 🚀")

@bot.message_handler(commands=['help'])
def help_cmd(message):
    lang = current_lang(message.chat.id)
    it = ("Comandi:\n"
          "/start — verifica bot\n"
          "/fase <it|en> — cambia lingua\n"
          "/bozza <brief> — 3 varianti con RAG\n"
          "/ricorda|/memorizza <testo> — memorizza\n"
          "/cerca <query> — cerca nei ricordi\n"
          "/svuota — cancella memoria chat\n"
          "/esporta <txt|csv> — scarica memoria\n"
          "/gif <prompt> — immagine stile GIF\n"
          "/glitch <testo> — immagine glitch")
    en = ("Commands:\n"
          "/start — check bot\n"
          "/fase <it|en> — switch language\n"
          "/bozza <brief> — 3 variants with RAG\n"
          "/ricorda|/memorizza <text> — remember\n"
          "/cerca <query> — search memory\n"
          "/svuota — clear chat memory\n"
          "/esporta <txt|csv> — export memory\n"
          "/gif <prompt> — GIF-like image\n"
          "/glitch <text> — glitch image")
    bot.reply_to(message, it if lang=="IT" else en)

@bot.message_handler(commands=['fase'])
def fase_cmd(message):
    args = message.text.split(maxsplit=1)
    if len(args) == 1:
        bot.reply_to(message, f"Fase attuale: {current_lang(message.chat.id)}"); return
    val = args[1].strip().lower()
    if val in ("it","en"):
        LANG_RUNTIME[message.chat.id] = val.upper()
        bot.reply_to(message, f"Fase impostata: {val.upper()}")
    else:
        bot.reply_to(message, "Usa: /fase it oppure /fase en")

# ---- RAG + BOZZA ----
@bot.message_handler(commands=['bozza'])
def bozza_cmd(message):
    if not OPENAI_API_KEY:
        bot.reply_to(message, "OpenAI non configurato."); return
    brief = message.text.replace('/bozza','',1).strip()
    if not brief:
        bot.reply_to(message, "Usa così:\n/bozza teaser sul secondo picco" if current_lang(message.chat.id)=="IT"
                     else "Usage:\n/bozza teaser about second peak"); return
    lang = current_lang(message.chat.id)
    namespace = str(message.chat.id)
    context_chunks = []
    if pindex:
        matches = pinecone_query(brief, namespace, top_k=12)
        context_chunks = [t for t,_ in top_unique_matches(matches, k=5)]

    sys_it = ("Sei l’editor ufficiale di 2Peak. Tono: criptico, selettivo, anti-hype. "
              "Frasi brevi, pause. Non spiegare mai il 'secondo picco'. "
              "Intesse sottilmente il contesto senza citarlo.")
    sys_en = ("You are 2Peak’s in-house editor. Tone: cryptic, selective, anti-hype. "
              "Short lines, intentional pauses. Never explain the 'second peak'. "
              "Subtly weave in the context without explicit citations.")
    sys_prompt = sys_it if lang=="IT" else sys_en

    user_it = ("Obiettivo: {brief}\nContesto (se utile):\n{ctx}\n"
               "Produci 3 varianti, max 240 caratteri ciascuna, con a-capo intenzionali. "
               "Niente hashtag o emoji.")
    user_en = ("Goal: {brief}\nContext (if useful):\n{ctx}\n"
               "Return 3 variants, max 240 chars each, with intentional line breaks. "
               "No hashtags or emoji.")
    user_prompt = (user_it if lang=="IT" else user_en).format(
        brief=brief, ctx="\n".join(f"- {c}" for c in context_chunks) if context_chunks else "(no context)")

    try:
        bot.send_chat_action(message.chat.id, 'typing')
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"system","content":sys_prompt},{"role":"user","content":user_prompt}],
            temperature=0.7,
        )
        text = (resp.choices[0].message.content or "").strip()
        # parse 3 varianti
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        variants, current = [], []
        for l in lines:
            if l[:2] in ("1.","2.","3.") or l[:2].isdigit() or l.startswith(("•","- ")):
                if current: variants.append(" ".join(current).strip()); current=[]
                l = l.lstrip("•-123). ").strip(); current.append(l)
            else:
                current.append(l)
        if current: variants.append(" ".join(current).strip())
        bot.reply_to(message, "\n\n".join(f"• {v}" for v in variants[:3]) or ("Nessuna variante." if lang=="IT" else "No variants."))
    except Exception as e:
        bot.reply_to(message, f"Errore generazione: {e}")

# ---- MEMORIZZA / RICORDA ----
@bot.message_handler(commands=['ricorda','memorizza'])
def ricorda_cmd(message):
    if not pindex:
        bot.reply_to(message, "Memoria non configurata."); return
    text = message.text.split(' ',1)[1].strip() if ' ' in message.text else ""
    if not text:
        bot.reply_to(message, "Usa così:\n/ricorda testo" if current_lang(message.chat.id)=="IT" else "Usage:\n/ricorda text"); return
    try:
        bot.send_chat_action(message.chat.id, 'typing')
        vec = embed_text(text)
        namespace = str(message.chat.id)
        vec_id = f"{message.chat.id}-{message.message_id}-{int(time.time())}"
        pindex.upsert(vectors=[{"id": vec_id, "values": vec, "metadata": {"text": text}}], namespace=namespace)
        bot.reply_to(message, "Memorizzato ✅" if current_lang(message.chat.id)=="IT" else "Saved ✅")
    except Exception as e:
        bot.reply_to(message, f"Errore memorizzazione: {e}")

# ---- CERCA ----
@bot.message_handler(commands=['cerca'])
def cerca_cmd(message):
    if not pindex:
        bot.reply_to(message, "Memoria non configurata."); return
    query = message.text.split(' ',1)[1].strip() if ' ' in message.text else ""
    if not query:
        bot.reply_to(message, "Usa così:\n/cerca query" if current_lang(message.chat.id)=="IT" else "Usage:\n/cerca query"); return
    try:
        bot.send_chat_action(message.chat.id, 'typing')
        selected = top_unique_matches(pinecone_query(query, str(message.chat.id), top_k=20), k=3, score_min=SCORE_MIN)
        if not selected:
            bot.reply_to(message, "Nessun risultato." if current_lang(message.chat.id)=="IT" else "No results."); return
        reply = "\n\n".join(f"• {t}\n  (score: {s:.3f})" for t,s in selected)
        if selected and selected[0][1] < SCORE_MIN:
            reply += ("\n\n_(nessun match ≥ soglia; mostrato il migliore disponibile)_" 
                      if current_lang(message.chat.id)=="IT"
                      else "\n\n_(no match ≥ threshold; showing best available)_")
        bot.reply_to(message, reply)
    except Exception as e:
        bot.reply_to(message, f"Errore ricerca: {e}")

# ---- SVUOTA ----
@bot.message_handler(commands=['svuota'])
def svuota_cmd(message):
    if not pindex:
        bot.reply_to(message, "Memoria non configurata."); return
    try:
        pindex.delete(namespace=str(message.chat.id), delete_all=True)
        bot.reply_to(message, "Memoria di questa chat svuotata ✅" if current_lang(message.chat.id)=="IT" else "Chat memory cleared ✅")
    except Exception as e:
        bot.reply_to(message, f"Errore svuotamento: {e}")

# ---- ESPORTA ----
@bot.message_handler(commands=['esporta'])
def esporta_cmd(message):
    if not pindex:
        bot.reply_to(message, "Memoria non configurata."); return
    fmt = (message.text.split(' ',1)[1].strip().lower() if ' ' in message.text else "txt")
    if fmt not in ("txt","csv"):
        bot.reply_to(message, "Usa: /esporta txt | /esporta csv"); return
    try:
        matches = pindex.query(vector=[0]*1536, top_k=10000, include_metadata=True, namespace=str(message.chat.id))
        matches = matches.get("matches", []) if isinstance(matches, dict) else matches.matches
        rows = []
        for m in matches:
            md = m["metadata"] if isinstance(m, dict) else (m.metadata or {})
            t = (md.get("text","") if isinstance(md, dict) else md.get("text","")).strip()
            if t: rows.append(t)
        if not rows:
            bot.reply_to(message, "Nessun dato."); return
        if fmt=="txt":
            path = save_temp_bytes(("\n".join(rows)).encode("utf-8"), ".txt")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                w = csv.writer(tmp); w.writerow(["text"])
                for r in rows: w.writerow([r])
                path = tmp.name
        with open(path, "rb") as f:
            bot.send_document(message.chat.id, f)
        os.remove(path)
    except Exception as e:
        bot.reply_to(message, f"Errore export: {e}")

# ---- MEDIA (PNG immagini) ----
@bot.message_handler(commands=['gif'])
def gif_cmd(message):
    lang = current_lang(message.chat.id)
    prompt = message.text.replace('/gif','',1).strip()
    if not prompt:
        bot.reply_to(message, "Scrivi: /gif <prompt>" if lang=="IT" else "Type: /gif <prompt>"); return
    style = ("minimale, alto contrasto, coerente brand 2Peak, inquadratura cinematografica, grana leggera, senza scritte"
             if lang=="IT" else
             "minimal, high-contrast, 2Peak brand-consistent, cinematic frame, subtle grain, no text overlays")
    try:
        bot.send_chat_action(message.chat.id, 'upload_photo')
        img = gen_image_png(f"{prompt}. {style}.")
        p = save_temp_bytes(img, ".png")
        with open(p,"rb") as fh: bot.send_photo(message.chat.id, fh, caption=("Immagine pronta." if lang=="IT" else "Image ready."))
        os.remove(p)
    except RuntimeError as e:
        msg = str(e).lower()
        if "organization must be verified" in msg or "403" in msg:
            bot.reply_to(message, ("⚠️ Immagini non abilitate: verifica l’organizzazione in OpenAI e riprova."
                                   if lang=="IT" else "⚠️ Images not enabled: verify your OpenAI organization and try again."))
        else:
            bot.reply_to(message, f"Errore generazione: {e}")

@bot.message_handler(commands=['glitch'])
def glitch_cmd(message):
    lang = current_lang(message.chat.id)
    text = message.text.replace('/glitch','',1).strip()
    if not text:
        bot.reply_to(message, "Scrivi: /glitch <testo>" if lang=="IT" else "Type: /glitch <text>"); return
    prompt = (f"Immagine glitch con la frase '{text}', cyberpunk, righe di scansione, aberrazione cromatica, neon, estetica 2Peak, senza watermark"
              if lang=="IT" else
              f"Glitch art of the phrase '{text}', cyberpunk, scanlines, chromatic aberration, neon, 2Peak aesthetic, no watermark")
    try:
        bot.send_chat_action(message.chat.id, 'upload_photo')
        img = gen_image_png(prompt)
        p = save_temp_bytes(img, ".png")
        with open(p,"rb") as fh: bot.send_photo(message.chat.id, fh, caption=("Glitch pronto." if lang=="IT" else "Glitch ready."))
        os.remove(p)
    except RuntimeError as e:
        msg = str(e).lower()
        if "organization must be verified" in msg or "403" in msg:
            bot.reply_to(message, ("⚠️ Immagini non abilitate: verifica l’organizzazione in OpenAI e riprova."
                                   if lang=="IT" else "⚠️ Images not enabled: verify your OpenAI organization and try again."))
        else:
            bot.reply_to(message, f"Errore generazione: {e}")

# ========= WEBHOOK =========
@app.route(f"/{TOKEN}", methods=['POST'])
def receive_update():
    update = telebot.types.Update.de_json(request.get_data().decode('utf-8'))
    bot.process_new_updates([update])
    return '', 200

@app.route("/")
def webhook():
    base_url = os.environ.get("RENDER_EXTERNAL_URL") or request.url_root
    if not base_url.endswith("/"): base_url += "/"
    bot.remove_webhook(); bot.set_webhook(url=base_url + TOKEN)
    return "Webhook set!", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
