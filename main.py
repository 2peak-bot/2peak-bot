import os
import telebot
from flask import Flask, request
from openai import OpenAI
from pinecone import Pinecone

# ============== ENV VARS ==============
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")               # richiesto
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")          # richiesto
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")  # modello chat

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")      # richiesto
PINECONE_HOST = os.environ.get("PINECONE_HOST")            # richiesto: https://<index>.svc.<region>.pinecone.io

# soglia minima per mostrare i risultati della ricerca (override con env SEARCH_SCORE_MIN)
SCORE_MIN = float(os.environ.get("SEARCH_SCORE_MIN", "0.60"))

bot = telebot.TeleBot(TOKEN)
app = Flask(__name__)

# ============== CLIENTS ==============
client = OpenAI(api_key=OPENAI_API_KEY)

pc = None
pindex = None
if PINECONE_API_KEY and PINECONE_HOST:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pindex = pc.Index(host=PINECONE_HOST)

# ============== UTILS ==============
def embed_text(txt: str) -> list:
    """Embedding 1536-dim con OpenAI (serve billing attivo)."""
    resp = client.embeddings.create(model="text-embedding-3-small", input=[txt])
    return resp.data[0].embedding


# ============== HANDLERS BASE ==============
@bot.message_handler(commands=['start'])
def start_cmd(message):
    bot.reply_to(message, "Ciao! Il bot 2Peak è attivo 🚀")

@bot.message_handler(commands=['help'])
def help_cmd(message):
    bot.reply_to(
        message,
        "Comandi:\n"
        "/start — verifica bot\n"
        "/bozza <brief> — 3 varianti in stile 2Peak\n"
        "/ricorda <testo> — memorizza nel knowledge base\n"
        "/cerca <query> — cerca tra i contenuti memorizzati\n"
        "/svuota — cancella la memoria di questa chat"
    )


# ============== /bozza (OpenAI chat) ==============
@bot.message_handler(commands=['bozza'])
def bozza_cmd(message):
    brief = message.text.replace('/bozza', '', 1).strip()
    if not brief:
        bot.reply_to(message, "Usa così:\n/bozza teaser sul secondo picco")
        return

    bot.send_chat_action(message.chat.id, 'typing')

    system_prompt = (
        "Sei l’editor ufficiale di 2Peak. Tono: criptico, selettivo, zero hype. "
        "Regole: frasi brevi, pause intenzionali, nessuna spiegazione del 'secondo picco'. "
        "Niente emoji (salvo necessità), niente hashtag salvo richiesta. "
        "Rispondi con 3 varianti per social, max 240 caratteri, con eventuali a-capo intenzionali."
    )
    user_prompt = f"Obiettivo: {brief}\nOutput: 3 varianti."

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
        )
        text = (resp.choices[0].message.content or "").strip()

        # parsing semplice in 3 varianti
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        variants, current = [], []
        for l in lines:
            if l[:2] in ("1.", "2.", "3.") or l[:2].isdigit() or l.startswith(("•", "- ")):
                if current:
                    variants.append(" ".join(current).strip())
                    current = []
                l = l.lstrip("•-123). ").strip()
                current.append(l)
            else:
                current.append(l)
        if current:
            variants.append(" ".join(current).strip())

        reply = "\n\n".join(f"• {v}" for v in variants[:3]) or "Nessuna variante generata."
        bot.reply_to(message, reply)

    except Exception as e:
        bot.reply_to(message, f"Errore generazione: {e}")


# ============== /ricorda (memorizza su Pinecone) ==============
@bot.message_handler(commands=['ricorda'])
def ricorda_cmd(message):
    if not pindex:
        bot.reply_to(message, "Memoria non configurata: imposta PINECONE_API_KEY e PINECONE_HOST su Render.")
        return

    text = message.text.replace('/ricorda', '', 1).strip()
    if not text:
        bot.reply_to(message, "Usa così:\n/ricorda testo da memorizzare")
        return

    try:
        bot.send_chat_action(message.chat.id, 'typing')
        vec = embed_text(text)
        namespace = str(message.chat.id)  # memoria separata per chat
        vec_id = f"{message.chat.id}-{message.message_id}"
        pindex.upsert(
            vectors=[{"id": vec_id, "values": vec, "metadata": {"text": text}}],
            namespace=namespace
        )
        bot.reply_to(message, "Memorizzato ✅")
    except Exception as e:
        bot.reply_to(message, f"Errore memorizzazione: {e}")


# ============== /cerca (dedup + soglia + fallback) ==============
@bot.message_handler(commands=['cerca'])
def cerca_cmd(message):
    if not pindex:
        bot.reply_to(message, "Memoria non configurata: imposta PINECONE_API_KEY e PINECONE_HOST su Render.")
        return

    query = message.text.replace('/cerca', '', 1).strip()
    if not query:
        bot.reply_to(message, "Usa così:\n/cerca domanda o parola chiave")
        return

    try:
        bot.send_chat_action(message.chat.id, 'typing')
        qvec = embed_text(query)
        namespace = str(message.chat.id)

        res = pindex.query(vector=qvec, top_k=20, include_metadata=True, namespace=namespace)
        matches = res.get("matches", []) if isinstance(res, dict) else res.matches
        if not matches:
            bot.reply_to(message, "Nessun risultato.")
            return

        def _score(m):
            return m.get("score", 0.0) if isinstance(m, dict) else getattr(m, "score", 0.0)

        # ordina per score e rimuovi duplicati testuali
        matches = sorted(matches, key=_score, reverse=True)
        seen, selected = set(), []
        for m in matches:
            text = (m["metadata"]["text"] if isinstance(m, dict)
                    else (m.metadata.get("text", "") if m.metadata else "")).strip()
            if not text or text in seen:
                continue
            s = _score(m)
            if s >= SCORE_MIN:
                selected.append((text, s))
                seen.add(text)
                if len(selected) == 3:
                    break

        # fallback: se niente supera la soglia, mostra il migliore unico
        if not selected:
            for m in matches:
                text = (m["metadata"]["text"] if isinstance(m, dict)
                        else (m.metadata.get("text", "") if m.metadata else "")).strip()
                if text:
                    selected = [(text, _score(m))]
                    break

        if not selected:
            bot.reply_to(message, "Nessun risultato.")
            return

        reply = "\n\n".join(f"• {t}\n  (score: {s:.3f})" for t, s in selected)
        if selected and selected[0][1] < SCORE_MIN:
            reply += "\n\n_(nessun match ≥ soglia; mostrato il migliore disponibile)_"
        bot.reply_to(message, reply)

    except Exception as e:
        bot.reply_to(message, f"Errore ricerca: {e}")


# ============== /svuota (pulisce namespace chat) ==============
@bot.message_handler(commands=['svuota'])
def svuota_cmd(message):
    if not pindex:
        bot.reply_to(message, "Memoria non configurata.")
        return
    try:
        namespace = str(message.chat.id)
        pindex.delete(namespace=namespace, delete_all=True)
        bot.reply_to(message, "Memoria di questa chat svuotata ✅")
    except Exception as e:
        bot.reply_to(message, f"Errore svuotamento: {e}")


# ============== WEBHOOK (Telegram) ==============
@app.route(f"/{TOKEN}", methods=['POST'])
def receive_update():
    update = telebot.types.Update.de_json(request.get_data().decode('utf-8'))
    bot.process_new_updates([update])
    return '', 200

@app.route("/")
def webhook():
    base_url = os.environ.get("RENDER_EXTERNAL_URL") or request.url_root
    if not base_url.endswith("/"):
        base_url += "/"
    bot.remove_webhook()
    bot.set_webhook(url=base_url + TOKEN)
    return "Webhook set!", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
