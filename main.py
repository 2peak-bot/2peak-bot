import os
import telebot
from flask import Flask, request
from openai import OpenAI

# === Env ===
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")  # su Render dev'essere impostata così
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

bot = telebot.TeleBot(TOKEN)
app = Flask(__name__)

# OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# ===== Handlers base =====
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Ciao! Il bot 2Peak è attivo 🚀")

@bot.message_handler(commands=['help'])
def help_cmd(message):
    bot.reply_to(message, "Comandi:\n/start — verifica bot\n/bozza <brief> — 3 varianti in stile 2Peak")

# ===== /bozza =====
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
        "Niente emoji (salvo 🧠 se già previsto), niente hashtag salvo richiesta. "
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

        # Split semplice in bullet/righe
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        variants = []
        current = []
        for l in lines:
            if (l[:2].isdigit() or l[:2] in ("1.", "2.", "3.", "1)", "2)", "3)")) or l.startswith(("•","- ")):
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

# ===== Webhook endpoints =====
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
