import os
import telebot
import requests
from flask import Flask, request
from openai import OpenAI
import base64

# ==========================================
# CONFIGURAZIONE
# ==========================================
BOT_TOKEN = os.getenv("BOT_TOKEN")       # inserisci il token Bot Telegram
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # inserisci la tua API key OpenAI

bot = telebot.TeleBot(BOT_TOKEN)
app = Flask(__name__)
client = OpenAI(api_key=OPENAI_API_KEY)

# ==========================================
# FUNZIONI UTILI
# ==========================================

def gen_image_png(prompt):
    """Genera un'immagine PNG da un prompt"""
    try:
        result = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size="512x512"
        )
        img_b64 = result.data[0].b64_json
        return base64.b64decode(img_b64)
    except Exception as e:
        return f"[Errore generazione immagine: {str(e)}]".encode()

def gen_gif_glitch(prompt):
    """Genera una GIF animata glitch"""
    try:
        # Simuliamo output glitch con immagini sequenziali → GIF
        result = client.images.generate(
            model="gpt-image-1",
            prompt=f"{prompt}, stile glitch, animazione frame",
            size="512x512",
            n=4
        )
        frames = [base64.b64decode(x.b64_json) for x in result.data]
        # Qui potresti usare PIL.Image per unire frames in una GIF
        return frames[0]  # per ora restituiamo solo il primo frame
    except Exception as e:
        return f"[Errore generazione GIF: {str(e)}]".encode()

# ==========================================
# COMANDI TELEGRAM
# ==========================================

@bot.message_handler(commands=["start"])
def cmd_start(message):
    bot.reply_to(message, "👋 Benvenuto! Sono il bot 2Peak.\nUsa /memorizza o /cerca.")

@bot.message_handler(commands=["memorizza"])
def cmd_memorizza(message):
    testo = message.text.replace("/memorizza", "").strip()
    if not testo:
        bot.reply_to(message, "❌ Devi scrivere qualcosa dopo /memorizza")
    else:
        # Placeholder per memoria su Pinecone
        bot.reply_to(message, f"📝 Memorizzato: {testo}")

@bot.message_handler(commands=["cerca"])
def cmd_cerca(message):
    query = message.text.replace("/cerca", "").strip()
    if not query:
        bot.reply_to(message, "❌ Devi scrivere qualcosa dopo /cerca")
    else:
        # Placeholder → query Pinecone
        bot.reply_to(message, f"🔍 Risultato della ricerca per: {query}")

@bot.message_handler(commands=["img"])
def cmd_img(message):
    prompt = message.text.replace("/img", "").strip()
    if not prompt:
        bot.reply_to(message, "❌ Devi scrivere un prompt dopo /img")
        return
    bot.reply_to(message, "🎨 Generazione immagine in corso...")
    img_data = gen_image_png(prompt)
    try:
        bot.send_photo(message.chat.id, img_data)
    except Exception as e:
        bot.reply_to(message, f"Errore invio immagine: {e}")

@bot.message_handler(commands=["gif"])
def cmd_gif(message):
    prompt = message.text.replace("/gif", "").strip()
    if not prompt:
        bot.reply_to(message, "❌ Devi scrivere un prompt dopo /gif")
        return
    bot.reply_to(message, "🎬 Generazione GIF/glitch in corso...")
    img_data = gen_gif_glitch(prompt)
    try:
        bot.send_document(message.chat.id, ("glitch.gif", img_data))
    except Exception as e:
        bot.reply_to(message, f"Errore invio GIF: {e}")

# ==========================================
# WEBHOOK FLASK
# ==========================================

@app.route(f"/{BOT_TOKEN}", methods=["POST"])
def webhook():
    json_str = request.get_data().decode("UTF-8")
    update = telebot.types.Update.de_json(json_str)
    try:
        bot.process_new_updates([update])
    except Exception as e:
        print(f"Errore update Telegram: {e}")
    return "!", 200

@app.route("/")
def index():
    return "Bot 2Peak attivo 🚀", 200

# ==========================================
# AVVIO LOCALE
# ==========================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
