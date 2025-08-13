import os
import telebot
from flask import Flask, request
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
bot = telebot.TeleBot(TOKEN)

app = Flask(__name__)

# Comando /start
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Ciao! Il bot 2Peak è attivo 🚀")

# Ricezione di tutti gli altri messaggi
@bot.message_handler(func=lambda message: True)
def echo_all(message):
    bot.reply_to(message, message.text)

# Endpoint per Telegram (webhook)
@app.route(f"/{TOKEN}", methods=['POST'])
def receive_update():
    json_str = request.get_data().decode('UTF-8')
    update = telebot.types.Update.de_json(json_str)
    bot.process_new_updates([update])
    return '', 200

# Endpoint per impostare il webhook
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

