"""
Telegram bot wrapper for the Gilded Age RAG query engine.

Prerequisites:
  1. Create a bot via @BotFather on Telegram and get your token.
  2. Add TELEGRAM_BOT_TOKEN=<your_token> to the .env file.
  3. Make sure the vector index has been built (run ingest.py first).

Usage:
  python telegram_bot.py
"""
import logging
import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from query_engine import get_query_engine

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Suppress httpx logs so the bot token never appears in stdout/log files
logging.getLogger("httpx").setLevel(logging.WARNING)

# Load once at startup — avoids per-request ChromaDB connection overhead
print("Loading query engine…")
_engine = get_query_engine()
print("Query engine ready.")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Welcome! I am your Gilded Age culinary historian.\n\n"
        "Ask me about presidential dinners, Delmonico's, period recipes, "
        "service à la Russe, food styling for period productions, and more.\n\n"
        "What would you like to know?"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Just send me any question about Gilded Age dining and cuisine!\n\n"
        "Example questions:\n"
        "• What was served at Grover Cleveland's state dinners?\n"
        "• How did service à la Russe differ from French service?\n"
        "• What wines were fashionable at elite Gilded Age banquets?\n"
        "• How would I style a Delmonico's-era table for a period drama?"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_text = update.message.text.strip()
    if not user_text:
        return

    # Show typing indicator while querying
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action="typing"
    )

    try:
        response = _engine.query(user_text)
        answer = str(response).strip()
        if not answer:
            answer = "I couldn't find a clear answer in my sources. Could you rephrase or ask something more specific?"
    except Exception as e:
        logger.error("Query error: %s", e)
        answer = "I encountered an error processing your question. Please try again."

    # Telegram has a 4096 char limit per message
    if len(answer) > 4000:
        for i in range(0, len(answer), 4000):
            await update.message.reply_text(answer[i : i + 4000])
    else:
        await update.message.reply_text(answer)


def main() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise ValueError(
            "TELEGRAM_BOT_TOKEN not set. Add it to your .env file.\n"
            "Get a token from @BotFather on Telegram."
        )

    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Bot is running. Press Ctrl+C to stop.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
