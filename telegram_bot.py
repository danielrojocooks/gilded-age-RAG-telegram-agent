"""
Telegram bot with per-user chat sessions and 30-minute inactivity timeout.

Each user gets their own CondensePlusContextChatEngine with separate memory.
Sessions expire after SESSION_TIMEOUT seconds of inactivity and are recreated
on the next message, giving a fresh conversation context.
"""
import logging
import os
import time
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from query_engine import get_index, get_chat_engine

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

SESSION_TIMEOUT = 30 * 60  # seconds of inactivity before resetting conversation

# Load the vector index once — shared across all user sessions
print("Loading index…")
_index = get_index()
print("Index ready.")

# Per-user sessions: {user_id: {"engine": engine, "last_active": timestamp}}
_sessions: dict = {}


def _get_user_engine(user_id: int):
    """Return the chat engine for this user, creating or resetting as needed."""
    now = time.time()
    session = _sessions.get(user_id)

    if session and (now - session["last_active"]) < SESSION_TIMEOUT:
        session["last_active"] = now
        return session["engine"]

    # New session or expired — create a fresh engine with empty memory
    engine = get_chat_engine(_index)
    _sessions[user_id] = {"engine": engine, "last_active": now}
    return engine


def _reset_user_session(user_id: int):
    """Discard the user's current session so the next message starts fresh."""
    _sessions.pop(user_id, None)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _reset_user_session(update.effective_user.id)
    await update.message.reply_text(
        "Welcome! I'm your Gilded Age culinary research assistant.\n\n"
        "Ask me about period menus, recipes, food styling, yields, "
        "service styles, food safety, or anything else in the knowledge base.\n\n"
        "I'll remember what we've discussed during our conversation. "
        "Send /start any time to reset.\n\n"
        "What would you like to know?"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Ask me anything about Gilded Age dining, food styling, or production research.\n\n"
        "Example questions:\n"
        "• What was served at Grover Cleveland's state dinners?\n"
        "• How did service à la Russe differ from French service?\n"
        "• What's the yield on a whole beef tenderloin?\n"
        "• What are the RI food code rules for cooling cooked food?\n"
        "• How would I style a Delmonico's-era table for a period drama?\n\n"
        "I remember context within our conversation. Send /start to reset."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_text = update.message.text.strip()
    if not user_text:
        return

    user_id = update.effective_user.id

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action="typing"
    )

    try:
        engine = _get_user_engine(user_id)
        response = engine.chat(user_text)
        answer = str(response).strip()
        if not answer:
            answer = "I couldn't find a clear answer. Could you rephrase or be more specific?"
    except Exception as e:
        logger.error("Chat error for user %s: %s", user_id, e)
        answer = "I encountered an error processing your question. Please try again."

    if len(answer) > 4000:
        for i in range(0, len(answer), 4000):
            await update.message.reply_text(answer[i : i + 4000])
    else:
        await update.message.reply_text(answer)


def main() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise ValueError(
            "TELEGRAM_BOT_TOKEN not set. Add it to your .env file."
        )

    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Bot is running. Press Ctrl+C to stop.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
