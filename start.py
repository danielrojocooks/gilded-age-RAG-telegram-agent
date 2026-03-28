"""
Startup script for Railway deployment.
Runs ingestion to build the ChromaDB index, then launches the Telegram bot.
"""
import subprocess
import sys
import os

print("=== Gilded Age RAG Agent Startup ===")
print("Step 1: Building vector index from KB...")
result = subprocess.run([sys.executable, "ingest.py"], check=True)
print("Step 2: Starting Telegram bot...")
os.execv(sys.executable, [sys.executable, "telegram_bot.py"])
