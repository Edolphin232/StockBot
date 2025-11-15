#!/usr/bin/env python3
"""
Discord bot that posts top signals every day at 10:15 ET.

Requires:
    pip install discord.py pytz schedule

Bot token is stored in environment variable:
    export DISCORD_TOKEN="your_bot_token_here"
    export DISCORD_CHANNEL_ID="1234567890"
"""

import os
import asyncio
import discord
import pytz
import schedule
from datetime import datetime, date
from top_signals import process_day  # <-- your top-3 script

ET = pytz.timezone("US/Eastern")

TOKEN = os.getenv("DISCORD_TOKEN")
CHANNEL_ID = int(os.getenv("DISCORD_CHANNEL_ID"))

if not TOKEN:
    raise RuntimeError("Missing DISCORD_TOKEN env variable")

if not CHANNEL_ID:
    raise RuntimeError("Missing DISCORD_CHANNEL_ID env variable")

intents = discord.Intents.default()
client = discord.Client(intents=intents)


def get_signal_text(d: date):
    """
    Run your top signal engine and return a nicely-formatted
    Discord-friendly message string.
    """
    df_calls, df_puts, message = process_day(d, return_message=True)
    return message


async def send_daily_signal():
    """
    Calls process_day(), builds message, and sends it.
    """
    await client.wait_until_ready()
    channel = client.get_channel(CHANNEL_ID)

    if not channel:
        print("[ERROR] Could not find channel")
        return

    today = datetime.now(ET).date()
    msg = get_signal_text(today)
    await channel.send(msg)
    print(f"[DISCORD] Sent signals for {today}")


def schedule_loop():
    """
    Scheduled job runner — runs in a thread.
    """
    while True:
        schedule.run_pending()
        time.sleep(1)


@client.event
async def on_ready():
    print(f"[BOT] Logged in as {client.user}")

    # Schedule the job
    schedule.every().day.at("10:15").do(
        lambda: asyncio.create_task(send_daily_signal())
    )

    # Run scheduler loop (async)
    asyncio.create_task(asyncio.to_thread(schedule_loop))


if __name__ == "__main__":
    client.run(TOKEN)
