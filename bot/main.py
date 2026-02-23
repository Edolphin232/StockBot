# bot/main.py
"""
Main entry point — schedules all jobs and runs the Discord bot.
"""
import asyncio
import pytz
import discord
import os
from datetime import datetime, time, timedelta
from dotenv import load_dotenv

load_dotenv(override=True)

from bot.discord_client import (
    client, send_message,
    premarket_embed, orb_embed, signal_embed
)
from bot.scanner import (
    get_today_store,
    run_premarket_scan,
    run_orb_scan,
    run_signal_scan,
)
from bot.commands import handle_testspy, handle_earnings
from strategy.technical_signals import check_all_signals, TechnicalSignal

EASTERN = pytz.timezone("US/Eastern")

# State shared across jobs
state = {
    "store":          None,
    "date":           None,
    "prev_close":     None,
    "vix":            None,
    "premarket_done": False,
    "orb_done":       False,
    "signal_fired":   False,   # only fire once per day
    "technical_signals_active": False,  # Start monitoring after signal fires
    "orb_range":      None,    # ORB range for technical signals
    "sent_signals":   set(),   # Track sent technical signals to avoid duplicates
}

@client.event
async def on_message(message: discord.Message):
    # Ignore messages from the bot itself
    if message.author == client.user:
        return

    # Debug: log all messages (remove in production if needed)
    print(f"[Discord] Message from {message.author}: {message.content[:50]}")

    if message.content.startswith("!testspy"):
        args = message.content.split()[1:]
        print(f"[Discord] Handling !testspy command with args: {args}")
        await handle_testspy(message, args)
    elif message.content.startswith("!earnings"):
        args = message.content.split()[1:]
        print(f"[Discord] Handling !earnings command with args: {args}")
        await handle_earnings(message, args)

async def wait_until(target: time):
    """Sleep until a specific Eastern time today."""
    now = datetime.now(EASTERN)
    target_dt = EASTERN.localize(datetime.combine(now.date(), target))
    if target_dt < now:
        return  # already past
    delta = (target_dt - now).total_seconds()
    await asyncio.sleep(delta)


async def wait_until_next_weekday(target_weekday: int, target_time: time):
    """
    Wait until the next occurrence of a specific weekday and time.

    Args:
        target_weekday: 0=Monday, 6=Sunday
        target_time: time object (e.g., time(20, 0) for 8:00 PM)
    """
    now = datetime.now(EASTERN)
    days_ahead = target_weekday - now.weekday()

    # If today is the target weekday but time has passed, or if target is earlier in week
    if days_ahead < 0 or (days_ahead == 0 and now.time() >= target_time):
        days_ahead += 7  # Move to next week

    target_date = now.date() + timedelta(days=days_ahead)
    target_dt = EASTERN.localize(datetime.combine(target_date, target_time))

    delta = (target_dt - now).total_seconds()
    if delta > 0:
        await asyncio.sleep(delta)


async def wait_until_next_trading_day(target_time: time):
    """Sleep until target_time on the next weekday (skips weekends, not holidays)."""
    now = datetime.now(EASTERN)
    next_day = now.date() + timedelta(days=1)
    while next_day.weekday() >= 5:  # Skip Saturday (5) and Sunday (6)
        next_day += timedelta(days=1)
    target_dt = EASTERN.localize(datetime.combine(next_day, target_time))
    delta = (target_dt - now).total_seconds()
    if delta > 0:
        await asyncio.sleep(delta)


async def refresh_store():
    """Reload bars from Alpaca — called every minute."""
    store, date, prev_close, vix = get_today_store()
    
    # Reset daily state if date changed
    if state["date"] != date and state["date"] is not None:
        state["sent_signals"] = set()
        state["signal_fired"] = False
        state["technical_signals_active"] = False
        state["premarket_done"] = False
        state["orb_done"] = False
        state["orb_range"] = None
        print(f"[State] Date changed from {state['date']} to {date}, resetting daily flags")
    
    state["store"]      = store
    state["date"]       = date
    state["prev_close"] = prev_close
    state["vix"]        = vix


async def job_premarket():
    """9:45 AM every trading day — send premarket summary."""
    while True:
        await wait_until(time(9, 45))

        try:
            await refresh_store()

            if state["store"] is None:
                error_msg = f"❌ **Premarket Error** — No data store available for {state.get('date', 'today')}"
                await send_message(error_msg)
                print(f"[{state.get('date', 'unknown')}] Premarket: store is None")
            else:
                bars = state["store"].get_day_bars(state["date"])
                if bars.empty:
                    error_msg = f"⚠️ **Premarket Warning** — No bars available yet for {state['date']}. Market may not be open or data not available."
                    await send_message(error_msg)
                    print(f"[{state['date']}] Premarket: bars are empty")
                else:
                    premarket = run_premarket_scan(
                        state["store"], state["date"],
                        state["prev_close"], state["vix"]
                    )
                    if premarket:
                        embed = premarket_embed(state["date"], premarket, state["vix"] or 0.0)
                        await send_message(embed=embed)
                        print(f"[{state['date']}] Premarket sent")
                    else:
                        error_msg = f"⚠️ **Premarket Warning** — Premarket scan returned None for {state['date']}"
                        await send_message(error_msg)
                        print(f"[{state['date']}] Premarket scan returned None")

                    state["premarket_done"] = True
        except Exception as e:
            error_msg = f"❌ **Premarket Error** — Failed to generate premarket report: {str(e)}"
            await send_message(error_msg)
            print(f"[{state.get('date', 'unknown')}] Error in job_premarket: {e}")
            import traceback
            traceback.print_exc()

        await wait_until_next_trading_day(time(9, 45))


async def job_orb():
    """10:01 AM every trading day — send ORB analysis.
    Starts at 10:01 (not 10:00) so all 30 opening-range bars are guaranteed available."""
    while True:
        await wait_until(time(10, 1))

        try:
            await refresh_store()

            if state["store"] is None:
                error_msg = f"❌ **ORB Error** — No data store available for {state.get('date', 'today')}"
                await send_message(error_msg)
                print(f"[{state.get('date', 'unknown')}] ORB: store is None")
            else:
                bars = state["store"].get_day_bars(state["date"])
                if bars.empty or len(bars) < 30:
                    error_msg = f"⚠️ **ORB Warning** — Only {len(bars)} bars available for {state['date']}. Need at least 30."
                    await send_message(error_msg)
                    print(f"[{state['date']}] ORB: not enough bars ({len(bars)} < 30)")
                else:
                    orb_range, orb_result = run_orb_scan(state["store"], state["date"])
                    if orb_range and orb_result:
                        embed = orb_embed(state["date"], orb_result, orb_range)
                        await send_message(embed=embed)
                        print(f"[{state['date']}] ORB sent")
                        state["orb_range"] = orb_range
                    else:
                        error_msg = f"⚠️ **ORB Warning** — ORB scan returned None for {state['date']}. May need more data."
                        await send_message(error_msg)
                        print(f"[{state['date']}] ORB scan returned None")

                    state["orb_done"] = True
        except Exception as e:
            error_msg = f"❌ **ORB Error** — Failed to generate ORB analysis: {str(e)}"
            await send_message(error_msg)
            print(f"[{state.get('date', 'unknown')}] Error in job_orb: {e}")
            import traceback
            traceback.print_exc()

        await wait_until_next_trading_day(time(10, 1))


async def job_scan_loop():
    """10:00 AM → 12:00 PM every trading day — scan every minute for signal."""
    while True:
        await wait_until(time(10, 0))

        # Inner loop: scan every minute until noon
        while True:
            now = datetime.now(EASTERN)

            # Stop at noon
            if now.time() >= time(12, 0):
                await send_message("🛑 Scan window closed for today (12:00 PM)")
                print(f"[{state['date']}] Scan window closed")
                break

            # Only fire once per day
            if not state["signal_fired"]:
                try:
                    await refresh_store()

                    if state["store"] is None:
                        error_msg = f"❌ **Signal Scan Error** — No data store available for {state.get('date', 'today')}"
                        await send_message(error_msg)
                        print(f"[{state.get('date', 'unknown')}] Signal scan: store is None")
                    else:
                        bars = state["store"].get_day_bars(state["date"])
                        if bars.empty:
                            if now.time() >= time(9, 50):
                                print(f"[{state['date']}] Signal scan: bars are empty (may be normal early in day)")
                        elif len(bars) < 10:
                            print(f"[{state['date']}] Signal scan: not enough bars ({len(bars)} < 10)")
                        else:
                            signal = run_signal_scan(
                                state["store"], state["date"],
                                state["prev_close"], state["vix"]
                            )
                            if signal and signal.triggered:
                                bars = state["store"].get_day_bars(state["date"])
                                embed = signal_embed(signal, bars=bars)
                                await send_message(embed=embed)
                                state["signal_fired"] = True
                                state["technical_signals_active"] = True
                                print(f"[{state['date']}] Signal fired: {signal.direction}")
                except Exception as e:
                    error_msg = f"❌ **Signal Scan Error** — Failed to scan for signals: {str(e)}"
                    await send_message(error_msg)
                    print(f"[{state.get('date', 'unknown')}] Error in signal scan: {e}")
                    import traceback
                    traceback.print_exc()

            # Check technical signals only after signal fired and before 12:00 PM
            if state.get("technical_signals_active", False):
                now = datetime.now(EASTERN)
                if now.time() < time(12, 0):
                    await check_technical_signals()
                else:
                    state["technical_signals_active"] = False
                    print(f"[{state['date']}] Technical signal monitoring stopped (12:00 PM)")
            else:
                if state.get("signal_fired", False):
                    print(f"[{state['date']}] Signal fired but technical_signals_active is False")
                else:
                    print(f"[{state['date']}] No signal fired yet, skipping technical signal check")

            await asyncio.sleep(60)

        await wait_until_next_trading_day(time(10, 0))


async def check_technical_signals():
    """Check for technical signals and send notifications."""
    try:
        if state["store"] is None:
            print(f"[{state['date']}] check_technical_signals: store is None")
            return
        
        bars = state["store"].get_day_bars(state["date"])
        if bars.empty:
            print(f"[{state['date']}] check_technical_signals: bars are empty")
            return
        
        if len(bars) < 2:
            print(f"[{state['date']}] check_technical_signals: not enough bars ({len(bars)})")
            return
        
        # Get ORB range if available
        orb_range = state.get("orb_range")
        prev_close = state.get("prev_close")
        
        print(f"[{state['date']}] check_technical_signals: checking signals (bars={len(bars)}, prev_close={prev_close}, orb_range={orb_range is not None})")
        
        # Check all technical signals
        signals = check_all_signals(bars, prev_close, orb_range)
        
        print(f"[{state['date']}] check_technical_signals: found {len(signals)} signals")
        
        # Track which signals we've already sent (avoid duplicates)
        if "sent_signals" not in state:
            state["sent_signals"] = set()
        
        for signal in signals:
            # Create unique key for this signal
            signal_key = f"{signal.signal_type}_{signal.level:.2f}"
            
            # Only send if we haven't sent this signal before
            if signal_key not in state["sent_signals"]:
                from bot.discord_client import send_message, technical_signal_embed
                
                embed = technical_signal_embed(signal)
                await send_message(embed=embed)
                state["sent_signals"].add(signal_key)
                print(f"[{state['date']}] Technical signal sent: {signal.description}")
            else:
                print(f"[{state['date']}] Technical signal already sent: {signal_key}")
    except Exception as e:
        error_msg = f"❌ **Technical Signals Error** — Failed to check technical signals: {str(e)}"
        await send_message(error_msg)
        print(f"[{state.get('date', 'unknown')}] Error in check_technical_signals: {e}")
        import traceback
        traceback.print_exc()


async def job_sunday_report():
    """Every Sunday at 8:00 PM — send weekly risk report."""
    # Wait until next Sunday at 8:00 PM
    await wait_until_next_weekday(6, time(20, 0))  # 6 = Sunday, 20:00 = 8:00 PM
    
    # Then run every week
    while True:
        try:
            from sunday_report.sunday_report import generate_sunday_report, create_sunday_report_embed
            
            macro, earnings, dashboard = generate_sunday_report()
            embed = create_sunday_report_embed(macro, earnings, dashboard)
            
            if embed:
                await send_message(embed=embed)
                print(f"[Sunday Report] Weekly report sent successfully")
            else:
                # Fallback to plain text
                from sunday_report.sunday_report import format_for_discord
                text = format_for_discord(macro, earnings, dashboard)
                await send_message(text)
                print(f"[Sunday Report] Weekly report sent (plain text)")
        except Exception as e:
            print(f"[Sunday Report] Error: {e}")
        
        # Wait until next Sunday at 8:00 PM
        await wait_until_next_weekday(6, time(20, 0))


@client.event
async def on_ready():
    print(f"[Discord] Logged in as {client.user}")
    asyncio.create_task(job_premarket())
    asyncio.create_task(job_orb())
    asyncio.create_task(job_scan_loop())
    asyncio.create_task(job_sunday_report())


if __name__ == "__main__":
    client.run(os.getenv("DISCORD_TOKEN"))