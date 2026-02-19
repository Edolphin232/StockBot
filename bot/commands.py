# bot/commands.py
"""
Discord bot commands.
!testspy YYYY-MM-DD — runs full signal pipeline for a given date and sends results.
"""
import discord
import pytz
import pandas as pd
from datetime import datetime, time
from data.store import DataStore
from fetchers.alpaca_client import fetch_bars as alpaca_fetch
from fetchers.yahoo_client import fetch_bars as yf_fetch
from fetchers.catalyst_client import fetch_catalysts
from filters.premarket_filter import run_premarket_filter
from calculators.orb import calculate_opening_range, detect_breakout
from strategy.signal_generator import generate_signal
from strategy.technical_signals import check_all_signals, TechnicalSignal
from bot.discord_client import premarket_embed, orb_embed, signal_embed, technical_signal_embed

EASTERN = pytz.timezone("US/Eastern")


def prev_trading_day(date_str: str, max_lookback_days: int = 10) -> str:
    """
    Return the actual previous trading day by checking for data.
    Accounts for weekends AND market holidays.
    
    Args:
        date_str: Date in YYYY-MM-DD format
        max_lookback_days: Maximum days to look back (default 10)
    
    Returns:
        Previous trading day as YYYY-MM-DD string
    """
    from datetime import timedelta
    target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    
    # Try each day going back, checking if it has trading data
    for i in range(1, max_lookback_days + 1):
        check_date = target_date - timedelta(days=i)
        
        # Skip weekends (but still check in case of weird data)
        if check_date.weekday() >= 5:
            continue
        
        # Try to fetch a small amount of data for this date to verify it's a trading day
        check_start = EASTERN.localize(datetime.combine(check_date, time(9, 30)))
        check_end = EASTERN.localize(datetime.combine(check_date, time(10, 0)))  # Just 30 min to verify
        
        try:
            test_bars = alpaca_fetch("SPY", check_start, check_end, timeframe="1m")
            if test_bars is not None and not test_bars.empty:
                # Found a trading day with data
                return check_date.strftime("%Y-%m-%d")
        except Exception:
            # If fetch fails, continue to next day
            continue
    
    # Fallback: if we can't find any data, just return the day before (skipping weekends)
    d = target_date - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d.strftime("%Y-%m-%d")


async def handle_testspy(message: discord.Message, args: list):
    """
    Handle !testspy YYYY-MM-DD command.
    Runs full signal pipeline for the given date and sends results to Discord.
    """
    # Validate args
    if not args:
        await message.channel.send("❌ Usage: `!testspy YYYY-MM-DD`")
        return

    date_str = args[0]
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        await message.channel.send("❌ Invalid date format. Use `YYYY-MM-DD`")
        return

    await message.channel.send(f"🔍 Running signal pipeline for `{date_str}`...")

    # Build date range
    prev_date = prev_trading_day(date_str)
    target    = datetime.strptime(date_str, "%Y-%m-%d")

    prev_start = EASTERN.localize(datetime.combine(
        datetime.strptime(prev_date, "%Y-%m-%d").date(), time(9, 30)
    ))
    prev_end = EASTERN.localize(datetime.combine(
        datetime.strptime(prev_date, "%Y-%m-%d").date(), time(16, 0)
    ))
    day_start = EASTERN.localize(datetime.combine(target.date(), time(9, 30)))
    day_end   = EASTERN.localize(datetime.combine(target.date(), time(16, 0)))

    # Fetch data
    try:
        prev_bars  = alpaca_fetch("SPY", prev_start, prev_end, timeframe="1m")
        today_bars = alpaca_fetch("SPY", day_start, day_end, timeframe="1m")
        vix_data   = yf_fetch("^VIX", prev_date, date_str)
    except Exception as e:
        await message.channel.send(f"❌ Data fetch failed: {e}")
        return

    if today_bars is None or today_bars.empty:
        await message.channel.send(f"❌ No SPY bars found for `{date_str}` — market may have been closed.")
        return

    # Load store
    store = DataStore()
    if prev_bars is not None and not prev_bars.empty:
        store.load_spy(pd.concat([prev_bars, today_bars]))
    else:
        store.load_spy(today_bars)
    store.load_vix(vix_data)

    bars = store.get_day_bars(date_str)
    
    # Calculate prev_close directly from prev_bars (more reliable than store method)
    if prev_bars is not None and not prev_bars.empty:
        prev_close = float(prev_bars.iloc[-1]["close"])
    else:
        # Fallback to store method if prev_bars not available
        prev_close = store.get_prev_close(date_str)
        if prev_close is None and not bars.empty:
            # Last resort: use first bar open as proxy
            prev_close = float(bars.iloc[0]["open"])
    
    vix = store.get_vix(date_str) or store.get_vix(prev_date)

    if bars.empty:
        await message.channel.send(f"❌ Could not slice bars for `{date_str}`")
        return

    # 1. Premarket
    current_open = float(bars.iloc[0]["open"])
    
    # Fetch catalysts for this date
    catalyst_events = fetch_catalysts(target_date=date_str)
    
    premarket = run_premarket_filter(
        current_open=current_open,
        prev_close=prev_close,
        vix_level=vix or 0.0,
        date=date_str,
        catalyst_events=catalyst_events,
    )
    await message.channel.send(embed=premarket_embed(date_str, premarket, vix or 0.0))

    # 2. ORB (10:00)
    orb_range  = calculate_opening_range(bars)
    orb_result = detect_breakout(bars, orb_range) if orb_range else None
    if orb_range and orb_result:
        await message.channel.send(embed=orb_embed(date_str, orb_result, orb_range))
    else:
        await message.channel.send("⚠️ Could not compute ORB for this date.")

    # 3. Full signal (9:50-12:00 scan)
    signal = generate_signal(
        date=date_str,
        bars=bars,
        prev_close=prev_close,
        vix_level=vix or 0.0,
        catalyst_events=catalyst_events,
    )

    if signal.triggered:
        await message.channel.send(embed=signal_embed(signal, bars=bars))
        
        # Simulate technical signals from signal trigger to 12:00 PM
        await simulate_technical_signals(message, signal, bars, orb_range, prev_close, date_str)
    else:
        await message.channel.send(
            f"📭 No signal triggered for `{date_str}`\n"
            f"**Reason:** {signal.reason}\n"
            f"**Score:** {signal.total_score:.1f}"
        )


async def simulate_technical_signals(
    message: discord.Message,
    signal,
    bars: pd.DataFrame,
    orb_range,
    prev_close: float,
    date_str: str
):
    """
    Simulate technical signals from signal trigger point to 12:00 PM.
    Shows signal notifications as they would have occurred.
    """
    if bars.empty or signal.trigger_bar_idx < 0 or signal.trigger_bar_idx >= len(bars):
        return
    
    await message.channel.send(f"📊 Simulating technical signals from entry to 12:00 PM...")
    
    # Get bars from signal trigger point onwards
    entry_bars = bars.iloc[signal.trigger_bar_idx:]
    
    if entry_bars.empty:
        return
    
    # Track sent signals to avoid duplicates
    sent_signals = set()
    
    # Check every 5 minutes from entry point until 12:00 PM
    for i in range(0, len(entry_bars), 5):
        if i == 0:
            continue  # Skip first bar (entry point)
        
        # Get bars up to this point
        current_bars = entry_bars.iloc[:i+1]
        
        # Get timestamp for this bar
        bar_index = current_bars.index[-1]
        if isinstance(bar_index, tuple):
            bar_time = bar_index[1]  # MultiIndex: (symbol, timestamp)
        else:
            bar_time = bar_index
        
        # Stop at 12:00 PM
        if hasattr(bar_time, 'hour') and bar_time.hour >= 12:
            break
        
        # Check all technical signals
        signals = check_all_signals(current_bars, prev_close, orb_range)
        
        for tech_signal in signals:
            # Create unique key for this signal
            signal_key = f"{tech_signal.signal_type}_{tech_signal.level:.2f}"
            
            # Only send if we haven't sent this signal before
            if signal_key not in sent_signals:
                # Update signal time to match bar time
                if hasattr(bar_time, 'tz_localize') or hasattr(bar_time, 'tz_convert'):
                    if hasattr(bar_time, 'tz') and bar_time.tz is None:
                        tech_signal.time = EASTERN.localize(bar_time)
                    elif hasattr(bar_time, 'tz_convert'):
                        tech_signal.time = bar_time.tz_convert(EASTERN)
                    else:
                        tech_signal.time = bar_time
                else:
                    tech_signal.time = bar_time
                
                embed = technical_signal_embed(tech_signal)
                await message.channel.send(embed=embed)
                sent_signals.add(signal_key)


async def handle_earnings(message: discord.Message, args: list):
    """
    Handle !earnings command - manually trigger Sunday report or lookup earnings.
    Usage:
        !earnings                    - Full weekly report (default)
        !earnings YYYY-MM-DD         - Report for specific date
        !earnings TICKER             - Next earnings + EPS history + implied move for any ticker
        !earnings {int}              - Earnings with ImpactScore >= {int}
    Sends report to the channel where the command was issued.
    """
    try:
        from sunday_report.sunday_report import fetch_earnings, filter_earnings_for_window
        from sunday_report.sunday_report import generate_sunday_report, create_sunday_report_embed, format_for_discord
        from datetime import date, timedelta
        
        if not args:
            # No arguments - full weekly report
            await message.channel.send("📊 Generating weekly risk report...")
            macro, earnings, dashboard = generate_sunday_report(None)
            embed = create_sunday_report_embed(macro, earnings, dashboard)
            if embed:
                await message.channel.send(embed=embed)
                await message.channel.send("✅ Weekly report sent!")
            else:
                text = format_for_discord(macro, earnings, dashboard)
                await message.channel.send(text)
                await message.channel.send("✅ Weekly report sent!")
            return
        
        arg = args[0].upper().strip()
        
        # Check if argument is an integer (ImpactScore threshold)
        try:
            impact_threshold = int(arg)
            await message.channel.send(f"🔍 Searching for earnings with ImpactScore >= {impact_threshold}...")
            
            # Fetch earnings for next 90 days
            start_date = date.today()
            end_date = start_date + timedelta(days=90)
            earnings_all = fetch_earnings(start_date, end_date)
            
            if earnings_all.empty:
                await message.channel.send(f"❌ No earnings data available")
                return
            
            # Filter by ImpactScore
            filtered = earnings_all[earnings_all["ImpactScore"] >= impact_threshold]
            
            if filtered.empty:
                await message.channel.send(f"❌ No earnings found with ImpactScore >= {impact_threshold}")
                return
            
            # Create embed for filtered earnings
            embed = discord.Embed(
                title=f"💣 Earnings (ImpactScore >= {impact_threshold})",
                color=discord.Color.blue()
            )
            
            earnings_text = []
            for _, r in filtered.head(20).iterrows():  # Limit to top 20
                ticker = r['Ticker']
                date_str = r['Date']
                time_str = r.get('Time', '')
                revenue = r['RevenueEst'] / 1e9
                impact = r['ImpactScore']
                earnings_text.append(f"• {ticker} — {date_str} ({time_str}) — ${revenue:.0f}B (Score: {impact:.2f})")
            
            embed.add_field(
                name=f"Found {len(filtered)} earnings",
                value="\n".join(earnings_text) if earnings_text else "None",
                inline=False
            )
            
            await message.channel.send(embed=embed)
            return
        
        except ValueError:
            # Not an integer, check if it's a date
            try:
                datetime.strptime(arg, "%Y-%m-%d")
                # It's a date - generate report for that date
                await message.channel.send("📊 Generating weekly risk report...")
                macro, earnings, dashboard = generate_sunday_report(arg)
                embed = create_sunday_report_embed(macro, earnings, dashboard)
                if embed:
                    await message.channel.send(embed=embed)
                    await message.channel.send("✅ Weekly report sent!")
                else:
                    text = format_for_discord(macro, earnings, dashboard)
                    await message.channel.send(text)
                    await message.channel.send("✅ Weekly report sent!")
                return
            except ValueError:
                # Not a date either - treat as ticker
                ticker = arg
                await message.channel.send(f"🔍 Looking up earnings for **{ticker}**...")

                from sunday_report.next_earnings import fetch_next_earnings
                info = fetch_next_earnings(ticker)

                if info is None:
                    await message.channel.send(f"❌ No upcoming earnings found for **{ticker}**")
                    return

                embed = discord.Embed(
                    title=f"📅 Next Earnings: {info['ticker']}",
                    color=discord.Color.green()
                )
                embed.add_field(name="Date", value=info["date"], inline=True)
                embed.add_field(name="Session", value=info["session"], inline=True)

                if info["eps_estimate"] is not None:
                    embed.add_field(name="EPS Estimate", value=f"{info['eps_estimate']:.2f}", inline=True)

                if info["beat_count"] is not None:
                    beat_str = f"{info['beat_count']}/{info['total_quarters']}  avg {info['avg_beat_pct']:+.1f}%"
                    embed.add_field(name="Beat History", value=beat_str, inline=True)

                if info["implied_move_pct"] is not None:
                    move_str = f"±${info['implied_move_dollar']}  ({info['implied_move_pct']:.1f}%)"
                    embed.add_field(name="Implied Move", value=move_str, inline=True)

                await message.channel.send(embed=embed)
                return
            
    except Exception as e:
        await message.channel.send(f"❌ Error: {e}")
        print(f"[!earnings] Error: {e}")
        import traceback
        traceback.print_exc()