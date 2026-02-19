# bot/discord_client.py
"""
Discord bot client — sends messages using bot token.
"""
import discord
import os


DISCORD_TOKEN   = os.getenv("DISCORD_TOKEN")
DISCORD_CHANNEL = int(os.getenv("DISCORD_CHANNEL_ID"))

intents = discord.Intents.default()
intents.message_content = True  # Required to read message content in discord.py 2.x
client  = discord.Client(intents=intents)


async def send_message(message: str = None, embed: discord.Embed = None):
    channel = client.get_channel(DISCORD_CHANNEL)
    if channel is None:
        print(f"[DISCORD] Channel {DISCORD_CHANNEL} not found")
        return
    if embed:
        await channel.send(embed=embed)
    elif message:
        await channel.send(message)


def premarket_embed(date: str, premarket, vix: float) -> discord.Embed:
    from calculators.gap import GapResult
    gap = premarket.gap

    color = discord.Color.green() if gap.gap_direction == "up" else \
            discord.Color.red()   if gap.gap_direction == "down" else \
            discord.Color.default()

    embed = discord.Embed(
        title=f"📊 SPY Pre-Market Summary — {date}",
        color=color
    )
    embed.add_field(name="Gap",       value=f"{gap.gap_pct:+.2f}% ({gap.gap_direction})", inline=True)
    embed.add_field(name="Qualifies", value="✅" if gap.gap_qualifies else "❌",           inline=True)
    embed.add_field(name="VIX",       value=f"{vix:.1f} — {'✅ in range' if premarket.vix_in_range else '❌ out of range'}", inline=True)
    embed.add_field(name="Catalyst",  value=premarket.catalyst_name or "None",             inline=True)
    embed.add_field(name="Score",     value=f"{premarket.score:.1f}",                      inline=True)
    embed.add_field(name="Pass",      value="✅" if premarket.pass_filter else "❌",        inline=True)
    return embed


def orb_embed(date: str, orb_result, orb_range) -> discord.Embed:
    color = discord.Color.green() if orb_result.direction == "long" else \
            discord.Color.red()   if orb_result.direction == "short" else \
            discord.Color.blue()

    embed = discord.Embed(
        title=f"📐 SPY ORB Analysis — {date}",
        color=color
    )
    embed.add_field(name="ORB High",   value=f"{orb_range.high:.2f}",                          inline=True)
    embed.add_field(name="ORB Low",    value=f"{orb_range.low:.2f}",                           inline=True)
    embed.add_field(name="Range %",    value=f"{orb_range.range_pct:.3f}%",                    inline=True)
    embed.add_field(name="Breakout",   value="✅" if orb_result.breakout else "❌ Not yet",     inline=True)
    embed.add_field(name="Direction",  value=orb_result.direction.upper(),                      inline=True)
    if orb_result.breakout:
        embed.add_field(name="Trigger Price", value=f"{orb_result.trigger_price:.2f}",         inline=True)
    return embed


def signal_embed(signal, bars=None) -> discord.Embed:
    """
    Create a Discord embed for a trade signal.
    
    Args:
        signal: TradeSignal object
        bars: Optional DataFrame with bars (used to extract trigger time if ORB doesn't have it)
    """
    color = discord.Color.green() if signal.direction == "long" else discord.Color.red()

    embed = discord.Embed(
        title=f"🚨 SIGNAL — SPY {signal.direction.upper()}",
        color=color
    )
    
    # Get trigger time from ORB result or bars DataFrame
    trigger_time_str = "N/A"
    if signal.intraday and signal.intraday.orb:
        trigger_time = signal.intraday.orb.trigger_time
        
        # If ORB has trigger_time, use it
        if trigger_time is not None:
            import pytz
            import pandas as pd
            EASTERN = pytz.timezone("US/Eastern")
            
            # Handle different timestamp types
            if isinstance(trigger_time, pd.Timestamp):
                if trigger_time.tz is None:
                    trigger_time = EASTERN.localize(trigger_time)
                else:
                    trigger_time = trigger_time.tz_convert(EASTERN)
                trigger_time_str = trigger_time.strftime("%I:%M:%S %p ET")
        # Fallback: extract from bars DataFrame if available
        elif bars is not None and signal.trigger_bar_idx >= 0 and signal.trigger_bar_idx < len(bars):
            try:
                import pytz
                EASTERN = pytz.timezone("US/Eastern")
                # Get timestamp from bars index
                bar_index = bars.index[signal.trigger_bar_idx]
                if isinstance(bar_index, tuple):
                    # MultiIndex: (symbol, timestamp)
                    ts = bar_index[1]
                else:
                    ts = bar_index
                
                if hasattr(ts, 'tz_localize') or hasattr(ts, 'tz_convert'):
                    if ts.tz is None:
                        ts = EASTERN.localize(ts)
                    else:
                        ts = ts.tz_convert(EASTERN)
                    trigger_time_str = ts.strftime("%I:%M:%S %p ET")
            except Exception as e:
                print(f"[Discord] Could not extract trigger time from bars: {e}")
    
    embed.add_field(name="Entry Price",  value=f"{signal.entry_price:.2f}",    inline=True)
    embed.add_field(name="Trigger Time", value=trigger_time_str,                inline=True)
    embed.add_field(name="Score",        value=f"{signal.total_score:.1f}",    inline=True)
    embed.add_field(name="Confidence",   value=signal.confidence,              inline=True)
    embed.add_field(name="VWAP Side",    value=signal.intraday.vwap.side,      inline=True)
    embed.add_field(name="Vol Confirm",  value="✅" if signal.intraday.volume_confirmed else "❌", inline=True)
    embed.add_field(name="Gap Held",     value="✅" if signal.intraday.gap_held else "❌",         inline=True)
    embed.add_field(name="Reason",       value=signal.reason,                  inline=False)
    return embed


def exit_embed(exit_signal, position) -> discord.Embed:
    """
    Create a Discord embed for position exit notifications.
    
    Args:
        exit_signal: ExitSignal object
        position: Position object
    """
    import discord
    
    color = discord.Color.blue() if exit_signal.tier == 1 else \
            discord.Color.orange() if exit_signal.tier == 2 else \
            discord.Color.red() if exit_signal.tier == 3 else \
            discord.Color.dark_red()
    
    title = f"📤 EXIT — Tier {exit_signal.tier}"
    if exit_signal.tier == 0:
        title = "🛑 STOP LOSS"
    
    embed = discord.Embed(title=title, color=color)
    embed.add_field(name="Reason", value=exit_signal.reason, inline=False)
    embed.add_field(name="Exit Price", value=f"${exit_signal.price:.2f}", inline=True)
    embed.add_field(name="Entry Price", value=f"${position.entry_price:.2f}", inline=True)
    
    # Format exit time
    if hasattr(exit_signal.time, 'strftime'):
        exit_time_str = exit_signal.time.strftime("%I:%M:%S %p ET")
    else:
        exit_time_str = str(exit_signal.time)
    
    # Format entry time
    if hasattr(position.entry_time, 'strftime'):
        entry_time_str = position.entry_time.strftime("%I:%M:%S %p ET")
    else:
        entry_time_str = str(position.entry_time)
    
    # Calculate P&L
    pnl_pct = position.current_pnl_pct(exit_signal.price)
    pnl_dollar = abs(exit_signal.price - position.entry_price)
    
    if exit_signal.exit_all:
        embed.add_field(name="P&L", value=f"{pnl_pct:+.2f}% (${pnl_dollar:.2f})", inline=True)
    else:
        # Partial exit (Tier 1 = 33%, Tier 2 = 33%)
        if exit_signal.tier == 1:
            pct_exit = 33
            remaining = 67
        else:  # Tier 2
            pct_exit = 33
            remaining = 34
        embed.add_field(name=f"P&L ({pct_exit}%)", value=f"{pnl_pct:+.2f}% (${pnl_dollar * pct_exit / 100:.2f})", inline=True)
        embed.add_field(name="Remaining", value=f"{remaining}% position", inline=True)
    
    embed.add_field(name="Entry Time", value=entry_time_str, inline=True)
    embed.add_field(name="Exit Time", value=exit_time_str, inline=True)
    
    # Add position details
    direction_emoji = "📈" if position.direction == "long" else "📉"
    embed.add_field(
        name="Position",
        value=f"{direction_emoji} {position.direction.upper()} | Risk: ${position.risk:.2f}",
        inline=False
    )
    
    return embed


def technical_signal_embed(signal) -> discord.Embed:
    """
    Create a Discord embed for technical signal notifications.
    
    Args:
        signal: TechnicalSignal object
    """
    color = discord.Color.green() if signal.direction == "bullish" else discord.Color.red()
    
    # Signal type emoji
    emoji_map = {
        "ma20_cross": "📈",
        "resistance_break": "🚀",
        "support_break": "📉",
    }
    emoji = emoji_map.get(signal.signal_type, "📊")
    
    title = f"{emoji} Technical Signal — {signal.signal_type.replace('_', ' ').title()}"
    
    embed = discord.Embed(title=title, color=color)
    embed.add_field(name="Description", value=signal.description, inline=False)
    embed.add_field(name="Level", value=f"${signal.level:.2f}", inline=True)
    embed.add_field(name="Price", value=f"${signal.price:.2f}", inline=True)
    embed.add_field(name="Direction", value=signal.direction.upper(), inline=True)
    
    # Format time
    if hasattr(signal.time, 'strftime'):
        time_str = signal.time.strftime("%I:%M:%S %p ET")
    else:
        time_str = str(signal.time)
    embed.add_field(name="Time", value=time_str, inline=True)
    
    return embed