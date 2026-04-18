---
name: weather
description: "Get weather forecasts and current conditions for any city or location"
user-invocable: true
---

# Weather Skill

When the user asks about weather, temperature, or forecasts, use the `exec_command` tool.

## Current Weather

```bash
curl -s "wttr.in/{location}?format=3"
```

Example: `curl -s "wttr.in/Shanghai?format=3"`

## Detailed Forecast

```bash
curl -s "wttr.in/{location}"
```

This returns a 3-day forecast with ASCII art.

## JSON Format

```bash
curl -s "wttr.in/{location}?format=j1"
```

Returns structured JSON with temperature, humidity, wind, and description.

## Notes

- Replace `{location}` with city name, airport code, or coordinates
- For Chinese cities, use pinyin: Beijing, Shanghai, Guangzhou
- If the user asks about multiple locations, run separate commands for each
