# EdgeFinder

Prediction market edge analysis tool. Paste a Kalshi market URL, get a structured analysis of whether the contract is mispriced.

## How it works

1. Extracts the market ticker from a Kalshi URL
2. Pulls **live market data** from the Kalshi API (prices, volume, order book) — no auth required
3. Searches **DuckDuckGo News** for recent articles about the market topic
4. Sends everything to **Claude** for structured analysis
5. Returns an edge signal, evidence breakdown, probability comparison, and suggested position

Supports both binary (YES/NO) and multi-outcome markets (e.g. "Who will be the next AG?" with named candidates).

## Stack

- **Backend:** Flask (Python)
- **Frontend:** Vanilla HTML/CSS/JS — dark terminal aesthetic
- **AI:** Claude Sonnet 4 via Anthropic API
- **Data:** Kalshi API (free, no auth) + DuckDuckGo search (free, no key)

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Add your Anthropic API key to .env
python3 app.py
```

Open `http://localhost:8080`

## Screens

- **Landing page** (`/`) — paste a Kalshi URL, click "Find edge"
- **Analysis** (`/analysis`) — full breakdown with edge signal, evidence, model vs market bars, suggested position
- **History** (`/history`) — saved analyses with filters and stats
