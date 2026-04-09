from flask import Flask, request, jsonify
from flask_cors import CORS
from anthropic import Anthropic
import os
import json
import re
import requests as http_requests
from ddgs import DDGS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder='static')
CORS(app)
client = Anthropic()

KALSHI_BASE = 'https://api.elections.kalshi.com/trade-api/v2'


def extract_tickers(url):
    """Extract market ticker AND event ticker from a Kalshi URL."""
    url = url.strip().rstrip('/')
    clean = url.replace('https://', '').replace('http://', '')
    parts = [p for p in clean.split('/') if p and p != 'markets' and '.' not in p.replace('kalshi', '')]

    # parts might be like ['kxnextag', 'next-ag', 'kxnextag-29']
    # or ['kxfedrate', 'kxfedrate-25jun']
    tickers = []
    event_ticker = None
    for part in parts:
        upper = part.upper()
        if upper.startswith('KX') or upper.startswith('INX') or '-' in upper:
            tickers.append(upper)
            # The shortest KX* one without a dash is likely the event/series ticker
            if upper.startswith('KX') and '-' not in upper and not event_ticker:
                event_ticker = upper

    # Market ticker is the most specific one (usually last, with a dash)
    market_ticker = tickers[-1] if tickers else url.split('/')[-1].upper()

    return market_ticker, event_ticker


def parse_market(m):
    """Parse a Kalshi market object into our format."""
    # yes_sub_title has the candidate/option name for multi-outcome markets
    option_name = m.get('yes_sub_title', '') or m.get('subtitle', '')
    return {
        'ticker': m.get('ticker', ''),
        'title': m.get('title', ''),
        'subtitle': m.get('subtitle', ''),
        'option_name': option_name,
        'yes_bid': m.get('yes_bid_dollars', ''),
        'yes_ask': m.get('yes_ask_dollars', ''),
        'no_bid': m.get('no_bid_dollars', ''),
        'no_ask': m.get('no_ask_dollars', ''),
        'last_price': m.get('last_price_dollars', ''),
        'volume': m.get('volume', 0),
        'open_interest': m.get('open_interest', 0),
        'status': m.get('status', ''),
        'close_time': m.get('close_time', ''),
        'category': m.get('category', ''),
        'rules': m.get('rules_primary', ''),
    }


def fetch_kalshi_market(market_ticker, event_ticker=None):
    """Pull real market data from Kalshi API. Returns (single_market, sibling_markets)."""
    single = None
    siblings = []

    # Try fetching the specific market
    try:
        r = http_requests.get(f'{KALSHI_BASE}/markets/{market_ticker}', timeout=10)
        if r.status_code == 200:
            single = parse_market(r.json().get('market', {}))
    except Exception:
        pass

    # Try fetching sibling markets via event ticker (for multi-outcome markets)
    if event_ticker:
        try:
            r = http_requests.get(f'{KALSHI_BASE}/markets', params={
                'event_ticker': event_ticker,
                'status': 'open',
                'limit': 20,
            }, timeout=10)
            if r.status_code == 200:
                for m in r.json().get('markets', []):
                    parsed = parse_market(m)
                    siblings.append(parsed)
                    # If we didn't get a single market, use the one matching our ticker
                    if not single and m.get('ticker', '').upper() == market_ticker:
                        single = parsed
        except Exception:
            pass

    # If we still don't have a single market, try the event ticker as a series
    if not single and event_ticker:
        try:
            r = http_requests.get(f'{KALSHI_BASE}/markets', params={
                'series_ticker': event_ticker,
                'status': 'open',
                'limit': 20,
            }, timeout=10)
            if r.status_code == 200:
                markets = r.json().get('markets', [])
                if markets:
                    siblings = [parse_market(m) for m in markets]
                    # Pick the one matching our ticker, or first one
                    for m in markets:
                        if m.get('ticker', '').upper() == market_ticker:
                            single = parse_market(m)
                            break
                    if not single:
                        single = siblings[0]
        except Exception:
            pass

    return single, siblings


def simplify_search_query(title, option_name=''):
    """Extract key terms from a Kalshi market title for better search results.

    Strips filler words like 'Will', 'by', dates, question marks, and other
    noise that causes DuckDuckGo to return 0 results.
    """
    q = title
    # Remove leading "Will" and trailing question marks
    q = re.sub(r'^Will\s+', '', q, flags=re.IGNORECASE)
    q = q.replace('?', '')
    # Remove date-like fragments: "by December 31, 2026", "on June 5", "before March 2025", etc.
    q = re.sub(r'\b(by|before|after|on|during)\s+\w+\s+\d{1,2},?\s*\d{0,4}', '', q, flags=re.IGNORECASE)
    q = re.sub(r'\b(by|before|after|on|during)\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s*\d{0,4}', '', q, flags=re.IGNORECASE)
    q = re.sub(r'\b(by|before|after|on)\s+end\s+of\s+\d{4}', '', q, flags=re.IGNORECASE)
    # Remove standalone years
    q = re.sub(r'\b20\d{2}\b', '', q)
    # Remove filler words
    q = re.sub(r'\b(be|the|a|an|in|at|to|for|of|next|this)\b', '', q, flags=re.IGNORECASE)
    # Collapse whitespace
    q = re.sub(r'\s+', ' ', q).strip()
    # For multi-outcome markets, prepend the candidate/option name if not already present
    if option_name and option_name.lower() not in q.lower():
        q = option_name + ' ' + q
    return q


def search_news(query, max_results=8):
    """Search DuckDuckGo for recent news on the topic.

    Tries the given query first; if no results, falls back to a broader
    version using only the first few keywords.
    """
    def _fetch(q, timelimit, n):
        try:
            results = DDGS().news(
                q,
                timelimit=timelimit,
                max_results=n
            )
            return [{'title': r.get('title', ''), 'body': r.get('body', ''), 'source': r.get('source', ''), 'date': r.get('date', '')} for r in results]
        except Exception as e:
            print(f'[search_news] Error for "{q}": {e}')
            return []

    # Try week-scoped first (most relevant)
    results = _fetch(query, 'w', max_results)
    if results:
        return results
    # Broaden: try month-scoped with same query
    results = _fetch(query, 'm', max_results)
    if results:
        return results
    # Broaden further: use only first 4 keywords, month-scoped
    shorter = ' '.join(query.split()[:4])
    if shorter != query:
        results = _fetch(shorter, 'm', max_results)
        if results:
            return results
    return []


def search_web(query, max_results=5):
    """Search DuckDuckGo for web results on the topic.

    Falls back to a shorter query if the initial one returns nothing.
    """
    def _fetch(q, timelimit, n):
        try:
            results = DDGS().text(
                q,
                timelimit=timelimit,
                max_results=n
            )
            return [{'title': r.get('title', ''), 'body': r.get('body', ''), 'href': r.get('href', '')} for r in results]
        except Exception as e:
            print(f'[search_web] Error for "{q}": {e}')
            return []

    results = _fetch(query, 'w', max_results)
    if results:
        return results
    results = _fetch(query, 'm', max_results)
    if results:
        return results
    # Broaden: fewer keywords, no timelimit
    shorter = ' '.join(query.split()[:4])
    if shorter != query:
        results = _fetch(shorter, None, max_results)
        if results:
            return results
    return []


def build_prompt(url, kalshi_data, siblings, news, web_results):
    """Build an enriched prompt with real data for Claude."""
    sections = []

    sections.append(f'You are EdgeFinder, a prediction market analysis tool. A user submitted this Kalshi URL: {url}')

    is_multi = len(siblings) > 1

    if kalshi_data:
        last = kalshi_data.get('last_price', '')
        yes_pct = ''
        if last:
            try:
                yes_pct = str(round(float(last) * 100))
            except (ValueError, TypeError):
                pass

        sections.append(f"""
LIVE KALSHI MARKET DATA (real-time, pulled just now):
- Title: {kalshi_data.get('title', 'Unknown')}
- Subtitle: {kalshi_data.get('subtitle', '')}
- Last traded price: {last} (YES at {yes_pct}%)
- YES bid/ask: {kalshi_data.get('yes_bid', '?')} / {kalshi_data.get('yes_ask', '?')}
- NO bid/ask: {kalshi_data.get('no_bid', '?')} / {kalshi_data.get('no_ask', '?')}
- Volume: {kalshi_data.get('volume', '?')} contracts
- Open interest: {kalshi_data.get('open_interest', '?')} contracts
- Status: {kalshi_data.get('status', '?')}
- Closes: {kalshi_data.get('close_time', '?')}
- Rules: {kalshi_data.get('rules', '')[:500]}""")

    if is_multi:
        sections.append(f"""
THIS IS A MULTI-OUTCOME MARKET with {len(siblings)} candidates/options:""")
        for s in sorted(siblings, key=lambda x: float(x.get('last_price', '0') or '0'), reverse=True):
            sp = ''
            try:
                sp = str(round(float(s.get('last_price', '0')) * 100))
            except (ValueError, TypeError):
                sp = '?'
            name = s.get('option_name', '') or s.get('title', 'Unknown')
            if int(sp or 0) >= 2:  # Only show options with at least 2%
                sections.append(f'  - {name}: YES at {sp}% (vol: {s.get("volume", "?")})')

        sections.append("""
For multi-outcome markets, your suggested_position MUST name the specific candidate/option to buy, e.g. "Buy YES on Lee Zeldin" or "Buy YES on Todd Blanche". Do NOT just say "Buy YES" without specifying who/what.""")
    else:
        if kalshi_data:
            sections.append('\nUse these REAL market prices as the market_yes_pct in your response. Do NOT guess the market price.')

    if not kalshi_data:
        sections.append('Note: Could not fetch live Kalshi data for this market. Estimate the market price based on the URL slug.')

    if news:
        sections.append('\nRECENT NEWS (from the past month, via DuckDuckGo News):')
        for i, article in enumerate(news[:8], 1):
            sections.append(f'{i}. [{article["source"]}] {article["title"]}')
            if article['body']:
                sections.append(f'   {article["body"][:200]}')

    if web_results:
        sections.append('\nWEB SEARCH RESULTS (recent, from DuckDuckGo):')
        for i, r in enumerate(web_results[:5], 1):
            sections.append(f'{i}. {r["title"]}')
            if r['body']:
                sections.append(f'   {r["body"][:200]}')

    sections.append("""
Using the REAL market data above and the recent news context, analyze whether this contract is mispriced. Ground your analysis in the specific news articles and data provided — do not make up facts.

Respond ONLY with valid JSON in exactly this format, no other text:
{{
  "contract_title": "Plain English description of what the contract is asking",
  "category": "One of: ECONOMICS, MARKETS, CRYPTO, SPORTS, TECH, POLITICS, CLIMATE",
  "market_yes_pct": <integer — use the REAL last traded price from the data above>,
  "model_yes_pct": <integer — your estimate of the TRUE probability based on the news and data>,
  "edge_signal": "One of: EDGE FOUND, SLIGHT EDGE, NO EDGE",
  "confidence": "One of: High, Medium, Low",
  "verdict_summary": "2 sentences max. Reference specific news articles or data points.",
  "evidence": [
    {{
      "signal": "One of: SUPPORTING, OPPOSING, NEUTRAL",
      "title": "Short label for this piece of evidence, 4-6 words",
      "detail": "1-2 sentences. Cite specific news sources or data."
    }},
    {{
      "signal": "One of: SUPPORTING, OPPOSING, NEUTRAL",
      "title": "Short label",
      "detail": "1-2 sentences."
    }},
    {{
      "signal": "One of: SUPPORTING, OPPOSING, NEUTRAL",
      "title": "Short label",
      "detail": "1-2 sentences."
    }}
  ],
  "suggested_position": "For multi-outcome: 'Buy YES on [specific name/option]'. For binary: 'Buy YES', 'Buy NO', or 'No position'",
  "entry_price": "Use the real YES ask price, e.g. $0.47 per contract",
  "target_exit": "Based on your model probability, e.g. $0.65 if probability converges"
}}

Edge signal rules:
- EDGE FOUND: model differs from market by 8+ percentage points
- SLIGHT EDGE: model differs by 4-7 percentage points
- NO EDGE: model differs by less than 4 percentage points""")

    return '\n'.join(sections)


@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/analysis')
def analysis():
    resp = app.send_static_file('analysis.html')
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    return resp


@app.route('/history')
def history():
    return app.send_static_file('history.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    body = request.get_json(silent=True)
    if not body or not body.get('url'):
        return jsonify({'success': False, 'error': 'Missing url parameter'}), 400

    url = body['url'].strip()

    # 1. Extract tickers and fetch real market data
    market_ticker, event_ticker = extract_tickers(url)
    kalshi_data, siblings = fetch_kalshi_market(market_ticker, event_ticker)

    # For multi-outcome markets, use the TOP candidate as the focal market
    if siblings and len(siblings) > 1:
        top = max(siblings, key=lambda x: float(x.get('last_price', '0') or '0'))
        # If our specific ticker is a low-probability sub-market, switch to the leader
        if kalshi_data:
            try:
                our_pct = float(kalshi_data.get('last_price', '0') or '0')
                top_pct = float(top.get('last_price', '0') or '0')
                if our_pct < 0.05 and top_pct > our_pct:
                    kalshi_data = top
            except (ValueError, TypeError):
                kalshi_data = top
        else:
            kalshi_data = top

    # 2. Build a search query from the market title or URL
    raw_title = ''
    option_name = ''
    if kalshi_data and kalshi_data.get('title'):
        raw_title = kalshi_data['title']
        option_name = kalshi_data.get('option_name', '')
    elif siblings and siblings[0].get('title'):
        raw_title = siblings[0]['title']
        option_name = siblings[0].get('option_name', '')

    if raw_title:
        search_query = simplify_search_query(raw_title, option_name)
    else:
        # Derive query from URL slug
        slug = market_ticker.lower().replace('-', ' ').replace('kx', '')
        search_query = slug

    # 3. Fetch news and web results
    news = search_news(search_query)
    web_results = search_web(search_query + ' prediction forecast')

    # 4. Build enriched prompt and call Claude
    prompt = build_prompt(url, kalshi_data, siblings, news, web_results)

    try:
        message = client.messages.create(
            model='claude-sonnet-4-20250514',
            max_tokens=1024,
            messages=[{
                'role': 'user',
                'content': prompt
            }],
            timeout=30.0
        )

        raw = message.content[0].text.strip()

        # Strip markdown fencing if Claude wraps it
        if raw.startswith('```'):
            raw = raw.split('\n', 1)[1]
            if raw.endswith('```'):
                raw = raw[:-3]
            raw = raw.strip()

        data = json.loads(raw)

        # Build similar markets from siblings (excluding the current market)
        similar = []
        for sib in siblings:
            if sib.get('ticker', '').upper() == market_ticker.upper():
                continue
            try:
                yes_pct = int(round(float(sib.get('last_price', '0') or '0') * 100))
            except (ValueError, TypeError):
                yes_pct = 0
            has_edge = abs(yes_pct - 50) > 8
            name = sib.get('option_name', '') or sib.get('title', 'Unknown')
            similar.append({
                'name': name,
                'yes_pct': yes_pct,
                'has_edge': has_edge,
            })
            if len(similar) >= 3:
                break
        data['similar_markets'] = similar

        # Attach metadata about what data sources we used
        data['_sources'] = {
            'kalshi_live': kalshi_data is not None,
            'news_count': len(news),
            'web_count': len(web_results),
            'ticker': market_ticker,
            'event_ticker': event_ticker,
            'sibling_count': len(siblings),
        }

        return jsonify({'success': True, 'data': data})

    except json.JSONDecodeError:
        return jsonify({'success': False, 'error': 'Failed to parse model response as JSON'}), 502
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=int(os.getenv('PORT', 8080)))
