"""
Agent-based edge detection system for prediction markets.

Architecture:
  1. Four specialized research agents run in parallel (using ThreadPoolExecutor)
  2. A synthesis engine combines their reports into an independent probability estimate
     WITHOUT seeing the market price (prevents anchoring bias)
  3. An edge scorer compares the model's estimate against the actual market price
"""

import json
import re
import time
from datetime import datetime, timezone
from anthropic import Anthropic
from ddgs import DDGS

client = Anthropic()

HAIKU_MODEL = 'claude-haiku-4-5-20251001'
SONNET_MODEL = 'claude-sonnet-4-20250514'


# ---------------------------------------------------------------------------
# Market Classification
# ---------------------------------------------------------------------------

def classify_market(kalshi_data, siblings, option_name=''):
    """Detect market type to configure the agent pipeline.

    Returns a profile dict with:
      timeframe: flash/short/medium/long
      structure: binary/range_bucket/above_below/candidate
      category: from Kalshi API
      skip_contrarian: bool
      skip_cross_market: bool
      needs_current_price: bool
      current_price: float or None (estimated from sibling distribution)
    """
    now = datetime.now(timezone.utc)
    category = (kalshi_data or {}).get('category', 'MARKETS') or 'MARKETS'

    # --- Timeframe ---
    timeframe = 'long'
    close_time = (kalshi_data or {}).get('close_time', '')
    if close_time:
        try:
            close_dt = datetime.fromisoformat(close_time.replace('Z', '+00:00'))
            delta = close_dt - now
            hours = delta.total_seconds() / 3600
            if hours <= 4:
                timeframe = 'flash'
            elif hours <= 7 * 24:
                timeframe = 'short'
            elif hours <= 90 * 24:
                timeframe = 'medium'
        except Exception:
            pass

    # --- Structure ---
    is_multi = siblings and len(siblings) > 1
    structure = 'binary'
    if is_multi and option_name:
        name_lower = option_name.lower()
        # Detect above/below: "$80,800 or above", "72,000 or below", "or above", "or below"
        if 'or above' in name_lower or 'or below' in name_lower or 'or more' in name_lower or 'or less' in name_lower:
            structure = 'above_below'
        # Detect numeric ranges: "70,000 to 74,999.99", "0.2% to 0.3%"
        elif re.search(r'\d.*to\s+\d', name_lower):
            structure = 'range_bucket'
        # Detect "at least X" patterns (e.g., "At least 95 days")
        elif 'at least' in name_lower:
            structure = 'above_below'
        else:
            # Named candidates/options
            structure = 'candidate'
    elif is_multi:
        structure = 'range_bucket'  # multi-outcome without clear option name

    # --- Needs current price? ---
    price_categories = {'CRYPTO', 'MARKETS'}
    price_structures = {'above_below', 'range_bucket'}
    needs_current_price = category in price_categories and structure in price_structures

    # --- Estimate current price from sibling distribution ---
    current_price = None
    if needs_current_price and siblings:
        current_price = _estimate_current_price(siblings)

    # --- Agent skip logic ---
    skip_contrarian = timeframe == 'flash'
    skip_cross_market = timeframe in ('flash', 'short')

    return {
        'timeframe': timeframe,
        'structure': structure,
        'category': category,
        'needs_current_price': needs_current_price,
        'current_price': current_price,
        'skip_contrarian': skip_contrarian,
        'skip_cross_market': skip_cross_market,
    }


def _estimate_current_price(siblings):
    """Estimate the current underlying price from the sibling distribution.

    For above/below markets, find the contract closest to 50% — that threshold
    is approximately the current price. For range buckets, find the highest-
    probability bucket's midpoint.
    """
    best_contract = None
    best_distance = 100

    for s in siblings:
        try:
            pct = float(s.get('last_price', '0') or '0') * 100
        except (ValueError, TypeError):
            continue

        name = s.get('option_name', '') or s.get('title', '')
        distance = abs(pct - 50)

        if distance < best_distance:
            best_distance = distance
            best_contract = name

    if not best_contract:
        return None

    # Extract a number from the contract name
    # e.g., "$72,000 or above" → 72000, "70,000 to 74,999.99" → 72500
    numbers = re.findall(r'[\d,]+\.?\d*', best_contract.replace(',', ''))
    if not numbers:
        return None

    try:
        nums = [float(n) for n in numbers]
        if len(nums) >= 2:
            return round((nums[0] + nums[1]) / 2, 2)  # midpoint of range
        return nums[0]
    except (ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# Shared search helpers
# ---------------------------------------------------------------------------

def _search_news(query, timelimit='w', max_results=8):
    """Search DuckDuckGo News with timeout protection."""
    try:
        results = list(DDGS().news(query, timelimit=timelimit, max_results=max_results))
        return [{'title': r.get('title', ''), 'body': r.get('body', ''),
                 'source': r.get('source', ''), 'date': r.get('date', '')} for r in results]
    except Exception as e:
        print(f'[_search_news] Error for "{query}": {e}')
        return []


def _search_web(query, timelimit='w', max_results=5):
    """Search DuckDuckGo Web with timeout protection."""
    try:
        results = list(DDGS().text(query, timelimit=timelimit, max_results=max_results))
        return [{'title': r.get('title', ''), 'body': r.get('body', ''),
                 'href': r.get('href', '')} for r in results]
    except Exception as e:
        print(f'[_search_web] Error for "{query}": {e}')
        return []


def _call_llm(prompt, model=HAIKU_MODEL, max_tokens=1024):
    """Call Claude and return the text response."""
    try:
        msg = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{'role': 'user', 'content': prompt}],
            timeout=20.0,
        )
        return msg.content[0].text.strip()
    except Exception as e:
        print(f'[_call_llm] Error: {e}')
        return ''


def _parse_json(text):
    """Best-effort JSON extraction from LLM output."""
    text = text.strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[1]
        if text.endswith('```'):
            text = text[:-3]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Agent 1 — News Researcher
# ---------------------------------------------------------------------------

def _extract_search_question(question):
    """Strip multi-outcome context from question for search queries.

    The question may contain '— Specific contract: ...' and distribution info
    appended by the orchestrator. We want the base question for web searches.
    """
    base = question.split('—')[0].split('\n')[0].strip()
    return base


def agent_news_researcher(question, category, option_name=''):
    """Deep multi-query news research from different angles."""
    t0 = time.time()

    search_q = _extract_search_question(question)

    # 1 news + 1 web search to minimize DDG load
    all_news = _search_news(search_q, 'w', 8)
    if not all_news:
        all_news = _search_news(search_q, 'm', 8)
    all_web = _search_web(search_q, 'w', 5)

    # Summarize with LLM
    news_text = '\n'.join(
        f'- [{a["source"]}] {a["title"]}: {a["body"][:200]}'
        for a in all_news[:12]
    )
    web_text = '\n'.join(
        f'- {r["title"]}: {r["body"][:200]}'
        for r in all_web[:8]
    )

    prompt = f"""You are a news research agent for prediction market analysis.

QUESTION: {question}
CATEGORY: {category}

NEWS ARTICLES:
{news_text or '(no news found)'}

WEB RESULTS:
{web_text or '(no web results)'}

Analyze these sources and extract the key facts relevant to this prediction market question.

Respond ONLY with JSON:
{{
  "facts": ["fact 1 with source citation", "fact 2", ...],
  "sentiment": "bullish or bearish or neutral",
  "key_developments": ["most important recent development 1", "development 2"],
  "source_count": <number of unique sources found>,
  "freshness": "hours or days or weeks since most recent relevant news"
}}"""

    raw = _call_llm(prompt, max_tokens=512)
    report = _parse_json(raw) or {
        'facts': [a['title'] for a in all_news[:5]],
        'sentiment': 'neutral',
        'key_developments': [],
        'source_count': len(all_news),
        'freshness': 'unknown',
    }
    report['_agent'] = 'news_researcher'
    report['_duration'] = round(time.time() - t0, 1)
    report['_raw_news_count'] = len(all_news)
    report['_raw_web_count'] = len(all_web)
    return report


# ---------------------------------------------------------------------------
# Agent 2 — Category-Specific Data Researcher
# ---------------------------------------------------------------------------

CATEGORY_QUERIES = {
    'POLITICS': [
        '{q} polling data',
        '{q} approval rating',
        '{q} political analysis forecast',
    ],
    'ECONOMICS': [
        '{q} economic data indicators',
        '{q} Fed forecast analyst',
        '{q} economic outlook projection',
    ],
    'MARKETS': [
        '{q} market analysis forecast',
        '{q} analyst prediction outlook',
        '{q} financial data trend',
    ],
    'CRYPTO': [
        '{q} crypto market analysis',
        '{q} blockchain regulatory news',
        '{q} on-chain data sentiment',
    ],
    'SPORTS': [
        '{q} stats odds analysis',
        '{q} injury report lineup',
        '{q} betting line prediction',
    ],
    'TECH': [
        '{q} tech industry analysis',
        '{q} product launch timeline',
        '{q} analyst forecast prediction',
    ],
    'CLIMATE': [
        '{q} weather forecast data NOAA',
        '{q} climate science projection',
        '{q} environmental data trend',
    ],
}


def agent_data_researcher(question, category, option_name=''):
    """Category-specific data gathering — polls, economic indicators, stats, etc."""
    t0 = time.time()

    search_q = _extract_search_question(question)
    query_templates = CATEGORY_QUERIES.get(category, CATEGORY_QUERIES['MARKETS'])
    base_q = search_q
    if option_name and option_name.lower() not in search_q.lower():
        base_q = f'{option_name} {search_q}'

    # Single search with the most relevant category query
    q = query_templates[0].format(q=base_q)
    all_results = _search_web(q, 'm', 8)

    results_text = '\n'.join(
        f'- {r["title"]}: {r["body"][:250]}'
        for r in all_results[:15]
    )

    cat_instruction = {
        'POLITICS': 'Focus on polling data, approval ratings, endorsements, and political expert predictions.',
        'ECONOMICS': 'Focus on economic indicators (CPI, jobs, GDP), Fed commentary, and analyst forecasts.',
        'MARKETS': 'Focus on market data, analyst price targets, earnings forecasts, and financial trends.',
        'CRYPTO': 'Focus on on-chain data, regulatory developments, exchange flows, and market sentiment.',
        'SPORTS': 'Focus on team/player stats, injury reports, betting lines, and historical matchup data.',
        'TECH': 'Focus on product timelines, company financials, regulatory developments, and industry analysis.',
        'CLIMATE': 'Focus on weather data, scientific measurements, NOAA forecasts, and climate model projections.',
    }.get(category, 'Focus on quantitative data, expert analysis, and trend data.')

    prompt = f"""You are a data research agent specialized in {category} markets.

QUESTION: {question}
CATEGORY: {category}

{cat_instruction}

SEARCH RESULTS:
{results_text or '(no results found)'}

Extract quantitative data points and expert forecasts relevant to this prediction market question.

Respond ONLY with JSON:
{{
  "data_points": ["specific data point 1 with number/date", "data point 2", ...],
  "base_rate": "<historical base rate if applicable, e.g. 'Fed has cut rates in 8 of last 10 recessions', or null>",
  "trend": "up or down or flat",
  "expert_forecasts": ["expert/org forecast 1", "forecast 2"],
  "key_number": "<the single most important number for this question, e.g. '3.2% CPI' or '52% approval'>"
}}"""

    raw = _call_llm(prompt, max_tokens=512)
    report = _parse_json(raw) or {
        'data_points': [],
        'base_rate': None,
        'trend': 'flat',
        'expert_forecasts': [],
        'key_number': None,
    }
    report['_agent'] = 'data_researcher'
    report['_duration'] = round(time.time() - t0, 1)
    report['_result_count'] = len(all_results)
    return report


# ---------------------------------------------------------------------------
# Agent 3 — Cross-Market / Forecaster Researcher
# ---------------------------------------------------------------------------

def agent_cross_market(question, category, option_name=''):
    """Check other prediction platforms and expert forecaster consensus."""
    t0 = time.time()

    search_q = _extract_search_question(question)

    # 1 broad prediction search
    all_results = _search_web(f'{search_q} prediction market forecast probability', 'm', 8)

    results_text = '\n'.join(
        f'- {r["title"]}: {r["body"][:250]}'
        for r in all_results[:12]
    )

    prompt = f"""You are a cross-market research agent that checks what OTHER prediction platforms and expert forecasters think.

QUESTION: {question}

SEARCH RESULTS:
{results_text or '(no results found)'}

Look for:
1. Probability estimates from Metaculus, Polymarket, Good Judgment, PredictIt, or similar platforms
2. Superforecaster or expert probability estimates
3. Wisdom-of-crowds aggregations
4. Any quantitative probability estimates from credible sources

Respond ONLY with JSON:
{{
  "other_estimates": [
    {{"source": "platform/expert name", "probability": <0-100 or null>, "context": "brief note"}},
    ...
  ],
  "expert_views": ["expert opinion 1 with attribution", "opinion 2"],
  "consensus_probability": <best guess at forecaster consensus 0-100, or null if insufficient data>,
  "agreement_level": "strong or moderate or weak or no_data"
}}"""

    raw = _call_llm(prompt, max_tokens=512)
    report = _parse_json(raw) or {
        'other_estimates': [],
        'expert_views': [],
        'consensus_probability': None,
        'agreement_level': 'no_data',
    }
    report['_agent'] = 'cross_market'
    report['_duration'] = round(time.time() - t0, 1)
    report['_result_count'] = len(all_results)
    return report


# ---------------------------------------------------------------------------
# Agent 4 — Contrarian Analyst
# ---------------------------------------------------------------------------

def agent_contrarian(question, category, option_name=''):
    """Find reasons the market consensus could be WRONG."""
    t0 = time.time()

    search_q = _extract_search_question(question)

    # 1 contrarian-angled search
    all_results = _search_web(f'{search_q} risks contrarian bear case', 'm', 8)

    results_text = '\n'.join(
        f'- {r["title"]}: {r["body"][:250]}'
        for r in all_results[:12]
    )

    prompt = f"""You are a contrarian analyst. Your job is to find reasons why the CURRENT MARKET CONSENSUS could be WRONG.

QUESTION: {question}
CATEGORY: {category}

Think about:
- What risks or factors is the market likely underweighting?
- Are there historical analogies where consensus was wrong in similar situations?
- What would cause a surprise outcome?
- What tail risks exist that aren't priced in?

SEARCH RESULTS (looking for contrarian angles):
{results_text or '(no results found)'}

Respond ONLY with JSON:
{{
  "contrarian_factors": ["reason market could be wrong 1", "reason 2", ...],
  "historical_analogies": ["similar past situation where consensus was wrong 1", ...],
  "risk_factors": ["underweighted risk 1", "risk 2"],
  "surprise_probability": "<your estimate of probability that the consensus is significantly wrong, e.g. 'moderate' or 'low' or 'high'>"
}}"""

    raw = _call_llm(prompt, max_tokens=512)
    report = _parse_json(raw) or {
        'contrarian_factors': [],
        'historical_analogies': [],
        'risk_factors': [],
        'surprise_probability': 'unknown',
    }
    report['_agent'] = 'contrarian'
    report['_duration'] = round(time.time() - t0, 1)
    report['_result_count'] = len(all_results)
    return report


# ---------------------------------------------------------------------------
# Synthesis Rules Builder
# ---------------------------------------------------------------------------

def _build_synthesis_rules(profile):
    """Build context-specific rules for the synthesis prompt based on market profile."""
    if not profile:
        profile = {}

    rules = ['CRITICAL RULES:']
    timeframe = profile.get('timeframe', 'long')
    structure = profile.get('structure', 'binary')

    # Timeframe rules
    if timeframe == 'flash':
        rules.append(
            '1. TIMEFRAME: This market closes in HOURS. Only immediate factors matter. '
            'Large price moves are nearly impossible in this window. If the outcome requires '
            'a >5% move, probability should be <3%. If >10% move needed, probability should be <1%.'
        )
    elif timeframe == 'short':
        rules.append(
            '1. TIMEFRAME: This market closes in DAYS. Focus on near-term catalysts only. '
            'Discount long-term forecasts heavily. If a large move is needed in days, '
            'probability should be very low unless there is an imminent catalyst.'
        )
    elif timeframe == 'medium':
        rules.append(
            '1. TIMEFRAME: This market closes in weeks/months. Balance near-term catalysts '
            'with medium-term trends. Long-term forecasts have limited relevance.'
        )
    else:
        rules.append(
            '1. TIMEFRAME: This market has a long time horizon. Full analysis applies. '
            'Weight long-term trends, structural factors, and base rates.'
        )

    # Structure rules
    if structure == 'above_below':
        rules.append(
            '2. STRUCTURE: This is a THRESHOLD contract (above/below a specific level). '
            'Estimate the probability the value will cross the stated threshold. '
            'If a CURRENT PRICE is mentioned, calculate how far the threshold is from current '
            'price and factor in typical volatility for the timeframe.'
        )
    elif structure == 'range_bucket':
        rules.append(
            '2. STRUCTURE: This is ONE RANGE BUCKET in a multi-outcome market. '
            'Estimate probability the value falls in THIS EXACT RANGE — not above or below it. '
            'Consider the full distribution of outcomes. A single bucket rarely exceeds 15-25%.'
        )
    elif structure == 'candidate':
        rules.append(
            '2. STRUCTURE: This is one CANDIDATE/OPTION among competitors. '
            'Estimate this specific option\'s win probability relative to all competitors.'
        )
    else:
        rules.append(
            '2. STRUCTURE: This is a simple binary YES/NO question. '
            'Estimate the probability of YES.'
        )

    # Current price rule (for price-based markets)
    if profile.get('current_price'):
        rules.append(
            f'3. CURRENT PRICE: The underlying asset is currently at ~{profile["current_price"]:,.0f}. '
            f'Use this as your anchor. Calculate the percentage move required to reach the target.'
        )
    else:
        rules.append(
            '3. If the question mentions current prices or levels, use them as your anchor.'
        )

    # Platform assumption rule
    rules.append(
        '4. PLATFORM CONTEXT: Kalshi only creates markets for CONFIRMED or SCHEDULED events. '
        'If the question asks about what someone will SAY or DO during an event (speech, podcast, '
        'earnings call, press conference), assume the event IS happening. The question is about '
        'the CONTENT of the event, not whether the event occurs. Do NOT discount probability '
        'based on "no confirmation the event will happen" — it will happen.'
    )

    return '\n'.join(rules)


# ---------------------------------------------------------------------------
# Synthesis Engine — combines all agent reports WITHOUT market price
# ---------------------------------------------------------------------------

def synthesize_reports(question, category, reports, profile=None):
    """Combine all agent reports into an independent probability estimate.

    CRITICAL: This function does NOT receive the market price.
    The LLM must estimate probability purely from the gathered evidence.
    """
    # Build a CONCISE summary — keep it short to minimize LLM latency
    bullets = []

    news = reports.get('news_researcher', {})
    if news.get('facts'):
        bullets.extend(news['facts'][:4])
    if news.get('key_developments'):
        bullets.extend(news['key_developments'][:2])

    data = reports.get('data_researcher', {})
    if data.get('key_number'):
        bullets.append(f'Key stat: {data["key_number"]}')
    if data.get('base_rate'):
        bullets.append(f'Base rate: {data["base_rate"]}')
    bullets.extend(data.get('expert_forecasts', [])[:2])

    cross = reports.get('cross_market', {})
    for est in cross.get('other_estimates', [])[:3]:
        prob = est.get('probability', '?')
        bullets.append(f'{est.get("source", "?")}: {prob}%')
    if cross.get('consensus_probability') is not None:
        bullets.append(f'Forecaster consensus: ~{cross["consensus_probability"]}%')

    contra = reports.get('contrarian', {})
    bullets.extend(contra.get('contrarian_factors', [])[:2])
    bullets.extend(contra.get('risk_factors', [])[:2])

    evidence_block = '\n'.join(f'- {b}' for b in bullets) if bullets else '(Limited data)'

    prompt = f"""You are a master forecaster synthesizing research from multiple specialist agents.

QUESTION: "{question}"
CATEGORY: {category}

IMPORTANT: You must estimate the TRUE probability of this event based ONLY on the evidence below.
You do NOT know what the market price is. Do NOT guess or reference any market price.
Think like a superforecaster: decompose the question, weigh the evidence, consider base rates, and account for contrarian factors.

{_build_synthesis_rules(profile)}

EVIDENCE FROM RESEARCH AGENTS:
{evidence_block}

Based on ALL the evidence above, estimate the probability that the answer to the QUESTION is YES.

IMPORTANT: "model_probability" must be the chance of YES happening (0 = definitely won't happen,
100 = definitely will happen). If something is nearly impossible, use 1-3. If very unlikely, use 5-15.
Make sure your number MATCHES your reasoning — if you say "nearly impossible", don't output 40.

Respond ONLY with JSON:
{{
  "model_probability": <integer 0-100, probability of YES>,
  "confidence": "High or Medium or Low",
  "reasoning": "2-3 sentence explanation citing specific evidence",
  "key_factors": [
    {{"factor": "short description", "direction": "supports_yes or supports_no or neutral", "weight": "high or medium or low"}},
    ...
  ]
}}"""

    raw = _call_llm(prompt, model=HAIKU_MODEL, max_tokens=512)
    result = _parse_json(raw)

    if not result or 'model_probability' not in result:
        # Synthesis failed — mark it so the edge scorer knows not to trust it
        fallback_prob = cross.get('consensus_probability')
        result = {
            'model_probability': fallback_prob,  # None if no consensus
            'confidence': 'Low',
            'reasoning': 'Unable to synthesize a confident estimate from available data.',
            'key_factors': [],
            '_failed': True,
        }

    return result


# ---------------------------------------------------------------------------
# Edge Scorer — compares model estimate vs market price
# ---------------------------------------------------------------------------

# Finance/economics markets are more efficient (smarter participants)
# Sports/entertainment have more behavioral bias (longshot bias, fan bias)
CATEGORY_THRESHOLDS = {
    'ECONOMICS': {'edge': 10, 'slight': 6},
    'MARKETS':   {'edge': 10, 'slight': 6},
    'CRYPTO':    {'edge': 8, 'slight': 4},
    'POLITICS':  {'edge': 8, 'slight': 4},
    'SPORTS':    {'edge': 6, 'slight': 3},
    'TECH':      {'edge': 8, 'slight': 4},
    'CLIMATE':   {'edge': 8, 'slight': 4},
}

DEFAULT_THRESHOLDS = {'edge': 8, 'slight': 4}


def score_edge(synthesis, market_data, category):
    """Compare the model's independent probability estimate against market price.

    Returns the final analysis result matching the existing frontend format.
    """
    # Get market YES probability
    market_pct = 50  # fallback
    if market_data and market_data.get('last_price'):
        try:
            market_pct = round(float(market_data['last_price']) * 100)
        except (ValueError, TypeError):
            pass

    # If synthesis failed, use market price as model estimate (no edge by definition)
    synthesis_failed = synthesis.get('_failed', False)
    model_pct = synthesis.get('model_probability')
    if model_pct is None or synthesis_failed:
        model_pct = market_pct

    gap = abs(model_pct - market_pct)
    thresholds = dict(CATEGORY_THRESHOLDS.get(category, DEFAULT_THRESHOLDS))

    # High-volume markets are more efficient — raise the bar
    volume = (market_data or {}).get('volume', 0) or 0
    if volume > 10000:
        thresholds['edge'] += 2
        thresholds['slight'] += 1

    if synthesis_failed:
        edge_signal = 'NO EDGE'
    elif gap >= thresholds['edge']:
        edge_signal = 'EDGE FOUND'
    elif gap >= thresholds['slight']:
        edge_signal = 'SLIGHT EDGE'
    else:
        edge_signal = 'NO EDGE'

    # Determine position
    if edge_signal == 'NO EDGE':
        suggested_position = 'No position'
    elif model_pct > market_pct:
        suggested_position = 'Buy YES'
    else:
        suggested_position = 'Buy NO'

    # Build evidence array from key_factors
    evidence = []
    signal_map = {'supports_yes': 'SUPPORTING', 'supports_no': 'OPPOSING', 'neutral': 'NEUTRAL'}
    for kf in synthesis.get('key_factors', [])[:4]:
        evidence.append({
            'signal': signal_map.get(kf.get('direction', 'neutral'), 'NEUTRAL'),
            'title': kf.get('factor', 'Unknown factor')[:50],
            'detail': f'Weight: {kf.get("weight", "medium")}. From agent research.',
        })

    # Ensure at least 3 evidence items
    while len(evidence) < 3:
        evidence.append({
            'signal': 'NEUTRAL',
            'title': 'Limited data available',
            'detail': 'Insufficient evidence gathered for this factor.',
        })

    # Entry/exit prices
    entry_price = ''
    if market_data:
        ask = market_data.get('yes_ask', '')
        if ask:
            entry_price = f'${ask} per contract'
        elif market_data.get('last_price'):
            entry_price = f'${market_data["last_price"]} per contract'

    target_exit = ''
    if model_pct and model_pct != market_pct:
        target_exit = f'${model_pct / 100:.2f} if probability converges'

    return {
        'market_yes_pct': market_pct,
        'model_yes_pct': model_pct,
        'edge_signal': edge_signal,
        'confidence': synthesis.get('confidence', 'Medium'),
        'verdict_summary': synthesis.get('reasoning', 'Analysis complete.'),
        'evidence': evidence[:4],
        'suggested_position': suggested_position,
        'entry_price': entry_price,
        'target_exit': target_exit,
    }


# ---------------------------------------------------------------------------
# Orchestrator — runs all agents and produces final result
# ---------------------------------------------------------------------------

def run_deep_analysis(question, category, market_data=None, siblings=None,
                      option_name='', profile=None, on_progress=None):
    """Run research agents, synthesize, and score.

    Args:
        question: The market question (with context appended by orchestrator)
        category: Market category (POLITICS, ECONOMICS, etc.)
        market_data: Parsed Kalshi market data dict
        siblings: List of sibling markets for multi-outcome
        option_name: For multi-outcome markets, the specific option name
        profile: Market classification dict from classify_market()
        on_progress: Optional callback(agent_name, status) for progress updates

    Returns:
        Final analysis dict matching the existing frontend format
    """
    if not profile:
        profile = classify_market(market_data, siblings, option_name)

    def _progress(agent, status):
        if on_progress:
            try:
                on_progress(agent, status)
            except Exception:
                pass

    # Phase 1: Select agents based on market profile
    agent_fns = {
        'news_researcher': agent_news_researcher,
        'data_researcher': agent_data_researcher,
    }
    # Only add cross_market and contrarian when useful
    if not profile.get('skip_cross_market'):
        agent_fns['cross_market'] = agent_cross_market
    if not profile.get('skip_contrarian'):
        agent_fns['contrarian'] = agent_contrarian

    _progress('orchestrator', 'dispatching_agents')

    reports = {}
    for name, fn in agent_fns.items():
        _progress(name, 'started')
        try:
            reports[name] = fn(question, category, option_name)
            _progress(name, 'completed')
        except Exception as e:
            print(f'[orchestrator] Agent {name} failed: {e}')
            reports[name] = {'_agent': name, '_error': str(e)}
            _progress(name, 'failed')

    # Phase 2: Synthesize (without market price)
    _progress('synthesizer', 'started')
    synthesis = synthesize_reports(question, category, reports, profile)
    _progress('synthesizer', 'completed')

    # Phase 3: Score edge (now we compare to market)
    _progress('scorer', 'started')
    detected_category = category
    if not detected_category and market_data:
        detected_category = market_data.get('category', 'MARKETS')
    result = score_edge(synthesis, market_data, detected_category or 'MARKETS')
    _progress('scorer', 'completed')

    # Add contract metadata
    # Use clean title (strip distribution context appended by orchestrator)
    clean_title = question.split('\n')[0].strip()
    result['contract_title'] = clean_title
    result['category'] = detected_category or 'MARKETS'

    # For multi-outcome: include option name in position
    if option_name and result['suggested_position'] not in ('No position',):
        result['suggested_position'] = f'{result["suggested_position"]} on {option_name}'

    # Attach agent metadata for the frontend
    agent_meta = {}
    for name, report in reports.items():
        agent_meta[name] = {
            'duration': report.get('_duration', 0),
            'status': 'error' if '_error' in report else 'ok',
        }
        if name == 'news_researcher':
            agent_meta[name]['news_count'] = report.get('_raw_news_count', 0)
            agent_meta[name]['web_count'] = report.get('_raw_web_count', 0)
        elif name in ('data_researcher', 'cross_market', 'contrarian'):
            agent_meta[name]['result_count'] = report.get('_result_count', 0)

    result['_agents'] = agent_meta
    result['_synthesis'] = {
        'model_probability': synthesis.get('model_probability'),
        'confidence': synthesis.get('confidence'),
        'key_factors': synthesis.get('key_factors', []),
    }

    return result
