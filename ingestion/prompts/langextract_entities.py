"""
LangExtract prompt configuration for crypto chat entity extraction.

This file contains the prompt definition and examples for extracting entities
from cryptocurrency and finance-related chat conversations.
"""

import langextract as lx

# Define crypto-specific entity extraction prompt
LANGEXTRACT_PROMPT = """Extract entities from this crypto chat conversation.
Focus on identifying:
1. ASSET: BTC, Bitcoin, ETH, SOL, AVAX, POLYGON, ADA, XRP, DOT, etc.
2. PLATFORMS/EXCHANGES: Coinbase, Binance, DFX, StraitsX, UniSwap, IBKR, Fireblocks, etc.
3. ORGANIZATIONS: Tesla, MicroStrategy, Lweefinance, StraitsX, etc.
4. EVENTS: Bitcoin Halving, Ethereum Merge, etc.
5. LOCATIONS: Singapore, Hong Kong, etc.
6. PEOPLE/INFLUENCERS: marcus, tianyao, tw, tianwei, victor, henghong, shaun, jiawei, weeli, amanda, etc.
7. FINANCIAL_INSTRUMENTS: bonds, stocks, options, credit derivatives, etc.
8. TOOLS/APPS: TradingView, MetaMask, Telegram, etc.
8. MEMECOINS: PEPE, DOGE, etc.
9. NFTS: Azuki, CryptoPunks, Doodles, Pudgy Penguins, etc.
10. PROJECTS: LweeFinance, etc
11. FOOD: Pizza, Burger, etc.

Use entity normalization to merge similar cryptocurrency references:
- BTC, btc, $BTC, Bitcoin, bitcoin -> BTC
- ETH, eth, $ETH, Ethereum, ethereum -> ETH
- SOL, sol, $SOL, Solana, solana -> SOL
- ADA, ada, $ADA, Cardano, cardano -> ADA
- DOT, dot, $DOT, Polkadot, polkadot -> DOT
- AVAX, avax, $AVAX, Avalanche, avalanche -> AVAX
- MATIC, matic, $MATIC, Polygon, polygon -> MATIC
- USDC, usdc, $USDC, USD Coin, usd coin -> USDC
- USDT, usdt, $USDT, Tether, tether -> USDT
- WBTC, wbtc, $WBTC, Wrapped Bitcoin, wrapped bitcoin -> WBTC
- LINK, link, $LINK, Chainlink, chainlink -> LINK
- UNI, uni, $UNI, Uniswap, uniswap -> UNI
- AAVE, aave, $AAVE, Aave, aave -> AAVE
- SUSHI, sushi, $SUSHI, SushiSwap, sushi swap -> SUSHI
- CRV, crv, $CRV, Curve, curve -> CRV
- XRP, xrp, $XRP, Ripple, ripple -> XRP
- $JUP, Jup, jup -> JUP

use normalization for people:
- Jiawei, jiawei, Jiawei Climbing -> Jiawei-Lwee
- henghonglee, henghong, Heng Hong, HengHong Lee, Heng Hong Lee, Henghong Lee -> henghong-lee
- marcus, Marcus -> marcus
- tianyao, Tianyao -> tianyao
- tianwei, Tianwei, tw, TW -> tianwei
- victor, Victor -> victor
- shaun, Shaun -> shaun
- weeli, Weeli, Wei Li -> weeli
- amanda, Amanda -> amanda
- ng yang yi desmond, Ng Yang Yi Desmond, Desmond -> desmond
- lyn, Lyn -> lyn
- leon, Leon -> leon


Special cases:
- Jupiter Exchange, Jupiter, JupiterExchange, Jup Dapp -> Jupiter Exchange
- Jupiter Mobile, Jupiter Wallet -> Jupiter Mobile
- Jupiverse -> Jupiverse

For each entity, normalize to the standard ticker format and include both original_text and normalized_name.

DO NOT EXTRACT TOPICS. ONLY EXTRACT ENTITIES .
Extract exact text mentions and provide meaningful attributes.
Do not paraphrase. Preserve original casing for tickers.
"""

# Create examples based on the chat domain
LANGEXTRACT_EXAMPLES = [
    lx.data.ExampleData(
        text="I sold most of my ETH after the pump. Now looking at $SOL which hit ATH.",
        extractions=[
            lx.data.Extraction(
                extraction_class="ASSET",
                extraction_text="ETH",
                attributes={
                    "ticker": "ETH",
                    "action": "sell",
                    "sentiment": "neutral",
                },
            ),
            lx.data.Extraction(
                extraction_class="ASSET",
                extraction_text="SOL",
                attributes={
                    "ticker": "SOL",
                    "action": "watch",
                    "sentiment": "bullish",
                },
            ),
        ],
    ),
    lx.data.ExampleData(
        text="Jiawei plans to liquidate Lweefinance and buy Singapore bonds via StraitsX.",
        extractions=[
            lx.data.Extraction(
                extraction_class="ORGANIZATION",
                extraction_text="Lweefinance",
                attributes={"type": "investment_fund", "action": "liquidate"},
            ),
            lx.data.Extraction(
                extraction_class="FINANCIAL_INSTRUMENT",
                extraction_text="Singapore bonds",
                attributes={
                    "asset_type": "bond",
                    "country": "Singapore",
                    "action": "buy",
                },
            ),
            lx.data.Extraction(
                extraction_class="PLATFORM",
                extraction_text="StraitsX",
                attributes={"type": "exchange", "function": "fiat_gateway"},
            ),
        ],
    ),
]

# Model configuration
LANGEXTRACT_MODEL_ID = "gpt-4o-mini"
# LANGEXTRACT_MODEL_ID = "gpt-5-nano"
# LANGEXTRACT_MODEL_ID = "gemini-2.5-pro"
# Alternative models:
# LANGEXTRACT_MODEL_ID = "qwen3:8b"
# LANGEXTRACT_MODEL_ID = "llama3.2:3b"
# LANGEXTRACT_MODEL_ID = "gemma3:4b"
# LANGEXTRACT_MODEL_ID = "phi3:3.8b"


def get_langextract_config():
    """
    Returns the complete LangExtract configuration.

    Returns:
        tuple: (prompt, examples, model_id)
    """
    return LANGEXTRACT_PROMPT, LANGEXTRACT_EXAMPLES, LANGEXTRACT_MODEL_ID
