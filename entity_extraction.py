#!/usr/bin/env python3
"""
Entity extraction module using Gemini LangExtract.

This module provides functionality to extract entities from chat content
and enrich JSONL files with extracted entities.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterable


def extract_entities_with_langextract(page_content: str, metadata: dict) -> List[Dict[str, Any]]:
    """
    Extract entities from chat content using Gemini LangExtract.
    Returns a list of extracted entities with their attributes.
    """
    try:
        import langextract as lx
    except ImportError:
        print("❌ langextract not installed. Install with: pip install langextract")
        return []
    
    # Define crypto-specific entity extraction prompt
    prompt = """Extract entities from this crypto chat conversation.
    Focus on identifying:
    1. CRYPTOCURRENCIES: BTC, Bitcoin, ETH, SOL, AVAX, POLYGON, ADA, XRP, DOT, etc.
    2. PLATFORMS/EXCHANGES: Coinbase, Binance, DFX, StraitsX, UniSwap, IBKR, Fireblocks, etc.
    3. COMPANIES: Tesla, MicroStrategy, Lweefinance, StraitsX, etc.
    4. PEOPLE/INFLUENCERS: marcus, tianyao, tw, tianwei, victor, henghong, shaun, jiawei, weeli, amanda, etc.
    5. FINANCIAL_INSTRUMENTS: bonds, stocks, options, credit derivatives, etc. 
    6. TOOLS/APPS: TradingView, MetaMask, Telegram, etc.
    7. MEMECOINS: PEPE, DOGE, etc.
    8. NFTS: Azuki, CryptoPunks, Doodles, Pudgy Penguins, etc.
    9. PROJECTS: LweeFinance, etc
    
    DO NOT EXTRACT TOPICS. ONLY EXTRACT ENTITIES .
    Extract exact text mentions and provide meaningful attributes.
    Do not paraphrase. Preserve original casing for tickers.
    """
    
    # Create examples based on the chat domain
    examples = [
        lx.data.ExampleData(
            text="I sold most of my ETH after the pump. Now looking at SOL which hit ATH.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="CRYPTOCURRENCY",
                    extraction_text="ETH",
                    attributes={"ticker": "ETH", "action": "sell", "sentiment": "neutral"}
                ),
                lx.data.Extraction(
                    extraction_class="CRYPTOCURRENCY", 
                    extraction_text="SOL",
                    attributes={"ticker": "SOL", "action": "watch", "sentiment": "bullish"}
                ),
                lx.data.Extraction(
                    extraction_class="TOPIC",
                    extraction_text="pump",
                    attributes={"context": "price_movement", "type": "market_action"}
                ),
            ]
        ),
        lx.data.ExampleData(
            text="Jiawei plans to liquidate Lweefinance and buy Singapore bonds via StraitsX.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="COMPANY",
                    extraction_text="Lweefinance", 
                    attributes={"type": "investment_fund", "action": "liquidate"}
                ),
                lx.data.Extraction(
                    extraction_class="FINANCIAL_INSTRUMENT",
                    extraction_text="Singapore bonds",
                    attributes={"asset_type": "bond", "country": "Singapore", "action": "buy"}
                ),
                lx.data.Extraction(
                    extraction_class="PLATFORM",
                    extraction_text="StraitsX",
                    attributes={"type": "exchange", "function": "fiat_gateway"}
                ),
            ]
        )
    ]
    
    try:
        # Run entity extraction
        result = lx.extract(
            text_or_documents=page_content,
            prompt_description=prompt,
            examples=examples,
            model_id="gemini-2.5-flash",
        )
        
        # Convert to our format
        entities = []
        if hasattr(result, 'extractions') and result.extractions:
            for extraction in result.extractions:
                entity = {
                    "type": extraction.extraction_class,
                    "text": extraction.extraction_text,
                    "attributes": extraction.attributes if hasattr(extraction, 'attributes') else {},
                    "chunk_id": metadata.get("chunk_id"),
                    "chat_name": metadata.get("chat_name")
                }
                entities.append(entity)
            
        return entities
        
    except Exception as e:
        print(f"⚠️ Entity extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return []


def load_jsonl(path: Path) -> Iterable[dict]:
    """Load JSONL file and yield documents."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def filter_docs(stream: Iterable[dict], chat_name: Optional[str], chunk_id: Optional[str], limit: Optional[int], resume_from: Optional[str]) -> List[dict]:
    """Filter documents based on criteria."""
    out = []
    found_resume_point = resume_from is None  # If no resume point, start immediately
    
    for obj in stream:
        meta = obj.get("metadata", {})
        if chat_name and meta.get("chat_name") != chat_name:
            continue
        
        current_chunk_id = meta.get("chunk_id")
        
        # Handle resume logic
        if resume_from and not found_resume_point:
            if current_chunk_id == resume_from:
                found_resume_point = True
                print(f"🔄 Found resume point: {resume_from}")
            else:
                continue  # Skip until we find the resume point
        
        if chunk_id and current_chunk_id != chunk_id:
            continue
            
        out.append(obj)
        if limit and len(out) >= limit:
            break
    
    if resume_from and not found_resume_point:
        print(f"⚠️ Resume point '{resume_from}' not found in the filtered documents")
    
    return out


def enrich_jsonl_with_entities(input_jsonl: Path, output_jsonl: Path, chat_name: Optional[str] = None, chunk_id: Optional[str] = None) -> None:
    """
    Enrich existing JSONL with extracted entities and save to new file.
    """
    print(f"🧠 Starting entity extraction pass...")
    print(f"📂 Input: {input_jsonl}")
    print(f"📂 Output: {output_jsonl}")
    
    docs = filter_docs(load_jsonl(input_jsonl), chat_name, chunk_id, None, None)
    if not docs:
        print("❌ No documents to process for entity extraction")
        return
        
    print(f"✓ Found {len(docs)} document(s) to process")
    
    # Ensure output directory exists
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    with open(output_jsonl, 'w', encoding='utf-8') as outfile:
        for i, doc in enumerate(docs, start=1):
            print(f"\n--- Extracting entities from document {i}/{len(docs)} ---")
            
            page_content = doc.get("page_content", "")
            metadata = doc.get("metadata", {})
            
            chunk_id_current = metadata.get('chunk_id', 'unknown')
            chat_name_current = metadata.get('chat_name', 'unknown')
            print(f"📄 Processing: {chunk_id_current} from chat '{chat_name_current}'")
            
            # Extract entities
            entities = extract_entities_with_langextract(page_content, metadata)
            print(f"✓ Extracted {len(entities)} entities")
            
            # Enrich the document with entities
            enriched_doc = doc.copy()
            enriched_doc["extracted_entities"] = entities
            
            # Write to output file
            outfile.write(json.dumps(enriched_doc, ensure_ascii=False) + '\n')
            processed_count += 1
    
    print(f"\n🎉 Entity extraction completed! Processed {processed_count} documents")
    print(f"📄 Enriched JSONL saved to: {output_jsonl}")
