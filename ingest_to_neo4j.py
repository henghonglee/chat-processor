#!/usr/bin/env python3
"""
Neo4j chat ingestion system:
Generate Cypher from enriched JSONL using OpenAI and insert into Neo4j

Inputs
- Enriched JSONL with extracted entities: each line has { page_content, metadata, extracted_entities }

Environment
- OPENAI_API_KEY: OpenAI API key (required for Neo4j ingestion)
- NEO4J_URI (default bolt://localhost:7687)
- NEO4J_USER (default neo4j)
- NEO4J_PASSWORD (default password)

Usage examples
  # Process enriched JSONL to Neo4j
  python3 ingest_to_neo4j.py --jsonl exports/facebook-soldegen-enriched.jsonl \
    --chat-name facebook-soldegen --chunk-id chunk_001.txt --dry-run

  # Resume from a specific chunk (skip already processed chunks)
  python3 ingest_to_neo4j.py --jsonl exports/facebook-soldegen-enriched.jsonl \
    --chat-name facebook-soldegen --resume-from chunk_010.txt
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any


def load_jsonl(path: Path) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_system_prompt() -> str:
    """Load system prompt from chat_to_cypher_prompt.md (entire file)."""
    prompt_path = Path(__file__).parent / "chat_to_cypher_prompt.md"
    text = prompt_path.read_text(encoding="utf-8")
    # Use full file to preserve schema/rules; LLM instructed to output only Cypher.
    return text


def build_user_prompt(page_content: str, metadata: dict, extracted_entities: List[dict]) -> str:
    # Truncate very large chunks for LLM safety while preserving structure
    max_chars = 12000
    content = page_content if len(page_content) <= max_chars else page_content[:max_chars]

    meta_snippet = {
        "chat_name": metadata.get("chat_name"),
        "chunk_id": metadata.get("chunk_id"),
        "ts_start": metadata.get("ts_start"),
        "ts_end": metadata.get("ts_end"),
        "context": metadata.get("context"),
        "extracted_entities": extracted_entities,
        "message_count": metadata.get("message_count"),
    }
    print(f"Metadata: {meta_snippet}")
    return (
        "Metadata: " + json.dumps(meta_snippet, ensure_ascii=False) + "\n\n"
        "Chunk content follows (verbatim):\n" + content
    )


def call_openai_generate_cypher(model: str, system_prompt: str, user_prompt: str) -> str:
    """Use OpenAI Chat Completions API to generate Cypher."""
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in environment")

    client = OpenAI(api_key=api_key)

    # Use the standard Chat Completions API instead of Responses API
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_completion_tokens=6000
    )

    text = response.choices[0].message.content or ""
    
    # Remove code blocks if present
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def split_cypher_statements(cypher_blob: str) -> List[str]:
    """
    Split Cypher blob into individual statements, handling multiline statements properly.
    """
    # First, normalize line endings and split by semicolon
    normalized = cypher_blob.replace('\r\n', '\n').replace('\r', '\n')
    
    # Split by semicolon, but we need to be careful about semicolons inside strings
    statements = []
    current_statement = []
    in_string = False
    escape_next = False
    quote_char = None
    
    i = 0
    while i < len(normalized):
        char = normalized[i]
        
        if escape_next:
            escape_next = False
            current_statement.append(char)
        elif char == '\\' and in_string:
            escape_next = True
            current_statement.append(char)
        elif char in ('"', "'") and not in_string:
            in_string = True
            quote_char = char
            current_statement.append(char)
        elif char == quote_char and in_string:
            in_string = False
            quote_char = None
            current_statement.append(char)
        elif char == ';' and not in_string:
            # End of statement
            stmt = ''.join(current_statement).strip()
            if stmt:
                statements.append(stmt)
            current_statement = []
        else:
            current_statement.append(char)
        
        i += 1
    
    # Add any remaining statement
    stmt = ''.join(current_statement).strip()
    if stmt:
        statements.append(stmt)
    
    # Filter out empty statements and Browser-only directives
    filtered: List[str] = []
    for stmt in statements:
        if not stmt:
            continue
        if stmt.lstrip().startswith(":"):
            # Skip Neo4j Browser commands (not valid via Bolt)
            continue
        filtered.append(stmt)
    
    return filtered


def sanitize_node_variable(identifier: str) -> str:
    """
    Sanitize node variable names to be valid Neo4j identifiers.
    Replace hyphens with underscores to avoid syntax errors.
    """
    # Replace hyphens with underscores for Neo4j variable names
    return identifier.replace("-", "_")


def fix_cypher_node_syntax(statement: str) -> str:
    """
    Fix Cypher statement syntax by ensuring node variables don't contain hyphens.
    The error indicates that node variables like 'person-ng-yang-yi-desmond' are invalid.
    Neo4j expects valid identifiers as node variables.
    """
    # Pattern to match node references like (person-something:Label ...)
    # We need to fix the variable name part (before the colon)
    node_pattern = r'\(\s*([a-zA-Z][\w-]*)\s*(?=:)'
    
    def replace_node_var(match):
        var_name = match.group(1)
        sanitized_var = sanitize_node_variable(var_name)
        # Keep the ':' that follows due to lookahead
        return f'({sanitized_var}'
    
    # Fix node variable names in MERGE/CREATE/MATCH statements
    fixed_statement = re.sub(node_pattern, replace_node_var, statement)
    
    # Also fix relationship patterns where nodes are referenced by variable
    # Pattern like (person-something)-[...]->(other-thing)
    # Allow optional whitespace around relationship parts and arrow
    rel_pattern = r'\(\s*([a-zA-Z][\w-]*)\s*\)\s*(-\s*\[[^\]]*\]\s*-\s*>?)\s*\(\s*([a-zA-Z][\w-]*)\s*\)'
    
    def replace_rel_vars(match):
        var1 = sanitize_node_variable(match.group(1))
        rel_part = match.group(2)
        var2 = sanitize_node_variable(match.group(3))
        return f'({var1}){rel_part}({var2})'
    
    fixed_statement = re.sub(rel_pattern, replace_rel_vars, fixed_statement)
    
    # Handle single node references in relationships like (person-something)-[...]->
    single_node_rel_pattern = r'\(\s*([a-zA-Z][\w-]*)\s*\)\s*(-\s*\[[^\]]*\]\s*-\s*>?)'
    
    def replace_single_rel_var(match):
        var1 = sanitize_node_variable(match.group(1))
        rel_part = match.group(2)
        return f'({var1}){rel_part}'
    
    fixed_statement = re.sub(single_node_rel_pattern, replace_single_rel_var, fixed_statement)

    # Finally, sanitize standalone variable nodes like (claim_exit-crypto) with no labels/properties
    # This matches parentheses that contain only a variable identifier.
    standalone_var_pattern = r'\(\s*([a-zA-Z][\w-]*)\s*\)'

    def replace_standalone_var(match):
        var_name = match.group(1)
        # If the variable already looks sanitized, leave it
        if '-' not in var_name:
            return match.group(0)
        sanitized_var = sanitize_node_variable(var_name)
        return f'({sanitized_var})'

    fixed_statement = re.sub(standalone_var_pattern, replace_standalone_var, fixed_statement)
    
    return fixed_statement


def validate_and_fix_cypher(statement: str) -> Optional[str]:
    """
    Validate and fix common Cypher syntax issues.
    Returns the fixed statement or None if it cannot be fixed.
    """
    if not statement or not statement.strip():
        return None
    
    # Fix node variable syntax issues
    fixed = fix_cypher_node_syntax(statement)
    
    # Check for incomplete statements (missing quotes, etc.)
    # Count quotes to ensure they are balanced
    single_quotes = fixed.count("'")
    double_quotes = fixed.count('"')
    
    # If odd number of quotes, likely incomplete statement
    if single_quotes % 2 != 0 or double_quotes % 2 != 0:
        print(f"⚠️ Skipping statement with unbalanced quotes: {fixed[:100]}...")
        return None
    
    # Check for basic Cypher keywords
    cypher_keywords = ['MERGE', 'CREATE', 'MATCH', 'SET', 'RETURN', 'DELETE', 'REMOVE', 'WITH']
    has_keyword = any(keyword in fixed.upper() for keyword in cypher_keywords)
    
    if not has_keyword:
        print(f"⚠️ Skipping statement without Cypher keywords: {fixed[:100]}...")
        return None
    
    return fixed


def check_chunk_processed(chat_name: str, chunk_id: str) -> bool:
    """
    Check if a chunk has already been processed by querying Neo4j for data from this chunk.
    Returns True if the chunk appears to have been processed already.
    """
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(uri, auth=(user, password))
        
        with driver.session() as session:
            # Check for Claims from this chunk
            query = """
            MATCH (c:Claim)
            WHERE c.chat_name = $chat_name AND c.chunk_id = $chunk_id
            RETURN count(c) as claim_count
            """
            result = session.run(query, chat_name=chat_name, chunk_id=chunk_id)
            record = result.single()
            claim_count = record["claim_count"] if record else 0
            
            # Also check for any assertions from this chunk
            query2 = """
            MATCH ()-[r:ASSERTED]->()
            WHERE r.chat_name = $chat_name AND r.chunk_id = $chunk_id
            RETURN count(r) as assertion_count
            """
            result2 = session.run(query2, chat_name=chat_name, chunk_id=chunk_id)
            record2 = result2.single()
            assertion_count = record2["assertion_count"] if record2 else 0
            
        driver.close()
        
        # Consider chunk processed if it has claims or assertions
        has_data = claim_count > 0 or assertion_count > 0
        if has_data:
            print(f"📋 Chunk {chunk_id} already processed: {claim_count} claims, {assertion_count} assertions")
        return has_data
        
    except Exception as e:
        print(f"⚠️ Could not check processing status for {chunk_id}: {e}")
        return False


def run_cypher(statements: List[str], dry_run: bool = False) -> None:
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")

    # Validate and fix statements before execution
    valid_statements = []
    for stmt in statements:
        fixed_stmt = validate_and_fix_cypher(stmt)
        if fixed_stmt:
            valid_statements.append(fixed_stmt)

    if dry_run:
        print("-- Dry run: would execute the following statements --")
        for idx, s in enumerate(valid_statements, start=1):
            print(f"-- Statement {idx} --\n{s};\n")
        return

    if not valid_statements:
        print("⚠️ No valid Cypher statements to execute")
        return

    # Import Neo4j driver lazily to allow dry-run without the dependency installed
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(uri, auth=(user, password))
    executed = 0
    failed = 0
    failed_statements = []
    with driver.session() as session:
        for idx, s in enumerate(valid_statements, start=1):
            try:
                session.execute_write(lambda tx: tx.run(s))
                executed += 1
            except Exception as e:
                failed += 1
                failed_statements.append((s, str(e)))
                preview = (s[:200] + "...") if len(s) > 200 else s
                print(f"❌ Failed statement {idx}: {e}\n{preview}\n")
    driver.close()
    
    # Print summary of failed statements
    if failed_statements:
        print("\n📋 Failed Statements Summary:")
        print("=" * 50)
        for i, (stmt, error) in enumerate(failed_statements, start=1):
            print(f"\n--- Failed Statement {i} ---")
            print(f"Error: {error}")
            print(f"Statement:\n{stmt}")
            print("-" * 40)
    
    print(f"🧾 Execution summary: executed={executed}, failed={failed}")


def filter_docs(stream: Iterable[dict], chat_name: Optional[str], chunk_id: Optional[str], limit: Optional[int], resume_from: Optional[str]) -> List[dict]:
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


def main() -> None:
    # Load .env if present
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
        print("✓ Loaded environment variables from .env file")
    except Exception as e:
        print(f"⚠ Could not load .env file: {e}")
        pass

    parser = argparse.ArgumentParser(description="Generate Cypher via OpenAI and insert into Neo4j")
    parser.add_argument("--jsonl", required=True, help="Path to enriched JSONL file with extracted entities")
    parser.add_argument("--chat-name", help="Filter by chat name (folder name)")
    parser.add_argument("--chunk-id", help="Filter by chunk id (e.g., chunk_001.txt)")
    parser.add_argument("--limit", type=int, help="Limit number of docs to process")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use")
    parser.add_argument("--dry-run", action="store_true", help="Show Cypher instead of executing")
    parser.add_argument("--cypher-file", help="Optional path to a file containing raw Cypher to process instead of calling OpenAI")
    parser.add_argument("--resume-from", help="Resume processing from a specific chunk ID (skip already processed chunks)")
    parser.add_argument("--skip-processed", action="store_true", help="Automatically skip chunks that have already been processed")

    args = parser.parse_args()

    print(f"📁 Loading enriched JSONL file: {args.jsonl}")
    enriched_jsonl_path = Path(args.jsonl).resolve()
    if not enriched_jsonl_path.exists():
        print(f"❌ File not found: {enriched_jsonl_path}")
        raise FileNotFoundError(enriched_jsonl_path)
    
    print(f"✓ Found enriched JSONL file: {enriched_jsonl_path}")

    # Neo4j Ingestion
    print("\n" + "="*60)
    print("🎯 NEO4J INGESTION")
    print("="*60)

    print("🔍 Filtering documents...")
    print(f"   - Chat name filter: {args.chat_name or 'None'}")
    print(f"   - Chunk ID filter: {args.chunk_id or 'None'}")
    print(f"   - Limit: {args.limit or 'None'}")
    print(f"   - Resume from: {args.resume_from or 'None'}")
    print(f"   - Skip processed: {'ON' if args.skip_processed else 'OFF'}")

    docs = filter_docs(load_jsonl(enriched_jsonl_path), args.chat_name, args.chunk_id, args.limit, args.resume_from)
    if not docs:
        print("❌ No matching documents found.")
        return

    print(f"✓ Found {len(docs)} matching document(s)")
    print(f"🤖 Using OpenAI model: {args.model}")
    print(f"🔧 Dry run mode: {'ON' if args.dry_run else 'OFF'}")

    print("📋 Loading system prompt...")
    system_prompt = load_system_prompt()
    print(f"✓ System prompt loaded ({len(system_prompt)} characters)")

    print("\n🚀 Starting document processing...")
    processed_count = 0
    skipped_count = 0
    
    for i, doc in enumerate(docs, start=1):
        print(f"\n--- Processing document {i}/{len(docs)} ---")
        
        page_content = doc.get("page_content", "")
        metadata = doc.get("metadata", {})
        
        chunk_id = metadata.get('chunk_id', 'unknown')
        chat_name = metadata.get('chat_name', 'unknown')
        print(f"📄 Document: {chunk_id} from chat '{chat_name}'")
        
        # Check if this chunk has already been processed
        if args.skip_processed and not args.dry_run:
            if check_chunk_processed(chat_name, chunk_id):
                print(f"⏭️ Skipping already processed chunk: {chunk_id}")
                skipped_count += 1
                continue
        
        print(f"📝 Content length: {len(page_content)} characters")
        
        print("🔨 Building user prompt...")
        user_prompt = build_user_prompt(page_content, metadata, doc.get("extracted_entities", []))
        print(f"✓ User prompt built ({len(user_prompt)} characters)")
        
        if args.cypher_file:
            cypher_path = Path(args.cypher_file).resolve()
            if not cypher_path.exists():
                print(f"❌ Cypher file not found: {cypher_path}")
                continue
            cypher_blob = cypher_path.read_text(encoding="utf-8")
            print(f"✓ Loaded Cypher from file ({len(cypher_blob)} characters)")
        else:
            try:
                print("🧠 Calling OpenAI to generate Cypher...")
                cypher_blob = call_openai_generate_cypher(args.model, system_prompt, user_prompt)
                print(f"✓ Received response ({len(cypher_blob)} characters)")
            except Exception as e:
                print(f"❌ OpenAI API call failed: {e}")
                continue
        
        print("✂️ Splitting Cypher statements...")
        statements = split_cypher_statements(cypher_blob)
        if not statements:
            print(f"⚠️ No Cypher statements generated for {chunk_id}")
            continue
            
        print(f"✓ Generated {len(statements)} Cypher statement(s)")
        
        if args.dry_run:
            print("🔍 Dry run - showing statements that would be executed:")
        else:
            print("💾 Executing Cypher statements in Neo4j...")
            
        try:
            run_cypher(statements, dry_run=args.dry_run)
            if not args.dry_run:
                print(f"✅ Successfully processed {chunk_id}")
                processed_count += 1
        except Exception as e:
            print(f"❌ Neo4j execution failed: {e}")
            continue

    print(f"\n🎉 Completed processing:")
    print(f"   - Processed: {processed_count} documents")
    print(f"   - Skipped: {skipped_count} documents")
    print(f"   - Total considered: {len(docs)} documents")


if __name__ == "__main__":
    main()