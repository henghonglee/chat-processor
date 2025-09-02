#!/usr/bin/env python3
"""
Step 7: Cypher Query Post-Processing and Neo4j Ingestion

Combined step that post-processes raw Cypher queries and ingests them into Neo4j.
This combines the functionality of g_cypher_query_post_processing and h_ingestion.

Post-processing includes:
1. Split Cypher blob into individual statements
2. Fix node variable syntax (remove hyphens)
3. Sanitize SAID/MENTION relationship conflicts
4. Validate and filter invalid statements
5. Extract and embed full_text content
6. Extract and embed entity information
7. Save processed statements

Then ingests the processed statements into Neo4j database.

Inputs:
- Raw Cypher queries from cypher-queries/CHAT_NAME/

Outputs:
- Processed Cypher queries saved to cypher-processed/CHAT_NAME/
- Vector embeddings stored in ChromaDB collections
- Data stored in Neo4j database

Environment Variables:
- NEO4J_URI (default bolt://localhost:7687)
- NEO4J_USER (default neo4j)
- NEO4J_PASSWORD (default password)
- EMBEDDING_PROVIDER (default openai)
- EMBEDDING_MODEL (default text-embedding-3-large)
- OLLAMA_URL (default http://localhost:11434)

Usage examples:
  # Process and ingest all Cypher files for a chat
  python3 steps/g_ingestion.py facebook-soldegen

  # Process specific chunk
  python3 steps/g_ingestion.py facebook-soldegen --chunk-id chunk_001.txt

  # Dry run to see what would be processed
  python3 steps/g_ingestion.py facebook-soldegen --dry-run

  # Post-process only (no ingestion)
  python3 steps/g_ingestion.py facebook-soldegen --post-process-only
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import chromadb
import openai
import requests
from dotenv import load_dotenv

load_dotenv()

# Initialize embedding configuration
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")  # "openai" or "ollama"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# Initialize OpenAI client for embeddings
if EMBEDDING_PROVIDER == "openai":
    openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./vector_store")


def get_embedding(text: str) -> List[float]:
    """Generate embedding for text using configured provider (OpenAI or Ollama)."""
    try:
        if EMBEDDING_PROVIDER == "openai":
            response = openai.embeddings.create(model=EMBEDDING_MODEL, input=text)
            return response.data[0].embedding
        elif EMBEDDING_PROVIDER == "ollama":
            response = requests.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": EMBEDDING_MODEL, "prompt": text},
                timeout=30,
            )
            response.raise_for_status()
            return response.json()["embedding"]
        else:
            print(f"❌ Unsupported embedding provider: {EMBEDDING_PROVIDER}")
            return []
    except Exception as e:
        print(f"❌ Error generating embedding with {EMBEDDING_PROVIDER}: {e}")
        return []


def sanitize_node_variable(identifier: str) -> str:
    """
    Sanitize node variable names to be valid Neo4j identifiers.
    Replace hyphens with underscores to avoid syntax errors.
    """
    # Replace hyphens with underscores for Neo4j variable names
    return identifier.replace("-", "_")


def sanitize_said_and_mention_relationships(raw_cypher: str) -> str:
    """
    Remove MENTION relationships where the same person already has a SAID relationship
    to the same claim node. SAID takes precedence over MENTION.

    Example:
    If we have (p2)-[:SAID]->(c3) and (c3)-[:MENTION]->(p2),
    then remove the MENTION relationship since SAID takes precedence.
    """
    # Split raw_cypher into individual statements - each line starting with MERGE
    statements = []
    for line in raw_cypher.strip().split("\n"):
        line = line.strip()
        if line and line.upper().startswith("MERGE"):
            statements.append(line)

    # Parse all statements to find SAID and MENTION relationships
    said_relationships = set()  # (person_var, claim_var)
    mention_statements_to_remove = set()  # indices to remove

    # First pass: identify all SAID relationships
    for _, stmt in enumerate(statements):
        # Pattern for SAID relationships: (person)-[:SAID...]->(claim)
        said_pattern = r"MERGE\s+\(\s*([a-zA-Z][\w_]*)\s*\)\s*-\s*\[\s*:SAID\b[^\]]*\]\s*->\s*\(\s*([a-zA-Z][\w_]*)\s*\)"
        said_match = re.search(said_pattern, stmt, re.IGNORECASE)
        if said_match:
            person_var = said_match.group(1)
            claim_var = said_match.group(2)
            said_relationships.add((person_var, claim_var))

    # Second pass: identify MENTION relationships that conflict with SAID
    for i, stmt in enumerate(statements):
        # Pattern for MENTION relationships: (claim)-[:MENTION]->(person)
        mention_pattern = r"MERGE\s+\(\s*([a-zA-Z][\w_]*)\s*\)\s*-\s*\[\s*:MENTION\s*\]\s*->\s*\(\s*([a-zA-Z][\w_]*)\s*\)"
        mention_match = re.search(mention_pattern, stmt, re.IGNORECASE)
        if mention_match:
            claim_var = mention_match.group(1)
            target_var = mention_match.group(2)

            # Check if this MENTION relationship conflicts with a SAID relationship
            # If (person)-[:SAID]->(claim) exists, then (claim)-[:MENTION]->(person) should be removed
            if (target_var, claim_var) in said_relationships:
                mention_statements_to_remove.add(i)
                print(
                    f"🧹 Removing conflicting MENTION relationship: ({claim_var})-[:MENTION]->({target_var}) "
                    f"because ({target_var})-[:SAID]->({claim_var}) exists"
                )

    # Return statements with conflicting MENTION relationships removed
    sanitized_statements = []
    for i, stmt in enumerate(statements):
        if i not in mention_statements_to_remove:
            sanitized_statements.append(stmt)
        else:
            print(f"⚠️ Skipping statement {i+1}: {stmt[:100]}...")

    return "\n".join(sanitized_statements)


def extract_full_text_from_statements(
    statements: List[str],
) -> Dict[str, Dict[str, str]]:
    """
    Extract full_text content from Cypher statements and map to node IDs with type.
    Concatenates arrays of full_text with full stops.
    """
    print(f"🔍 Extracting full_text content from {len(statements)} statements")
    full_text_data = {}

    # First pass: store all claim IDs for each variable
    variable_to_claim_id = {}

    for stmt in statements:
        # Extract claim variable and ID mapping
        claim_pattern = r"MERGE\s+\(\s*([a-zA-Z][\w_]*)\s*:\s*Claim\s*\{[^}]*\bid\s*:\s*\"([^\"]+)\""
        claim_match = re.search(claim_pattern, stmt, re.IGNORECASE)
        if claim_match:
            claim_var = claim_match.group(1)
            claim_id = claim_match.group(2)
            variable_to_claim_id[claim_var] = claim_id

    print(f"🔍 Variable to claim ID: {variable_to_claim_id}")
    # Second pass: extract full_text content from SAID and REACTED relationships
    for stmt in statements:
        # Find SAID relationships with full_text arrays
        said_pattern = r"MERGE\s+\([^)]+\)\s*-\s*\[\s*:SAID\s*\{[^}]*full_text\s*:\s*\[([^\]]*)\][^}]*\}\s*\]\s*->\s*\(\s*([a-zA-Z][\w_]*)\s*\)"
        said_match = re.search(said_pattern, stmt, re.IGNORECASE)
        if said_match and said_match.group(2) in variable_to_claim_id:
            claim_var = said_match.group(2)
            claim_id = variable_to_claim_id[claim_var]
            full_text_array = said_match.group(1)

            # Extract individual strings from the array
            text_items = re.findall(r'"([^"]*)"', full_text_array)
            combined_text = ". ".join(text_items)

            if claim_id not in full_text_data and combined_text:
                full_text_data[claim_id] = {
                    "text": combined_text,
                    "node_type": "claim",
                }
                print(
                    f"   📝 Extracted SAID text for {claim_id}: {combined_text[:100]}..."
                )
            elif claim_id in full_text_data:
                # Append to existing text
                existing_text = full_text_data[claim_id]["text"]
                full_text_data[claim_id]["text"] = f"{existing_text}. {combined_text}"
                print(
                    f"   📝 Appended SAID text for {claim_id}: {combined_text[:100]}..."
                )

        # Find REACTED relationships with full_text arrays
        reacted_pattern = r"MERGE\s+\([^)]+\)\s*-\s*\[\s*:REACTED\s*\{[^}]*full_text\s*:\s*\[([^\]]*)\][^}]*\}\s*\]\s*->\s*\(\s*([a-zA-Z][\w_]*)\s*\)"
        reacted_match = re.search(reacted_pattern, stmt, re.IGNORECASE)
        if reacted_match and reacted_match.group(2) in variable_to_claim_id:
            claim_var = reacted_match.group(2)
            claim_id = variable_to_claim_id[claim_var]
            full_text_array = reacted_match.group(1)

            # Extract individual strings from the array
            text_items = re.findall(r'"([^"]*)"', full_text_array)
            combined_text = ". ".join(text_items)

            if claim_id not in full_text_data and combined_text:
                full_text_data[claim_id] = {
                    "text": combined_text,
                    "node_type": "claim",
                }
                print(
                    f"   📝 Extracted REACTED text for {claim_id}: {combined_text[:100]}..."
                )
            elif claim_id in full_text_data:
                # Append to existing text
                existing_text = full_text_data[claim_id]["text"]
                full_text_data[claim_id]["text"] = f"{existing_text}. {combined_text}"
                print(
                    f"   📝 Appended REACTED text for {claim_id}: {combined_text[:100]}..."
                )

    print(f"✅ Extracted full_text from {len(full_text_data)} items")
    return full_text_data


def extract_entities_from_statements(
    statements: List[str],
) -> Dict[str, Dict[str, str]]:
    """
    Extract entity information (id only) from Cypher statements.
    """
    entity_data = {}

    for stmt in statements:
        # Pattern for Entity nodes
        entity_pattern = r"MERGE\s+\([^:]+:Entity[^}]*\{[^}]*\bid\s*:\s*\"([^\"]+)\""
        entity_matches = re.finditer(entity_pattern, stmt, re.IGNORECASE)

        for match in entity_matches:
            node_id = match.group(1)

            entity_data[node_id] = {
                "id": node_id,
                "node_type": "entity",
            }

    return entity_data


def store_full_text_embeddings(
    full_text_data: Dict[str, Dict[str, str]], chat_name: str
):
    """
    Store full_text embeddings in ChromaDB full_text collection.
    """
    try:
        collection = chroma_client.get_or_create_collection(
            name="full_text",
            metadata={"description": "Full text content from chat conversations"},
        )

        for node_id, data in full_text_data.items():
            full_text = data["text"]
            node_type = data["node_type"]

            if full_text.strip():
                embedding = get_embedding(full_text)
                if embedding:
                    collection.add(
                        embeddings=[embedding],
                        documents=[full_text],
                        metadatas=[
                            {
                                "node_id": node_id,
                                "chat_name": chat_name,
                                "node_type": node_type,
                            }
                        ],
                        ids=[f"{chat_name}_{node_id}"],
                    )
                    print(f"📄 Embedded full_text for {node_type} node: {node_id}")

        print(f"✅ Stored {len(full_text_data)} full_text embeddings")

    except Exception as e:
        print(f"❌ Error storing full_text embeddings: {e}")


def store_entity_embeddings(entity_data: Dict[str, Dict[str, str]], chat_name: str):
    """
    Store entity embeddings in ChromaDB entity collection.
    """
    try:
        collection = chroma_client.get_or_create_collection(
            name="entity",
            metadata={"description": "Entity information from chat conversations"},
        )

        for node_id, _data in entity_data.items():

            # Combine name and type for embedding
            combined_text = f"{node_id}".strip()

            if combined_text:
                embedding = get_embedding(combined_text)
                if embedding:
                    collection.add(
                        embeddings=[embedding],
                        documents=[combined_text],
                        metadatas=[
                            {
                                "node_id": node_id,
                                "chat_name": chat_name,
                                "node_type": "entity",
                            }
                        ],
                        ids=[f"{chat_name}_{node_id}"],
                    )
                    print(f"🏷️ Embedded entity:({node_id})")

        print(f"✅ Stored {len(entity_data)} entity embeddings")

    except Exception as e:
        print(f"❌ Error storing entity embeddings: {e}")


def save_processed_cypher_to_file(statements: str, output_path: Path) -> None:
    """Save processed Cypher statements to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(statements)


def load_processed_cypher(cypher_file: Path) -> List[str]:
    """
    Load processed Cypher statements from a file.

    Args:
        cypher_file: Path to processed Cypher file

    Returns:
        List of Cypher statements
    """
    if not cypher_file.exists():
        return []

    content = cypher_file.read_text(encoding="utf-8").strip()
    if not content:
        return []

    # Split on individual lines for processed statements
    statements = [stmt.strip() for stmt in content.split("\n") if stmt.strip()]
    return statements


def execute_cypher_statements(
    statements: List[str], chat_name: str, chunk_id: str
) -> Tuple[int, int]:
    """
    Execute Cypher statements against Neo4j.

    Args:
        statements: List of Cypher statements to execute
        chat_name: Chat name for logging
        chunk_id: Chunk ID for logging

    Returns:
        tuple: (successful_count, failed_count)
    """
    if not statements:
        return 0, 0

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")

    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("❌ neo4j package not installed. Run: pip install neo4j")
        return 0, len(statements)

    successful = 0
    failed = 0

    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))

        with driver.session() as session:
            for i, statement in enumerate(statements, 1):
                try:
                    print(f"   Executing statement {i}/{len(statements)}...")
                    result = session.run(statement)

                    # Consume the result to ensure it's executed
                    summary = result.consume()

                    # Log execution details
                    if hasattr(summary, "counters"):
                        counters = summary.counters
                        if (
                            counters.nodes_created > 0
                            or counters.relationships_created > 0
                        ):
                            print(
                                f"     ✓ Created {counters.nodes_created} nodes, {counters.relationships_created} relationships"
                            )

                    successful += 1

                except Exception as e:
                    print(f"     ❌ Statement {i} failed: {e}")
                    print(f"     Statement: {statement[:100]}...")
                    failed += 1
                    continue

        driver.close()

    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")
        return 0, len(statements)

    return successful, failed


def process_single_chunk_post_processing(
    cypher_file: Path, chat_name: str, dry_run: bool = False
) -> bool:
    """
    Post-process a single chunk file.

    Returns:
        True if successful, False otherwise
    """
    chunk_id = cypher_file.stem
    print(f"\n📦 Post-processing chunk: {chunk_id}")

    if dry_run:
        print(f"   [DRY RUN] Would post-process: {cypher_file}")
        return True

    try:
        # Read raw Cypher from file
        with open(cypher_file, "r", encoding="utf-8") as f:
            raw_cypher = f.read()

        if not raw_cypher or not raw_cypher.strip():
            print(f"⚠️ Empty Cypher file: {cypher_file}")
            return False

        print(f"   📄 Read {len(raw_cypher)} characters")

        # Split into individual statements
        statements = []
        for line in raw_cypher.strip().split("\n"):
            line = line.strip()
            if line and line.upper().startswith("MERGE"):
                statements.append(line)
        print(f"   📝 Split into: {len(statements)} statements")

        # Sanitize relationships
        sanitized_cypher = sanitize_said_and_mention_relationships(raw_cypher)
        print(f"   🧹 After sanitization: {len(sanitized_cypher)} characters")

        # Extract and store embeddings
        full_text_data = extract_full_text_from_statements(statements)
        print(f"   📄 Extracted full_text from {len(full_text_data)} items")
        store_full_text_embeddings(full_text_data, chat_name)

        entity_data = extract_entities_from_statements(statements)
        print(f"   🏷️ Extracted entity data from {len(entity_data)} items")
        store_entity_embeddings(entity_data, chat_name)

        # Create output directory and save processed statements
        output_dir = Path("cypher-processed") / chat_name
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"{chunk_id}.cypher"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(sanitized_cypher)
        print(f"   💾 Saved processed Cypher to: {output_file}")

        return True

    except Exception as e:
        print(f"   ❌ Error processing chunk {chunk_id}: {e}")
        return False


def process_single_chunk_ingestion(
    chunk_id: str, chat_name: str, dry_run: bool = False
) -> Tuple[int, int]:
    """
    Ingest a single processed chunk.

    Returns:
        Tuple of (successful_count, failed_count)
    """
    print(f"\n🔗 Ingesting chunk: {chunk_id}")

    # Check if processed file exists
    processed_file = Path("cypher-processed") / chat_name / f"{chunk_id}.cypher"
    if not processed_file.exists():
        print(f"   ❌ Processed file not found: {processed_file}")
        return 0, 1

    if dry_run:
        print(f"   [DRY RUN] Would ingest: {processed_file}")
        return 1, 0

    try:
        # Load statements from processed file
        statements = load_processed_cypher(processed_file)

        if not statements:
            print(f"   ⚠️ No statements found in: {processed_file}")
            return 0, 0

        print(f"   📄 Executing {len(statements)} statements")

        # Execute statements
        successful, failed = execute_cypher_statements(statements, chat_name, chunk_id)

        if successful > 0:
            print(f"   ✅ Successfully executed {successful} statements")
        if failed > 0:
            print(f"   ❌ Failed to execute {failed} statements")

        return successful, failed

    except Exception as e:
        print(f"   ❌ Error ingesting chunk {chunk_id}: {e}")
        return 0, 1


def get_available_chats() -> List[str]:
    """Get list of available chat names from cypher-queries directory."""
    cypher_queries_dir = Path("cypher-queries")
    if not cypher_queries_dir.exists():
        return []

    chats = []
    for item in cypher_queries_dir.iterdir():
        if item.is_dir():
            chats.append(item.name)

    return sorted(chats)


def get_chunk_files(chat_name: str) -> List[Path]:
    """Get sorted list of chunk files for a chat."""
    cypher_queries_dir = Path("cypher-queries") / chat_name
    if not cypher_queries_dir.exists():
        return []

    chunk_files = list(cypher_queries_dir.glob("chunk_*.cypher"))
    # Sort by chunk number
    chunk_files.sort(key=lambda x: int(x.stem.split('_')[1]))

    return chunk_files


def process_cypher_chunks(
    chat_name: str,
    dry_run: bool = False,
    post_process_only: bool = False,
    ingest_only: bool = False,
    start_chunk: int = None,
    end_chunk: int = None
) -> bool:
    """
    Process Cypher chunks through post-processing and ingestion steps.

    Args:
        chat_name: Name of the chat to process
        dry_run: Show what would be processed without executing
        post_process_only: Run only post-processing step
        ingest_only: Run only ingestion step (requires post-processed files)
        start_chunk: Start processing from chunk number N
        end_chunk: Stop processing at chunk number N

    Returns:
        True if successful, False otherwise
    """
    print(f"🚀 Processing chat: {chat_name}")

    # Get chunk files
    chunk_files = get_chunk_files(chat_name)
    if not chunk_files:
        print(f"❌ No chunk files found for chat: {chat_name}")
        return False

    # Apply chunk range filters
    if start_chunk is not None or end_chunk is not None:
        filtered_files = []
        for chunk_file in chunk_files:
            chunk_num = int(chunk_file.stem.split('_')[1])
            if start_chunk is not None and chunk_num < start_chunk:
                continue
            if end_chunk is not None and chunk_num > end_chunk:
                continue
            filtered_files.append(chunk_file)
        chunk_files = filtered_files

    if not chunk_files:
        print(f"❌ No chunks found in specified range for chat: {chat_name}")
        return False

    print(f"📁 Processing {len(chunk_files)} chunks for chat: {chat_name}")
    if dry_run:
        print("🔍 DRY RUN MODE - No actual processing will occur")

    # Processing statistics
    post_process_success = 0
    post_process_failed = 0
    total_statements_executed = 0
    total_statements_failed = 0

    # Process each chunk
    for chunk_file in chunk_files:
        chunk_id = chunk_file.stem

        # Post-processing step
        if not ingest_only:
            success = process_single_chunk_post_processing(chunk_file, chat_name, dry_run)
            if success:
                post_process_success += 1
            else:
                post_process_failed += 1
                if not dry_run:
                    print(f"   ⚠️ Skipping ingestion for {chunk_id} due to post-processing failure")
                    continue

        # Ingestion step
        if not post_process_only:
            successful, failed = process_single_chunk_ingestion(chunk_id, chat_name, dry_run)
            total_statements_executed += successful
            total_statements_failed += failed

    # Final summary
    print(f"\n🎯 Processing Summary for {chat_name}:")
    print(f"📊 Total chunks processed: {len(chunk_files)}")

    if not ingest_only:
        print(f"✅ Post-processing successful: {post_process_success}")
        if post_process_failed > 0:
            print(f"❌ Post-processing failed: {post_process_failed}")

    if not post_process_only:
        print(f"✅ Statements executed successfully: {total_statements_executed}")
        if total_statements_failed > 0:
            print(f"❌ Statements failed: {total_statements_failed}")

    if dry_run:
        print("🔍 This was a dry run - no actual changes were made")

    return post_process_failed == 0 and total_statements_failed == 0


# Legacy functions to maintain compatibility
def post_process_cypher_data(chat_name: str) -> bool:
    """
    Legacy function for backward compatibility.
    Now calls the combined process_cypher_chunks function with post_process_only=True.
    """
    return process_cypher_chunks(chat_name, post_process_only=True)


def ingest_processed_cypher(chat_name: str) -> bool:
    """
    Legacy function for backward compatibility.
    Now calls the combined process_cypher_chunks function with ingest_only=True.
    """
    return process_cypher_chunks(chat_name, ingest_only=True)
