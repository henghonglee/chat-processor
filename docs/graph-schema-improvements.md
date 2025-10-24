# Graph Schema Improvement Plan

**Document Version:** 1.0
**Last Updated:** 2025-10-21
**Status:** Draft
**Owner:** Architecture Team

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current State Analysis](#current-state-analysis)
3. [Improvement Roadmap](#improvement-roadmap)
4. [Phase 1: Core Schema Enhancements](#phase-1-core-schema-enhancements)
5. [Phase 2: Relationship Expansion](#phase-2-relationship-expansion)
6. [Phase 3: Advanced Features](#phase-3-advanced-features)
7. [Migration Strategy](#migration-strategy)
8. [Performance Considerations](#performance-considerations)
9. [Implementation Checklist](#implementation-checklist)

---

## Executive Summary

### Purpose
This document outlines a comprehensive plan to enhance the Neo4j graph schema used in the chat-processor system. The improvements aim to increase expressiveness, query performance, and analytical capabilities while maintaining backward compatibility.

### Key Goals
- **Richer Node Properties**: Extend Person, Claim, and Entity nodes with metadata for better context
- **Expanded Relationships**: Add relationship types beyond SAID/REACTED/MENTION
- **Schema Validation**: Implement Neo4j constraints and indexes for data integrity
- **Temporal Tracking**: Enable time-series analysis of claims and relationships
- **Entity-to-Entity Links**: Support direct relationships between entities (not just through claims)

### Success Metrics
- **Query Performance**: 50% reduction in average query time for common patterns
- **Schema Coverage**: 90%+ of chat interactions representable in graph
- **Data Quality**: Zero duplicate entities, validated entity types
- **Query Complexity**: Support for 3-hop relationship queries with <500ms response time

---

## Current State Analysis

### Existing Schema Overview

#### Node Types (3)
| Node Type | Properties | Count (Est.) | Limitations |
|-----------|-----------|--------------|-------------|
| **Person** | `id`, `name` | ~100-1000 | No role, bio, or metrics |
| **Claim** | `id`, `text`, `valid_at`, `url` | ~10,000-100,000 | No sentiment, category, confidence |
| **Entity** | `id`, `name`, `type`, `description` | ~1,000-10,000 | Loose type validation, no metadata |

#### Relationship Types (3)
| Relationship | Pattern | Properties | Limitations |
|--------------|---------|-----------|-------------|
| **SAID** | `Person→Claim` | `full_text[]`, `valid_at` | Only originating speaker |
| **REACTED** | `Person→Claim` | `full_text[]`, `valid_at` | Only for reactions |
| **MENTION** | `Claim→Entity/Person` | `full_text[]`, `valid_at` | No Person→Person or Person→Entity |

#### Entity Type Categories (12)
```
ASSET, PLATFORM, ORGANIZATION, EVENT, LOCATION, PERSON,
FINANCIAL_INSTRUMENT, TOOL, MEMECOIN, NFT, PROJECT, FOOD
```

### Critical Gaps

#### 1. Limited Expressiveness
- Cannot represent person-to-person relationships (follows, collaborates, works_at)
- Cannot represent entity-to-entity relationships (belongs_to, competes_with)
- All semantics forced through intermediate Claim nodes

#### 2. Missing Metadata
- **Person**: No role, organization, first/last mention timestamps
- **Claim**: No sentiment score, confidence level, category classification
- **Entity**: No sector, market cap, founding date, or domain-specific attributes

#### 3. Schema Inconsistencies
- Property names differ between prompt (`summary_text`) and code (`text`)
- Entity types not enforced as enum - any string accepted
- No unique constraints on node IDs (duplicates possible)
- No indexes defined for common query patterns

#### 4. Query Limitations
- Graph searcher uses substring matching (`CONTAINS`) - inefficient for large datasets
- No full-text search indexes on Claim.text in Neo4j
- Relationship arrays (`full_text[]`) not indexed
- Cannot efficiently query temporal ranges or time-series trends

---

## Improvement Roadmap

### Three-Phase Approach

```
Phase 1 (4-6 weeks)          Phase 2 (6-8 weeks)          Phase 3 (8-12 weeks)
Core Schema Enhancements  →  Relationship Expansion    →  Advanced Features
- Extend node properties     - Add new relationship types - Temporal versioning
- Add constraints/indexes    - Entity-to-entity links     - Sentiment analysis
- Fix inconsistencies        - Person-to-person links     - Confidence scoring
- Validate entity types      - Relationship metadata      - Community detection
                             - Multi-hop queries          - Cross-chat linking
```

---

## Phase 1: Core Schema Enhancements

**Timeline:** 4-6 weeks
**Priority:** HIGH
**Goal:** Stabilize and enrich the existing schema without breaking changes

### 1.1 Person Node Enhancements

#### New Properties
```cypher
// Existing
(p:Person {
  id: "person-henghong-lee",
  name: "Henghong Lee"
})

// Enhanced
(p:Person {
  id: "person-henghong-lee",
  name: "Henghong Lee",

  // Identity & Role
  role: "trader",                    // enum: trader, developer, influencer, observer
  organization: "LweeFinance",       // optional affiliation
  bio: "Crypto trader since 2017",   // optional short biography

  // Temporal Tracking
  first_mentioned: datetime("2023-01-15T10:30:00Z"),
  last_mentioned: datetime("2025-10-21T15:45:00Z"),

  // Metrics
  mention_count: 247,                // total mentions across all chats
  claim_count: 89,                   // total claims made
  reaction_count: 158,               // total reactions given

  // Metadata
  created_at: datetime("2023-01-15T10:30:00Z"),
  updated_at: datetime("2025-10-21T15:45:00Z")
})
```

#### Schema Constraints
```cypher
CREATE CONSTRAINT person_id_unique IF NOT EXISTS
FOR (p:Person) REQUIRE p.id IS UNIQUE;

CREATE CONSTRAINT person_name_required IF NOT EXISTS
FOR (p:Person) REQUIRE p.name IS NOT NULL;

CREATE INDEX person_role_index IF NOT EXISTS
FOR (p:Person) ON (p.role);

CREATE INDEX person_organization_index IF NOT EXISTS
FOR (p:Person) ON (p.organization);
```

#### Migration Script
```cypher
// Add default values for existing Person nodes
MATCH (p:Person)
WHERE NOT EXISTS(p.role)
SET p.role = "observer",
    p.created_at = datetime(),
    p.updated_at = datetime()

// Calculate metrics from existing relationships
MATCH (p:Person)
WITH p,
     size((p)-[:SAID]->()) as claims,
     size((p)-[:REACTED]->()) as reactions
SET p.claim_count = claims,
    p.reaction_count = reactions,
    p.mention_count = claims + reactions
```

---

### 1.2 Claim Node Enhancements

#### New Properties
```cypher
// Existing
(c:Claim {
  id: "claim-btc-rally-2025",
  text: "BTC will reach $100k by end of 2025",
  valid_at: datetime("2024-03-15T14:30:00Z"),
  url: "https://twitter.com/..."
})

// Enhanced
(c:Claim {
  id: "claim-btc-rally-2025",
  text: "BTC will reach $100k by end of 2025",
  valid_at: datetime("2024-03-15T14:30:00Z"),
  url: "https://twitter.com/...",

  // Classification
  category: "price_prediction",      // enum: price_prediction, technical_analysis,
                                     //       news, opinion, question, event, other
  topic: "market_sentiment",         // optional: refined categorization

  // Confidence & Sentiment
  sentiment: "bullish",              // enum: bullish, bearish, neutral, mixed
  sentiment_score: 0.75,             // float: -1.0 (very bearish) to 1.0 (very bullish)
  confidence_score: 0.6,             // float: 0.0 (speculation) to 1.0 (verified fact)

  // Source Tracking
  source_chunk_id: "chunk-20240315-143000",
  source_chat: "crypto-alpha-group",

  // Metrics
  mention_count: 12,                 // times this claim was mentioned/reacted to
  entity_count: 3,                   // number of entities mentioned

  // Metadata
  created_at: datetime("2024-03-15T14:30:00Z"),
  updated_at: datetime("2024-03-15T14:30:00Z"),
  archived: false                    // for claim versioning
})
```

#### Schema Constraints
```cypher
CREATE CONSTRAINT claim_id_unique IF NOT EXISTS
FOR (c:Claim) REQUIRE c.id IS UNIQUE;

CREATE CONSTRAINT claim_text_required IF NOT EXISTS
FOR (c:Claim) REQUIRE c.text IS NOT NULL;

CREATE FULLTEXT INDEX claim_text_fulltext IF NOT EXISTS
FOR (c:Claim) ON EACH [c.text];

CREATE INDEX claim_category_index IF NOT EXISTS
FOR (c:Claim) ON (c.category);

CREATE INDEX claim_sentiment_index IF NOT EXISTS
FOR (c:Claim) ON (c.sentiment);

CREATE INDEX claim_valid_at_index IF NOT EXISTS
FOR (c:Claim) ON (c.valid_at);
```

#### Enum Validation (Application Layer)
```python
# ingestion/validation/claim_validator.py
VALID_CLAIM_CATEGORIES = {
    "price_prediction", "technical_analysis", "news",
    "opinion", "question", "event", "announcement", "other"
}

VALID_SENTIMENTS = {"bullish", "bearish", "neutral", "mixed"}

def validate_claim_properties(claim_dict):
    if claim_dict.get("category") not in VALID_CLAIM_CATEGORIES:
        raise ValueError(f"Invalid category: {claim_dict.get('category')}")

    if claim_dict.get("sentiment") not in VALID_SENTIMENTS:
        raise ValueError(f"Invalid sentiment: {claim_dict.get('sentiment')}")

    if not (-1.0 <= claim_dict.get("sentiment_score", 0) <= 1.0):
        raise ValueError("sentiment_score must be between -1.0 and 1.0")
```

---

### 1.3 Entity Node Enhancements

#### New Properties
```cypher
// Existing
(e:Entity {
  id: "entity-asset-btc",
  name: "BTC",
  type: "asset",
  description: "Bitcoin cryptocurrency"
})

// Enhanced
(e:Entity {
  id: "entity-asset-btc",
  name: "BTC",
  type: "asset",
  description: "Bitcoin cryptocurrency",

  // Type-Specific Metadata
  // For ASSET type:
  symbol: "BTC",
  full_name: "Bitcoin",
  sector: "cryptocurrency",
  market_cap_tier: "large",          // large, mid, small, micro

  // For ORGANIZATION type:
  // industry: "exchange",
  // founded_year: 2012,
  // headquarters: "San Francisco, CA",

  // For PLATFORM type:
  // platform_type: "centralized_exchange",
  // supported_assets: ["BTC", "ETH", "SOL"],

  // Common Metadata
  aliases: ["Bitcoin", "bitcoin"],   // alternative names for entity resolution
  canonical_name: "Bitcoin",         // preferred display name

  // Metrics
  mention_count: 523,                // total mentions across all claims
  first_mentioned: datetime("2023-01-15T10:00:00Z"),
  last_mentioned: datetime("2025-10-21T16:00:00Z"),

  // Metadata
  created_at: datetime("2023-01-15T10:00:00Z"),
  updated_at: datetime("2025-10-21T16:00:00Z"),
  verified: true                     // manual verification flag
})
```

#### Schema Constraints
```cypher
CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
FOR (e:Entity) REQUIRE e.id IS UNIQUE;

CREATE CONSTRAINT entity_name_required IF NOT EXISTS
FOR (e:Entity) REQUIRE e.name IS NOT NULL;

CREATE CONSTRAINT entity_type_required IF NOT EXISTS
FOR (e:Entity) REQUIRE e.type IS NOT NULL;

CREATE INDEX entity_type_index IF NOT EXISTS
FOR (e:Entity) ON (e.type);

CREATE INDEX entity_symbol_index IF NOT EXISTS
FOR (e:Entity) ON (e.symbol);

CREATE INDEX entity_canonical_name_index IF NOT EXISTS
FOR (e:Entity) ON (e.canonical_name);

CREATE FULLTEXT INDEX entity_name_fulltext IF NOT EXISTS
FOR (e:Entity) ON EACH [e.name, e.canonical_name, e.description];
```

#### Validated Entity Types
```python
# ingestion/validation/entity_validator.py
VALID_ENTITY_TYPES = {
    "asset", "platform", "organization", "event", "location",
    "person", "financial_instrument", "tool", "memecoin",
    "nft", "project", "food", "concept"
}

ASSET_SUBTYPES = {"cryptocurrency", "token", "stablecoin", "wrapped_asset"}
PLATFORM_SUBTYPES = {"centralized_exchange", "dex", "lending_protocol", "bridge"}

def validate_entity_type(entity_dict):
    entity_type = entity_dict.get("type")
    if entity_type not in VALID_ENTITY_TYPES:
        raise ValueError(f"Invalid entity type: {entity_type}")
```

---

### 1.4 Fix Schema Inconsistencies

#### Issues to Resolve

**Issue 1: Property Name Mismatch**
- **Current:** Prompt uses `summary_text`, code uses `text`
- **Fix:** Standardize on `text` in both prompt and code

```json
// Update: ingestion/prompts/cypher_prompt.json
{
  "claim_properties": {
    "text": "The consolidated claim statement"  // Was: summary_text
  }
}
```

**Issue 2: Relationship Property Standardization**
- **Current:** All relationships use `full_text` as array
- **Fix:** Consider renaming to `excerpts` for clarity

```cypher
// Before
(p)-[:SAID {full_text: ["quote 1", "quote 2"]}]->(c)

// After
(p)-[:SAID {excerpts: ["quote 1", "quote 2"], valid_at: datetime()}]->(c)
```

**Issue 3: Timestamp Consistency**
- **Current:** `valid_at` used inconsistently across nodes/relationships
- **Fix:** Standardize on `timestamp` for relationships, `valid_at` for temporal claims

---

### 1.5 Implementation Checklist

- [ ] Update `cypher_prompt.json` with new property definitions
- [ ] Modify Cypher generation step to include new properties
- [ ] Update entity extraction to classify claim categories
- [ ] Implement sentiment analysis for claims (using LLM or rule-based)
- [ ] Add validation layer for enum properties
- [ ] Create migration scripts for existing data
- [ ] Add Neo4j constraints and indexes
- [ ] Update query templates to use new properties
- [ ] Update graph searcher to leverage full-text indexes
- [ ] Add unit tests for property validation
- [ ] Document new schema in README

---

## Phase 2: Relationship Expansion

**Timeline:** 6-8 weeks
**Priority:** MEDIUM
**Goal:** Add new relationship types for richer graph semantics

### 2.1 Person-to-Person Relationships

#### New Relationship Types

**FOLLOWS** (Person → Person)
```cypher
(alice:Person)-[:FOLLOWS {
  since: datetime("2024-01-15T00:00:00Z"),
  strength: 0.8,                     // 0.0-1.0: interaction frequency
  context: "crypto_trading"
}]->(bob:Person)
```

**COLLABORATES_WITH** (Person ↔ Person)
```cypher
(alice:Person)-[:COLLABORATES_WITH {
  project: "DeFi Protocol",
  since: datetime("2024-03-01T00:00:00Z"),
  role: "co-founder"
}]-(bob:Person)
```

**DISAGREES_WITH** (Person → Person) - via Claims
```cypher
// Pattern: Alice disagrees with Bob's claim
MATCH (alice:Person)-[:REACTED]->(claim:Claim)<-[:SAID]-(bob:Person)
WHERE claim.sentiment_score < -0.5  // Negative reaction
CREATE (alice)-[:DISAGREES_WITH {
  on_claim_id: claim.id,
  timestamp: claim.valid_at,
  severity: 0.7
}]->(bob)
```

**WORKS_AT** (Person → Organization Entity)
```cypher
(person:Person)-[:WORKS_AT {
  role: "Head of Trading",
  since: datetime("2023-06-01T00:00:00Z"),
  current: true
}]->(org:Entity {type: "organization"})
```

---

### 2.2 Entity-to-Entity Relationships

#### New Relationship Types

**BELONGS_TO** (Entity → Entity)
```cypher
// Asset belongs to platform
(asset:Entity {type: "token"})-[:BELONGS_TO {
  relation_type: "native_token"
}]->(platform:Entity {type: "platform"})

// Example: USDC belongs to Circle
(usdc:Entity)-[:BELONGS_TO {relation_type: "issued_by"}]->(circle:Entity)
```

**COMPETES_WITH** (Entity ↔ Entity)
```cypher
(uniswap:Entity {type: "platform"})-[:COMPETES_WITH {
  market_segment: "DEX",
  intensity: 0.9
}]-(sushiswap:Entity {type: "platform"})
```

**BUILT_ON** (Entity → Entity)
```cypher
// Protocol built on blockchain
(protocol:Entity {type: "project"})-[:BUILT_ON {
  layer: "L2",
  since: datetime("2023-01-01T00:00:00Z")
}]->(ethereum:Entity {type: "asset"})
```

**ACQUIRED_BY** (Entity → Entity)
```cypher
(ftx:Entity {type: "platform"})-[:ACQUIRED_BY {
  date: datetime("2022-01-01T00:00:00Z"),
  price: "420M USD"
}]->(binance:Entity {type: "platform"})
```

---

### 2.3 Claim-to-Claim Relationships

#### New Relationship Types

**SUPPORTS** (Claim → Claim)
```cypher
(claim1:Claim)-[:SUPPORTS {
  confidence: 0.8,
  reasoning: "Provides corroborating evidence"
}]->(claim2:Claim)
```

**CONTRADICTS** (Claim ↔ Claim)
```cypher
(claim1:Claim)-[:CONTRADICTS {
  aspect: "price_direction",
  severity: 0.9
}]-(claim2:Claim)
```

**BUILDS_ON** (Claim → Claim)
```cypher
// Claim extends or refines another claim
(refinement:Claim)-[:BUILDS_ON {
  timestamp: datetime()
}]->(original:Claim)
```

**UPDATES** (Claim → Claim)
```cypher
// Later claim updates/corrects earlier claim
(new_claim:Claim)-[:UPDATES {
  reason: "price_change",
  timestamp: datetime()
}]->(old_claim:Claim)

// Set old claim as archived
SET old_claim.archived = true
```

---

### 2.4 Enhanced Relationship Properties

#### Standard Metadata for All Relationships
```cypher
// Every relationship should have:
{
  created_at: datetime(),           // when relationship was created
  updated_at: datetime(),           // last modification
  source_chunk_id: "chunk-xyz",     // originating chunk
  confidence: 0.85,                 // 0.0-1.0: extraction confidence
  verified: false                   // manual verification flag
}
```

---

### 2.5 Auto-Relationship Inference

#### Algorithms to Derive Implicit Relationships

**1. Co-Mention Analysis** (Person-Person)
```cypher
// Find people who frequently appear in same claims
MATCH (p1:Person)-[:SAID|REACTED]->(c:Claim)<-[:SAID|REACTED]-(p2:Person)
WHERE p1.id < p2.id  // avoid duplicates
WITH p1, p2, count(c) as co_mentions
WHERE co_mentions > 5
MERGE (p1)-[:CO_MENTIONED_WITH {
  frequency: co_mentions,
  created_at: datetime()
}]-(p2)
```

**2. Entity Affiliation** (Person-Entity)
```cypher
// Infer person's affiliation from frequent mentions
MATCH (p:Person)-[:SAID|REACTED]->(c:Claim)-[:MENTION]->(e:Entity)
WITH p, e, count(c) as mentions
WHERE mentions > 10
MERGE (p)-[:FREQUENTLY_MENTIONS {
  count: mentions,
  created_at: datetime()
}]->(e)
```

**3. Sentiment Clusters** (Person-Person)
```cypher
// Find people with similar sentiment patterns
MATCH (p1:Person)-[:SAID|REACTED]->(c:Claim)
WITH p1, avg(c.sentiment_score) as p1_avg_sentiment
MATCH (p2:Person)-[:SAID|REACTED]->(c2:Claim)
WITH p1, p1_avg_sentiment, p2, avg(c2.sentiment_score) as p2_avg_sentiment
WHERE abs(p1_avg_sentiment - p2_avg_sentiment) < 0.2
  AND p1.id < p2.id
MERGE (p1)-[:SIMILAR_SENTIMENT {
  difference: abs(p1_avg_sentiment - p2_avg_sentiment),
  created_at: datetime()
}]-(p2)
```

---

### 2.6 Implementation Checklist

- [ ] Define new relationship types in schema documentation
- [ ] Update Cypher prompt with relationship extraction rules
- [ ] Implement relationship inference algorithms
- [ ] Add relationship property validation
- [ ] Create indexes for new relationship types
- [ ] Update query patterns to leverage new relationships
- [ ] Add graph visualization support for new relationships
- [ ] Write tests for relationship inference
- [ ] Document new query patterns in README

---

## Phase 3: Advanced Features

**Timeline:** 8-12 weeks
**Priority:** LOW
**Goal:** Advanced analytics and temporal features

### 3.1 Temporal Versioning

#### Claim History Tracking
```cypher
// Version 1 of claim
(c1:Claim {
  id: "claim-btc-price-v1",
  text: "BTC will hit $50k",
  version: 1,
  valid_at: datetime("2024-01-01T00:00:00Z"),
  superseded: true
})<-[:UPDATES {
  reason: "price_adjustment",
  timestamp: datetime("2024-02-01T00:00:00Z")
}]-(c2:Claim {
  id: "claim-btc-price-v2",
  text: "BTC will hit $60k",
  version: 2,
  valid_at: datetime("2024-02-01T00:00:00Z"),
  superseded: false
})
```

#### Time-Series Queries
```cypher
// Find sentiment evolution over time
MATCH (p:Person)-[:SAID]->(c:Claim)-[:MENTION]->(e:Entity {id: "entity-asset-btc"})
WHERE c.valid_at >= datetime("2024-01-01T00:00:00Z")
  AND c.valid_at <= datetime("2024-12-31T23:59:59Z")
RETURN
  date.truncate('month', c.valid_at) as month,
  avg(c.sentiment_score) as avg_sentiment,
  count(c) as claim_count
ORDER BY month
```

---

### 3.2 Sentiment Analysis Integration

#### LLM-Based Sentiment Extraction
```python
# ingestion/steps/sentiment_analyzer.py
from openai import OpenAI

def analyze_claim_sentiment(claim_text: str) -> dict:
    """
    Extract sentiment and confidence from claim text.

    Returns:
        {
            "sentiment": "bullish" | "bearish" | "neutral" | "mixed",
            "sentiment_score": -1.0 to 1.0,
            "confidence_score": 0.0 to 1.0,
            "reasoning": "..."
        }
    """
    client = OpenAI()

    prompt = f"""
    Analyze the sentiment and confidence of this crypto-related claim:

    "{claim_text}"

    Provide:
    1. Overall sentiment (bullish/bearish/neutral/mixed)
    2. Sentiment score (-1.0 very bearish to 1.0 very bullish)
    3. Confidence score (0.0 speculation to 1.0 verified fact)
    4. Brief reasoning

    Respond in JSON format.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)
```

---

### 3.3 Community Detection

#### Identify Person Clusters
```cypher
// Using Neo4j Graph Data Science Library
CALL gds.graph.project(
  'person-network',
  'Person',
  {
    FOLLOWS: {orientation: 'NATURAL'},
    COLLABORATES_WITH: {orientation: 'UNDIRECTED'},
    CO_MENTIONED_WITH: {orientation: 'UNDIRECTED'}
  }
)

CALL gds.louvain.stream('person-network')
YIELD nodeId, communityId
WITH gds.util.asNode(nodeId) AS person, communityId
SET person.community_id = communityId
RETURN communityId, collect(person.name) as members
ORDER BY size(members) DESC
```

---

### 3.4 Cross-Chat Entity Linking

#### Merge Duplicate Entities
```cypher
// Find potential duplicates using fuzzy matching
CALL apoc.text.sorensenDiceSimilarity(e1.name, e2.name) as similarity
MATCH (e1:Entity), (e2:Entity)
WHERE e1.id < e2.id
  AND similarity > 0.85
  AND e1.type = e2.type
RETURN e1.name, e2.name, similarity
ORDER BY similarity DESC

// Manual merge after review
MATCH (e1:Entity {id: "entity-asset-btc"}),
      (e2:Entity {id: "entity-asset-bitcoin"})
CALL apoc.refactor.mergeNodes([e1, e2], {properties: "combine"})
YIELD node
RETURN node
```

---

### 3.5 Implementation Checklist

- [ ] Implement claim versioning system
- [ ] Add sentiment analysis to ingestion pipeline
- [ ] Install Neo4j GDS library for community detection
- [ ] Create entity deduplication workflow
- [ ] Add temporal query templates
- [ ] Implement time-series aggregation functions
- [ ] Create dashboards for sentiment trends
- [ ] Add graph visualization for communities
- [ ] Document advanced query patterns
- [ ] Performance test with large datasets

---

## Migration Strategy

### Backward Compatibility Approach

#### 1. Additive Changes Only (Phases 1-2)
- Add new properties without removing old ones
- Maintain existing relationship types
- Support both old and new property names during transition

#### 2. Dual-Write Period
```python
# During migration: write to both old and new schema
def create_claim_node(claim_data):
    cypher = """
    MERGE (c:Claim {id: $id})
    ON CREATE SET
      c.text = $text,                    // New standard
      c.summary_text = $text,            // Deprecated: for backward compat
      c.valid_at = $valid_at,
      c.category = $category,            // New property
      c.sentiment = $sentiment,          // New property
      c.created_at = datetime()
    ON MATCH SET
      c.updated_at = datetime()
    """
```

#### 3. Migration Scripts

**Script 1: Add Default Values**
```cypher
// Run after schema changes deployed
MATCH (p:Person)
WHERE NOT EXISTS(p.role)
SET p.role = "observer",
    p.created_at = datetime(),
    p.mention_count = 0,
    p.claim_count = 0,
    p.reaction_count = 0
RETURN count(p) as updated_persons;
```

**Script 2: Calculate Metrics**
```cypher
// Populate mention_count for existing entities
MATCH (e:Entity)
OPTIONAL MATCH (c:Claim)-[:MENTION]->(e)
WITH e, count(c) as mentions
SET e.mention_count = mentions,
    e.updated_at = datetime()
RETURN count(e) as updated_entities;
```

**Script 3: Migrate Property Names**
```cypher
// Rename summary_text to text (if needed)
MATCH (c:Claim)
WHERE EXISTS(c.summary_text) AND NOT EXISTS(c.text)
SET c.text = c.summary_text
REMOVE c.summary_text
RETURN count(c) as migrated_claims;
```

#### 4. Rollback Plan
- Keep backups of Neo4j database before migration
- Document rollback scripts for each phase
- Test migration on staging environment first
- Use feature flags to toggle new schema features

---

## Performance Considerations

### 1. Indexing Strategy

#### Critical Indexes (Phase 1)
```cypher
// Node property indexes
CREATE INDEX person_name_index IF NOT EXISTS FOR (p:Person) ON (p.name);
CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.type);
CREATE INDEX claim_valid_at_index IF NOT EXISTS FOR (c:Claim) ON (c.valid_at);

// Composite indexes for common query patterns
CREATE INDEX person_role_org_index IF NOT EXISTS
FOR (p:Person) ON (p.role, p.organization);

CREATE INDEX claim_category_sentiment_index IF NOT EXISTS
FOR (c:Claim) ON (c.category, c.sentiment);

// Full-text search indexes
CREATE FULLTEXT INDEX claim_text_fulltext IF NOT EXISTS
FOR (c:Claim) ON EACH [c.text];

CREATE FULLTEXT INDEX entity_name_fulltext IF NOT EXISTS
FOR (e:Entity) ON EACH [e.name, e.canonical_name];
```

#### Relationship Indexes (Phase 2)
```cypher
// Index relationship properties for filtering
CREATE INDEX FOR ()-[r:SAID]-() ON (r.valid_at);
CREATE INDEX FOR ()-[r:FOLLOWS]-() ON (r.since);
CREATE INDEX FOR ()-[r:BELONGS_TO]-() ON (r.relation_type);
```

### 2. Query Optimization Patterns

#### Before (Inefficient)
```cypher
// Substring search on all entities
MATCH (e:Entity)
WHERE toLower(e.id) CONTAINS toLower($term)
RETURN e
```

#### After (Optimized)
```cypher
// Use full-text index
CALL db.index.fulltext.queryNodes('entity_name_fulltext', $term)
YIELD node, score
RETURN node, score
ORDER BY score DESC
LIMIT 10
```

### 3. Batch Processing Guidelines

#### Ingestion Optimization
```python
# Use batched Cypher transactions
def batch_create_claims(claims: List[dict], batch_size=100):
    """Create claims in batches to avoid memory issues."""
    for i in range(0, len(claims), batch_size):
        batch = claims[i:i + batch_size]

        cypher = """
        UNWIND $batch as claim_data
        MERGE (c:Claim {id: claim_data.id})
        ON CREATE SET c = claim_data
        """

        session.run(cypher, batch=batch)
```

### 4. Monitoring Metrics

Track these metrics post-deployment:
- Average query response time (target: <500ms for 90th percentile)
- Index hit rate (target: >80%)
- Database size growth (baseline)
- Relationship traversal performance (2-hop queries <100ms)
- Full-text search performance (<200ms)

---

## Implementation Checklist

### Phase 1: Core Schema Enhancements
- [ ] **Week 1-2: Schema Design**
  - [ ] Finalize new property definitions
  - [ ] Document validation rules
  - [ ] Create migration scripts
  - [ ] Review with stakeholders

- [ ] **Week 3-4: Implementation**
  - [ ] Update `cypher_prompt.json`
  - [ ] Modify Cypher generation code
  - [ ] Add validation layer
  - [ ] Implement sentiment analysis
  - [ ] Create Neo4j constraints

- [ ] **Week 5-6: Testing & Deployment**
  - [ ] Test on staging database
  - [ ] Run migration scripts
  - [ ] Validate data quality
  - [ ] Update documentation
  - [ ] Deploy to production

### Phase 2: Relationship Expansion
- [ ] **Week 7-9: Relationship Design**
  - [ ] Define new relationship types
  - [ ] Create extraction rules
  - [ ] Implement inference algorithms

- [ ] **Week 10-12: Implementation**
  - [ ] Update Cypher prompt
  - [ ] Add relationship extraction
  - [ ] Create indexes
  - [ ] Update query patterns

- [ ] **Week 13-14: Testing & Deployment**
  - [ ] Test relationship inference
  - [ ] Validate graph structure
  - [ ] Deploy to production

### Phase 3: Advanced Features
- [ ] **Week 15-18: Temporal & Analytics**
  - [ ] Implement versioning
  - [ ] Add time-series queries
  - [ ] Community detection setup

- [ ] **Week 19-22: Integration & Polish**
  - [ ] Cross-chat linking
  - [ ] Performance optimization
  - [ ] Dashboard creation

- [ ] **Week 23-26: Final Testing**
  - [ ] Load testing
  - [ ] User acceptance testing
  - [ ] Documentation finalization
  - [ ] Production deployment

---

## Success Criteria

### Phase 1 (Core Schema)
- ✅ All Person nodes have `role`, `mention_count`, timestamps
- ✅ All Claim nodes have `category`, `sentiment`, `confidence_score`
- ✅ All Entity nodes have validated `type` enum
- ✅ Neo4j constraints created and enforced
- ✅ Query performance baseline established
- ✅ Zero breaking changes to existing queries

### Phase 2 (Relationships)
- ✅ Person-Person relationships extractable from chat data
- ✅ Entity-Entity relationships defined and populated
- ✅ Relationship inference algorithms running
- ✅ Multi-hop queries supported (3+ hops)
- ✅ Query response time <500ms for 90% of queries

### Phase 3 (Advanced)
- ✅ Claim versioning operational
- ✅ Time-series queries functional
- ✅ Community detection identifying clusters
- ✅ Sentiment trends visualized
- ✅ Entity deduplication workflow in place

---

## Appendix

### A. Reference Files
- `/home/user/chat-processor/ingestion/prompts/cypher_prompt.json` - Current schema definition
- `/home/user/chat-processor/ingestion/prompts/langextract_entities.py` - Entity types
- `/home/user/chat-processor/ingestion/steps/f_cypher_query_generation.py` - Cypher generation
- `/home/user/chat-processor/query/steps/b_cypher_generation.py` - Query patterns

### B. Useful Neo4j Queries

**View Schema**
```cypher
CALL db.schema.visualization()
```

**Count Nodes by Type**
```cypher
MATCH (n) RETURN labels(n), count(n) ORDER BY count(n) DESC
```

**Count Relationships by Type**
```cypher
MATCH ()-[r]->() RETURN type(r), count(r) ORDER BY count(r) DESC
```

**Find Orphan Nodes**
```cypher
MATCH (n) WHERE NOT (n)--() RETURN labels(n), count(n)
```

### C. Resources
- Neo4j Documentation: https://neo4j.com/docs/
- Graph Data Science Library: https://neo4j.com/docs/graph-data-science/
- Cypher Query Language: https://neo4j.com/docs/cypher-manual/
- LangExtract (Entity Extraction): https://github.com/google/langextract

---

**Document End**
