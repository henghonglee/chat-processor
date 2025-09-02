# Vector Search and Cypher Generation Improvements

## Executive Summary

This document outlines comprehensive improvements for the `a_vector_search.py` and `b_cypher_generation.py` query processing steps. The proposed enhancements focus on performance optimization, query intelligence, and system robustness while maintaining the existing architecture patterns.

## Current System Analysis

### Vector Search (`a_vector_search.py`) Current State
- **Strengths**: Clean ChromaDB integration, configurable embedding providers, structured result organization
- **Limitations**: Single query embedding, fixed similarity thresholds, basic result ranking, no caching
- **Performance**: ~267 lines, straightforward but not optimized for complex queries

### Cypher Generation (`b_cypher_generation.py`) Current State
- **Strengths**: Clear template structure, good separation of entity/claim queries, comprehensive logging
- **Limitations**: Static templates, no query optimization, limited error handling, no batching
- **Performance**: ~213 lines, functional but lacks intelligence and optimization

---

## Part 1: Vector Search Improvements

### 1.1 Query Enhancement and Preprocessing

**Priority**: High
**Impact**: Significant improvement in retrieval quality
**Effort**: Medium

#### Current Issue
```python
# Line 126 in a_vector_search.py
query_embedding = self._get_embedding(user_query)
```
Single embedding approach misses nuanced search opportunities.

#### Proposed Solution
```python
def _preprocess_query(self, user_query: str) -> Dict[str, Any]:
    """Generate enhanced query variations for better retrieval."""
    return {
        'original': user_query,
        'keywords': self._extract_keywords(user_query),
        'entities': self._extract_potential_entities(user_query),
        'intent': self._analyze_query_intent(user_query),
        'variations': self._generate_query_variations(user_query)
    }
def _extract_potential_entities(self, text: str) -> List[str]:
    """Extract potential entities using langextract for better entity recognition."""
    try:
        import langextract
        
        # Use langextract to detect and extract entities
        # This library is particularly good at identifying names, places, and organizations
        entities = []
        
        # Extract potential entities using different extraction methods
        # langextract.extract_entities returns a list of potential entities
        extracted = langextract.extract_entities(text)
        
        if extracted:
            for entity in extracted:
                # Clean and validate the entity
                entity_clean = entity.strip()
                if len(entity_clean) > 2:  # Minimum length filter
                    entities.append(entity_clean)
        
        # Also try extracting capitalized words as potential entities
        import re
        capitalized_words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        for word in capitalized_words:
            if word not in entities and len(word) > 2:
                entities.append(word)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity.lower() not in seen:
                seen.add(entity.lower())
                unique_entities.append(entity)
        
        self.logger.debug(f"Extracted entities: {unique_entities}")
        return unique_entities[:10]  # Limit to top 10 entities
        
    except Exception as e:
        self.logger.error(f"Error extracting entities: {e}")
        return []


def _extract_keywords(self, text: str) -> List[str]:
    """Extract key terms using TF-IDF scoring."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        import re
        
        # Preprocess text: lowercase, remove punctuation, split into words
        cleaned_text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = cleaned_text.split()
        
        # Need at least 2 words for TF-IDF
        if len(words) < 2:
            return words
        
        # Create documents by treating each word as a separate document
        # and the full text as context for IDF calculation
        documents = [text.lower(), ' '.join(words)]
        
        # Initialize TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=5,
            stop_words='english',
            ngram_range=(1, 2),  # Include both unigrams and bigrams
            min_df=1,
            token_pattern=r'\b[a-zA-Z]{3,}\b'  # Words with 3+ characters
        )
        
        # Fit and transform the documents
        tfidf_matrix = vectorizer.fit_transform(documents)
        
        # Get feature names (keywords)
        feature_names = vectorizer.get_feature_names_out()
        
        # Get TF-IDF scores for the original text (first document)
        scores = tfidf_matrix[0].toarray()[0]
        
        # Create keyword-score pairs and sort by score
        keyword_scores = list(zip(feature_names, scores))
        keyword_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Extract top keywords with non-zero scores
        keywords = [keyword for keyword, score in keyword_scores if score > 0]
        
        return keywords[:5]  # Return top 5 keywords
        
    except Exception as e:
        self.logger.error(f"Error extracting keywords with TF-IDF: {e}")
        # Fallback to simple word extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        return list(set(words))[:5]

def _generate_query_variations(self, user_query: str) -> List[str]:
    """Create query variations using LLM-powered expansion."""
    variations = [user_query]
    
    # Generate LLM-powered query expansions
    expansion_prompt = f"""
    Given this user query, generate 3-4 alternative phrasings that would help find relevant information:
    
    Original query: "{user_query}"
    
    Generate variations that:
    1. Rephrase using different terminology
    2. Break down complex queries into simpler components
    3. Add context or specify the type of information sought
    4. Convert questions to declarative statements (and vice versa)
    
    Return only the variations, one per line, without numbering or explanation.
    """
    
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a query expansion expert. Generate concise, focused query variations."},
            {"role": "user", "content": expansion_prompt}
        ],
        temperature=0.3,
        max_tokens=200
    )
    
    llm_variations = response.choices[0].message.content.strip().split('\n')
    llm_variations = [v.strip() for v in llm_variations if v.strip() and len(v.strip()) > 5]
    variations.extend(llm_variations)
    
    self.logger.info(f"Generated {len(llm_variations)} LLM variations for query")

    # Remove duplicates while preserving order
    seen = set()
    unique_variations = []
    for variation in variations:
        if variation not in seen:
            seen.add(variation)
            unique_variations.append(variation)
    
    return unique_variations

def process(self, user_query: str) -> VectorSearchResults:
    """Enhanced process method with query preprocessing."""
    self.logger.info(f"Performing enhanced vector search for: {user_query}")
    
    # Preprocess query
    query_analysis = self._preprocess_query(user_query)
    
    # Search with multiple query variations
    all_results = []
    for variation in query_analysis['variations']:
        query_embedding = self._get_embedding(variation)
        if query_embedding:
            results = self._search_all_collections(query_embedding)
            all_results.extend(results)
    
    # Deduplicate and organize results
    vector_results = self._organize_results(all_results, user_query)
    
    self.logger.info(
        f"Found {len(vector_results.entity_ids)} entities, "
        f"{len(vector_results.claim_ids)} claims, "
        f"{len(vector_results.full_text_ids)} full_text matches"
    )
    
    return vector_results
```

#### Implementation Plan
1. Add new methods to `VectorSearcher` class
2. Modify `process()` method to use query preprocessing
3. Update imports to include NLP dependencies
4. Add configuration options for keyword extraction

#### Testing Strategy
- Test with various query types (questions, statements, keywords)
- Compare results quality before/after implementation
- Measure performance impact of multiple embeddings
