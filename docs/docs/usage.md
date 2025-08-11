# Usage Examples

## Basic Search

Search for relevant RFC sections using semantic similarity:

```bash
python -m rfc_rag.modeling.predict search "TCP congestion control"
```

Output example:
```
Top 5 results for query: 'TCP congestion control'
================================================================================

Rank 1 | Similarity: 0.8234
RFC: rfc8312 - CUBIC for Fast and Long-Distance Networks
Section: 2. Conventions
Text: CUBIC is a congestion control algorithm for TCP that is designed to be more scalable than the traditional TCP congestion control algorithms...
--------------------------------------------------------------------------------
```

## Advanced Search Options

### Increase number of results
```bash
python -m rfc_rag.modeling.predict search "IPv6 addressing" --k 10
```

### Use different embedding model
```bash
python -m rfc_rag.modeling.predict search "HTTP/2 protocol" --model-name "all-mpnet-base-v2"
```

## AI-Powered Q&A

Get intelligent answers from Claude based on RFC context:

```bash
python -m rfc_rag.modeling.predict answer "How does TCP handle packet loss?"
```

Output example:
```
Claude's Answer:
================================================================================
Based on the RFC documentation provided, TCP handles packet loss through several mechanisms:

1. **Retransmission**: When TCP detects that a packet has been lost (through timeouts or duplicate ACKs), it retransmits the lost packet.

2. **Congestion Control**: TCP reduces its sending rate when packet loss is detected, assuming that loss indicates network congestion...
================================================================================

Performance Metrics:
--------------------------------------------------------------------------------
Index/metadata load time: 0.234s
Model load time: 1.456s
Query embedding time: 0.089s
FAISS search time: 0.012s
Context building time: 0.003s
Claude API time: 2.345s
Total end-to-end time: 4.139s
--------------------------------------------------------------------------------

Relevance Metrics:
--------------------------------------------------------------------------------
Average similarity score: 0.7234
Min similarity score: 0.6123
Max similarity score: 0.8456
High relevance chunks (>0.7): 3/5
Medium relevance chunks (0.5-0.7): 2/5
Low relevance chunks (<0.5): 0/5
--------------------------------------------------------------------------------

Sources (top 5 matches):
1. RFC 793 - Transmission Control Protocol
   Section: 3.7 Data Communication (similarity: 0.8456 - HIGH)
2. RFC 5681 - TCP Congestion Control
   Section: 3.1 Slow Start and Congestion Avoidance (similarity: 0.7845 - HIGH)
...
```

## Query Examples by Topic

### Network Protocols
```bash
# General protocol information
python -m rfc_rag.modeling.predict answer "What is the difference between TCP and UDP?"

# Specific protocol features
python -m rfc_rag.modeling.predict answer "How does HTTP/2 multiplexing work?"

# Protocol security
python -m rfc_rag.modeling.predict answer "What are the security considerations for TLS 1.3?"
```

### Addressing and Routing
```bash
# IPv6 questions
python -m rfc_rag.modeling.predict answer "How does IPv6 address auto-configuration work?"

# BGP routing
python -m rfc_rag.modeling.predict answer "What is BGP route reflection?"

# DHCP
python -m rfc_rag.modeling.predict answer "How does DHCP lease renewal work?"
```

### Performance and Optimization
```bash
# Congestion control
python -m rfc_rag.modeling.predict answer "What are the advantages of CUBIC over traditional TCP?"

# Quality of Service
python -m rfc_rag.modeling.predict answer "How does DiffServ provide quality of service?"

# Load balancing
python -m rfc_rag.modeling.predict answer "What are the methods for HTTP load balancing?"
```

## Advanced Usage

### Custom Model Selection
```bash
# Use Claude Sonnet for more detailed answers
python -m rfc_rag.modeling.predict answer "Explain QUIC protocol design decisions" --claude-model "claude-3-sonnet-20240229"

# Use different embedding model for search
python -m rfc_rag.modeling.predict answer "BGP path selection" --model-name "all-mpnet-base-v2"
```

### Adjusting Context Size
```bash
# Use more context chunks for complex questions
python -m rfc_rag.modeling.predict answer "Compare TCP variants" --k 10

# Use fewer chunks for simple questions (faster, cheaper)
python -m rfc_rag.modeling.predict answer "What port does HTTP use?" --k 2
```

## Tips for Better Results

### Query Formulation
- **Good**: "How does TCP congestion control work?"
- **Better**: "What algorithms does TCP use for congestion control?"
- **Best**: "Explain TCP congestion control algorithms and their differences"

### Search vs Answer
- Use `search` to explore what's available on a topic
- Use `answer` when you have a specific question
- Search is faster and doesn't require API calls

### Performance Optimization
- First search shows what RFCs are relevant
- Then ask specific questions about those RFCs
- Use appropriate `--k` values (more context = better answers but slower)

## Interpreting Results

### Similarity Scores
- **>0.7**: High relevance, very likely to contain your answer
- **0.5-0.7**: Medium relevance, may contain useful context
- **<0.5**: Low relevance, probably not helpful

### Performance Metrics
- **Total time**: Complete end-to-end response time
- **Claude API time**: Usually the slowest component
- **Model load time**: One-time cost, faster on subsequent queries
- **Search time**: FAISS is very fast, usually <50ms
