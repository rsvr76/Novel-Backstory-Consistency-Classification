# Novel--Backstory Consistency Classification

**BDH-Inspired Persistent State Architecture for Long-Context Narrative
Reasoning**

A Track B solution for the **Kharagpur Data Science Hackathon 2026**
that classifies whether a hypothetical character backstory is consistent
with a complete long-form novel.

## Problem Statement

The task provides:

-   A complete long-form novel.
-   A hypothetical backstory for one of the novel's central characters.

The system must determine whether the proposed backstory is:

-   `1` --- **Consistent** with the novel
-   `0` --- **Contradictory** to the novel

The main challenge is long-context reasoning. A novel can contain
information about a character's timeline, relationships, family,
occupation, location, and major events across many parts of the text. A
useful system therefore needs to accumulate information while processing
the narrative instead of relying only on a single local passage.


## Approach

This project implements a **BDH-inspired reasoning architecture** rather
than training or reproducing the complete BDH model.

The implementation follows three principles from the Track B
requirements:

1.  **Persistent internal state** --- maintain information while
    processing narrative chunks.
2.  **Selective/sparse updates** --- update only the most important
    dimensions of the state.
3.  **Incremental belief formation** --- gradually update beliefs about
    the character as more narrative evidence is processed.

### Pipeline

``` text
Full Novel
    │
    ▼
Text Chunking
    │
    ▼
SentenceTransformer Encoding
    │
    ▼
Projection to State Space
    │
    ▼
BDH-Inspired Persistent State
    │
    ├── Gated Updates
    ├── Sparse Dimension Selection
    └── Incremental Belief Tracking
    │
    ▼
Character Narrative State
    │
    ├──────────────────────────┐
    │                          │
    ▼                          ▼
Backstory Encoding       Evidence Retrieval
    │                          │
    └────────────┬─────────────┘
                 ▼
       Hybrid Compatibility
            Classification
                 │
        ┌────────┴────────┐
        ▼                 ▼
 Neural Scoring      Rule-Based Signals
        │                 │
        └────────┬────────┘
                 ▼
        Final Prediction
        ┌────────┴────────┐
        ▼                 ▼
   1 = Consistent   0 = Contradictory
```
<br>

## 1. Novel Processing

### Text Chunking

The complete novel is divided into overlapping chunks.

``` python
chunk_novel(text, chunk_size=200, overlap=50)
```

Overlapping chunks help preserve information around chunk boundaries.

### Semantic Encoding

Each chunk is encoded using:

``` text
SentenceTransformer
all-mpnet-base-v2
```

This converts textual chunks into dense semantic embeddings.

### State Projection

The 768-dimensional SentenceTransformer embedding is projected into a
smaller state representation:

``` text
768 → 256
```

The resulting vector becomes the input to the persistent state
mechanism.

<br>

## 2. BDH-Inspired Persistent State

The `BDHState` module maintains a persistent state while the novel is
processed sequentially.

For each chunk:

``` text
Previous State + Current Chunk
              │
              ▼
        Importance Scorer
              │
              ▼
      Select Important Dimensions
              │
              ▼
          Update Gate
              │
              ▼
       Candidate State
              │
              ▼
      Sparse State Update
```

The state therefore carries information from earlier parts of the
narrative into later processing.

### Gated Update

The implementation uses a gated update mechanism:

``` text
candidate = (1 - z) * previous_state + z * candidate_state
```

where `z` controls how strongly the new information modifies the
previous state.

### Selective Sparse Updates

An importance scorer assigns an importance value to state dimensions.

Only the top portion of dimensions is allowed to update, while the
remaining dimensions retain their previous values.

This explicitly implements the **selective/sparse update principle**
required by the Track B BDH-inspired option.

<br>

## 3. Incremental Belief Formation

The system maintains structured character-related beliefs while reading
the novel.

Examples include:

-   Alive/dead status
-   Birth year
-   Death year
-   Home/location
-   Family status
-   Occupation

Each belief contains:

``` text
Value
Confidence
Evidence
Locked status
```

As new chunks are processed, the beliefs are updated incrementally.

High-confidence beliefs can become locked, preventing weaker later
evidence from unnecessarily replacing stronger earlier evidence.

<br>

## 4. Character-Aware Narrative Processing

The system identifies chunks relevant to the target character using
character-name matching.

This allows the persistent state to focus primarily on narrative
passages associated with the character rather than processing every
passage equally during character-state construction.

<br>

## 5. State Trajectory

The system maintains a history of state and belief updates.

This makes it possible to inspect how the system's internal
representation changes as the narrative progresses.

The implementation records:

-   Updated state dimensions
-   Importance scores
-   Sparsity statistics
-   Belief changes
-   Confidence trajectory
-   Evidence associated with beliefs

This provides an interpretable view of the incremental reasoning
process.

<br>

## 6. Backstory Representation

The hypothetical backstory is encoded using the same SentenceTransformer
representation.

The resulting representation is projected into the model's state space
and compared with the accumulated narrative state.

The classifier combines:

``` text
Narrative State
      +
Backstory Representation
      +
Extracted Narrative Features
```

to produce a compatibility score.

<br>

## 7. Hybrid Classification

The final decision combines two types of signals.

### Neural Signal

The neural classifier learns compatibility between:

-   Narrative state
-   Backstory representation
-   Extracted features

### Rule-Based Signals

Explicit contradiction patterns are also checked for cases such as:

-   Death vs. survival
-   Location conflicts
-   Birth information
-   Family information
-   Other explicit textual inconsistencies

The neural prediction and rule-based signals are then combined to
produce the final binary prediction.

<br>

## 8. Feature Extraction

The `UniversalFeatureExtractor` extracts task-specific features from the
backstory and retrieved evidence.

The current feature groups include:

-   Birth claims
-   Death claims
-   Birth contradictions
-   Death contradictions
-   Family claims
-   Location claims
-   Profession claims
-   Personality claims
-   Wealth claims
-   Negation intensity
-   Backstory specificity
-   Evidence semantic similarity

These features provide additional structured information to the
classifier.

<br>

## 9. Evidence Retrieval

For prediction generation, the system retrieves semantically similar
narrative chunks using SentenceTransformer embeddings.

The retrieved passages are used to provide additional context for
rule-based checks and rationale generation.

The implementation uses cosine similarity to identify relevant passages.

<br>

## 10. Data Augmentation

The training pipeline includes lightweight textual augmentation.

The current augmentation strategies include:

-   Paraphrasing selected phrases
-   Negating selected statements
-   Emphasizing selected attributes

Contradictory examples can therefore be generated from consistent
examples in specific cases.

Augmentation is applied only to the training split, while validation
data remains unchanged.

<br>

## 11. Training Strategy

The pipeline separates narrative-state construction from classifier
training.

The workflow is:

``` text
Training Data
     │
     ▼
Character-Aware Novel Processing
     │
     ▼
Pre-computed Narrative States
     │
     ▼
Train / Validation Split
     │
     ▼
Training Augmentation
     │
     ▼
Compatibility Classifier
     │
     ▼
Validation
     │
     ▼
Best Model Checkpoint
```

The classifier is trained using:

-   `AdamW`
-   `BCEWithLogitsLoss`
-   Class weighting
-   Learning-rate scheduling
-   Gradient clipping
-   Early stopping

## Technologies

| Technology | Purpose |
|---|---|
| Python | Main implementation |
| PyTorch | Neural modules, training, and classification |
| SentenceTransformers | Semantic text embeddings |
| `all-mpnet-base-v2` | Dense sentence/document representation |
| NumPy | Numerical operations and feature processing |
| Pandas | Dataset loading and manipulation |
| scikit-learn | Train/validation splitting and ML utilities |
| Regular Expressions | Pattern-based information extraction |
| Kaggle | Development and GPU execution environment |

The exact directory structure may vary depending on the final notebook
and submission organization.

<br>

## Output

The prediction pipeline produces:

``` text
results.csv
```

with the required prediction information:

``` text
Story ID | Prediction
```

where:

``` text
1 → Consistent
0 → Contradictory
```

A short rationale can also be generated by the implementation.

<br>

## Why This Approach?

A complete 100k+ word narrative cannot be treated like a normal
short-text classification problem.

The approach therefore combines:

-   **Semantic representation** for understanding textual meaning
-   **Persistent state** for carrying information across the narrative
-   **Sparse updates** for selective information retention
-   **Incremental beliefs** for accumulating character-level knowledge
-   **State trajectory analysis** for long-range narrative reasoning
-   **Rule-based checks** for explicit contradictions
-   **Neural classification** for learned compatibility patterns

The goal is to combine learned semantic similarity with persistent,
structured, incremental reasoning.

<br>

## BDH Relationship

This project does **not** attempt to reproduce the full BDH architecture
or train a large-scale BDH model.

Instead, it implements the Track B option based on **reasoning
components explicitly inspired by BDH principles**:

``` text
Persistent Internal State
          +
Selective / Sparse Updates
          +
Incremental Belief Formation
          =
BDH-Inspired Narrative Reasoning
```

This design focuses on applying those principles to long-context
narrative consistency classification.

<br>

## Reproducibility

The notebook fixes random seeds for:

-   Python
-   NumPy
-   PyTorch
-   CUDA

The training pipeline also uses:

-   Saved best-model checkpoints
-   Explicit train/validation splitting
-   Training-only augmentation
-   Deterministic configuration where supported

<br>  

## Limitations

The current implementation has several limitations:

-   Rule-based extraction is pattern-dependent.
-   Character-name matching can miss references expressed through
    aliases or pronouns.
-   Semantic similarity does not guarantee logical consistency.
-   The belief extractor currently covers a limited set of attributes.
-   The sparse state mechanism is an explicit BDH-inspired design rather
    than the original BDH architecture.
-   More sophisticated temporal, relational and causal reasoning could
    improve difficult cases.

<br>

## Future Improvements

Potential improvements include:

1.  Better entity and coreference resolution.
2.  Explicit temporal reasoning across events.
3.  Character relationship graphs.
4.  More robust contradiction detection.
5.  Multi-hop evidence retrieval.
6.  Better calibration of confidence scores.
7.  Learned evidence weighting.
8.  More comprehensive belief types.
9.  Stronger causal reasoning over narrative events.
10. Ablation studies comparing persistent, sparse and incremental
    components.



