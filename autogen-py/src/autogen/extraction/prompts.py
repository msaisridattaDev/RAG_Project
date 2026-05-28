"""LLM prompt templates for entity/relationship extraction (Phase 3 Days 12-13).

Mirrors the .NET source's prompt chain: EntityExtraction, EntityContinueExtraction,
ExtractMissingEntities, NormalizeNames, plus medical entity types.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Token delimiter constants — mirrored from .NET prompt templates
# ---------------------------------------------------------------------------
RESPONSE_DELIMITER = "<|>"
TUPLE_DELIMITER = "##"
COMPLETION_DELIMITER = "<|COMPLETE|>"

# ---------------------------------------------------------------------------
# Canonical entity types for medical domain
# ---------------------------------------------------------------------------
MEDICAL_ENTITY_TYPES: list[str] = [
    "DRUG",
    "ENZYME",
    "DISEASE",
    "SYMPTOM",
    "TREATMENT",
    "PROCEDURE",
    "ANATOMY",
    "GENE",
    "PROTEIN",
    "PATHWAY",
    "BIOMARKER",
    "CONDITION",
    "COMPOUND",
    "MECHANISM",
    "ORGAN",
    "CELL_TYPE",
    "RECEPTOR",
    "HORMONE",
    "ELECTROLYTE",
    "VITAMIN",
]

# ---------------------------------------------------------------------------
# Primary extraction prompt — first pass
# ---------------------------------------------------------------------------
ENTITY_EXTRACTION_PROMPT = """\
You are a medical knowledge extraction assistant. Read the text below and extract:
1. **Entities**: named medical concepts (drugs, diseases, enzymes, genes, proteins, 
   symptoms, treatments, procedures, anatomical structures, pathways, biomarkers, 
   etc.) with their type and a short description.
2. **Relationships**: connections between entities, with a description, keywords, 
   and a strength score (0.0-1.0).
3. **Content keywords**: important topic keywords for this chunk.

{entity_types_section}

--- DELIMITERS ---
Entity delimiter: {response_delimiter}
Relationship delimiter: {response_delimiter}
Keyword delimiter: {response_delimiter}
Tuple delimiter: {tuple_delimiter}
Completion marker: {completion_delimiter}

--- TEXT ---
{input_text}

--- INSTRUCTIONS ---
1. Identify all MEDICAL entities in the text. For each entity, output:
   ({tuple_delimiter}name{tuple_delimiter}type{tuple_delimiter}description)
2. Identify all RELATIONSHIPS between entities. For each relationship, output:
   ({tuple_delimiter}source{tuple_delimiter}target{tuple_delimiter}description{tuple_delimiter}keywords{tuple_delimiter}strength)
   Keywords should be comma-separated. Strength should be 0.0-1.0.
3. Identify CONTENT KEYWORDS important for this chunk. Output as comma-separated list.
4. Use {response_delimiter} to separate entities, relationships, and keywords sections.
5. End your response with {completion_delimiter}

Begin your extraction:
"""

ENTITY_EXTRACTION_WITH_TYPES_SECTION = """\
Valid entity types: {entity_types}
"""

# ---------------------------------------------------------------------------
# Gleaning prompt — "are there others?"
# ---------------------------------------------------------------------------
ENTITY_CONTINUE_EXTRACTION_PROMPT = """\
You previously extracted the following entities from the text:
{previous_entities}

Are there any ADDITIONAL medical entities in this text that you missed? 
Look carefully for diseases, drugs, enzymes, genes, proteins, symptoms, treatments,
procedures, anatomical structures, pathways, biomarkers, and other medical concepts.

{entity_types_section}

--- DELIMITERS ---
Entity delimiter: {response_delimiter}
Tuple delimiter: {tuple_delimiter}
Completion marker: {completion_delimiter}

--- TEXT ---
{input_text}

--- INSTRUCTIONS ---
Output ONLY additional entities you missed before. Use the same format:
({tuple_delimiter}name{tuple_delimiter}type{tuple_delimiter}description)
End with {completion_delimiter}
If there are no additional entities, just output {completion_delimiter}
"""

EXTRACT_MISSING_ENTITIES_PROMPT = """\
A previous extraction found relationships referencing these entities,
but the entities themselves were not extracted from this text chunk:
{missing_names}

Please extract these specific entities from the text below:

--- DELIMITERS ---
Entity delimiter: {response_delimiter}
Tuple delimiter: {tuple_delimiter}
Completion marker: {completion_delimiter}

--- TEXT ---
{input_text}

--- INSTRUCTIONS ---
For each entity in the list above that appears in the text, output:
({tuple_delimiter}name{tuple_delimiter}type{tuple_delimiter}description)
End with {completion_delimiter}
If none of the entities appear in the text, just output {completion_delimiter}
"""

# ---------------------------------------------------------------------------
# Name normalization prompt — global synonym clustering
# ---------------------------------------------------------------------------
NORMALIZE_NAMES_PROMPT = """\
You are a medical terminology normalization assistant. Given a list of entity names 
extracted from medical documents, group them by concept — synonyms, abbreviations, 
alternate spellings, and case variants should all map to a single canonical name.

For each group, choose the most standard/formal name as the canonical form.
Return a JSON mapping from each original name to its canonical form.

--- ENTITY NAMES ---
{entity_names}

--- INSTRUCTIONS ---
Return a JSON object where:
- Keys are the original extracted names
- Values are the canonical names they should map to
- Names that are already canonical should map to themselves
- Example: {{"MI": "Myocardial Infarction", "myocardial infarct": "Myocardial Infarction"}}

Return ONLY valid JSON, no other text.
"""

# ---------------------------------------------------------------------------
# EntityTypeResolver prompt — canonical type normalization
# ---------------------------------------------------------------------------
ENTITY_TYPE_RESOLVER_PROMPT = """\
You are a medical ontology normalizer. Map the following raw entity type to one of 
the canonical types from the list below. The raw type may be a synonym, abbreviation, 
or related concept.

Valid canonical types:
{valid_types}

Raw type: {raw_type}
Entity context (optional): {context}

--- INSTRUCTIONS ---
Return ONLY the canonical type string, nothing else.
If no canonical type matches, return "CONCEPT" as the fallback.
"""