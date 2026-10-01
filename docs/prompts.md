# Exact Prompts for Journal Appendix

## Provenance and Scope

The prompt bodies below are transcribed from the source constants and builders used by the saved run stages. Dynamic document text, column definitions, candidate answers, evidence, page images, and blinded score batches are shown as named runtime insertions; the exact source builders are included. No inference, retrieval, verification, or judging was run for this appendix.

Run-linked model identities and stage runs:

| Family | Exact model name | B1 run | Agent A run | Agent B / E run | Per-paper stage prompt rules |
|---|---|---|---|---|---|
| Qwen | `Qwen/Qwen3.6-27B` | `e1-qwen36-full-b1-r1` | `schema-mhspc-trials-20260919020503-v4` | `schema-mhspc-trials-20260919020503-v4` | B1 `v5`; A/B/E `notes:853a2edb7ecd` |
| Mistral | `mistralai/Mistral-Small-3.2-24B-Instruct-2506` | `e1-mistral-small32-full-b1-r1` | `e1-mistral-small32-full-a-r1` | `e1-mistral-small32-full-e-r1` | all stages `notes:853a2edb7ecd` |
| Gemma | `google/gemma-4-31B-it` | `e1-gemma4-31b-full-b1-r1` | `e1-gemma4-31b-full-a-opt-r1` | `e1-gemma4-31b-full-e-opt-r1` | all stages `notes:853a2edb7ecd` |
| Llama | `RedHatAI/Llama-4-Scout-17B-16E-Instruct-quantized.w4a16` | `e1-llama4-scout-full-b1-r1` | `e1-llama4-scout-full-a-r1` | `e1-llama4-scout-full-e-r1` | all stages `notes:853a2edb7ecd` |

Run headers are `new_pipeline_outputs/benchmark_runs/<run>.json`; per-paper records are `new_pipeline_outputs/results/<paper>/runs/<run>/<stage>/extraction_metadata.json`. For every paper, stage metadata records Qwen B1 `v5`, Qwen A/B/E `notes:853a2edb7ecd`, and all Mistral/Gemma/Llama B1/A/B/E stages `notes:853a2edb7ecd`. This conflicts with top-level Mistral/Gemma/Llama benchmark headers that say `extraction_rules: v5`; this appendix follows per-paper stage metadata attached to completed outputs. Run headers record `git.dirty: true`.

Input variants across all ten papers: Qwen and Mistral Agent A used `markdown_images`; Gemma and Llama Agent A used `markdown`. Qwen and Mistral reconciliation/verifier calls had page images (`page_image_scale: 2.0`); Gemma and Llama did not (`page_image_scale: null`). All four model catalog entries support tools and JSON schema. Search uses page text and does not attach images.

Qwen A/B/E has no saved raw per-call prompt/conversation transcripts. Its stage metadata pins model, input mode, note fingerprint, and reconciler version, but not historical request/schema serialization. The Qwen B1 `raw_llm_responses.json` stores responses, not sent prompts. The source-defined prompt templates are reproduced below; Qwen A historical enum/schema state cannot be independently verified. The judge reports record model, temperature, scorer, and rubric path but no prompt hash.

The fixed evaluation judge is Vertex `gemini-3.5-flash` at temperature `0.0`; `Qwen/Qwen3-Embedding-8B` is the shared embedding model, not an evaluated chat model.

## 1. Single-Pass Baseline B1

Source:
`src/evisearch/services/markdown_baseline.py`

Defined in:
`build_prompt`, `ChatMarkdownProvider.query_markdown_with_schema`, `build_json_schema_for_group`, `run_baseline_stage`

Used by final runs:
Qwen `e1-qwen36-full-b1-r1`; Mistral `e1-mistral-small32-full-b1-r1`; Gemma `e1-gemma4-31b-full-b1-r1`; Llama `e1-llama4-scout-full-b1-r1`.

Prompt role:
One user message with two text parts in order: group prompt, then parsed-document part. No system message. One call per label group.

Prompt text from `build_prompt` with runtime values marked; the numbered item block repeats once for every `item in items`:
````text
Extract values for the following columns (Label: {{label}}):

1. {{item['column']}}: {{item['definition']}}
   If not present, use value: 'not found' and reasoning: 'not found'.
{{rules}}

============================================================
Pay special attention to table and figure captions to check if the results are reported for the whole population or sub-group wise. If values are reported for sub-groups in different tables, and the query asks for the whole population, combine values from logical subgroups that make up the whole population. Output a single JSON object. For each column provide 'value' (the extracted value or 'not found') and 'reasoning' (where you found it and how you derived it, or 'not found').
============================================================
````
Second text part of the same user message:
````text
---

DOCUMENT:

{markdown_text}
````

### Verbatim runtime prompt builder
````python
def build_prompt(label: str, items: List[Dict[str, str]], rules: str = "") -> str:
    lines = [f"Extract values for the following columns (Label: {label}):\n"]
    for i, item in enumerate(items, 1):
        lines.append(
            f"{i}. {item['column']}: {item['definition']}\n"
            "   If not present, use value: 'not found' and reasoning: 'not found'."
        )
    if rules:
        lines.append(rules)
    lines.append("\n" + "=" * 60)
    lines.append(
        "Pay special attention to table and figure captions to check if the results are reported for the whole population or sub-group wise. "
        "If values are reported for sub-groups in different tables, and the query asks for the whole population, combine values from logical subgroups that make up the whole population. "
        "Output a single JSON object. For each column provide "
        "'value' (the extracted value or 'not found') and "
        "'reasoning' (where you found it and how you derived it, or 'not found')."
    )
    lines.append("=" * 60)
    return "\n".join(lines)
````

### Verbatim message construction
````python
    def query_markdown_with_schema(
        self, prompt: str, markdown_text: str, json_schema: Dict[str, Any]
    ) -> Tuple[str, int, int]:
        result = self.chat.chat(
            [Message.user(prompt, "---\n\nDOCUMENT:\n\n" + markdown_text)],
            response_schema=json_schema,
            max_tokens=MAX_TOKENS["baseline"],
        )
        with self._lock:
            self.usage.add(result.usage)
            self.calls.append(result.call_record())
        return result.text, result.usage.input_tokens, result.usage.output_tokens
````

### Verbatim structured-output schema builder
````python
def build_json_schema_for_group(columns: List[str]) -> Dict[str, Any]:
    properties = {}
    for col in columns:
        properties[col] = {
            "type": "object",
            "properties": {
                "value": {
                    "type": "string",
                    "description": VALUE_DESCRIPTION,
                },
                "reasoning": {
                    "type": "string",
                    "description": REASONING_DESCRIPTION,
                },
            },
            "required": ["value", "reasoning"],
        }
    return {
        "type": "object",
        "properties": properties,
        "required": list(columns),
    }
````

Generated response schema shape for one runtime column (the object is repeated for every name in `columns`):
````json
{
  "type": "object",
  "properties": {
    "{{column_name}}": {
      "type": "object",
      "properties": {
        "value": {
          "type": "string",
          "description": "The extracted value exactly as in the document (e.g. number, percentage, text); use 'not found' if not reported."
        },
        "reasoning": {
          "type": "string",
          "description": "Brief reasoning on where in the document you found the value and how you derived it; or 'not found' if not reported."
        }
      },
      "required": ["value", "reasoning"]
    }
  },
  "required": ["{{column_name}}"]
}
````

B1 inserts `shared_rules()` without a column filter. Qwen B1 inserts `RULES["v5"]`; the other three B1 runs insert all agent-role frozen notes. The API also receives the generated JSON schema out-of-band. Exact rule insertions are in [Frozen Shared-Note Insertions](#frozen-shared-note-insertions).

## 2. Agent A / PDF Query Agent

Source:
`src/evisearch/services/pdf_query.py`; shared rules: `src/evisearch/services/extraction_rules.py`, `src/evisearch/knowledge/notes.py`

Defined in:
`SYSTEM_PROMPT`, `IMAGE_RULES`, `system_prompt_text`, `build_columns_prompt`, `extraction_schema`, `build_document_input`, `run_pdf_query.ask`

Used by final runs:
Qwen `schema-mhspc-trials-20260919020503-v4`; Mistral `e1-mistral-small32-full-a-r1`; Gemma `e1-gemma4-31b-full-a-opt-r1`; Llama `e1-llama4-scout-full-a-r1`.

Prompt role:
System message followed by one user message. The user message has document text/image parts first, then the batch-column prompt. JSON schema is sent out-of-band.

System prompt, image-capable run variant (Qwen and Mistral):
````text
You extract clinical trial data from a research paper.

Use ONLY the document provided. Return a value for every requested column.

Rules:
- Use "Not reported" and found=false when the document does not report the value.
- For N (%) columns include both the count and the percentage when reported.
- Check table and figure captions for scope: use overall/all-patient values for overall columns, and the matching
  subgroup for subgroup columns. When only subgroups are reported and the column asks for the whole population,
  combine the subgroups that make up the whole population.
- Attribution lists the 1-based page number(s) the value came from, with modality "table", "figure" or "text", and
  evidence: the text on that page that supports the value, copied as printed (the sentence; for a table, the row
  label, the column header and the cell; for a figure, its label and what you read from it). Every value is checked
  against the page and evidence you give, so cite the page that actually shows it.
- Pages come with their parsed text and, where included, their rendered image. Use the parsed text for exact wording
  and numbers in text and tables; use the image for figures (Kaplan-Meier curves, forest plots, flow diagrams) and to
  check table layout. When the parsed text and the image disagree about a value, trust the image.
````

System prompt, markdown-only run variant (Gemma and Llama):
````text
You extract clinical trial data from a research paper.

Use ONLY the document provided. Return a value for every requested column.

Rules:
- Use "Not reported" and found=false when the document does not report the value.
- For N (%) columns include both the count and the percentage when reported.
- Check table and figure captions for scope: use overall/all-patient values for overall columns, and the matching
  subgroup for subgroup columns. When only subgroups are reported and the column asks for the whole population,
  combine the subgroups that make up the whole population.
- Attribution lists the 1-based page number(s) the value came from, with modality "table", "figure" or "text", and
  evidence: the text on that page that supports the value, copied as printed (the sentence; for a table, the row
  label, the column header and the cell; for a figure, its label and what you read from it). Every value is checked
  against the page and evidence you give, so cite the page that actually shows it.
````

Both append `shared_rules(columns=names)` to the system message; frozen note text and selection logic are in [Frozen Shared-Note Insertions](#frozen-shared-note-insertions).

User prompt from `build_columns_prompt` with empty preferences (saved preferences file is empty); the column block repeats per `batch_columns` item:
````text

COLUMNS TO EXTRACT:
---
Column 1: {{col.get('column_name', '')}}
Definition: {{col.get('definition', '')}}

Return JSON: {"columns": [{"column": <exact column name>, "value": ..., "reasoning": ..., "found": true|false, "attribution": [{"page": N, "modality": "text"|"table"|"figure", "evidence": ...}]}]}
````

Document-part framing sent before the column prompt:
````text
Image mode, all pages included:
DOCUMENT: {len(pages)} pages, each given as its parsed text followed by its rendered image.

=== PAGE {number}: parsed text ===
{text}
=== PAGE {number}: image ===
[ImagePart(images[number].png)]

END OF DOCUMENT.

Markdown-only mode:
DOCUMENT: {len(pages)} pages of parsed text.

=== PAGE {number}: parsed text ===
{text}

END OF DOCUMENT.
````

### Verbatim column-prompt builder
````python
def build_columns_prompt(batch_columns: List[Dict[str, Any]], preferences: str) -> str:
    blocks = [
        f"---\nColumn {i}: {col.get('column_name', '')}\nDefinition: {col.get('definition', '')}"
        for i, col in enumerate(batch_columns, 1)
    ]
    preference_block = f"\nHUMAN EXTRACTION PREFERENCES:\n{preferences}\n" if preferences else ""
    return (
        f"{preference_block}\nCOLUMNS TO EXTRACT:\n" + "\n".join(blocks) + "\n\n"
        'Return JSON: {"columns": [{"column": <exact column name>, "value": ..., "reasoning": ..., '
        '"found": true|false, "attribution": [{"page": N, "modality": "text"|"table"|"figure", "evidence": ...}]}]}'
    )
````

### Verbatim document-part builder
````python
def build_document_input(
    doc_id: str, input_mode: str, token_budget: Optional[int] = None, image_tokens: Optional[ImageTokens] = None
) -> DocumentInput:
    """Document parts for one call. In markdown_images mode every page gets its image; when that would exceed
    token_budget, images are limited to pages with figures or tables, then dropped (info["fallback"] says which).
    image_tokens is the model's image token cost (catalog.yaml models.<key>.image_tokens)."""
    pages = load_markdown_pages(doc_id)
    texts = {number: f"=== PAGE {number}: parsed text ===\n{text}" for number, text in enumerate(pages, 1)}
    text_tokens = sum(len(text) for text in texts.values()) // CHARS_PER_TEXT_TOKEN
    info: Dict[str, Any] = {
        "input_mode": input_mode,
        "pages": len(pages),
        "image_pages": [],
        "page_image_scale": None,
        "fallback": None,
        "estimated_tokens": text_tokens,
        "token_budget": token_budget,
        "warnings": [],
    }
    if input_mode == "markdown":
        parts = [TextPart(f"DOCUMENT: {len(pages)} pages of parsed text.")]
        parts += [TextPart(texts[number]) for number in sorted(texts)]
        return DocumentInput(parts + [TextPart("END OF DOCUMENT.")], info)
    if input_mode != "markdown_images":
        raise ConfigError(f"pdf_query_input={input_mode!r}: choose markdown_images or markdown")

    pdf_path = resolve_pdf_path(doc_id)
    if not pdf_path or not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF not found for {doc_id}; markdown_images renders the page images from it")
    pdf_pages = pdf_page_count(Path(pdf_path))
    if pdf_pages != len(pages):
        info["warnings"].append(f"page count mismatch: markdown has {len(pages)} pages, PDF has {pdf_pages}")
    images = {image.page: image for image in render_pages(Path(pdf_path), list(range(1, pdf_pages + 1)), PAGE_IMAGE_SCALE)}
    candidates = {
        None: sorted(images),
        "figure_table_pages": [number for number in figure_or_table_pages(pages) if number in images],
        "markdown_only": [],
    }
    for fallback in FALLBACK_STEPS:
        image_pages = candidates[fallback]
        estimate = text_tokens + sum(images[number].estimated_tokens(image_tokens) for number in image_pages)
        fits = token_budget is None or estimate <= token_budget
        if (fits and len(image_pages) <= PDF_QUERY_MAX_PAGE_IMAGES) or fallback == "markdown_only":
            break
    if token_budget is not None and estimate > token_budget:
        info["warnings"].append(f"document needs ~{estimate} tokens but only {token_budget} are available")
    info.update(image_pages=image_pages, fallback=fallback, estimated_tokens=estimate,
                page_image_scale=PAGE_IMAGE_SCALE if image_pages else None)

    if fallback is None:
        header = f"DOCUMENT: {len(pages)} pages, each given as its parsed text followed by its rendered image."
    elif image_pages:
        header = f"DOCUMENT: {len(pages)} pages of parsed text; rendered images follow the pages with figures or tables ({', '.join(map(str, image_pages))})."
    else:
        header = f"DOCUMENT: {len(pages)} pages of parsed text (page images left out: they do not fit the model's context)."
    parts: List[Any] = [TextPart(header)]
    for number in range(1, max(len(pages), pdf_pages) + 1):
        text = texts.get(number, f"=== PAGE {number}: parsed text ===\n(no parsed text for this page)")
        if number in image_pages:
            parts += [TextPart(f"{text}\n=== PAGE {number}: image ==="), ImagePart(images[number].png)]
        else:
            parts.append(TextPart(text))
    return DocumentInput(parts + [TextPart("END OF DOCUMENT.")], info)
````

### Verbatim system-prompt composition
````python
def system_prompt_text(images: bool = True, columns: Optional[Iterable[str]] = None) -> str:
    """Agent A's system prompt: base rules, the page-image rules when images are sent, then the shared conventions
    (those for `columns` under scoped delivery)."""
    return SYSTEM_PROMPT + (IMAGE_RULES if images else "") + shared_rules(columns=columns)
````

### Verbatim structured-output schema builder
````python
def extraction_schema(names: List[str]) -> Dict[str, Any]:
    """One entry per requested column; Vertex rejects some otherwise valid multi-value column-name enums."""
    items = {**extraction_items_schema(names), "minItems": len(names), "maxItems": len(names)}
    items["items"]["properties"]["column"].pop("enum", None)
    return {"type": "object", "properties": {"columns": items}, "required": ["columns"]}
````

Generated response schema shape for one requested column (runtime `minItems` and `maxItems` equal the requested batch size; `column` has no enum in this builder):
````json
{
  "type": "object",
  "properties": {
    "columns": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "column": {"type": "string"},
          "value": {"type": "string", "description": "Extracted value, or 'Not reported'"},
          "reasoning": {"type": "string", "description": "Where the value was found and how it was derived"},
          "found": {"type": "boolean"},
          "attribution": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "page": {"type": "integer", "description": "1-based page number"},
                "modality": {"type": "string", "enum": ["text", "table", "figure"]},
                "evidence": {"type": "string", "description": "The text on this page that supports the value, copied as printed: the sentence, or the table row label with the column header and cell, or the figure label and what you read from it"}
              },
              "required": ["page", "modality", "evidence"]
            }
          }
        },
        "required": ["column", "value", "reasoning", "found", "attribution"]
      },
      "minItems": 1,
      "maxItems": 1
    }
  },
  "required": ["columns"]
}
````

Runtime names include `batch_columns`, `preferences`/`prefs`, `names`, `document.parts`, `system`, `asked`, and `schema`. The selected models advertise `json_schema: true`; the current extraction schema has no enum on `column`, and `minItems`/`maxItems` equal the requested-column count.

## 3. Agent B / Search Agent

Source:
`src/evisearch/services/search.py`; shared rules: `src/evisearch/services/extraction_rules.py`, `src/evisearch/knowledge/notes.py`; tool loop: `src/inference/tool_loop.py`

Defined in:
`SYSTEM_PROMPT`, `FOLLOW_UP`, `tool_specs`, `run_search_agent`

Used by final runs:
Qwen `schema-mhspc-trials-20260919020503-v4`; Mistral `e1-mistral-small32-full-e-r1`; Gemma `e1-gemma4-31b-full-e-opt-r1`; Llama `e1-llama4-scout-full-e-r1`.

Prompt role:
System and user messages with three tool declarations; each tool response is followed by `FOLLOW_UP` unless the tool stops the loop. Tool schemas are out-of-band.

System prompt (then agent-role `shared_rules(columns=names)`):
````text
You extract clinical trial values from document pages.

WORKFLOW (follow this order):
1. Load initial pages: get_chunks_by_page([1, 2]) first.
2. Extract from what you have: Fill as many columns as possible from pages 1-2 before calling search_chunks.
3. Identify gaps: Note which columns are still blank or unclear.
4. Search only for gaps: Call search_chunks only for those specific columns. Do not search for info you may already have.
5. Submit extraction when done.

EXTRACT-FIRST POLICY:
- Do not call search_chunks until you have attempted extraction from the pages you already have.
- Clinical trial papers often have title, authors, endpoints, eligibility, and key design info in the first few pages. Use them first.
- Before each search_chunks call: only use it for columns you cannot find or are unclear in your current content.

DOMAIN POLICY:
- Informational columns (trial name, treatment arm, control arm, phase, design): Start with get_chunks_by_page([1, 2, 3]) where this info usually appears.
- Specific columns (demographics, outcomes, adverse events): Extract from pages 1-3 first; use search_chunks only if still missing.

TABLE SCOPE AND SUBGROUP POLICY:
- Always check table headers for scope: does the table show "All Patients", "Overall", or subgroup-specific columns (e.g. "High Volume", "Low Volume")?
- For columns requesting overall/all population: use the "All Patients" or "Overall" column if present. If not present but subgroups are reported (e.g. High Volume, Low Volume), sum the subgroup values to derive overall (e.g. sum N and recalculate %).
- For columns requesting subgroup-specific data: use the matching subgroup column only.

RULES:
- Do not request pages you already have. We will tell you "already provided; check your context" for pages already sent.
- Submit when you have enough information for all columns (values or "Not reported").
- For N (%) columns include both count and percentage.
- Do NOT include "treatment" or "control" in your search queries as they are generic. Use specific terms (drug names, region names, arm labels, column-specific terms).

Attribution: For each column, list sources as [{"page": N, "modality": "text"|"table"|"figure", "evidence": "..."}]. Use "table" for table content, "figure" for figures, "text" for prose. Evidence is the text on that page that supports the value, copied as printed (the sentence; for a table, the row label, column header and cell). Every value is checked against the page and evidence you give, so cite the page that actually shows it. If not found: value="Not reported", found=false.
````

User prompt template:
````text
Extract values for these columns. Document has {total_pages} pages.

COLUMNS:
{blocks}

For informational columns (trial name, arms, etc.), use get_chunks_by_page([1, 2]) first. For specific columns, use search_chunks. Submit when you have enough information.
````
`blocks` contains one exact block per batch column, assembled from `i`, `col.get("column_name", "")`, and `definitions_map.get(...)` with `col.get("definition", "")` as fallback.

Exact tool-loop follow-up:
````text
Summarize what you learned. Then: search for more columns, load more pages, or call submit_extraction when you have enough information.
````

### Verbatim tool schema/description builder
````python
def tool_specs(names: List[str]) -> List[ToolSpec]:
    return [
        ToolSpec(
            name="search_chunks",
            description="Semantic search over document pages. Returns the best matching pages with full content. Pages you already have show 'already provided; check your context'.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query, e.g. 'median overall survival abiraterone months'"},
                },
                "required": ["query"],
            },
        ),
        ToolSpec(
            name="get_chunks_by_page",
            description="Load the full content of specific pages. Use for informational columns (trial name, arms) that appear in the first pages, or when you know the page.",
            parameters={
                "type": "object",
                "properties": {
                    "page_numbers": {"type": "array", "items": {"type": "integer"}, "description": "1-based page numbers, e.g. [1, 2, 3]"},
                },
                "required": ["page_numbers"],
            },
        ),
        ToolSpec(
            name="submit_extraction",
            description="Submit the extracted values for all columns (values or 'Not reported'). Ends the task.",
            parameters={
                "type": "object",
                "properties": {"results": extraction_items_schema(names)},
                "required": ["results"],
            },
        ),
    ]
````

### Verbatim prompt and tool-loop assembly
````python
def run_search_agent(
    doc_id: str,
    batch_columns: List[Dict[str, Any]],
    definitions_map: Dict[str, str],
    log_path: Optional[Path] = None,
    model: Optional[str] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    """Run the search agent for one batch. Returns ({column: result}, usage); never raises for model errors."""
    names = column_names(batch_columns)
    try:
        chat = get_chat("search_agent", model)
    except (ConfigError, InferenceError) as exc:
        return fill_missing({}, names, f"search_agent not run: {exc}"), Usage().to_dict()

    total_pages = retriever.get_total_pages(doc_id)
    session = _SearchSession(doc_id, names, total_pages)
    blocks = "".join(
        f"\n---\nColumn {i}: {col.get('column_name', '')}\nDefinition: {definitions_map.get(col.get('column_name', ''), '') or col.get('definition', '')}"
        for i, col in enumerate(batch_columns, 1)
    )
    user_prompt = (
        f"Extract values for these columns. Document has {total_pages} pages.\n\nCOLUMNS:\n{blocks}\n\n"
        "For informational columns (trial name, arms, etc.), use get_chunks_by_page([1, 2]) first. "
        "For specific columns, use search_chunks. Submit when you have enough information."
    )
    specs = {spec.name: spec for spec in tool_specs(names)}
    loop = run_tool_loop(
        chat,
        system=SYSTEM_PROMPT + shared_rules(columns=names),
        user=user_prompt,
        tools=[
            Tool(specs["search_chunks"], session.search_chunks),
            Tool(specs["get_chunks_by_page"], session.get_chunks_by_page),
            Tool(specs["submit_extraction"], session.submit_extraction),
        ],
        max_turns=AGENT_MAX_TURNS,
        max_tool_calls=AGENT_MAX_TOOL_CALLS,
        max_tokens=MAX_TOKENS["search_agent"],
        follow_up=FOLLOW_UP,
        finish_tool="submit_extraction",
    )

    if session.submitted is None:
        # The loop can end on a submission that was sent back (turn or call limit); keep the columns already received.
        reason = f"Agent did not submit ({loop.stopped_by}{': ' + loop.error if loop.error else ''})"
        results = fill_missing(dict(session.recovered), names, reason)
    else:
        results = fill_missing(dict(session.submitted), names, "Not extracted")

    if log_path:
        write_json(
            log_path.with_name(log_path.stem + "_conversation.json"),
            {
                "doc_id": doc_id,
                "model": chat.key,
                "stopped_by": loop.stopped_by,
                "error": loop.error,
                "tool_calls_sequence": [{"name": e["name"], "args": e["args"]} for e in loop.transcript if e["role"] == "tool"],
                "conversation": loop.transcript,
                "calls": loop.calls,
                "results": results,
            },
        )
    return results, loop.usage.to_dict()
````

### Advertised tool declarations (placeholder name stands for the run-specific name enum)
#### search_chunks
````json
{
  "name": "search_chunks",
  "description": "Semantic search over document pages. Returns the best matching pages with full content. Pages you already have show 'already provided; check your context'.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Search query, e.g. 'median overall survival abiraterone months'"
      }
    },
    "required": [
      "query"
    ]
  }
}
````
#### get_chunks_by_page
````json
{
  "name": "get_chunks_by_page",
  "description": "Load the full content of specific pages. Use for informational columns (trial name, arms) that appear in the first pages, or when you know the page.",
  "parameters": {
    "type": "object",
    "properties": {
      "page_numbers": {
        "type": "array",
        "items": {
          "type": "integer"
        },
        "description": "1-based page numbers, e.g. [1, 2, 3]"
      }
    },
    "required": [
      "page_numbers"
    ]
  }
}
````
#### submit_extraction
````json
{
  "name": "submit_extraction",
  "description": "Submit the extracted values for all columns (values or 'Not reported'). Ends the task.",
  "parameters": {
    "type": "object",
    "properties": {
      "results": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "column": {
              "type": "string",
              "enum": [
                "{{column_name}}"
              ]
            },
            "value": {
              "type": "string",
              "description": "Extracted value, or 'Not reported'"
            },
            "reasoning": {
              "type": "string",
              "description": "Where the value was found and how it was derived"
            },
            "found": {
              "type": "boolean"
            },
            "attribution": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "page": {
                    "type": "integer",
                    "description": "1-based page number"
                  },
                  "modality": {
                    "type": "string",
                    "enum": [
                      "text",
                      "table",
                      "figure"
                    ]
                  },
                  "evidence": {
                    "type": "string",
                    "description": "The text on this page that supports the value, copied as printed: the sentence, or the table row label with the column header and cell, or the figure label and what you read from it"
                  }
                },
                "required": [
                  "page",
                  "modality",
                  "evidence"
                ]
              }
            }
          },
          "required": [
            "column",
            "value",
            "reasoning",
            "found",
            "attribution"
          ]
        }
      }
    },
    "required": [
      "results"
    ]
  }
}
````

`submit_extraction.results` uses `extraction_items_schema(names)` from `src/evisearch/columns.py`; its source is included in the tool-builder source. Tool names: `search_chunks`, `get_chunks_by_page`, `submit_extraction`.

## 4. Reconciliation Agent

Source:
`src/evisearch/services/reconciliation_v5.py`; session/tool machinery: `src/evisearch/services/reconciliation.py`; shared rules: `src/evisearch/services/extraction_rules.py`, `src/evisearch/knowledge/notes.py`

Defined in:
`RECONCILER_VERSION`, `FINDINGS_PROMPT`, `RECONCILE_PROMPT`, `FINDINGS_FOLLOW_UP`, `RECONCILE_FOLLOW_UP`, `_FindingsSession.user_prompt`, `_ReconcileSession.describe_own`, `_ReconcileSession.user_prompt`, `findings_spec`, `paper_tool_specs`, `submit_spec_v5`, `_read_on_its_own`, `run_reconciliation_agent`

Used by final runs:
Qwen `schema-mhspc-trials-20260919020503-v4`; Mistral `e1-mistral-small32-full-e-r1`; Gemma `e1-gemma4-31b-full-e-opt-r1`; Llama `e1-llama4-scout-full-e-r1`. All per-paper E metadata says `own_reading_v5_contested`.

Prompt role:
Two system/user conversations per batch when contested columns require an independent read. Phase 1 hides A/B; phase 2 shows its own findings and A/B. Both have tool declarations out-of-band. Phase 1 appends auditor-role notes filtered to `read_names`; phase 2 appends agent-role notes filtered to `names`.

### Phase 1: Independent Extraction

System prompt with `RECONCILIATION_MAX_PAGE_IMAGES == 6` expanded:
````text
You extract clinical trial values from a research paper.

You are given columns, each with a definition. Fill every column.

WORKFLOW
1. get_pages([1, 2]) first, and fill what those pages answer.
2. search_chunks with terms from a column's definition for the columns still open. Do not search for what you
   already have.
3. get_pages to read a page in full with its image. Use it for values printed in a figure, a Kaplan-Meier panel, or
   a table captured as a picture: the parsed text often loses those. At most 6 page
   images per batch, after which get_pages still returns the text.
4. verify_attribution on the values you found, to confirm the page and capture the quote the value will be cited
   with. It reads the page you name and writes the column's answer itself. Use it on values you have already
   located - it is not a way to find them.
5. submit_findings when every column has a value or has been established as not reported.

RULES
- Copy the value as the paper prints it.
- Every column gets either a value with the page it is on, or the list of pages you opened looking for it. Do not
  leave a column unattempted.
- Never write a value the paper does not state. Do not read a number off a curve. Do not compute one unless the
  knowledge notes license that computation; when they do, show the arithmetic and name the row and column of every
  number in it.
- Give the page(s) each value is on and the text or table cell you took it from, copied as printed.
- When the parsed text and the page image disagree, trust the image.
- Search where the answer would be, not only where a word matches: baseline tables for characteristics, results
  tables and figure panels for outcomes, the methods for design, the discussion for durations.
````
The system then appends auditor-role frozen notes selected for phase-1 `read_names`.

### Verbatim phase-1 user-prompt builder
````python
    def user_prompt(self) -> str:
        blocks = [f"\n---\nColumn {i}: {name}\nDefinition: {self.definitions.get(name, '')}"
                  for i, name in enumerate(self.names, 1)]
        return (
            f"Answer the following columns from the paper. It has {self.total_pages} pages.\n"
            f"\nCOLUMNS:{''.join(blocks)}\n\nAnswer every column, then submit them with submit_findings."
        )
````

The generated user text begins `Answer the following columns from the paper. It has {self.total_pages} pages.`, emits `COLUMNS:` and one `Column {i}: {name}` plus definition block per requested column, and ends `Answer every column, then submit them with submit_findings.`

Phase-1 follow-up:
````text
Continue. Answer the remaining columns and submit them with submit_findings.
````

### Phase 2: Final Adjudication

System prompt with `RECONCILIATION_MAX_PAGE_IMAGES == 6` expanded:
````text
You review two independent extractions of the same paper and decide each column's final value.

You have already extracted these columns yourself. For each column you now see your answer and two candidates, A and
B (anonymous), with their reasoning and the pages and evidence they cite.

WORKFLOW
1. Where all three agree, submit that value.
2. Where they differ, read the pages the differing values cite - with get_pages, or with verify_attribution, which
   gives you a second reader's answer for that column from those pages. You have a fresh budget of
   6 page images here; the pages you opened earlier are not in this context.
3. Every value you submit must have been through verify_attribution, so the cell ships with the page and the quote a
   reader can check it against.
4. submit_verification with, per column: the final value, which answer it came from, and a verdict on each of the
   three answers.

RULES
- verify_attribution is a second reader confined to the pages you name. Where its answer differs from a claim, that
  is a disagreement for you to settle by looking - not a ruling. Its failure to find a value is not evidence the
  paper omits one.
- Answers compatible at different levels of detail (a drug class and the drug, one endpoint variant and both, a
  count and the same count with its percentage) are merged into the complete value the definition asks for; set
  final_source to "merged".
- "Not reported" is a valid final value when you have read the pages a candidate cites and they do not state one.
  Give absence_basis: the pages you read and what they say instead.
- Judge each of the three answers on its merits - own_verdict, a_verdict, b_verdict, each one of correct,
  incomplete, wrong or no_answer. Your own answer gets the same treatment as the other two.
````
The system then appends agent-role frozen notes selected for all phase-2 `names`.

### Own-answer formatting
````python
    def describe_own(self, name: str) -> str:
        own = self.own.get(name) or {}
        if own.get("skipped"):
            return "  YOURS: (not read - the two extractions already agreed on this column)"
        if own.get("unread"):
            return "  YOURS: (you did not answer this column)"
        value = own.get("value") or ""
        head = f'  YOURS: "{value}"' if value else "  YOURS: the paper states no answer for this column"
        lines = [head]
        if own.get("pages"):
            lines.append(f"     on page(s) {own['pages']}" + (f", evidence \"{own['evidence']}\"" if own.get("evidence") else ""))
        if own.get("looked_at"):
            lines.append(f"     looked at page(s) {own['looked_at']}")
        if own.get("reasoning"):
            lines.append(f"     your reasoning: {own['reasoning'][:REASONING_CHARS]}")
        return "\n".join(lines)
````

### Agent A/B candidate formatting
````python
    def describe(self, name: str) -> str:
        lines = []
        for origin in ("A", "B"):
            source = self.sources[origin][name]
            lines.append(f'  {origin}: "{source["value"]}"')
            if source["reasoning"]:
                lines.append(f'     reasoning: {source["reasoning"][:REASONING_CHARS]}')
            for item in source["pages"]:
                evidence = f', evidence "{item["evidence"]}"' if item["evidence"] else ""
                lines.append(f'     cites page {item["page"]} ({item["modality"]}){evidence}')
            if not is_absence(source["value"]) and not source["pages"]:
                lines.append("     cites no page")
        return "\n".join(lines)
````

### Verbatim phase-2 user-prompt builder
````python
    def user_prompt(self) -> str:
        blocks = [
            f"\n---\nColumn {i}: {name}\nDefinition: {self.definitions.get(name, '')}\n"
            f"{self.describe_own(name)}\n{self.describe(name)}"
            for i, name in enumerate(self.names, 1)
        ]
        return (
            f"Decide the following columns. The paper has {self.total_pages} pages. YOURS is the answer you found "
            f"yourself; A and B are two other extractions and are anonymous.\n"
            f"\nCOLUMNS:{''.join(blocks)}\n\nDecide and submit every column."
        )
````

The generated user text begins `Decide the following columns. The paper has {self.total_pages} pages. YOURS is the answer you found yourself; A and B are two other extractions and are anonymous.`, emits each column definition, YOURS block and A/B blocks, and ends `Decide and submit every column.`

Phase-2 follow-up:
````text
Continue. Fix rejected columns, verify what you still need, and submit every remaining column.
````

Forced-finish user instruction, if the tool loop exhausts its budget:
````text
You have no tool calls left. Call {tool} now with every value you have found so far; use "Not reported" for anything you could not find.
````

### Phase-1 submit schema builder
````python
def findings_spec(names: List[str]) -> ToolSpec:
    return ToolSpec(
        name="submit_findings",
        description="Your own answers for the columns, before you see anyone else's.",
        parameters={
            "type": "object",
            "properties": {
                "findings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "column": {"type": "string", "enum": names},
                            "value": {"type": "string", "description": 'the answer as the paper prints it, or "" when the paper states none'},
                            "pages": {"type": "array", "items": {"type": "integer"}, "description": "page(s) the answer is on"},
                            "evidence": {"type": "string", "description": "the text or table cell the answer was taken from"},
                            "looked_at": {"type": "array", "items": {"type": "integer"}, "description": "pages you examined for this column"},
                            "reasoning": {"type": "string"},
                        },
                        "required": ["column", "value", "reasoning"],
                    },
                }
            },
            "required": ["findings"],
        },
    )
````

### Page/search tool schema builder
````python
def paper_tool_specs() -> List[ToolSpec]:
    """The two tools v5 reads the paper with, in both phases. search_chunks is Agent B's, verbatim, so the arbiter sees
    exactly what the agent it is judging saw; get_pages is Agent B's get_chunks_by_page with the page image added."""
    return [
        ToolSpec(
            name="search_chunks",
            description="Semantic search over the paper's pages. Returns the best matching pages with full content. "
                        "Pages you already have show 'already provided; check your context'.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query, e.g. 'median overall survival abiraterone months'"},
                },
                "required": ["query"],
            },
        ),
        ToolSpec(
            name="get_pages",
            description=f"Open pages by number: each page's full content and its rendered image. The image is the only "
                        f"way to read a value printed inside a figure or a table captured as a picture. At most "
                        f"{RECONCILIATION_MAX_PAGE_IMAGES} page images per batch; past that the text still comes back.",
            parameters={
                "type": "object",
                "properties": {
                    "page_numbers": {"type": "array", "items": {"type": "integer"}, "description": "1-based page numbers, e.g. [1, 2, 3]"},
                },
                "required": ["page_numbers"],
            },
        ),
    ]
````

### Phase-2 submit schema/description adaptation
````python
def submit_spec_v5(names: List[str]) -> ToolSpec:
    """v4's submit_verification with two changes: the description no longer says a value needs a supporting verdict
    (the second reading informs the decision, it does not rule on it), and an absence carries the reading that
    justifies it."""
    spec = next(s for s in tool_specs(names) if s.name == "submit_verification")
    item = spec.parameters["properties"]["results"]["items"]
    item["properties"]["absence_basis"] = {
        "type": "object",
        "description": 'Required with "Not reported" when another answer stated a value: the reading that rules it out.',
        "properties": {
            "pages": {"type": "array", "items": {"type": "integer"}, "description": "pages you read"},
            "page_says": {"type": "string", "description": "what those pages state for this column instead"},
        },
    }
    item["properties"]["final_source"] = {"type": "string", "enum": list(SOURCES)}
    for key in ("own_verdict", "a_verdict", "b_verdict"):
        item["properties"][key] = {"type": "string", "enum": list(VERDICTS)}
    return ToolSpec(
        name=spec.name,
        description=("Submit final values for one or more columns. Every value must have been through "
                     "verify_attribution, so the cell ships with the page and quote a reader can check it against. "
                     '"Not reported" needs absence_basis when another answer stated a value. The response lists '
                     "accepted and rejected columns."),
        parameters=spec.parameters,
    )
````

### Verifier tool schema source
````python
def tool_specs(names: List[str]) -> List[ToolSpec]:
    source = {
        "type": "object",
        "properties": {
            "page": {"type": "integer", "description": "1-based page that shows the value"},
            "pages": {"type": "array", "items": {"type": "integer"}, "description": "Instead of page: the pages of a value verified on several pages"},
            "modality": {"type": "string", "enum": list(MODALITIES)},
            "evidence": {"type": "string", "description": "Supporting text as printed on that page"},
        },
    }
    return [
        ToolSpec(
            name="ask_document",
            description=f"Ask a reader that has the whole paper (text and page images). At most {document_reader.QUESTIONS_PER_CALL} questions per call; each answer comes with pages and evidence.",
            parameters={
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "column": {"type": "string", "enum": list(names), "description": "Column the question is about (its definition is sent along)"},
                                "question": {"type": "string"},
                            },
                            "required": ["question"],
                        },
                    }
                },
                "required": ["questions"],
            },
        ),
        ToolSpec(
            name="search_pages",
            description="Semantic search over the paper's pages. Returns the best matching pages with their most relevant lines.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Specific terms, e.g. 'ECOG performance status baseline characteristics'"}},
                "required": ["query"],
            },
        ),
        ToolSpec(
            name="verify_attribution",
            description=f"Check whether a page supports a value for a column (at most {VERIFY_MAX_CLAIMS} claims per call). Returns verdict, page_value and evidence per claim.",
            parameters={
                "type": "object",
                "properties": {
                    "claims": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "column": {"type": "string", "enum": list(names)},
                                "value": {"type": "string"},
                                "page": {"type": "integer"},
                                "pages": {"type": "array", "items": {"type": "integer"}, "description": f"Instead of page: up to {MAX_CLAIM_PAGES} pages when the value combines numbers from several"},
                                "evidence": {"type": "string"},
                            },
                            "required": ["column", "value"],
                        },
                    }
                },
                "required": ["claims"],
            },
        ),
        ToolSpec(
            name="submit_verification",
            description="Submit final values for one or more columns. A value is accepted only when verify_attribution found it supported on its source page; a rejected value is never accepted; \"Not reported\" is accepted when every extracted value failed the check. The response lists accepted and rejected columns.",
            parameters={
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "column": {"type": "string", "enum": list(names)},
                                "value": {"type": "string"},
                                "reasoning": {"type": "string"},
                                "verification": {"type": "string", "enum": list(VERIFICATIONS)},
                                "source": source,
                                "review": {"type": "boolean", "description": "true: send this value to a human reviewer (a partial value, or a value no page shows); a value the verifier rejected is not accepted even with review"},
                                "review_reason": {"type": "string"},
                            },
                            "required": ["column", "value", "reasoning", "verification"],
                        },
                    }
                },
                "required": ["results"],
            },
        ),
    ]
````

### Verbatim phase orchestration and tool assembly
````python
def run_reconciliation_agent(
    doc_id: str,
    batch_columns: List[Dict[str, Any]],
    definitions_map: Dict[str, str],
    source_a_data: Dict[str, Dict[str, Any]],
    source_b_data: Dict[str, Dict[str, Any]],
    log_path: Optional[Path] = None,
    model: Optional[str] = None,
    own_reading: Optional[OwnReading] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    """Reconcile one batch in two phases. Returns ({column: result}, usage). `own_reading`, from read_on_its_own,
    stands in for phase 1 when it covers exactly the columns phase 1 would read."""
    try:
        names, chat, scale, specs = _setup(batch_columns, model)
    except (ConfigError, InferenceError) as exc:
        return {name: _not_run(f"reconciliation not run: {exc}") for name in column_names(batch_columns)}, Usage().to_dict()
    usage = Usage()

    # ---- phase 1: the stage's own reading, with no access to A or B ------------------------------------------
    # Which columns it reads for itself. Under `contested` the agreed ones are left to phase 2, where the agents'
    # matching answer stands: the R3 numbers say the arbiter is already at 99.9% on cells the agents agree and are
    # right about, so reading those again risks more than it can win.
    scope = own_reading_scope()
    to_read = batch_columns
    skipped: List[str] = []
    if scope != "all":
        to_read = [c for c in batch_columns
                   if needs_own_reading(c.get("column_name", ""), source_a_data, source_b_data, scope)]
        skipped = [c.get("column_name", "") for c in batch_columns if c not in to_read]
    not_read = {
        name: {"value": "", "pages": [], "evidence": "", "looked_at": [],
               "reasoning": "the two extractions agreed; this stage did not read it", "skipped": True}
        for name in skipped
    }
    phase1 = None
    if to_read:
        if own_reading is not None and own_reading.columns == column_names(to_read):
            reading, phase1 = own_reading.session, own_reading.loop  # made while A and B were still answering
        else:
            reading, phase1 = _read_on_its_own(chat, doc_id, to_read, definitions_map, scale, specs)
        own = {**not_read, **reading.findings()}
        usage.add(phase1.usage).add(reading.tool_usage)
    else:  # every column in this batch was agreed: there is no reading pass to make
        own = not_read

    # ---- phase 2: A and B revealed, its own answer already committed -----------------------------------------
    session = _ReconcileSession(chat, doc_id, batch_columns, definitions_map, source_a_data, source_b_data, scale, own=own)
    if phase1 is not None:
        # a page phase 1 already had read carries its record forward: the citation is paid for once, and phase 2 sees
        # what that reading said rather than calling for it again
        session.checks.update(reading.checks)
        session.verifier_calls.extend(reading.verifier_calls)
    phase2 = run_tool_loop(
        chat,
        system=RECONCILE_PROMPT + shared_rules(columns=names, role="agent"),
        user=session.user_prompt(),
        tools=[
            Tool(specs["search_chunks"], session.search_chunks),
            Tool(specs["get_pages"], session.get_pages),
            Tool(specs["verify_attribution"], session.verify_attribution),
            Tool(submit_spec_v5(names), session.submit_verification),
        ],
        max_turns=AGENT_MAX_TURNS,
        max_tool_calls=AGENT_MAX_TOOL_CALLS,
        max_tokens=MAX_TOKENS["reconciliation"],
        follow_up=RECONCILE_FOLLOW_UP,
        is_done=session.done,
        finish_tool="submit_verification",
    )
    reason = f"reconciler did not submit ({phase2.stopped_by}{': ' + phase2.error if phase2.error else ''})"
    for name in names:
        if name not in session.submitted:
            session.submitted[name] = session.unsubmitted(name, reason)
    usage.add(phase2.usage).add(session.tool_usage)
    results = session.results()

    if log_path:
        write_json(
            log_path.with_name(log_path.stem + "_conversation.json"),
            {
                "doc_id": doc_id,
                "model": chat.key,
                "reconciler": RECONCILER_VERSION,
                "phase1": {
                    "scope": scope,
                    "read": column_names(to_read),
                    "not_read_because_agreed": skipped,
                    "stopped_by": phase1.stopped_by if phase1 else "not run",
                    "error": phase1.error if phase1 else None,
                    "findings": own,
                    "page_images": len(reading.images_attached) if phase1 else 0,
                    "pages_with_images": reading.images_attached if phase1 else [],
                    "tool_calls_sequence": [{"name": e["name"], "args": e["args"]}
                                            for e in (phase1.transcript if phase1 else []) if e["role"] == "tool"],
                    "conversation": phase1.transcript if phase1 else [],
                },
                "stopped_by": phase2.stopped_by,
                "error": phase2.error,
                "page_images": len(session.images_attached),
                "pages_with_images": session.images_attached,
                "verifier_calls": session.verifier_calls,
                "checks": list(session.checks.values()),
                "tool_calls_sequence": [{"name": e["name"], "args": e["args"]} for e in phase2.transcript if e["role"] == "tool"],
                "conversation": phase2.transcript,
                "calls": phase2.calls,
                "tool_usage": session.tool_usage.to_dict(),
                "results": results,
            },
        )
    return results, usage.to_dict()
````

### Phase 1 advertised tools
#### search_chunks
````json
{
  "name": "search_chunks",
  "description": "Semantic search over the paper's pages. Returns the best matching pages with full content. Pages you already have show 'already provided; check your context'.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Search query, e.g. 'median overall survival abiraterone months'"
      }
    },
    "required": [
      "query"
    ]
  }
}
````
#### get_pages
````json
{
  "name": "get_pages",
  "description": "Open pages by number: each page's full content and its rendered image. The image is the only way to read a value printed inside a figure or a table captured as a picture. At most 6 page images per batch; past that the text still comes back.",
  "parameters": {
    "type": "object",
    "properties": {
      "page_numbers": {
        "type": "array",
        "items": {
          "type": "integer"
        },
        "description": "1-based page numbers, e.g. [1, 2, 3]"
      }
    },
    "required": [
      "page_numbers"
    ]
  }
}
````
#### verify_attribution
````json
{
  "name": "verify_attribution",
  "description": "Check whether a page supports a value for a column (at most 30 claims per call). Returns verdict, page_value and evidence per claim.",
  "parameters": {
    "type": "object",
    "properties": {
      "claims": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "column": {
              "type": "string",
              "enum": [
                "{{column_name}}"
              ]
            },
            "value": {
              "type": "string"
            },
            "page": {
              "type": "integer"
            },
            "pages": {
              "type": "array",
              "items": {
                "type": "integer"
              },
              "description": "Instead of page: up to 3 pages when the value combines numbers from several"
            },
            "evidence": {
              "type": "string"
            }
          },
          "required": [
            "column",
            "value"
          ]
        }
      }
    },
    "required": [
      "claims"
    ]
  }
}
````
#### submit_findings
````json
{
  "name": "submit_findings",
  "description": "Your own answers for the columns, before you see anyone else's.",
  "parameters": {
    "type": "object",
    "properties": {
      "findings": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "column": {
              "type": "string",
              "enum": [
                "{{column_name}}"
              ]
            },
            "value": {
              "type": "string",
              "description": "the answer as the paper prints it, or \"\" when the paper states none"
            },
            "pages": {
              "type": "array",
              "items": {
                "type": "integer"
              },
              "description": "page(s) the answer is on"
            },
            "evidence": {
              "type": "string",
              "description": "the text or table cell the answer was taken from"
            },
            "looked_at": {
              "type": "array",
              "items": {
                "type": "integer"
              },
              "description": "pages you examined for this column"
            },
            "reasoning": {
              "type": "string"
            }
          },
          "required": [
            "column",
            "value",
            "reasoning"
          ]
        }
      }
    },
    "required": [
      "findings"
    ]
  }
}
````

### Phase 2 advertised tools
#### search_chunks
````json
{
  "name": "search_chunks",
  "description": "Semantic search over the paper's pages. Returns the best matching pages with full content. Pages you already have show 'already provided; check your context'.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Search query, e.g. 'median overall survival abiraterone months'"
      }
    },
    "required": [
      "query"
    ]
  }
}
````
#### get_pages
````json
{
  "name": "get_pages",
  "description": "Open pages by number: each page's full content and its rendered image. The image is the only way to read a value printed inside a figure or a table captured as a picture. At most 6 page images per batch; past that the text still comes back.",
  "parameters": {
    "type": "object",
    "properties": {
      "page_numbers": {
        "type": "array",
        "items": {
          "type": "integer"
        },
        "description": "1-based page numbers, e.g. [1, 2, 3]"
      }
    },
    "required": [
      "page_numbers"
    ]
  }
}
````
#### verify_attribution
````json
{
  "name": "verify_attribution",
  "description": "Check whether a page supports a value for a column (at most 30 claims per call). Returns verdict, page_value and evidence per claim.",
  "parameters": {
    "type": "object",
    "properties": {
      "claims": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "column": {
              "type": "string",
              "enum": [
                "{{column_name}}"
              ]
            },
            "value": {
              "type": "string"
            },
            "page": {
              "type": "integer"
            },
            "pages": {
              "type": "array",
              "items": {
                "type": "integer"
              },
              "description": "Instead of page: up to 3 pages when the value combines numbers from several"
            },
            "evidence": {
              "type": "string"
            }
          },
          "required": [
            "column",
            "value"
          ]
        }
      }
    },
    "required": [
      "claims"
    ]
  }
}
````
#### submit_verification
````json
{
  "name": "submit_verification",
  "description": "Submit final values for one or more columns. Every value must have been through verify_attribution, so the cell ships with the page and quote a reader can check it against. \"Not reported\" needs absence_basis when another answer stated a value. The response lists accepted and rejected columns.",
  "parameters": {
    "type": "object",
    "properties": {
      "results": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "column": {
              "type": "string",
              "enum": [
                "{{column_name}}"
              ]
            },
            "value": {
              "type": "string"
            },
            "reasoning": {
              "type": "string"
            },
            "verification": {
              "type": "string",
              "enum": [
                "A_correct_B_wrong",
                "B_correct_A_wrong",
                "both_correct",
                "both_wrong"
              ]
            },
            "source": {
              "type": "object",
              "properties": {
                "page": {
                  "type": "integer",
                  "description": "1-based page that shows the value"
                },
                "pages": {
                  "type": "array",
                  "items": {
                    "type": "integer"
                  },
                  "description": "Instead of page: the pages of a value verified on several pages"
                },
                "modality": {
                  "type": "string",
                  "enum": [
                    "text",
                    "table",
                    "figure"
                  ]
                },
                "evidence": {
                  "type": "string",
                  "description": "Supporting text as printed on that page"
                }
              }
            },
            "review": {
              "type": "boolean",
              "description": "true: send this value to a human reviewer (a partial value, or a value no page shows); a value the verifier rejected is not accepted even with review"
            },
            "review_reason": {
              "type": "string"
            },
            "absence_basis": {
              "type": "object",
              "description": "Required with \"Not reported\" when another answer stated a value: the reading that rules it out.",
              "properties": {
                "pages": {
                  "type": "array",
                  "items": {
                    "type": "integer"
                  },
                  "description": "pages you read"
                },
                "page_says": {
                  "type": "string",
                  "description": "what those pages state for this column instead"
                }
              }
            },
            "final_source": {
              "type": "string",
              "enum": [
                "own",
                "A",
                "B",
                "merged"
              ]
            },
            "own_verdict": {
              "type": "string",
              "enum": [
                "correct",
                "incomplete",
                "wrong",
                "no_answer"
              ]
            },
            "a_verdict": {
              "type": "string",
              "enum": [
                "correct",
                "incomplete",
                "wrong",
                "no_answer"
              ]
            },
            "b_verdict": {
              "type": "string",
              "enum": [
                "correct",
                "incomplete",
                "wrong",
                "no_answer"
              ]
            }
          },
          "required": [
            "column",
            "value",
            "reasoning",
            "verification"
          ]
        }
      }
    },
    "required": [
      "results"
    ]
  }
}
````

Legacy `ask_document` and `search_pages` specs are not passed to these v5 phase loops. Phase 1 gets `submit_findings`; phase 2 gets the v5-modified `submit_verification`.

## 5. Attribution Verifier

Source:
`src/evisearch/services/evidence_check.py`; caller: `src/evisearch/services/reconciliation.py`; orchestration: `src/evisearch/services/reconciliation_v5.py`

Defined in:
`SYSTEM_PROMPT`, `_claims_block`, `response_schema`, `_verify_pages`

Used by final runs:
Nested calls in Qwen `schema-mhspc-trials-20260919020503-v4`, Mistral `e1-mistral-small32-full-e-r1`, Gemma `e1-gemma4-31b-full-e-opt-r1`, and Llama `e1-llama4-scout-full-e-r1`. Uses the reconciliation model.

Prompt role:
Separate system/user call per page set. User message order is cited page text and optional image parts, then claims block; response schema is out-of-band.

System prompt (then agent-role `shared_rules(columns=[claim.column ...])`):
````text
You judge whether values extracted from a clinical trial paper are the CORRECT ANSWER for a table column, using
the page or pages they were taken from.

You get each page's parsed text and, when available, its image, then a list of claims. Each claim names a column with
its definition, a value someone extracted for it, and the evidence they quoted. Judge every claim only from these pages.

For each claim, work in this order:
1. Read the definition and name exactly what it asks for: the statistic (count, percentage, median, rate, hazard ratio,
   name, yes/no), the endpoint or characteristic, the population or subgroup, the arm, the timepoint and the unit.
2. page_value: find the correct answer on the pages YOURSELF, before you look at the claimed value, and copy it as
   printed. Include every part the column asks for that the pages state (a count and its percentage, each arm or
   trial the definition asks for). When the definition asks for an endpoint and the pages report it only under named
   variants (for example progression-free survival asked, and the pages give biochemical PFS and radiographic PFS, or
   PSA-PFS and clinical PFS), the answer is every variant with its label, joined with "; " (for example
   "bPFS 22.9 months; rPFS 23.5 months"). Write "" when the pages do not state an answer for this column.
3. evidence and modality for page_value, then reason, then the verdict comparing the claimed value with your answer:
  "supported": the claimed value IS the correct answer: the same statistic, endpoint, population or subgroup, arm,
    timepoint and unit as the definition asks, with the same numbers. Also supported:
    - the same number in another format or rounding;
    - the page's value converted to the unit the column asks for, when the conversion is right (for example years
      times 12 for months, within 0.1 after rounding);
    - the population, subgroup, arm or event named differently on the page (a synonym or abbreviation for the same
      thing);
    - a count made by adding printed counts of mutually exclusive subgroups into the population the column asks for,
      and, for an "N (%)" column, its percentage computed from that count and the printed arm size: find each number
      on the pages, in the right rows and columns, and redo the arithmetic yourself.
  "partial": the claimed value is right but incomplete: it gives only some of the parts of your answer (one of several
    endpoint variants, the count without the percentage, one of several arms or trials the definition asks for).
  "not_supported": anything else. A number printed on the pages that answers a DIFFERENT question is NOT supported:
    - another statistic (a median in months in a rate column; a hazard ratio, odds ratio or p value in a per-arm
      column);
    - another endpoint (for example time to castration resistance or time to PSA progression in a PFS column, when the
      paper does not call it progression-free survival);
    - another population or subgroup (a subgroup value given for the whole population, or the reverse), another arm,
      another timepoint;
    - a value computed from other numbers that the paper does not state (a rate from event counts, medians or curves;
      a number read off a curve; "Not reached" or "Not estimable" that the paper does not state for that population).
- reason: start with one tag in brackets, then one short sentence: [match], [incomplete], [statistic], [endpoint],
  [population], [arm], [timepoint], [unit] or [not on pages].

A number being printed on the pages is never enough: the claimed value must answer this column. When the parsed text
and the image disagree, trust the image. Do not use knowledge from outside these pages.
````

Claims block for one runtime claim (actual call emits one block per claim):
````text
CLAIMS:

---
id: c1
Column: {{claim.column}}
Definition: {{definitions[claim.column]}}
Claimed value: {{claim.value}}
Claimed evidence: {{claim.evidence}}

Return JSON: {"results": [{"id": ..., "page_value": ..., "evidence": ..., "modality": ..., "reason": ..., "verdict": ...}]}
````

### Verbatim claims-block builder
````python
def _claims_block(claims: Sequence[Claim], definitions: Dict[str, str]) -> str:
    lines = ["CLAIMS:"]
    for index, claim in enumerate(claims, 1):
        lines.append(
            f"\n---\nid: c{index}\nColumn: {claim.column}\nDefinition: {definitions.get(claim.column, '')}\n"
            f"Claimed value: {claim.value}\nClaimed evidence: {claim.evidence or '(none given)'}"
        )
    lines.append('\nReturn JSON: {"results": [{"id": ..., "page_value": ..., "evidence": ..., "modality": ..., "reason": ..., "verdict": ...}]}')
    return "\n".join(lines)
````

### Verbatim response-schema builder
````python
def response_schema(ids: List[str]) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    # answer first: the checker writes what the pages state, its evidence and reason before the verdict
                    "properties": {
                        "id": {"type": "string", "enum": ids},
                        "page_value": {"type": "string"},
                        "evidence": {"type": "string"},
                        "modality": {"type": "string", "enum": list(MODALITIES)},
                        "reason": {"type": "string"},
                        "verdict": {"type": "string", "enum": list(VERDICTS)},
                    },
                    "required": ["id", "page_value", "evidence", "modality", "reason", "verdict"],
                },
            }
        },
        "required": ["results"],
    }
````

### Verbatim message assembly
````python
def _verify_pages(
    chat: ChatModel, pages: Pages, texts: Dict[int, str], images: Dict[int, bytes], claims: Sequence[Claim], definitions: Dict[str, str]
) -> Tuple[List[Dict[str, Any]], Usage, Dict[str, Any]]:
    ids = [f"c{i}" for i in range(1, len(claims) + 1)]
    parts: List[Any] = []
    for page in pages:
        parts.append(TextPart(f"=== PAGE {page}: parsed text ===\n{texts.get(page) or '(no parsed text for this page)'}"))
        if images.get(page):
            parts += [TextPart(f"=== PAGE {page}: image ==="), ImagePart(images[page])]
    parts.append(TextPart(_claims_block(claims, definitions)))
    messages = [Message.system(SYSTEM_PROMPT + shared_rules(columns=[claim.column for claim in claims])), Message.user(*parts)]
    schema = response_schema(ids) if chat.capabilities.json_schema else None
    pages_text = "\n".join(texts.get(page, "") for page in pages)
    usage = Usage()
    call: Dict[str, Any] = {"page": pages[0], "pages": list(pages), "claims": len(claims), "image": all(images.get(p) for p in pages)}
    try:
        result = chat.chat(messages, response_schema=schema, max_tokens=MAX_TOKENS["verifier"])
        usage.add(result.usage)
        call.update(result.call_record(), finish_reason=result.finish_reason)
        if str(result.finish_reason).lower() == "length":
            raise ValueError("reply cut off at max_tokens")
        parsed = result.json()
    except (InferenceError, ValueError) as exc:
        call["error"] = str(exc)
        return [_record(c, "error", f"verifier call failed: {exc}", pages_text) for c in claims], usage, call
    by_id = {str(item.get("id")): item for item in (parsed or {}).get("results", []) if isinstance(item, dict)}
    records = []
    for claim_id, claim in zip(ids, claims):
        item = by_id.get(claim_id)
        if item is None:
            records.append(_record(claim, "error", "verifier returned no verdict for this claim", pages_text))
            continue
        verdict = str(item.get("verdict", "")).strip()
        found = {key: item.get(key) for key in ("page_value", "evidence", "modality")}
        records.append(_record(claim, verdict if verdict in VERDICTS else "error", str(item.get("reason", "")).strip(), pages_text, **found))
    return records, usage, call
````

### Constrained verifier response schema example
````json
{
  "type": "object",
  "properties": {
    "results": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": {
            "type": "string",
            "enum": [
              "c1"
            ]
          },
          "page_value": {
            "type": "string"
          },
          "evidence": {
            "type": "string"
          },
          "modality": {
            "type": "string",
            "enum": [
              "text",
              "table",
              "figure"
            ]
          },
          "reason": {
            "type": "string"
          },
          "verdict": {
            "type": "string",
            "enum": [
              "supported",
              "partial",
              "not_supported"
            ]
          }
        },
        "required": [
          "id",
          "page_value",
          "evidence",
          "modality",
          "reason",
          "verdict"
        ]
      }
    }
  },
  "required": [
    "results"
  ]
}
````

Verdict vocabulary is `supported`, `partial`, `not_supported`. Deterministic number/evidence-in-text checks are recorded after the response and do not decide the verdict.

## 6. Evaluation / Scoring Judge

Source:
`experiment-scripts/scoring/RUBRIC.md`; wrapper: `experiment-scripts/score_e1_vertex.py`; score constant: `src/evaluation/claude_scoring.py`

Defined in:
`RUBRIC.md` category sections and conventions C1-C7; `score_e1_vertex.prompt`, `response_schema`, `validate_payload`, `score_batch`

Used by final runs:
All four model families; Vertex `gemini-3.5-flash`, temperature `0.0`, scorer `vertex-gemini-3.5-flash-temp0-v1`, as recorded in the saved E1 scoring reports.

Prompt role:
One user message, no system message: complete rubric with trailing whitespace stripped, then a blinded JSON batch wrapper. A constrained response schema is sent out-of-band.

Complete user message template, including the literal rubric text from the source file. `{batch}` is serialized by `json.dumps(batch, indent=2, ensure_ascii=False)`:
````text
# Scoring rubric

The evaluator_v2 prompts, verbatim. Use the section matching the batch category.

## Category: exact_match

```text
You are evaluating clinical trial data extraction for exact match columns (identifiers, categorical values, binary fields).

Compare Ground Truth (GT) vs Predicted (Pred) values for exact/semantic equivalence.

CRITICAL - Empty / missing (use for BOTH GT and Pred):
- Treat as equivalent to empty: "", "Not reported", "not reported", "not found", "Not found", "not applicable", "N/A", "na", "NaN", "not present", "Not reported", "Not applicable"
- When GT is empty and Pred is any of the above → correctness=1.0, completeness=1.0 (no hallucination)
- When GT is empty and Pred has a substantive value → correctness=0.0, completeness=0.0 (hallucination)
- When GT has a value and Pred is empty → correctness=0.0, completeness=0.0 (missing extraction)

Rules:
- Case-insensitive comparison
- Accept synonyms: "Yes"/"Y", "No"/"N", "Full Pub"/"Full Publication", "Phase 3"/"3.0", etc.
- For identifiers (NCT, Trial Name): must match exactly (ignoring case/whitespace)
- Consider the column definition to understand acceptable variations

For each column, evaluate:
- If values are equivalent (by rules above): correctness=1.0, completeness=1.0
- If values differ: correctness=0.0, completeness=0.0

Columns to evaluate:

For each column, provide your evaluation with reasoning, then output the final scores.
Be thorough in your reasoning but concise.

IMPORTANT: When returning results, use the EXACT column name shown above (without the number prefix).
For example, if the column is shown as "1. Control Arm - N:", you must return column name as "Control Arm - N".
```

## Category: numeric_tolerance

```text
You are evaluating clinical trial data extraction for numeric columns.

Compare Ground Truth (GT) vs Predicted (Pred) values using the column Definition to determine what is required.

CRITICAL RULES:

1) **What's required vs optional (check the Definition)**:
   - If definition says "include count AND percentage" or "N (%)": both the count (N) and the percentage numeric value are REQUIRED. The literal "%" symbol is OPTIONAL—e.g. Pred="103 (55)" has both N and percentage value; do NOT dock for missing "%".
   - If definition says "at X years" or "X-year rate": the timepoint is REQUIRED.
   - If definition says "median" without specifying CI/IQR: only the median number is required; CI, IQR, SD, range, p-values are OPTIONAL context.
   - **Multiple values in GT**: If GT contains multiple distinct required values (e.g. "bPFS 12.9 mo; rPFS 15.3 mo" or "X and Y"), pred must include ALL of them for completeness=1.0. If pred reports only one value and GT has two or more, score correctness based on whether the reported value matches one in GT, but set completeness=0.5 (missing other required value(s)).
   - Default: primary numbers are required; statistical context (CI, IQR, p-values) is optional unless definition explicitly asks for it.

2) **Tolerance**: ±0.1 for absolute values, ±2% relative for percentages.

3) **Correctness** (use only 0.0, 0.5, or 1.0):
   - 1.0 = all predicted numbers match GT numbers (within tolerance); extra optional stats (IQR, CI) in pred are fine
   - 0.5 = some predicted numbers match, some don't or contradict GT
   - 0.0 = no predicted numbers match GT, or predicted numbers contradict GT

4) **Completeness** (use only 0.0, 0.5, or 1.0):
   - 1.0 = all REQUIRED numbers from GT are present in pred (check definition to determine what's required)
   - 0.5 = some required numbers present, some missing
   - 0.0 = required numbers missing

5) **Empty handling** (treat as empty for BOTH GT and Pred):
   - Empty-equivalent: "", "Not reported", "not found", "Not found", "not applicable", "N/A", "na", "NaN", "not present"
   - When GT is empty and Pred is empty-equivalent → correctness=1.0, completeness=1.0 (no hallucination)
   - When GT is empty and Pred has a number or substantive value → correctness=0.0, completeness=0.0 (hallucination)
   - When GT has a value and Pred is empty-equivalent → correctness=0.0, completeness=0.0 (missing extraction)
   - When both have substantive values: apply tolerance and required-vs-optional rules above

EXAMPLES:
- GT="35.1 months (95% CI 29.9–43.6)", Pred="35.1 months", Def="median OS" → correctness=1.0 (35.1 matches), completeness=1.0 (CI is optional)
- GT="70% at 5 years", Pred="70%", Def="OS rate at 5 years" → correctness=1.0 (70% matches), completeness=0.5 (missing required timepoint)
- GT="250 (63.6%)", Pred="250", Def="include count and percentage" → correctness=1.0 (N matches), completeness=0.5 (missing percentage value)
- GT="103 (55%)", Pred="103 (55)", Def="include count and percentage" → correctness=1.0, completeness=1.0 (both N and percentage value present; "%" symbol optional)
- GT="250 (63.6%)", Pred="250 (64%)", Def="include count and percentage" → correctness=1.0 (both within tolerance), completeness=1.0 (both present)
- GT="High-volume: 92 (48%) Low-volume: 100 (52%)", Pred="92 (48%)", Def="by volume subgroup" → correctness=1.0 (high-volume matches), completeness=0.5 (missing low-volume)
- GT="bPFS 12.9 mo; rPFS 15.3 mo", Pred="12.9 mo", Def="median PFS (months)" → correctness=1.0 (12.9 matches), completeness=0.5 (missing rPFS 15.3)
- GT="bPFS 12.9 mo; rPFS 15.3 mo", Pred="12.9 mo; 15.3 mo" or "bPFS 12.9; rPFS 15.3", Def="median PFS" → correctness=1.0, completeness=1.0 (both values present)

Columns to evaluate:

For each column, provide your evaluation with reasoning, then output the final scores.
Be thorough in your reasoning but concise.

IMPORTANT: When returning results, use the EXACT column name shown above (without the number prefix).
For example, if the column is shown as "1. Control Arm - N:", you must return column name as "Control Arm - N".
```

## Category: structured_text

```text
You are evaluating clinical trial data extraction for structured text columns (treatments, regimens, endpoints, classifications).

Compare Ground Truth (GT) vs Predicted (Pred) for semantic information content using the column Definition to determine what is required.

CRITICAL RULES:

1) **Correctness = no contradiction** (use only 0.0, 0.5, or 1.0):
   - 1.0 = all information in Pred matches or is compatible with GT; extra correct detail (e.g., drug mechanism, trial name, expanded abbreviations) is fine and does NOT lower the score
   - 0.5 = some predicted info correct, some contradicts GT (mixed: core fact right but extra detail wrong)
   - 0.0 = predicted info contradicts GT or core facts are wrong

2) **Completeness = required facts only** (use only 0.0, 0.5, or 1.0):
   - Use the Definition to identify what information is REQUIRED for this column
   - For "treatment regimen": drug name and combination partner (e.g., ADT) are typically required; dose and schedule are required if definition implies detail (e.g., "describe the regimen") but optional if definition just asks "what treatment"
   - For "endpoint": endpoint name is required; timepoints/thresholds are required only if definition specifies
   - Use medical judgment based on the definition to decide what's required vs optional context
   - 1.0 = all required facts from GT are present in pred
   - 0.5 = some required facts present, some missing
   - 0.0 = required facts missing
   - Do NOT penalize for missing optional context (mechanism, rationale, historical notes, expanded forms)

3) **Empty handling** (treat as empty for BOTH GT and Pred):
   - Empty-equivalent: "", "Not reported", "not found", "Not found", "not applicable", "N/A", "na", "not present"
   - When GT is empty and Pred is empty-equivalent → correctness=1.0, completeness=1.0 (no hallucination)
   - When GT is empty and Pred has substantive text → correctness=0.0, completeness=0.0 (hallucination)
   - When GT has a value and Pred is empty-equivalent → correctness=0.0, completeness=0.0 (missing extraction)

4) **General**:
   - Be lenient with abbreviations (e.g., "ADT" = "Androgen Deprivation Therapy")
   - Accept rephrasing if meaning is preserved

EXAMPLES:
- GT="ADT", Pred="ADT (LHRH agonist for testosterone suppression)", Def="control arm regimen" → correctness=1.0 (extra mechanism is fine, no contradiction), completeness=1.0 (core fact "ADT" present)
- GT="Docetaxel 75 mg/m² every 21 days + ADT", Pred="Docetaxel + ADT", Def="treatment regimen (describe)" → correctness=1.0 (no contradiction), completeness=0.5 (missing dose/schedule which are required by "describe")
- GT="ADT", Pred="ADT with docetaxel", Def="control arm" → correctness=0.5 (ADT correct but extra "docetaxel" contradicts), completeness=1.0 (ADT present)
- GT="Overall survival", Pred="Overall survival at 3 years", Def="primary endpoint" → correctness=1.0 (extra timepoint is fine), completeness=1.0 (endpoint name present)

Columns to evaluate:

For each column, provide your evaluation with reasoning, then output the final scores.
Be thorough in your reasoning but concise.

IMPORTANT: When returning results, use the EXACT column name shown above (without the number prefix).
For example, if the column is shown as "1. Control Arm - N:", you must return column name as "Control Arm - N".
```

## How to return scores

You are scoring one queued batch. Every item is one column of one paper: its definition, the gold value (GT) and a
predicted value (Pred). Apply the rules above for the batch's category to each item on its own. You do not know
which system produced a prediction, and you must not look anything up: judge only GT vs Pred under the rules.

## Conventions (fixed after the pilot; they resolve ambiguities in the rules above, for every system alike)

- C1. Tolerance, by kind of number. We are not looking for exact decimals, but counts are different populations when
  they differ:
  counts (N) must match exactly (654 vs 655: no match; 1305 vs 1306: no match);
  percentages match within ±1 percentage point or ±2% relative, whichever is more lenient (4.3 vs 4.2: match;
  23 vs 22.3: match; 19.3 vs 18.7: match; 63.5 vs 67.5: no match);
  other values (medians, durations, rates in the paper's units) match within ±0.1 or at a different rounding
  (35 vs 35.1: match; 41 vs 41.9: no match).
- C2. For completeness, a required GT number counts as present only if Pred has it within tolerance; a wrong number is
  not "present". A count that matches with a percentage outside tolerance is 0.5 / 0.5 when the Definition requires
  both (see C7 for "count and/or percentage"). When GT gives only a pooled total, per-arm numbers whose sum equals it do
  not make it present; Pred must state the total. When GT lists each arm's number and also their total, and the
  Definition asks for arms separately, the per-arm numbers are what is required and the total is optional.
- C3. "Not reached", "NR (not reached)" and "not estimable" are substantive values, not empty-equivalent: GT empty and
  Pred "Not reached" is 0/0; GT "Not reached" and Pred "Not reached" is 1/1. A bare "NR" in a median-survival or
  time-to-event column means "not reached" (so GT "NR" vs Pred "Not reported" is a miss, 0/0).
- C4. Structured text: a category label implied by the components Pred names counts as present (e.g. GT "Triplet:
  ARPI + ADT + docetaxel" vs Pred "darolutamide plus ADT and docetaxel" is complete).
- C5. Exact match: parenthetical context in GT is optional ("2 arms (A vs B)" vs "2" matches); "Surname et al" formats
  match when the definition allows them.
- C6. In the yes/no columns ("Quality of Life reported", "Reporting by prognostic groups - Y/N | ..."), "No"/"N" and an
  empty value mean the same thing (not reported): GT empty vs Pred "No" is 1/1; GT "No" vs Pred "Not reported" is 1/1;
  "Yes" vs "No" or vs empty is 0/0.
- C7. When the Definition says "count and/or percentage", either the count or the percentage alone is complete (GT
  "143 (36.4%)" vs Pred "143" is 1/1). If Pred gives both and only one is within tolerance, correctness is 0.5 and
  completeness 1.0 (GT "397 (100%)" vs Pred "390 (98.2%)" is 0.5 / 1.0). When it says "count and percentage" or
  "N (%)" without "or", both are required.
  A Definition that says "title or identifier" is satisfied by the identifier alone.

Write a JSON file with exactly this shape, one entry per item id in the batch:

    {"batch": "<batch name>",
     "results": [{"id": "<item id>", "correctness": 0.0 | 0.5 | 1.0, "completeness": 0.0 | 0.5 | 1.0,
                  "reason": "<one short sentence>"}]}

## Blinded batch

```json
{json.dumps(batch, indent=2, ensure_ascii=False)}
```
````

### Verbatim prompt wrapper
````python
def prompt(rubric: str, batch: Dict[str, Any]) -> str:
    """The checked-in rubric verbatim, followed only by the blinded batch it asks the judge to score."""
    return rubric.rstrip() + "\n\n## Blinded batch\n\n```json\n" + json.dumps(batch, indent=2, ensure_ascii=False) + "\n```\n"
````

### Verbatim response schema constructor
````python
def response_schema(batch: Dict[str, Any]) -> Dict[str, Any]:
    """The constrained response accepted by ``claude_scoring.ingest``."""
    return {
        "type": "object",
        "properties": {
            "batch": {"type": "string"},
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "correctness": {"type": "number"},
                        "completeness": {"type": "number"},
                        "reason": {"type": "string"},
                    },
                    "required": ["id", "correctness", "completeness", "reason"],
                },
            },
        },
        "required": ["batch", "results"],
    }
````

### Verbatim score and ID validator
````python
def validate_payload(payload: Any, batch: Dict[str, Any]) -> Dict[str, Any]:
    """Fail closed before the existing ingester appends anything to the label store."""
    parsed = BatchJudgment.model_validate(payload)
    result = parsed.model_dump()
    expected = [item["id"] for item in batch["items"]]
    returned = [item["id"] for item in result["results"]]
    if result["batch"] != batch["batch"]:
        raise ValueError(f"returned batch {result['batch']!r}, expected {batch['batch']!r}")
    if len(returned) != len(set(returned)):
        raise ValueError("response contains duplicate item ids")
    if set(returned) != set(expected):
        raise ValueError(f"response ids differ from queue (missing={set(expected) - set(returned)}, "
                         f"unknown={set(returned) - set(expected)})")
    allowed = set(claude_scoring.SCORES)
    for item in result["results"]:
        if item["correctness"] not in allowed or item["completeness"] not in allowed:
            raise ValueError(f"{item['id']}: score outside {sorted(allowed)}")
        item["reason"] = item["reason"].strip()
        if not item["reason"]:
            raise ValueError(f"{item['id']}: empty reason")
    result["results"].sort(key=lambda item: expected.index(item["id"]))
    return result
````

### Constrained judge response schema
````json
{
  "type": "object",
  "properties": {
    "batch": {
      "type": "string"
    },
    "results": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": {
            "type": "string"
          },
          "correctness": {
            "type": "number"
          },
          "completeness": {
            "type": "number"
          },
          "reason": {
            "type": "string"
          }
        },
        "required": [
          "id",
          "correctness",
          "completeness",
          "reason"
        ]
      }
    }
  },
  "required": [
    "batch",
    "results"
  ]
}
````

Correctness and completeness each must be exactly `{0.0, 0.5, 1.0}`; `claude_scoring.SCORES = (0.0, 0.5, 1.0)` and `score_e1_vertex.validate_payload` rejects other values. Empty/non-reported handling and C1-C7 special comparison rules are included verbatim in the rubric block. `src/evaluation/evaluator_v2.py` is not the fixed E1 request wrapper; the saved reports identify `score_e1_vertex.py` and the fixed Vertex judge.

## Frozen Shared-Note Insertions

Exact full agent-role rendering of snapshot `notes:853a2edb7ecd`. B1 uses all agent-role notes when its per-stage metadata says this fingerprint. Agent A/B, phase 2 and verifier filter family/column notes for their current columns; global notes always apply.
````text


EXTRACTION KNOWLEDGE BASE
These notes are how this table is read. Each note covers one topic and states its own scope and exceptions.
When a note written for specific columns differs from a general one, the specific note governs those columns.
### endpoints
# Endpoint identity

- An endpoint keeps its identity when the paper names a variant of it: biochemical, radiographic, clinical or PSA
  progression-free survival is progression-free survival, and a paper that reports only such variants reports that
  endpoint. Give each variant with its label (for example: bPFS X months; rPFS Y months). Never answer "Not
  reported" because the paper's name for the endpoint adds a qualifier.
- Time to PSA progression is the endpoint a TTPSA column asks for. When the paper reports time to castration-resistant
  prostate cancer and defines that event by a PSA rise (alone or together with clinical or radiographic progression),
  that is this endpoint: give the value with the paper's own label (for example: 22.0 months (time to CRPC, PSA rise
  or clinical progression)). Only a time to an event the paper defines without any PSA criterion is a different
  endpoint.

# A median that was not reached

- A median the paper reports as not reached or not estimable **for the column's own population and arm** is a value:
  write "Not reached" or "Not estimable", never "Not reported".
- A not-reached median printed for a different population, analysis set, subgroup or timepoint than the column asks
  for is not this cell's value. When that is the only place the paper shows it, the cell is "Not reported". In
  particular, do not carry a not-reached median out of a subgroup or secondary-population table into a column that
  names no subgroup.

### figures-and-panels
# Values printed inside figures

- A number printed inside a figure is reported by the paper. Read figure panels, forest plots, Kaplan-Meier plots and
  the tables embedded in or beside them exactly as you read a table, and cite the figure and panel as the source.
- Look in figures when a table does not answer the column, in particular for: survival or event-free rates at a stated
  timepoint annotated on a Kaplan-Meier curve or printed in its panel; per-subgroup counts and events in a forest
  plot ("events/N"); and regional, country or subgroup breakdowns given only as a figure.
- Read a rate or count that the panel prints as a number. Do not estimate a value by reading a position off a curve or
  a bar - that is not a printed value, and it is not an answer.
- When the parsed page text and the page image disagree about what a figure prints, trust the image.

### populations-and-subgroups
# Answering a subgroup column

- A subgroup column is answered from the paper's subgroup that corresponds to it, also when the paper names or defines
  it differently (for example a split by extent of disease under other labels or criteria): give the value with the
  paper's own label.
- When the paper splits patients by disease burden under a scheme other than the one specified in the column definition
  (for example LATITUDE high-risk/low-risk or extensive/minimal disease), accept that split as equivalent to the
  requested volume category and append the paper's specific label and criteria to the value (for example: 241 (48%)
  (high-risk, LATITUDE criteria)).

# Give every part the column covers

- When the paper reports the column's value for more than one population, trial or analysis set that the column's
  definition covers, the cell holds **all** of them, each with the paper's own label, joined with "; ". Giving only
  the headline population is an incomplete answer.
- This applies when a paper analyses a nested population alongside the overall one (for example an overall population
  and the subset that also received a backbone chemotherapy, both printed in the same baseline table), when a report
  covers more than one randomised comparison or sub-trial, and when the paper reports both a planned and an actual
  figure for the same characteristic.
- It does not license reporting a population the definition excludes, and it does not license combining the parts into
  one number: keep them separate and labelled.

### statistics-and-units
# Which statistic a cell holds, and in which unit

- The column name says which statistic goes in the cell: "Rate (%)" is a percentage of patients that the paper
  states (give its timepoint; do not compute a rate from counts or medians), "N (%)" is a count and/or its percentage
  (give whichever the paper reports if it gives only one), "(mo)" is a duration in months, "N" is a count. When the
  definition text asks for a different statistic than the column name, follow the column name. If the paper reports
  only a different kind of statistic for the column (for example only a hazard ratio or a median where a Rate (%) is
  asked), answer "Not reported".
- A per-arm column (its name contains "| Treatment" or "| Control") holds each arm's own value; when several arms fit,
  list each one. Never put a between-arm statistic (hazard ratio, odds ratio, difference, p-value) in a per-arm
  column.
- Report a value in the column's unit, converting when the paper uses another unit (for example a median in years
  for a column in months: give both, X years, Y months). A different unit is never a reason for "Not reported".
- Total-participant and arm-size counts are the numbers randomised into the population this paper reports on (for
  example only the metastatic patients when the paper analyses that cohort of a larger trial), not a safety,
  per-protocol or evaluable subset of it, unless the column asks for analysed patients.

### study-and-treatment-labels
# Is this report a follow-up?

- A paper is a follow-up when it reports further, updated, long-term, post hoc or secondary analyses of a trial whose
  primary results were published earlier (for example "as previously reported"), whatever its article type. A post
  hoc table inside the trial's primary report does not make that report a follow-up.

# What is the add-on treatment?

- The add-on treatment is the agent or agents added to the shared backbone in the experimental arm(s), not the
  backbone itself. When both arms receive a chemotherapy or hormonal backbone and only the experimental arm receives a
  further agent, that further agent is the add-on.

# Doublet and triplet labels

- Label the regimen "Doublet therapy (ADT + <added agent class>)" when it is ADT plus one additional agent class, and
  "Triplet therapy (ADT + docetaxel + AR inhibitor)" when it is ADT, docetaxel and an androgen-receptor inhibitor,
  even if the paper does not use these terms.
- Name the agent class from the agent the paper actually reports. Do not reuse a class name from this note or from a
  column definition as the answer.

### tables-and-subgroups
# Reading counts out of tables and subgroup analyses

- Counts of patients with a characteristic can come from a subgroup analysis: when the baseline table does not list
  the characteristic but a subgroup forest plot or table gives each subgroup's patients per arm (for example the N in
  "events/N"), that N is the number of patients with that characteristic in that arm; give it as the count.
- When the trial design gives a treatment to every patient in an arm (it defines the arm or is part of the arm's
  protocol regimen), that arm's count for the treatment is the arm size with 100%; when the arm's regimen excludes it,
  0 (0%). This covers treatments the protocol assigns, not patient characteristics or eligibility criteria.

### thresholds-and-arithmetic
# Thresholds

- Thresholds ("grade 3 or higher", ">=X"): use a total the paper prints for that threshold (a "grade >=3", "grade 3
  or worse" or "grade 3-5" row or sentence) as it is. When one table gives mutually exclusive worst-grade rows, add the
  rows at or above the threshold, grade 5 included. Never add a separately reported fatal or grade 5 count to a
  printed total.

# What may be added

- Add counts only. Never add rows for different event types, and never add or average rates, medians or durations.
- Show the arithmetic whenever a value is assembled from printed numbers, naming the row and column each number came
  from. A derived value whose ingredients cannot be pointed at on the page is not an answer.

### metastasis-and-presentation  [applies to the Mode of metastases, Metastases - N (%) columns]
# Synchronous and metachronous presentation

- Synchronous means the patient had metastatic disease at initial diagnosis (de novo). Metachronous means the
  metastases appeared later, after non-metastatic disease.
- Answer these columns **only** from the paper's own reporting of how the disease presented: the words de novo,
  synchronous or metachronous, or the stage at initial diagnosis (see the M1/M0 mapping below).
- Prior local therapy is **not** evidence of presentation, and the absence of prior local therapy is not evidence of
  de novo disease. A patient may receive prostatectomy or radiotherapy and still have presented with metastases, and
  many patients who presented de novo receive local therapy afterwards. Never derive a synchronous or metachronous
  count from prostatectomy, primary radiation or "no local therapy" rows.
- When the paper reports only prior local therapy, or reports presentation for no population the column covers, the
  cell is "Not reported". Do not add up subgroup rows to manufacture an arm total the paper does not print.

# Stage at initial diagnosis (count and rate columns only)

- In the count and rate columns for presentation, map "M1" or "distant metastasis at initial diagnosis" to the
  Synchronous columns and "M0" (with later metastasis) to the Metachronous columns; exclude "MX" or "not assessed"
  rows from the count and percentage, and keep the paper's original label in parentheses.
- This mapping does not apply to the yes/no reporting columns, which ask whether the paper reports outcomes for the
  subgroup - see the prognostic-group-reporting note.

# Metastatic sites

- Map M1 substages to specific site columns: assign M1a (non-regional lymph nodes only) to Nodal, M1b (bone, with or
  without nodes) to Bone, and M1c (visceral) to Liver. Set Lungs to "Not reported" if only M1 substages are provided.

### prognostic-group-reporting  [applies to the Reporting by prognostic groups - Y/N columns]
# Reporting by prognostic groups (yes/no)

- These columns ask one question: **does this paper report outcomes for the named subgroup?** Answer "Yes" when the
  paper gives outcomes for that subgroup in a table, a forest plot or the text; otherwise answer "No". Never answer
  "Not reported" - the absence of such reporting is itself the answer, and it is "No".
- Judge it on outcomes, not on baseline characteristics. A paper that merely counts how many patients fall in the
  subgroup, without reporting any outcome for them, is "No".
- Do not put a count, a percentage or a stage mapping in these cells. The M1/M0 mapping and the subgroup-equivalence
  rules apply to the count and rate columns, not here.
- A subgroup the paper reports under its own equivalent scheme counts as reported: if outcomes are given for
  high-risk/low-risk or extensive/minimal disease where the column asks for high/low volume, the answer is "Yes",
  and the paper's label may be added in parentheses.

### regions  [applies to the Region - N (%) columns]
# Region columns

- A continent column holds the **sum of the paper's own regional or country groups that belong to that continent**.
  When the paper reports countries or multi-country regions, add the ones that fall in the continent and give the
  total with its percentage of that arm (for example a paper reporting Canada 107 and USA 22 gives North America
  129). Name the groups you added in parentheses.
- Whatever the paper reports in a group that cannot be attributed to a single continent belongs to the paper's own
  other or rest-of-world category, under that category's exact label. Do not split such a group across continents and
  do not invent a share for it.
- When the paper reports a continent only inside a broader category of its own and gives no separable figure for it,
  write "Included in [Category Name]" using the paper's exact label.
- When every site of the trial lies in one country or one continent (stated in the methods, the site list or the
  affiliations), that continent holds the whole arm: give the arm size with 100%. The other continents are then that
  paper's unreported regions, not zeros - answer "Not reported" for them unless the paper prints a figure.
- When the region is not mentioned anywhere in the paper, write "Not reported".

### arm-level-values  [applies to: Median On-Treatment Duration (mo) | Treatment, Median On-Treatment Duration (mo) | Control, Median Age (years) | Treatment, Median Age (years) | Control, Median OS (mo) | Overall | Treatment, Median OS (mo) | Overall | Control, Median PFS (mo) | Overall | Treatment, Median PFS (mo) | Overall | Control]
# Arm-level medians

- Do not combine subgroup medians, and do not copy a subgroup median or a pooled across-arms median into an arm's
  cell. Medians cannot be added or averaged.
- But a median the paper states for this arm **anywhere** is a printed arm-level value and belongs in the cell,
  including when it appears in the text, the abstract, a supplementary table, a figure caption, or a discussion
  sentence comparing this trial with another. Look there before answering "Not reported".
- Only when the paper gives this arm's median nowhere - so that the sole route to it would be combining or borrowing
  from subgroups or from the other arm - is the cell "Not reported".
- This note governs medians only. It does not restrict counts and percentages, which may be assembled from printed
  mutually exclusive rows under the threshold and subgroup rules.
````

Exact full auditor-role rendering. Phase 1 further filters family-scoped notes for `read_names`:
````text


EXTRACTION KNOWLEDGE BASE
These notes are how this table is read. Each note covers one topic and states its own scope and exceptions.
When a note written for specific columns differs from a general one, the specific note governs those columns.
### endpoints
# Endpoint identity

- An endpoint keeps its identity when the paper names a variant of it: biochemical, radiographic, clinical or PSA
  progression-free survival is progression-free survival, and a paper that reports only such variants reports that
  endpoint. Give each variant with its label (for example: bPFS X months; rPFS Y months). Never answer "Not
  reported" because the paper's name for the endpoint adds a qualifier.
- Time to PSA progression is the endpoint a TTPSA column asks for. When the paper reports time to castration-resistant
  prostate cancer and defines that event by a PSA rise (alone or together with clinical or radiographic progression),
  that is this endpoint: give the value with the paper's own label (for example: 22.0 months (time to CRPC, PSA rise
  or clinical progression)). Only a time to an event the paper defines without any PSA criterion is a different
  endpoint.

# A median that was not reached

- A median the paper reports as not reached or not estimable **for the column's own population and arm** is a value:
  write "Not reached" or "Not estimable", never "Not reported".
- A not-reached median printed for a different population, analysis set, subgroup or timepoint than the column asks
  for is not this cell's value. When that is the only place the paper shows it, the cell is "Not reported". In
  particular, do not carry a not-reached median out of a subgroup or secondary-population table into a column that
  names no subgroup.

### populations-and-subgroups
# Answering a subgroup column

- A subgroup column is answered from the paper's subgroup that corresponds to it, also when the paper names or defines
  it differently (for example a split by extent of disease under other labels or criteria): give the value with the
  paper's own label.
- When the paper splits patients by disease burden under a scheme other than the one specified in the column definition
  (for example LATITUDE high-risk/low-risk or extensive/minimal disease), accept that split as equivalent to the
  requested volume category and append the paper's specific label and criteria to the value (for example: 241 (48%)
  (high-risk, LATITUDE criteria)).

# Give every part the column covers

- When the paper reports the column's value for more than one population, trial or analysis set that the column's
  definition covers, the cell holds **all** of them, each with the paper's own label, joined with "; ". Giving only
  the headline population is an incomplete answer.
- This applies when a paper analyses a nested population alongside the overall one (for example an overall population
  and the subset that also received a backbone chemotherapy, both printed in the same baseline table), when a report
  covers more than one randomised comparison or sub-trial, and when the paper reports both a planned and an actual
  figure for the same characteristic.
- It does not license reporting a population the definition excludes, and it does not license combining the parts into
  one number: keep them separate and labelled.

### statistics-and-units
# Which statistic a cell holds, and in which unit

- The column name says which statistic goes in the cell: "Rate (%)" is a percentage of patients that the paper
  states (give its timepoint; do not compute a rate from counts or medians), "N (%)" is a count and/or its percentage
  (give whichever the paper reports if it gives only one), "(mo)" is a duration in months, "N" is a count. When the
  definition text asks for a different statistic than the column name, follow the column name. If the paper reports
  only a different kind of statistic for the column (for example only a hazard ratio or a median where a Rate (%) is
  asked), answer "Not reported".
- A per-arm column (its name contains "| Treatment" or "| Control") holds each arm's own value; when several arms fit,
  list each one. Never put a between-arm statistic (hazard ratio, odds ratio, difference, p-value) in a per-arm
  column.
- Report a value in the column's unit, converting when the paper uses another unit (for example a median in years
  for a column in months: give both, X years, Y months). A different unit is never a reason for "Not reported".
- Total-participant and arm-size counts are the numbers randomised into the population this paper reports on (for
  example only the metastatic patients when the paper analyses that cohort of a larger trial), not a safety,
  per-protocol or evaluable subset of it, unless the column asks for analysed patients.

### study-and-treatment-labels
# Is this report a follow-up?

- A paper is a follow-up when it reports further, updated, long-term, post hoc or secondary analyses of a trial whose
  primary results were published earlier (for example "as previously reported"), whatever its article type. A post
  hoc table inside the trial's primary report does not make that report a follow-up.

# What is the add-on treatment?

- The add-on treatment is the agent or agents added to the shared backbone in the experimental arm(s), not the
  backbone itself. When both arms receive a chemotherapy or hormonal backbone and only the experimental arm receives a
  further agent, that further agent is the add-on.

# Doublet and triplet labels

- Label the regimen "Doublet therapy (ADT + <added agent class>)" when it is ADT plus one additional agent class, and
  "Triplet therapy (ADT + docetaxel + AR inhibitor)" when it is ADT, docetaxel and an androgen-receptor inhibitor,
  even if the paper does not use these terms.
- Name the agent class from the agent the paper actually reports. Do not reuse a class name from this note or from a
  column definition as the answer.

### metastasis-and-presentation  [applies to the Mode of metastases, Metastases - N (%) columns]
# Synchronous and metachronous presentation

- Synchronous means the patient had metastatic disease at initial diagnosis (de novo). Metachronous means the
  metastases appeared later, after non-metastatic disease.
- Answer these columns **only** from the paper's own reporting of how the disease presented: the words de novo,
  synchronous or metachronous, or the stage at initial diagnosis (see the M1/M0 mapping below).
- Prior local therapy is **not** evidence of presentation, and the absence of prior local therapy is not evidence of
  de novo disease. A patient may receive prostatectomy or radiotherapy and still have presented with metastases, and
  many patients who presented de novo receive local therapy afterwards. Never derive a synchronous or metachronous
  count from prostatectomy, primary radiation or "no local therapy" rows.
- When the paper reports only prior local therapy, or reports presentation for no population the column covers, the
  cell is "Not reported". Do not add up subgroup rows to manufacture an arm total the paper does not print.

# Stage at initial diagnosis (count and rate columns only)

- In the count and rate columns for presentation, map "M1" or "distant metastasis at initial diagnosis" to the
  Synchronous columns and "M0" (with later metastasis) to the Metachronous columns; exclude "MX" or "not assessed"
  rows from the count and percentage, and keep the paper's original label in parentheses.
- This mapping does not apply to the yes/no reporting columns, which ask whether the paper reports outcomes for the
  subgroup - see the prognostic-group-reporting note.

# Metastatic sites

- Map M1 substages to specific site columns: assign M1a (non-regional lymph nodes only) to Nodal, M1b (bone, with or
  without nodes) to Bone, and M1c (visceral) to Liver. Set Lungs to "Not reported" if only M1 substages are provided.

### prognostic-group-reporting  [applies to the Reporting by prognostic groups - Y/N columns]
# Reporting by prognostic groups (yes/no)

- These columns ask one question: **does this paper report outcomes for the named subgroup?** Answer "Yes" when the
  paper gives outcomes for that subgroup in a table, a forest plot or the text; otherwise answer "No". Never answer
  "Not reported" - the absence of such reporting is itself the answer, and it is "No".
- Judge it on outcomes, not on baseline characteristics. A paper that merely counts how many patients fall in the
  subgroup, without reporting any outcome for them, is "No".
- Do not put a count, a percentage or a stage mapping in these cells. The M1/M0 mapping and the subgroup-equivalence
  rules apply to the count and rate columns, not here.
- A subgroup the paper reports under its own equivalent scheme counts as reported: if outcomes are given for
  high-risk/low-risk or extensive/minimal disease where the column asks for high/low volume, the answer is "Yes",
  and the paper's label may be added in parentheses.

### regions  [applies to the Region - N (%) columns]
# Region columns

- A continent column holds the **sum of the paper's own regional or country groups that belong to that continent**.
  When the paper reports countries or multi-country regions, add the ones that fall in the continent and give the
  total with its percentage of that arm (for example a paper reporting Canada 107 and USA 22 gives North America
  129). Name the groups you added in parentheses.
- Whatever the paper reports in a group that cannot be attributed to a single continent belongs to the paper's own
  other or rest-of-world category, under that category's exact label. Do not split such a group across continents and
  do not invent a share for it.
- When the paper reports a continent only inside a broader category of its own and gives no separable figure for it,
  write "Included in [Category Name]" using the paper's exact label.
- When every site of the trial lies in one country or one continent (stated in the methods, the site list or the
  affiliations), that continent holds the whole arm: give the arm size with 100%. The other continents are then that
  paper's unreported regions, not zeros - answer "Not reported" for them unless the paper prints a figure.
- When the region is not mentioned anywhere in the paper, write "Not reported".
````

Exact Qwen B1 `RULES["v5"]` insertion:
````text


COLUMN AND TRIAL CONVENTIONS (apply to every column):
- The column name says which statistic goes in the cell: "Rate (%)" is a percentage of patients that the paper
  states (give its timepoint; do not compute a rate from counts or medians), "N (%)" is a count and/or its percentage
  (give whichever the paper reports if it gives only one), "(mo)" is a duration in months, "N" is a count. When the
  definition text asks for a different statistic than the column name, follow the column name. If the paper reports
  only a different kind of statistic for the column (for example only a hazard ratio or a median where a Rate (%) is
  asked), answer "Not reported".
- An endpoint keeps its identity when the paper names a variant of it: biochemical, radiographic, clinical or PSA
  progression-free survival is progression-free survival, and a paper that reports only such variants reports that
  endpoint. Give each variant with its label (for example: bPFS X months; rPFS Y months). Never answer "Not
  reported" because the paper's name for the endpoint adds a qualifier.
- A median that was not reached or is not estimable is a value: write "Not reached" or "Not estimable", never
  "Not reported".
- A per-arm column (its name contains "| Treatment" or "| Control") holds each arm's own value; when several arms fit,
  list each one. Never put a between-arm statistic (hazard ratio, odds ratio, difference, p-value) in a per-arm
  column.
- Counts of patients with a characteristic can come from a subgroup analysis: when the baseline table does not list
  the characteristic but a subgroup forest plot or table gives each subgroup's patients per arm (for example the N in
  "events/N"), that N is the number of patients with that characteristic in that arm; give it as the count.
- Report a value in the column's unit, converting when the paper uses another unit (for example a median in years
  for a column in months: give both, X years, Y months). A different unit is never a reason for "Not reported".
- A subgroup column is answered from the paper's subgroup that corresponds to it, also when the paper names or defines
  it differently (for example a split by extent of disease under other labels or criteria): give the value with the
  paper's own label.
- When the trial design gives a treatment to every patient in an arm (it defines the arm or is part of the arm's
  protocol regimen), that arm's count for the treatment is the arm size with 100%; when the arm's regimen excludes it,
  0 (0%). This covers treatments the protocol assigns, not patient characteristics or eligibility criteria.
- Total-participant and arm-size counts are the numbers randomised into the population this paper reports on (for
  example only the metastatic patients when the paper analyses that cohort of a larger trial), not a safety,
  per-protocol or evaluable subset of it, unless the column asks for analysed patients.
- Thresholds ("grade 3 or higher", ">=X"): use a total the paper prints for that threshold (a "grade >=3", "grade 3
  or worse" or "grade 3-5" row or sentence) as it is. When one table gives mutually exclusive worst-grade rows, add the
  rows at or above the threshold, grade 5 included. Never add a separately reported fatal or grade 5 count to a
  printed total.
- Add counts only. Never add rows for different event types, and never add or average rates, medians or durations.
- A paper is a follow-up when it reports further, updated, long-term, post hoc or secondary analyses of a trial whose
  primary results were published earlier (for example "as previously reported"), whatever its article type. A post
  hoc table inside the trial's primary report does not make that report a follow-up.
- The add-on treatment is the agent or agents added to the shared backbone in the experimental arm(s), not the
  backbone itself.
````

### Verbatim shared-rule selector
````python
def shared_rules(version: Optional[str] = None, columns: Optional[Iterable[str]] = None, role: str = "agent") -> str:
    """The knowledge text for a prompt.

    Every prompt receives all the notes its role may read, whatever the batch's columns (`columns` is accepted for the
    callers' convenience and does not narrow the notes). `role` decides which notes the caller may read: `agent` gets the definition and extraction notes, `auditor` gets
    the definition notes only, so the reconciliation stage's own reading pass does not inherit the agents' method and
    can disagree with them.
    """
    if version is None and knowledge_base_on():
        from src.evisearch.knowledge import notes

        return notes.render(notes.for_role(_kb_notes(), role))
    return RULES[version or rules_version()]
````

### Verbatim family/column selector
````python
def select_for(notes: Iterable[Note], columns: Optional[Iterable[str]] = None) -> List[Note]:
    """The notes a batch of columns receives. Global and table notes always; a family or column note only when the
    batch holds one of its columns, so a narrow rule never reaches a prompt for other columns (the failure Q10
    measured: an eligibility rule for the docetaxel columns was generalised to previous local therapy)."""
    notes = list(notes)
    if columns is None:
        return notes
    names = set(columns)

    def applies(note: Note) -> bool:
        if note.scope in ("global", "table"):
            return True
        if names & set(note.columns):
            return True
        return bool(note.families) and any(n.startswith(f) for f in note.families for n in names)

    return [n for n in notes if applies(n)]
````

### Verbatim note renderer
````python
def render(notes: Iterable[Note]) -> str:
    """The prompt text: the header, then each note's body verbatim under a scope label."""
    notes = list(notes)
    if not notes:
        return ""
    blocks = []
    for note in notes:
        where = ""
        if note.scope == "family" and note.families:
            where = f"  [applies to the {', '.join(note.families)} columns]"
        elif note.scope == "column" and note.columns:
            where = f"  [applies to: {', '.join(note.columns)}]"
        blocks.append(f"### {note.id}{where}\n{note.body}")
    return HEADER + "\n\n".join(blocks)
````

Frozen notes source: `new_pipeline_outputs/knowledge/note_snapshots/853a2edb7ecd.json`.

## Verification Notes

- Qwen A/B/E and all B1 response artifacts do not retain complete original request bodies. Mistral/Gemma/Llama A raw logs confirm per-paper image-mode variants; stage metadata confirms other prompt-rule settings.
- Top-level run headers and per-stage metadata conflict on the rule version for Mistral, Gemma, and Llama. Per-paper stage metadata consistently records the frozen notes fingerprint.
- All run headers are marked dirty. Missing historical request bytes cannot be reconstructed beyond the source-defined templates, schemas, fingerprints, and stage metadata.
