---
name: AlphaXiv Paper Lookup
description: Look up any arxiv paper on alphaxiv.org to get a structured AI-generated overview. This is faster and more reliable than trying to read a raw PDF.
---

# AlphaXiv Paper Lookup

Look up any arxiv paper on alphaxiv.org to get a structured AI-generated overview. This is faster and more reliable than trying to read a raw PDF.

## When to Use

- User shares an arxiv URL (e.g. `arxiv.org/abs/2401.12345`)
- User mentions a paper ID (e.g. `2401.12345`)
- User asks you to explain, summarize, or analyze a research paper
- User shares an alphaxiv URL (e.g. `alphaxiv.org/overview/2401.12345`)

## Workflow

### Step 1: Extract the paper ID

Parse the paper ID from whatever the user provides:

| Input                                      | Paper ID       |
| ------------------------------------------ | -------------- |
| `https://arxiv.org/abs/2401.12345`         | `2401.12345`   |
| `https://arxiv.org/pdf/2401.12345`         | `2401.12345`   |
| `https://alphaxiv.org/overview/2401.12345` | `2401.12345`   |
| `2401.12345v2`                             | `2401.12345v2` |
| `2401.12345`                               | `2401.12345`   |

### Step 2: Resolve the paper

```bash
curl -s "https://api.alphaxiv.org/papers/v3/{PAPER_ID}"
```

Extract `versionId` from the JSON response. This is the UUID needed for the next call.

If this returns 404, the paper hasn't been indexed on alphaxiv yet.

### Step 3: Fetch the AI overview

```bash
curl -s "https://api.alphaxiv.org/papers/v3/{VERSION_ID}/overview/en"
```

The response contains:

- **`intermediateReport`** — the machine-readable report (structured text, best for LLM consumption)
- **`overview`** — the full markdown blog post (human-readable)
- **`summary`** — structured summary with fields: `summary`, `originalProblem`, `solution`, `keyInsights`, `results`
- **`citations`** — list of cited papers with titles and justifications

**Prefer `intermediateReport`** when available — it's specifically formatted for machine consumption. Fall back to `summary` fields if `intermediateReport` is null.

## Error Handling

- **404 on Step 2**: Paper not indexed. Tell the user it's not available on alphaxiv yet.
- **404 on Step 3**: Overview not generated for this paper.
- **`intermediateReport` is null**: Use `summary` and `overview` fields instead.

## Notes

- No authentication required — these are public endpoints.
- Replace `en` with a language code (`fr`, `de`, `es`, `zh`, `ja`, `ar`, `hi`, `pt`) for translated overviews.
