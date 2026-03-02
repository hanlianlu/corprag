# Scripts

## CLI Client

`cli.py` is a unified CLI client for the corprag REST API:

```bash
# Ingest documents
uv run scripts/cli.py ingest ./docs
uv run scripts/cli.py ingest ./docs --replace

# Query (retrieve contexts + sources, no LLM answer)
uv run scripts/cli.py query "What are the key findings?"

# Answer (single-shot LLM-generated answer + contexts + sources)
uv run scripts/cli.py answer "What are the key findings?"
uv run scripts/cli.py answer "Summarize the report" --mode mix --top-k 30

# Chat (interactive multi-turn conversation)
uv run scripts/cli.py chat
uv run scripts/cli.py chat --mode mix
```

Inside `chat`, type questions to ask follow-up questions with conversation context. Use `/clear` to reset history, `/quit` to exit.

Set `CORPRAG_API_URL` (default `http://localhost:8100`) and `CORPRAG_API_AUTH_TOKEN` as needed, or place them in `.env`.

## Known Limitations

### Image Extraction Quality

RAGAnything delegates document parsing to MinerU or Docling via subprocess CLI calls.
There is **no passthrough configuration** for image resolution or DPI at any level.

| Component | Default | Configurable? |
|-----------|---------|---------------|
| MinerU | 200 DPI, max 3500px | No — hardcoded in `pdf_reader.py` |
| Docling | `images_scale=1.0` (72 DPI) | Python API only — CLI and RAGAnything ignore it |
| PaddleOCR | `scale=2.0` | No — hardcoded in RAGAnything `parser.py` |

**Why it can't be configured today:**

- RAGAnything calls parsers via `subprocess.run(["mineru", ...])` / `subprocess.run(["docling", ...])`,
  not through their Python APIs.
- `parse_document(**kwargs)` accepts extra kwargs but they are silently dropped when
  building CLI commands — only `backend`, `lang`, `device`, `start_page`, `end_page`,
  `formula`, `table`, `source` are recognized.
- MinerU has no `--dpi` CLI flag, no env var, and `~/.mineru.json` does not support DPI settings.
- Docling CLI does not expose `--images-scale`.

**Potential workarounds (not yet implemented):**

1. Monkey-patch `mineru.utils.pdf_reader.page_to_image` to override the default DPI.
2. Submit a PR to RAGAnything to support `parser_extra_args` or use the Python API instead of CLI.
3. Fork RAGAnything and modify `_run_mineru_command()` / `_run_docling_command()` directly.
