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
