from dlightrag.engine.answer.citations.sources import SourceDownloadLinkBuilder


def test_rest_source_link_uses_document_id_and_workspace() -> None:
    builder = SourceDownloadLinkBuilder()

    url = builder.resolve("doc-abc", workspace="finance")

    assert url == "/files/raw/doc-abc?workspace=finance"


def test_web_source_link_uses_web_owned_prefix() -> None:
    builder = SourceDownloadLinkBuilder(base_url="/web/api/files/raw")

    url = builder.resolve("doc-abc", workspace="finance")

    assert url == "/web/api/files/raw/doc-abc?workspace=finance"


def test_source_link_percent_encodes_identifiers() -> None:
    builder = SourceDownloadLinkBuilder()

    url = builder.resolve("doc 1", workspace="research & design")

    assert url == "/files/raw/doc%201?workspace=research%20%26%20design"


def test_source_link_requires_document_and_workspace() -> None:
    builder = SourceDownloadLinkBuilder()

    assert builder.resolve("", workspace="default") is None
    assert builder.resolve("doc-abc", workspace="") is None
