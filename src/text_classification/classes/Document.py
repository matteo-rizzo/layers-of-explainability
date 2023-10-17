from __future__ import annotations


class Document:
    def __init__(self, pid: str, body: str | None = None, title: str | None = None, keywords: list = list(), timestamp: int | None = None):
        self.id: str = pid  # Scopus ID
        self.body: str | None = body
        self.title: str | None = title
        self.keywords: list = keywords
        self.timestamp: int | None = timestamp
