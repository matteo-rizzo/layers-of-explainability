from __future__ import annotations

import re

import pandas as pd

from topic_extraction.classes.Document import Document


def metadata_extraction():
    df = pd.read_csv("data/TwinTransitionEstrazione7mar2023.csv")

    print(df[['Authors', 'Author(s) ID']].head())
    print(df.columns)

    years: list[str] = [y.strip() for y in df["Year"].astype(str).tolist()]
    non_years = [y for y in years if not (isinstance(y, str) and len(y) == 4)]
    print(non_years)

    non_keywords: list[str] = [s for s in df["Author Keywords"].tolist() if not (isinstance(s, str) and len(s) > 0)]
    print(non_keywords)

    years: list[int] = df["Year"].astype(int).tolist()
    print(f"Min/max year: {min(years), max(years)}")


def text_extraction() -> tuple[list[str], list[str], list[str], list[list[str]]]:
    df = pd.read_csv("data/TwinTransitionEstrazione7mar2023.csv", usecols=["Title", "Abstract", "Year", "Author Keywords"])
    df["all"] = df["Title"].str.cat(df["Abstract"], sep=". ")

    keywords: list[list[str]] = [[c.strip() for c in s.split(";")] for s in df["Author Keywords"].fillna("").tolist()]
    years: list[str] = [y.strip() for y in df["Year"].astype(str).tolist()]
    return df["all"].tolist(), df["Title"].tolist(), years, keywords


def extract_scopus_id(eid: str) -> str:
    sid = re.findall("^2-s2.0-(\\d+)$", eid)
    s = ""
    if sid:
        s = sid[0]
    return s


def extract_scopus_id_from_link(df: pd.DataFrame) -> list[str | None]:
    values = df.to_records(index=False).tolist()
    sids: list[str | None] = list()
    for eid, link in values:
        sid = re.findall("^2-s2.0-(\\d+)$", eid)
        if sid:
            sids.append(sid[0])
        else:
            sid = re.findall("eid=2-s2.0-(\\d+)", link)
            if sid:
                sids.append(sid[0])
            else:
                sids.append(None)
    return sids


def document_extraction(text_body: list[str], clean: bool = True) -> list[Document]:
    text_order: list[str] = list()
    for t in text_body:
        match t:
            case "a":
                j = "Abstract"
            case "t":
                j = "Title"
            case _:
                j = t
        text_order.append(j)
    text_body = text_order

    df = pd.read_csv("data/TwinTransitionEstrazione7mar2023.csv", usecols=["Title", "Abstract", "Year", "Author Keywords", "EID", "Link"])

    keywords: list[list[str]] = [[c.strip() for c in s.split(";")] for s in df["Author Keywords"].fillna("").tolist()]
    years: list[int] = [y for y in df["Year"].astype(int).tolist()]
    ids: list[str] = extract_scopus_id_from_link(df[["EID", "Link"]].fillna(""))  # [extract_scopus_id(s) for s in df[["EID", "Link"]].fillna("").tolist()]
    assert None not in ids, "There were None values in Scopus ids"

    df["all"] = df[text_body[0]]
    for t in text_body[1:]:
        if t == "k":
            k_series = pd.Series(["; ".join(ks) for ks in keywords])
            df["all"] = df["all"].str.cat(k_series, sep=". ")
        else:
            df["all"] = df["all"].str.cat(df[t], sep=". ")

    texts: list[str] = df["all"].tolist()
    if clean:
        texts = [re.sub(" +", " ", re.sub(r"[^A-Za-z\s.;]+", "", t)) for t in texts]

    docs = [Document(i, b, t, k, y) for i, b, t, y, k in zip(ids, texts, df["Title"].tolist(), years, keywords)]
    return docs


if __name__ == "__main__":
    metadata_extraction()
