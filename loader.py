import json
from pathlib import Path
from collections import namedtuple

Document = namedtuple("Document", ["page_content", "metadata"])

def load_clinic_docs(data_path: str = None):
    candidates = []
    if data_path:
        candidates.append(Path(data_path))

    candidates += [
        Path("data/info.JSON"),
        Path("data/info.json"),
        Path("data/clinic_info.json"),
        Path("data/clinic_info.JSON"),
    ]

    p = None
    for c in candidates:
        if c.exists():
            p = c
            break

    if p is None:
        raise FileNotFoundError(
            "Could not find any of the expected data files:\n" +
            "\n".join(str(c) for c in candidates)
        )

    raw = json.loads(p.read_text(encoding="utf-8"))
    docs = []

    for section, mapping in raw.items():
        for key, text in mapping.items():
            content = f"{section} | {key}: {text}"
            docs.append(Document(
                page_content=content,
                metadata={"section": section, "key": key}
            ))

    return docs
