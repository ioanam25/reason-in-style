"""Load unique SCAS validation problems from SFT parquet for pass@k eval."""

from __future__ import annotations

import json
from pathlib import Path


def load_scas_val_problems(parquet_path: str | Path, max_problems: int = 0) -> list[dict]:
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    seen: set[str] = set()
    problems: list[dict] = []
    for _, row in df.iterrows():
        qid = f"{row['source_dataset']}::{row['question_id']}"
        if qid in seen:
            continue
        seen.add(qid)
        msgs = json.loads(row["messages"])
        question = str(msgs[0]["content"])
        problems.append(
            {
                "question_id": qid,
                "question": question,
                "answer": str(row["answer"]),
                "subject": str(row.get("source_dataset", "")),
                "level": None,
            }
        )
    if max_problems and max_problems > 0:
        problems = problems[:max_problems]
    return problems
