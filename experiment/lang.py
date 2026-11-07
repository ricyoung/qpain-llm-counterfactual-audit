"""Prompts in Spanish, Portuguese and Japanese for the cross-lingual run.

The cases and prompt come from languages.json (draft translations; see its note).
The demographic line is built per language, with grammatical gender agreement in
Spanish and Portuguese. The answer schema stays in English, so analyze.py scores
every language with one parser.
"""
from __future__ import annotations

import json
from pathlib import Path

LANGS = ('es', 'pt', 'ja')
TABLE = json.loads((Path(__file__).parent / 'languages.json').read_text())


def patient_line(lang: str, race: str, sex: str, age: int) -> str:
    spec = TABLE[lang]
    if lang == 'ja':
        return spec['patient'].format(age=age, race=spec['race'][race], sex=spec['sex'][sex])
    adjective = (spec['race_f'] if sex == 'woman' else spec['race'])[race]
    return spec['patient'].format(article=spec['sex'][sex], race=adjective, age=age)


def build_prompt(lang: str, vignette: dict, race: str, sex: str, analgesics: list[str]) -> str:
    spec = TABLE[lang]
    return spec['prompt'].format(analgesics=analgesics,
                                 patient=patient_line(lang, race, sex, vignette['age']),
                                 case=spec['cases'][vignette['id']])
