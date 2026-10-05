ROMAN_URDU_MARKERS = [
    " aap ", " ap ", " kya ", " ki ", " ka ", " ke ", " hai ", " hain ",
    " nahi ", " rabta ", " baray ", " sawal ", " meherbani ", " hun ",
]


def needs_english_rewrite(text: str) -> bool:
    normalized = f" {text.lower()} "
    return any(marker in normalized for marker in ROMAN_URDU_MARKERS)


def enforce_english(llm, text: str) -> str:
    if not needs_english_rewrite(text):
        return text

    prompt = f"""Rewrite the response below in clear English only.
Do not include Urdu or Roman Urdu words.
Keep the same clinic facts, names, phone numbers, prices, and timings.

Response:
{text}

English response:"""

    return llm.invoke(prompt).content.strip()


def translate_query_to_english(llm, query: str) -> str:
    if not needs_english_rewrite(query):
        return query

    prompt = f"""Translate this patient query into clear English only.
Preserve the patient's intent and any medical terms.
Return only the translated query.

Patient query: {query}

English query:"""

    return llm.invoke(prompt).content.strip()
