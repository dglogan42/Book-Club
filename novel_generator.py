#!/usr/bin/env python3
"""
Generate a ~30,000-word Young Adult crime novel from a premise.

Pipeline (each stage is saved so a crashed run can resume):
  1. Story bible   - detective, victim, suspects, killer, secrets, clue ledger, timeline
  2. Outline       - one beat sheet per chapter, with clue plants and payoffs
  3. Chapters      - written one at a time with a rolling summary for continuity,
                     and re-expanded if a chapter comes in well under its word target
  4. Assembly      - novel.md plus a word-count report

Setup:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...

Usage:
    python novel_generator.py                      # default premise, 30,000 words
    python novel_generator.py --words 30000 --chapters 20 --out my_novel
    python novel_generator.py --premise "..." --setting "Auckland, New Zealand"
    python novel_generator.py --resume             # continue from saved state
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

try:
    import anthropic
except ImportError:
    sys.exit("Missing dependency. Run: pip install anthropic")

MODEL = "claude-sonnet-5-5"

DEFAULT_PREMISE = (
    "A single, socially awkward, autistic male detective sees the things everyone "
    "else misses, but misses the things that everyone else sees. He is brought in to "
    "investigate a school murder and is drawn into a web of intrigue as he uncovers "
    "the dirty secrets of teachers, students and parents."
)

# The craft rules of the crime-serial-detective skill, condensed into a system prompt.
CRAFT_RULES = """
You are a novelist writing an original Young Adult crime novel (readers aged 13-18).

THE DETECTIVE
- One lone, single adult male detective whose mind works differently (autistic-coded).
- He notices what others filter out (a scuff, a timestamp, a smell, the wording of a
  sentence, what is ABSENT) and misses what others read instantly (sarcasm, flirtation,
  performed vs. real grief, who lies out of shame rather than guilt).
- Specific beats superhuman: he reads evidence, never minds. Give him concrete habits of
  attention, routines and sensory needs, and one warm, uncomplicated relationship.
- Write him as a whole person: not robotic, not a punchline, not "cured" by romance.
  He is funny on purpose, rude by accident, loyal, and sometimes wrong. Neurotypical
  characters must not be merely cruel.

TWO-SIDED BLINDNESS (the central mechanic)
- Every case needs (1) a clue only he sees, planted fairly so a careful reader could too,
  and (2) an obvious social/emotional fact he misses, which sends him down a wrong path and
  costs him something concrete. The turning point is when another character (ideally a
  student) says the obvious thing out loud. His observation proves the solution; the human
  insight tells him where to look.

THE SCHOOL WEB
- Secrets in three circles (teachers, students, parents). Each secret is NOT the murder,
  makes its holder look guilty, and explains a clue. 3-5 genuine suspects, each with
  motive, opportunity and a lie.
- The killer is someone the story treated kindly (helper, witness, ally) with a small,
  human motive. No coincidences, hidden twins, or information the reader never had.

YA CARE
- Students are minors. Violence is restrained and mostly off the page. Never sexualise a
  child. Abuse, grooming or exploitation, if touched at all, is handled seriously at the
  level of consequence and investigation, with no explicit detail. No graphic gore.
- Teen voices are slippery, funny, coded and real. Keep the school specific: bells, noise,
  floor polish, laminated notices, group chats, timetables.

PROSE
- Close third person, alternating where the outline says so between the detective and a
  teenage student guide. Precise, literal, observational voice for the detective, with
  social cues arriving as raw data. Show the reader everything he sees; withhold only his
  interpretation. Dialogue carries subtext that the reader catches before he does.
- Grounded, humane, quietly sharp. Scenes, not summary. End every chapter on a turn.
- Everything is original. Do not reuse existing detectives, catchphrases or plots.
"""


# --------------------------------------------------------------------------- helpers

def make_client():
    return anthropic.Anthropic()


def call(client, system, user, max_tokens=8000, temperature=1.0, retries=4):
    """One model call with simple exponential-backoff retries. Returns text."""
    for attempt in range(retries):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return "".join(b.text for b in resp.content if b.type == "text").strip()
        except (anthropic.RateLimitError, anthropic.APIConnectionError,
                anthropic.InternalServerError) as e:
            wait = 2 ** attempt * 5
            print(f"  ! {type(e).__name__}; retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("Model call failed after retries")


def extract_json(text):
    """Pull the first JSON object/array out of a model reply."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    start = min([i for i in (text.find("{"), text.find("[")) if i != -1], default=-1)
    if start == -1:
        raise ValueError("No JSON found in model output")
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(text[start:])
    return obj


def word_count(text):
    return len(re.findall(r"\b[\w'’-]+\b", text))


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, (dict, list)):
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        path.write_text(data, encoding="utf-8")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# --------------------------------------------------------------------------- stage 1

def build_bible(client, premise, setting, title_hint):
    print("Stage 1/4: story bible")
    user = f"""Premise: {premise}
Setting: {setting}
Title idea (optional): {title_hint or 'invent a strong YA crime title'}

Design the story bible for a ~30,000-word YA crime novel. Reply with JSON only, with these keys:

title, logline,
detective: {{name, age, rank, why_single, perception_habits[3-4], blind_spots[3-4], routines_and_sensory_needs, warm_relationship, how_colleagues_treat_him}},
translator_colleague: {{name, role, how_they_help}},
teen_guide: {{name, age (15-17), year_level, voice, why_they_talk_to_the_detective, their_own_secret}},
school: {{name, place, feel, key_locations[5+], cctv_gap}},
victim: {{name, role, public_face, private_truth, time_and_place_of_death, cause (non-graphic)}},
suspects: [3-5 x {{name, role, circle (teacher/student/parent), secret_not_the_murder, motive, opportunity, lie, what_clue_it_explains}}],
killer: {{name, role, how_they_were_treated_kindly_in_story, small_human_motive, how_they_did_it, mistake_that_exposes_them}},
clue_only_he_sees: {{description, where_planted, why_others_miss_it, how_it_proves_the_solution}},
clue_he_misses_socially: {{the_obvious_social_fact, who_says_it_out_loud, what_his_miss_costs}},
red_herrings: [3+ x {{description, true_innocent_explanation}}],
clue_ledger: [8-12 x {{id, clue, planted_in_chapter_hint, meaning, payoff_hint}}],
timeline: [ordered list of dated/timed events from the day before the murder to the arrest],
personal_thread: {{what_continues_beyond_the_case}}

Make names ordinary, believable and distinct from each other (different first letters). Make every secret matter twice."""
    raw = call(client, CRAFT_RULES, user, max_tokens=8000)
    return extract_json(raw)


# --------------------------------------------------------------------------- stage 2

def build_outline(client, bible, n_chapters, words_per_chapter):
    print("Stage 2/4: chapter outline")
    user = f"""Here is the story bible:
{json.dumps(bible, indent=2)}

Write a chapter-by-chapter outline for exactly {n_chapters} chapters of about {words_per_chapter} words each.
Follow this arc, spreading the beats across the chapters in proportion:
1 The body and the room (open on what only he sees; hook)
2 The school as a suspect (he misses a social signal)
3 The students talk (the teen guide becomes his unlikely ally)
4 The first secret breaks (a reveal that is NOT the killer; the school closes ranks)
5 The wrong path (his blind spot costs something concrete)
6 Fall (he is sidelined or blamed; the person closest names the obvious thing)
7 Re-read (the two kinds of seeing meet)
8 The confrontation (quiet, evidence-driven, no stage monologue)
9 Aftermath (what was lost; personal thread; hook for the next case)

Alternate point of view between the detective and the teen guide where it suits the scene.
Reply with JSON only: a list of {n_chapters} objects with keys
number, title, pov (detective|teen_guide), beats[3-5 specific scene beats], clues_planted[clue ids], clues_paid_off[clue ids],
reveal_or_turn (the final-beat hook), word_target ({words_per_chapter}).
Every clue id in the bible's clue_ledger must be planted at least once and paid off by the final third."""
    raw = call(client, CRAFT_RULES, user, max_tokens=8000, temperature=0.8)
    outline = extract_json(raw)
    if isinstance(outline, dict):
        outline = outline.get("chapters") or outline.get("outline") or list(outline.values())[0]
    return outline


# --------------------------------------------------------------------------- stage 3

def summarise(client, chapter_text, number):
    """Short continuity summary: events, facts revealed, clues shown, emotional state."""
    user = f"""Summarise chapter {number} in at most 120 words for continuity. Include: what happens,
which facts or clues were shown to the reader (as the detective perceived them), what each
major character now believes, and any physical details that must stay consistent
(clothes, times, weather, locations). Plain text only.

CHAPTER:
{chapter_text}"""
    return call(client, "You write precise continuity notes for a novelist.", user,
                max_tokens=400, temperature=0.2)


def write_chapter(client, bible, outline, idx, summaries, words_per_chapter):
    ch = outline[idx]
    prev = "\n".join(f"Ch {i+1}: {s}" for i, s in enumerate(summaries)) or "(this is the first chapter)"
    nxt = outline[idx + 1] if idx + 1 < len(outline) else None
    user = f"""STORY BIBLE (canon, keep every name, age, time and detail consistent):
{json.dumps(bible, indent=2)}

PREVIOUS CHAPTERS (continuity notes):
{prev}

NOW WRITE CHAPTER {ch['number']} of {len(outline)}: "{ch['title']}"
POV: {ch['pov']}
Beats: {json.dumps(ch['beats'])}
Clues to plant (show them fairly as concrete sensory detail, do not interpret them): {ch.get('clues_planted')}
Clues to pay off: {ch.get('clues_paid_off')}
Final-beat turn: {ch['reveal_or_turn']}
{"Next chapter begins with: " + nxt['title'] if nxt else "This is the final chapter: close the case and the personal thread, leave a small hook."}

Requirements:
- About {words_per_chapter} words (not fewer than {int(words_per_chapter*0.9)}).
- Full scenes with dialogue, setting detail and interiority; no summary-only passages, no outline-style writing.
- Begin with the chapter heading as: ## Chapter {ch['number']}: {ch['title']}
- Output only the chapter text."""
    text = call(client, CRAFT_RULES, user, max_tokens=7000)
    return text


def expand_if_short(client, chapter_text, target, bible, ch):
    """If a chapter is well under target, ask for a fuller rewrite of the same chapter."""
    wc = word_count(chapter_text)
    if wc >= target * 0.85:
        return chapter_text
    print(f"    chapter is {wc} words (target {target}); expanding")
    user = f"""This chapter came in at {wc} words; it needs about {target}. Rewrite it in full, keeping every
plot event, clue and the final turn, but deepening scenes: more dialogue with subtext, sensory detail
as the point-of-view character perceives it, interiority, and the school's texture. Do not add new plot
facts that contradict the bible. Output only the full revised chapter, beginning with its ## heading.

BIBLE (names and facts):
{json.dumps({k: bible[k] for k in ('detective', 'teen_guide', 'school', 'victim') if k in bible}, indent=2)}

CHAPTER:
{chapter_text}"""
    revised = call(client, CRAFT_RULES, user, max_tokens=8000)
    return revised if word_count(revised) > wc else chapter_text


# --------------------------------------------------------------------------- stage 4

def consistency_report(client, bible, chapters):
    """Optional lightweight audit: clue plants/payoffs and timeline slips."""
    print("  consistency audit")
    digest = "\n\n".join(f"[Ch {i+1}] {c[:1800]}" for i, c in enumerate(chapters))
    user = f"""Audit this draft against the bible. List (a) any clue-ledger item never shown or never paid off,
(b) timeline or alibi contradictions, (c) name/age/detail inconsistencies, (d) anything that breaks YA care
(explicit violence or sexualised minors). Be specific and brief; if all is fine, say so.

BIBLE:
{json.dumps(bible)}

CHAPTER OPENINGS:
{digest}"""
    return call(client, "You are a meticulous mystery editor.", user, max_tokens=1500, temperature=0.2)


def assemble(out_dir, bible, chapters):
    title = bible.get("title", "Untitled")
    parts = [f"# {title}\n", f"*{bible.get('logline', '')}*\n"]
    parts += chapters
    novel = "\n\n".join(parts)
    save(out_dir / "novel.md", novel)
    return novel


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Generate a ~30,000-word YA crime novel.")
    ap.add_argument("--premise", default=DEFAULT_PREMISE)
    ap.add_argument("--setting", default="a mid-sized secondary school in a coastal New Zealand town")
    ap.add_argument("--title", default="", help="optional title idea")
    ap.add_argument("--words", type=int, default=30000, help="target total words")
    ap.add_argument("--chapters", type=int, default=20, help="number of chapters")
    ap.add_argument("--out", default="novel_out", help="output directory")
    ap.add_argument("--resume", action="store_true", help="reuse saved bible/outline/chapters")
    ap.add_argument("--audit", action="store_true", help="write a consistency report at the end")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wpc = args.words // args.chapters
    client = make_client()

    bible = load_json(out / "bible.json") if args.resume else None
    if not bible:
        bible = build_bible(client, args.premise, args.setting, args.title)
        save(out / "bible.json", bible)
    print(f"  title: {bible.get('title')}")

    outline = load_json(out / "outline.json") if args.resume else None
    if not outline:
        outline = build_outline(client, bible, args.chapters, wpc)
        save(out / "outline.json", outline)
    if len(outline) != args.chapters:
        print(f"  note: outline has {len(outline)} chapters (asked for {args.chapters})")

    print("Stage 3/4: writing chapters")
    chapters, summaries = [], []
    for i in range(len(outline)):
        ch_path = out / "chapters" / f"chapter_{i+1:02d}.md"
        sum_path = out / "chapters" / f"chapter_{i+1:02d}.summary.txt"
        if args.resume and ch_path.exists() and sum_path.exists():
            chapters.append(ch_path.read_text(encoding="utf-8"))
            summaries.append(sum_path.read_text(encoding="utf-8"))
            print(f"  chapter {i+1}: loaded ({word_count(chapters[-1])} words)")
            continue
        print(f"  chapter {i+1}/{len(outline)}: {outline[i]['title']}")
        text = write_chapter(client, bible, outline, i, summaries, wpc)
        text = expand_if_short(client, text, wpc, bible, outline[i])
        summary = summarise(client, text, i + 1)
        save(ch_path, text)
        save(sum_path, summary)
        chapters.append(text)
        summaries.append(summary)
        print(f"    {word_count(text)} words (running total {sum(word_count(c) for c in chapters)})")

    print("Stage 4/4: assembling")
    novel = assemble(out, bible, chapters)
    total = word_count(novel)
    report = [f"Title: {bible.get('title')}", f"Chapters: {len(chapters)}",
              f"Total words: {total}", f"Target: {args.words}"]
    if args.audit:
        report += ["", "Consistency audit:", consistency_report(client, bible, chapters)]
    save(out / "report.txt", "\n".join(report))
    print("\n".join(report))
    print(f"\nDone. Novel saved to {out / 'novel.md'}")


if __name__ == "__main__":
    main()
