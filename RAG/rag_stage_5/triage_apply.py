#!/usr/bin/env python3
"""Golden-set triage apply — run LOCALLY (needs sections.json parent text).

Turns qa.validated.json (68 validator flags) into a cleaned qa_stage5_v3.json,
doing ONLY the safe, reversible, verifiable edits automatically and leaving every
judgement call for you to eyeball. Nothing is silently guessed:

  ADD_SIBLING  -> add a same-document sibling to the label ONLY IF that sibling's
                 text actually contains the answer (token overlap >= --ground-min).
                 Source-match alone is NOT trusted.
  DROP_UNGROUNDED -> answer not in the labeled parent -> auto-drop (hallucinated Q).
  DROP_GENERIC / GRAY / DROP_NEG -> NOT touched. Printed to triage_todo.md for you.

Run:
  python triage_apply.py --qa qa.validated.json --sections sections.json \
                         --out qa_stage5_v3.json --version stage5_v3
"""
import argparse, json, re, collections
from pathlib import Path

WORD = re.compile(r"[a-z0-9]+")
STOP = set("the a an of to in for on and or is are was were be been this that these "
           "those with as by at from it its what which who how why when where does do "
           "did can could would should will has have had not no than then into about "
           "company companys reported amount".split())
def toks(s): return {t for t in WORD.findall((s or "").lower()) if t not in STOP}

PID  = re.compile(r'[A-Za-z0-9][\w./-]*#\d+')
def doc(pid): return pid.rsplit('#',1)[0] if pid else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--qa', default='qa.validated.json')
    ap.add_argument('--sections', default='sections.json')
    ap.add_argument('--out', default='qa_stage5_v3.json')
    ap.add_argument('--version', default='stage5_v3')
    ap.add_argument('--ground-min', type=float, default=0.5,
                    help='min answer->sibling token overlap to trust an ADD_SIBLING')
    a = ap.parse_args()

    data = json.loads(Path(a.qa).read_text())
    secs = json.loads(Path(a.sections).read_text())
    ptext = {f"{s['source']}#{s['section_idx']}": s.get('text','') for s in secs}

    added=dropped=0
    todo=collections.defaultdict(list)   # bucket -> [(id, why)]
    kept=[]

    for q in data['questions']:
        notes = q.get('_validate', [])
        gold  = (q.get('relevant_parent_ids') or [None])[0]

        # --- auto-drop hallucinated factuals ---
        if any('low-groundedness' in n for n in notes):
            dropped += 1
            todo['DROPPED_UNGROUNDED'].append((q['id'], 'answer not in labeled parent'))
            continue

        # --- verified sibling add ---
        did_add=False
        for n in notes:
            if 'gold not rank-1' not in n and 'gold absent' not in n:
                continue
            ans = toks(q.get('_answer',''))
            if not ans:
                todo['GRAY_no_answer_field'].append((q['id'], q['query'][:80])); break
            cands=[c for c in PID.findall(n) if c!=gold and doc(c)==doc(gold)]
            good=[c for c in cands
                  if c in ptext and len(ans & toks(ptext[c]))/len(ans) >= a.ground_min]
            if good:
                cur=set(q.get('relevant_parent_ids') or [])
                q['relevant_parent_ids']=list(cur | set(good))
                added+=len(set(good)-cur); did_add=True
                todo['ADDED_SIBLING'].append((q['id'], f'+{good}'))
            else:
                # no verified same-doc sibling -> your call
                gen = re.search(r'the company|percentage increase|net income|total assets|'
                                r'cash flow|unrecognized tax|shares outstanding|dividend|revenue',
                                q['query'], re.I)
                todo['GRAY_generic?' if gen else 'GRAY_specific_miss'].append(
                    (q['id'], q['query'][:80]))
            break

        # negatives the validator thinks are answerable
        if any('possibly ANSWERABLE' in n for n in notes):
            todo['REVIEW_NEGATIVE'].append((q['id'], q['query'][:80]))

        q.pop('_validate', None)
        kept.append(q)

    data['version']=a.version
    data['questions']=kept
    Path(a.out).write_text(json.dumps(data, indent=2, ensure_ascii=False))

    with open('triage_todo.md','w') as f:
        f.write(f'# Triage TODO — hand-verify these ({a.version})\n\n')
        for b in ['GRAY_generic?','GRAY_specific_miss','GRAY_no_answer_field',
                  'REVIEW_NEGATIVE','ADDED_SIBLING','DROPPED_UNGROUNDED']:
            rows=todo.get(b,[])
            if not rows: continue
            f.write(f'\n## {b} ({len(rows)})\n\n')
            for qid,why in rows: f.write(f'- `{qid}`  {why}\n')

    print(f'auto-added {added} verified sibling labels, dropped {dropped} ungrounded')
    for b,rows in todo.items(): print(f'  {len(rows):>3}  {b}')
    print(f'\nwrote {a.out} ({len(kept)} Q, version {a.version}) + triage_todo.md')
    print('NEXT: read triage_todo.md — resolve GRAY_* (drop generic / keep miss) and '
          'REVIEW_NEGATIVE by hand, then re-run eval on qa_stage5_v3.json.')

if __name__=='__main__':
    main()
