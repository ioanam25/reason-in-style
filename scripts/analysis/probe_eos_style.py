#!/usr/bin/env python3
import os
'''Phase-0 probe: does the student stop correctly, and does the style prefix change generation?

Logs finish_reason, exact token counts, and whether the chat-template end token
(<|im_end|>) is emitted but not treated as a stop token.
'''
import argparse, json, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, os.environ.get("REPO_ROOT", str(Path(__file__).resolve().parents[1])))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--styles', nargs='+', default=['style_1', 'style_6'])
    ap.add_argument('--n-problems', type=int, default=50)
    ap.add_argument('--n-samples', type=int, default=16)
    ap.add_argument('--max-tokens', type=int, default=4096)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tag', default='probe')
    ap.add_argument('--eos-fix', action='store_true',
                    help='Stop on im_end and endoftext (production EOS_FIX)')
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    im_end = tok.convert_tokens_to_ids('<|im_end|>')
    endoftext = tok.convert_tokens_to_ids('<|endoftext|>')
    print(f'tokenizer eos={tok.eos_token!r} id={tok.eos_token_id} im_end={im_end} endoftext={endoftext}', flush=True)

    gc_path = Path(args.checkpoint) / 'generation_config.json'
    gc = json.loads(gc_path.read_text()) if gc_path.exists() else {}
    print('generation_config eos_token_id =', gc.get('eos_token_id'), flush=True)

    ds = load_dataset('HuggingFaceH4/MATH-500', split='test')
    probs = [ds[i] for i in range(min(args.n_problems, len(ds)))]

    prompts, meta = [], []
    for pi, p in enumerate(probs):
        for st in args.styles:
            msgs = [{'role': 'user', 'content': f'[{st}]\n' + p['problem']}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            prompts.append(text)
            meta.append((pi, st))

    llm = LLM(model=args.checkpoint, tensor_parallel_size=1,
              max_model_len=args.max_tokens + 1024, gpu_memory_utilization=0.85)

    sp_kw = dict(temperature=1.0, top_p=1.0, max_tokens=args.max_tokens, n=args.n_samples)
    if args.eos_fix:
        stop_ids = sorted({i for i in (im_end, endoftext, tok.eos_token_id) if i is not None and i >= 0})
        sp_kw['stop_token_ids'] = stop_ids
        print(f'EOS_FIX stop_token_ids={stop_ids}', flush=True)
    else:
        print('no extra stop tokens (production-like)', flush=True)
    sp = SamplingParams(**sp_kw)
    outs = llm.generate(prompts, sp)

    rows = []
    for (pi, st), out in zip(meta, outs):
        for comp in out.outputs:
            ids = list(comp.token_ids)
            rows.append({
                'problem': pi, 'style': st,
                'finish_reason': comp.finish_reason,
                'n_tokens': len(ids),
                'has_im_end': im_end in ids,
                'im_end_pos': ids.index(im_end) if im_end in ids else -1,
                'chars': len(tok.decode(ids, skip_special_tokens=True)),
            })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows))

    print()
    print('=' * 70)
    print(f'PROBE {args.tag}: {args.checkpoint}')
    print('=' * 70)
    n = len(rows)
    fr = {}
    for r in rows:
        fr[r['finish_reason']] = fr.get(r['finish_reason'], 0) + 1
    print('finish_reason:', {k: f'{100*v/n:.1f}%' for k, v in fr.items()})
    hie = sum(r['has_im_end'] for r in rows)
    print(f'emitted <|im_end|> mid-generation: {100*hie/n:.1f}%')
    wasted = [r for r in rows if r['has_im_end'] and r['im_end_pos'] >= 0]
    if wasted:
        w = np.array([r['n_tokens'] - r['im_end_pos'] for r in wasted])
        print(f'  tokens generated AFTER <|im_end|>: mean={w.mean():.0f} median={np.median(w):.0f}')
    for st in args.styles:
        sub = [r for r in rows if r['style'] == st]
        t = np.array([r['n_tokens'] for r in sub])
        c = np.array([r['chars'] for r in sub])
        stop = 100 * np.mean([r['finish_reason'] == 'stop' for r in sub])
        print(f'  {st:9s} n={len(sub):5d} tokens mean={t.mean():7.1f} median={np.median(t):7.1f} '
              f'chars mean={c.mean():8.1f} stop={stop:5.1f}%')
    ts = [np.array([r['n_tokens'] for r in rows if r['style'] == s]) for s in args.styles]
    if len(ts) == 2 and ts[0].mean() > 0:
        sep = 100 * abs(ts[1].mean() - ts[0].mean()) / ts[0].mean()
        print(f'  >> per-style length separation: {sep:.1f}%')

if __name__ == '__main__':
    main()
