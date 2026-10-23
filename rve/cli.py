import argparse,json,os,statistics,sys
from .store import build_store,query
from .generate import generate
from .metrics import *
def run(args):
    docs=json.load(open(os.path.join('data','docs.json'))); data=json.load(open(os.path.join('data','eval.json'))); col=build_store(docs)
    rows=[]
    for q in data:
        hits=query(col,q['question'],args.k); ans=generate(q['question'],hits,args.hallucination)
        unanswerable=not q['relevant_doc_ids']
        rows.append({'question':q['question'],'answer':ans,'hits':hits,'ground_truth':q['ground_truth'],'context_precision':None if unanswerable else context_precision(hits,q['relevant_doc_ids'],args.k),'context_recall':None if unanswerable else context_recall(hits,q['relevant_doc_ids']),'mrr':None if unanswerable else mrr(hits,q['relevant_doc_ids']),'faithfulness':faithfulness(ans,hits),'answer_relevancy':answer_relevancy(ans,q['question']),'answer_correctness':f1(ans,q['ground_truth']),'abstention_accuracy':float(abstention_accuracy(ans,unanswerable)),'hallucination_on_unanswerable_rate':hallucination_on_unanswerable(ans,unanswerable)})
    answerable=[r for r in rows if r['context_precision'] is not None]
    agg={k:statistics.mean(r[k] for r in rows) for k in ('faithfulness','answer_relevancy','answer_correctness','abstention_accuracy','hallucination_on_unanswerable_rate')}
    agg.update({k:statistics.mean(r[k] for r in answerable) for k in ('context_precision','context_recall','mrr')})
    os.makedirs(args.out,exist_ok=True)
    with open(os.path.join(args.out,'results.jsonl'),'w') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    with open(os.path.join(args.out,'report.md'),'w') as f:
        f.write('# RAG evaluation report\n\n| Question | Precision | Recall | MRR | Faithfulness | Correctness |\n|---|---:|---:|---:|---:|---:|\n')
        def display(value): return '—' if value is None else '%.2f' % value
        for r in rows:f.write('| '+r['question'].replace('|','/')+' | %s | %s | %s | %s | %s |\n'%(display(r['context_precision']),display(r['context_recall']),display(r['mrr']),display(r['faithfulness']),display(r['answer_correctness'])))
        f.write('\n## Aggregate\n\n| Metric | Value |\n|---|---:|\n')
        for key,value in agg.items(): f.write('| '+key+' | %.3f |\n' % value)
        f.write('\n## How to read this\n\nRetrieval metrics are averaged only over answerable questions. For unanswerable questions, abstention accuracy measures whether the system correctly declines to answer, while hallucination-on-unanswerable rate measures the fraction that receive a non-abstaining answer (lower is better).\n')
    if args.baseline:
        base=json.load(open(args.baseline));
        if any(base[k]-agg[k]>args.max_drop for k in ('faithfulness','context_recall')): return 3
    return 0
def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='cmd',required=True); r=sub.add_parser('run'); r.add_argument('--k',type=int,default=3); r.add_argument('--hallucination',type=float,default=0); r.add_argument('--out',default='reports/clean'); r.add_argument('--baseline'); r.add_argument('--max-drop',type=float,default=.05); r.add_argument('--compare',action='store_true'); c=sub.add_parser('compare'); c.add_argument('--k',type=int,default=3)
    a=p.parse_args()
    if a.cmd=='compare' or getattr(a,'compare',False):
        a.baseline=None; a.max_drop=.05
        for h in (0,.3):
            a.hallucination=h;a.out='reports/compare-clean' if not h else 'reports/compare-hallucination';run(a)
        def aggregate_from_report(path):
            lines=open(path).read().split('## Aggregate\n\n',1)[1].split('## How',1)[0].strip().splitlines()[2:]
            return {p.split('|')[1].strip():p.split('|')[2].strip() for p in lines if p.strip()}
        left=aggregate_from_report('reports/compare-clean/report.md'); right=aggregate_from_report('reports/compare-hallucination/report.md')
        with open('reports/compare.md','w') as f:
            f.write('# Clean vs hallucination (0.3)\n\n| Metric | Clean | Hallucination 0.3 |\n|---|---:|---:|\n')
            for key in left: f.write('| %s | %s | %s |\n' % (key,left[key],right[key]))
        print('Compare complete: wrote reports/compare.md.'); return
    raise SystemExit(run(a))
if __name__=='__main__':main()
