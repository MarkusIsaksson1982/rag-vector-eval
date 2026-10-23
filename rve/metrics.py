import re
def toks(s): return set(re.findall(r"\w+",s.lower()))
def context_precision(hits, relevant, k=None):
    hs=hits[:k] if k else hits; return sum(h['id'] in relevant for h in hs)/len(hs) if hs else 0.0
def context_recall(hits,relevant): return len({h['id'] for h in hits}&set(relevant))/len(relevant) if relevant else 1.0
def mrr(hits,relevant):
    for i,h in enumerate(hits,1):
        if h['id'] in relevant:return 1/i
    return 0.0
def faithfulness(answer,hits,threshold=.5):
    ss=[s for s in re.split(r'(?<=[.!?])\s+',answer.strip()) if s]; ctx=toks(' '.join(h['text'] for h in hits))
    return sum(len(toks(s)&ctx)/len(toks(s)) >= threshold for s in ss)/len(ss) if ss else 1.0
def answer_relevancy(answer,question): return len(toks(answer)&toks(question))/len(toks(question)) if toks(question) else 0.0
def f1(a,b):
    x,y=toks(a),toks(b); p=len(x&y)/len(x) if x else 0; r=len(x&y)/len(y) if y else 0; return 2*p*r/(p+r) if p+r else 0.0
def abstention_accuracy(answer, unanswerable): return answer.lower().startswith(("i don't know","jag vet inte")) == unanswerable
def is_abstention(answer): return answer.lower().startswith(("i don't know","jag vet inte"))
def hallucination_on_unanswerable(answer, unanswerable):
    return float(unanswerable and not is_abstention(answer))
