import re, random
def generate(question, hits, hallucination=0.0, threshold=0.1, seed=1):
    if not hits or 1-hits[0]['distance'] < threshold: return "I don't know."
    words=set(re.findall(r"\w+",question.lower())); sentences=[]
    for h in hits:
        for s in re.split(r'(?<=[.!?])\s+',h['text']):
            if len(words & set(re.findall(r"\w+",s.lower()))) >= 2: sentences.append(s)
    answer=' '.join(sentences[:2]) or hits[0]['text']
    if random.Random(seed).random() < hallucination: answer += ' Nordwind Air guarantees a free hotel for every delay.'
    return answer
