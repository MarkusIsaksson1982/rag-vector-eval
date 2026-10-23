from typing import Protocol
class Metric(Protocol):
    def evaluate(self, item): ...
class RagasAdapter:
    def __init__(self):
        try: import ragas
        except ImportError as e: raise ImportError("RagasAdapter requires optional package 'ragas'; install ragas separately to use it") from e
    @staticmethod
    def sample(item): return {'question':item['question'],'answer':item['answer'],'contexts':[h['text'] for h in item['hits']],'ground_truth':item['ground_truth']}
class DeepEvalAdapter:
    def __init__(self):
        try: import deepeval
        except ImportError as e: raise ImportError("DeepEvalAdapter requires optional package 'deepeval'; install deepeval separately to use it") from e
    @staticmethod
    def sample(item): return {'input':item['question'],'actual_output':item['answer'],'retrieval_context':[h['text'] for h in item['hits']],'expected_output':item['ground_truth']}
