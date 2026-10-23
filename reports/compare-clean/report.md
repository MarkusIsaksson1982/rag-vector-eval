# RAG evaluation report

| Question | Precision | Recall | MRR | Faithfulness | Correctness |
|---|---:|---:|---:|---:|---:|
| How heavy can my cabin bag be? | 0.33 | 1.00 | 0.50 | 1.00 | 0.07 |
| Hur mycket får handbagaget väga? | 0.33 | 1.00 | 1.00 | 1.00 | 0.43 |
| What is the checked bag limit? | 0.33 | 1.00 | 1.00 | 1.00 | 0.74 |
| Can I get a refund on a Basic fare? | 0.33 | 1.00 | 1.00 | 1.00 | 0.77 |
| Kan jag ändra min bokning? | 0.33 | 1.00 | 1.00 | 1.00 | 0.00 |
| When does online check-in open? | 0.33 | 1.00 | 1.00 | 1.00 | 0.62 |
| När stänger online-incheckningen? | 0.33 | 1.00 | 0.50 | 1.00 | 0.21 |
| Are standard seats free? | 0.33 | 1.00 | 1.00 | 1.00 | 1.00 |
| What should I do during a delay? | 0.33 | 1.00 | 1.00 | 1.00 | 0.62 |
| Får små husdjur resa i kabinen? | 0.33 | 1.00 | 1.00 | 1.00 | 0.00 |
| When is support chat available? | 0.33 | 1.00 | 1.00 | 1.00 | 0.80 |
| När stänger gaten? | 0.33 | 1.00 | 1.00 | 1.00 | 0.13 |
| Does Nordwind have Wi-Fi? | 0.33 | 1.00 | 1.00 | 1.00 | 0.62 |
| Can a refundable fare be refunded before departure? | 0.33 | 1.00 | 1.00 | 1.00 | 0.59 |
| How early does the airport counter open? | 0.33 | 1.00 | 1.00 | 1.00 | 1.00 |
| Är fönsterplats gratis? | 0.33 | 1.00 | 1.00 | 1.00 | 0.09 |
| How early must I add a pet? | 0.33 | 1.00 | 0.33 | 0.00 | 0.00 |
| What document is needed at boarding? | 0.33 | 1.00 | 1.00 | 1.00 | 0.89 |
| Can I change after the flight departs? | — | — | — | 1.00 | 0.00 |
| What is Nordwind's loyalty program name? | — | — | — | 1.00 | 0.00 |
| Vilken terminal använder flyget? | — | — | — | 1.00 | 0.14 |
| Can I bring a bicycle for free? | — | — | — | 1.00 | 0.00 |
| What is the captain's favourite colour? | — | — | — | 1.00 | 0.00 |
| How much is extra legroom? | — | — | — | 1.00 | 0.00 |

## Aggregate

| Metric | Value |
|---|---:|
| faithfulness | 0.958 |
| answer_relevancy | 0.478 |
| answer_correctness | 0.363 |
| abstention_accuracy | 0.708 |
| hallucination_on_unanswerable_rate | 0.250 |
| context_precision | 0.333 |
| context_recall | 1.000 |
| mrr | 0.907 |

## How to read this

Retrieval metrics are averaged only over answerable questions. For unanswerable questions, abstention accuracy measures whether the system correctly declines to answer, while hallucination-on-unanswerable rate measures the fraction that receive a non-abstaining answer (lower is better).
