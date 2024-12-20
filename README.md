# MultiSynDS

From [1] diverse possibilites are proposed for **automatic evaluation**:

1. **Text Quality**
   1. **Word-overlap** based like ROUGE or BLEU. However, as we dont have a gold standard, **these metrics don't make much sense.**
   2. **Embedding-based** metrics like BERTScore, BLEURT or QuestEval. **These can be explored.**
2. **Medical concept correctness**
   1. **Negation Correctness**
      1. Get concepts coincident in source and output and then see how many of them were negated or not negated in both of the documents.
   2. **Concept Coverage**
      1. NER+L: Several studies have utilized concept correctness measures, such as F1score, precision, recall, and false positives, at various levels of granularity, including the report level and section level.
   3. **Fact Extraction**: The Fact-based metrics consist of two variants: Fact-Core, which relies on the extraction of seven core fact attributes, and Fact-Full, which combines these core facts and five additional attributes.

   The main problem of concept-based evaluation is the amount of False Positives.

3. **Auxiliary or intermediate tasks**
Examples in the survey are more related to speech and intent detection. I propose better:

   1. Applying a bi-encoder to measure text similarity and re-rank it using a cross-encoder
   2. applying clustering and ensuring that source and target are in the same one
In both cases, I would remove structure level features such as sections

For **human evaluation**, the following approaches may be considered:

1. **Intrinsic evaluation**: properties of the system’s output.

   1. **Text quality**: relevance, consistency, fluency, coherence, missing, hallucination, repetition and contraction. Srivastava et al. (2022) used four standard linguistic parameters: relevance (selection of relevant content), consistency (factual alignment between the summary and the source), fluency (linguistic quality of each sentence), and coherence (structure and organization of summary). In addition to these commonly used and well-studied criteria, the evaluation of MRG also concludes other medical correctness criteria, such as factually correct and medically relevant information.
   2. **Factually correct and medically relevant information**: Critical Omissions, Hallucinations, Correct Facts, Incorrect Facts based on fact extraction.
   3. **Relative evaluations**: Given 2 model outputs, choosing which one is better. In our situation it could be interesting when comparing diverse model outputs.

## References

[1] Zhou, Y., Ringeval, F., & Portet, F. (2023). A Survey of Evaluation Methods of Generated Medical Textual Reports. In T. Naumann, A. Ben Abacha, S. Bethard, K. Roberts, & A. Rumshisky (Eds.), Proceedings of the 5th Clinical Natural Language Processing Workshop (pp. 447–459). Association for Computational Linguistics. https://doi.org/10.18653/v1/2023.clinicalnlp-1.48