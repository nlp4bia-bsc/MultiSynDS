# MultiSynDS: Multilingual Synthetic Discharge Summary Generation

A reproducible and privacy‑preserving framework for generating, translating, and evaluating hospital‑style discharge summaries from clinical case reports using large language models and clinical NLP.

## Repository Structure

```
├── data                      # Input datasets and resources
│   ├── 0_raw                 # Raw original documents
│   ├── 1_original            
│   │   ├── metadata          # Case report metadata
│   │   └── txt               # Case report texts
│   ├── 2_generated           # Synthetic summaries
│   │   ├── 2step_transformation_dt4h_GeminiFlash
│   │   │   ├── en            # English summaries
│   │   │   └── nl            # Dutch summaries
│   │   └── 2step_transformation_dt4h_GPT4omini
│   │       ├── en
│   │       └── nl
│   ├── 3_toy_data            # Minimal examples for testing
│   ├── 4_gazetteers          # Gazetteer lists (e.g. medical terms)
│   │   └── en
│   ├── 5_abbreviations       # Clinical abbreviation lists
│   ├── human_eval            # Human evaluation responses
│   │   ├── gpt4omini
│   │   └── original
│   └── umls                  # UMLS resources
|
├── img                       # Figures and illustrations
│   ├── architecture          
│   ├── automatic_metric      
│   ├── data_analysis         
│   ├── external_figures      
│   └── results_analysis
│       └── form2
|
├── nbs                       # Notebooks for each phase
│   ├── data                  
│   │   ├── en
│   │   │   └── gpt40mini_comp
│   │   ├── nl
│   │   └── orig
│   ├── evaluation            
│   │   ├── automatic         # Automatic metric notebooks
│   │   │   ├── auto_eval_thres
│   │   │   ├── cardioner_auto_eval
│   │   │   └── cardioner_entities
│   │   ├── correlations      # Phase correlations
│   │   │   ├── phase_1
│   │   │   ├── phase_2
│   │   │   │   └── 70B
│   │   │   └── phase_3
│   │   └── human             # Human evaluation analysis
│   ├── generative            # Summary generation and judge evaluation
│   │   ├── Form_1
│   │   │   └── phase_2
│   │   ├── Form_2
│   │   │   └── phase_2
│   │   ├── Form_3
│   │   │   └── phase_3        # Contains LLM‑as‑judge for multiple models
│   │   └── other             # Auxiliary notebooks
│   └── ner                   # Named entity recognition tutorials
│       └── scispacy
│           └── tutorial
|
├── output                    # Generated outputs and evaluation results
│   ├── automatic_metric      # Metric scores and plots
│   ├── data                  # Synthetic summaries
│   │   ├── 2step_transformation_dt4h_GeminiFlash
│   │   │   └── en
│   │   └── 2step_transformation_dt4h_GPT4omini
│   │       └── en
│   ├── evaluation            # Evaluation outputs per form
│   │   ├── Form1
│   │   ├── Form2
│   │   └── Form3             # Phase 3 summaries and original texts
│   └── samples
│       └── en
│           ├── phase_1
│           ├── phase_2
│           └── phase_3       # Generated and original
|
├── scripts                   # Helper scripts for model inference
│   ├── llama_3B_inst_eval
│   ├── llama_doctor
│   ├── MMed-Llama-3-8B-EnIns
│   └── prometheus_2_mistral
|
├── src                       # Source code modules
│   └── __pycache__
|
├── tests                     # Unit tests
│   └── src
|
├── utils                     # Utilities, prompts, and templates
│   ├── prompts
│   └── templates
│       ├── basic
│       └── prometheus2_7b
|
├── requirements.txt          # Project dependencies
├── LICENSE                   # MIT License
└── README.md                 # This file
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-org/multisynDS.git
   cd multisynDS
   ```

2. Create and activate a Python 3.10+ environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Download SNOMED CT International Edition (2024AA) and set the `SNOMED_DIR` environment variable.

## Usage

### Automatic Evaluation

Execute automatic metrics on generated summaries in the `nbs\evaluation\automatic` directory.

### LLM‑as‑Judge Evaluation

Run the LLM‑as‑judge evaluation notebooks in `nbs\generative\Form_3\phase_3` to assess the quality of generated summaries.

### Figures Export

Other notebooks under `nbs` export plots to `img` subfolders. To regenerate, run the desired notebook; figures will save automatically.

## Contact

For questions or contributions, please contact alberto.becerra.tome@gmail.com.
