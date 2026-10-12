"""
Curated biomedical vocabularies used by the lexicon-based NER fallback
(see ner.py for why this exists instead of a downloaded BioBERT checkpoint).

These are well-known, publicly documented disease and drug/chemical names
(not sourced from any licensed database) used purely as a matching dictionary.
"""

DISEASES = [
    "type 2 diabetes", "type 1 diabetes", "hypertension", "rheumatoid arthritis",
    "psoriasis", "psoriatic arthritis", "alzheimer's disease", "parkinson's disease",
    "breast cancer", "lung cancer", "non-small cell lung cancer", "pulmonary fibrosis",
    "idiopathic pulmonary fibrosis", "multiple sclerosis", "asthma",
    "chronic obstructive pulmonary disease", "copd", "depression", "major depressive disorder",
    "crohn's disease", "ulcerative colitis", "inflammatory bowel disease", "covid-19",
    "sars-cov-2 infection", "migraine", "epilepsy", "osteoporosis",
    "chronic kidney disease", "heart failure", "atrial fibrillation", "hiv infection",
    "aids", "tuberculosis", "malaria", "obesity", "hyperlipidemia", "schizophrenia",
    "bipolar disorder", "anxiety disorder", "eczema", "atopic dermatitis",
    "ankylosing spondylitis", "systemic lupus erythematosus", "lupus", "sepsis",
    "stroke", "ischemic stroke", "coronary artery disease", "chronic hepatitis c",
    "non-alcoholic fatty liver disease", "prostate cancer", "colorectal cancer",
    "melanoma", "leukemia", "lymphoma", "cystic fibrosis", "huntington's disease",
    "amyotrophic lateral sclerosis", "als", "endometriosis", "fibromyalgia",
    "irritable bowel syndrome", "glaucoma", "macular degeneration", "chronic pain",
    "gastroesophageal reflux disease", "peptic ulcer disease", "osteoarthritis",
    "gout", "anemia", "thrombosis", "pulmonary hypertension", "sickle cell disease",
    "interstitial lung disease", "myocardial infarction",
]

CHEMICALS = [  # drugs / chemical compounds
    "metformin", "insulin", "atorvastatin", "simvastatin", "rosuvastatin",
    "lisinopril", "losartan", "amlodipine", "metoprolol", "warfarin", "aspirin",
    "ibuprofen", "prednisone", "dexamethasone", "methotrexate", "adalimumab",
    "infliximab", "etanercept", "rituximab", "baricitinib", "tofacitinib",
    "nintedanib", "pirfenidone", "sildenafil", "levothyroxine", "omeprazole",
    "amoxicillin", "azithromycin", "ciprofloxacin", "doxycycline", "remdesivir",
    "hydroxychloroquine", "chloroquine", "ritonavir", "lopinavir", "tenofovir",
    "emtricitabine", "efavirenz", "isoniazid", "rifampin", "sertraline",
    "fluoxetine", "escitalopram", "duloxetine", "gabapentin", "pregabalin",
    "levodopa", "donepezil", "memantine", "riluzole", "clopidogrel",
    "empagliflozin", "semaglutide", "liraglutide", "pembrolizumab", "nivolumab",
    "trastuzumab", "imatinib", "erlotinib", "tamoxifen", "cisplatin", "paclitaxel",
    "doxorubicin", "vincristine", "cyclophosphamide", "azathioprine",
    "cyclosporine", "tacrolimus", "sulfasalazine", "mesalamine", "colchicine",
    "allopurinol", "furosemide", "spironolactone", "hydrochlorothiazide",
    "digoxin", "amiodarone", "apixaban", "rivaroxaban", "heparin", "naproxen",
    "celecoxib", "montelukast", "albuterol", "salbutamol", "fluticasone",
    "budesonide", "tiotropium", "baclofen", "topiramate", "lamotrigine",
    "valproate", "carbamazepine", "latanoprost", "timolol", "finasteride",
    "tamsulosin", "metronidazole", "vancomycin", "meropenem",
]

GENES = [  # gene / protein symbols
    "brca1", "brca2", "tnf", "tnf-alpha", "il-6", "il-17", "il-23", "jak1", "jak2",
    "egfr", "her2", "vegf", "vegfr", "pd-1", "pd-l1", "tgf-beta", "fgfr",
    "ace2", "app", "snca", "lrrk2", "htt", "sod1", "cftr", "abl1", "kras",
    "tp53", "pdgfr", "il-1beta",
]
