
'''
All external data utilized in this work is detailed below: 

-------------------
⚠️  VERSION WARNING
-------------------
The following sources are living databases with no version pinned in the URL.
Files downloaded today WILL differ from those used in this study:
  - PDB          (accessed July 16, 2025)
  - ClinVar      (accessed Feb 3, 2026)
  - gene2refseq  (accessed Feb 3, 2026)
  - GRCh38       (accessed Feb 3, 2026)
  - DisProt      (accessed July 28, 2025)

--------------------------------------
⚠️ Reproducibility Instructions
--------------------------------------

1) To exactly reproduce the results in this paper, you can directly use the same sequences committed to this repo and skip the data gathering/filtering 
steps in clinvar_expanded.py and define_pdb_disprot_datasets.py. The exact sequences utilized in this study are stored below

    - ClinVar: data/clinvar_sequences.csv
    - PDB/DisProt: data/pdb_disprot.fasta and data/protein_input_sets.csv

If you do want to use these sequences derived for our study, you can skip running define_pdb_disprot_datasets.py 
and then you can choose derived = TRUE in clinvar_expanded.py and the code will be reproducible. 

2) For clinvar_expanded.py, to run the conservation regressions you will need to manually download the 9606_database file (see below)

3) Download STARLING cuda and experimental SAXS data, as well as IDRome sequence data using this script (python esm2_interp_data.py)

Code on how to access and download the source data we utilized is included, but again, given these are living databases, 
please refer to the committed sequences we utilized if you wish to reproduce our results. 

-------------------

DisProt (accessed July 28 2025)
- release = 2025_06, with ambiguous evidences 
- sequences utilized in this work are saved as data/disprot.fasta

PDB (accessed July 16, 2025)
- https://files.rcsb.org/pub/pdb/derived_data/pdb_seqres.txt.gz
- save to: data/pdb_seqres.txt.gz
- PDB sequences utilized in this work are saved to data/protein_input_sets.csv and data/pdb_disprot.fasta

---------------- ClinVar -----------------

ClinVar Variants (accessed Feb 3 2026)
- https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz
- save to data/variant_summary.txt.gz
- sequences utilized in this work are saved to data/clinvar_sequences.csv

gene2refseq (accessed Feb 3 2026)
- https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2refseq.gz 
- save to data/gene2refseq

GRCh38_latest_protein (accessed Feb 3 2026)
- https://ftp.ncbi.nlm.nih.gov/refseq/H_sapiens/annotation/GRCh38_latest/refseq_identifiers/GRCh38_latest_protein.faa.gz
- save to data/GRCh38_latest_protein_copy.faa

9606_database (accessed Feb 3 2026)
- https://biomine2.cs.vcu.edu/servers/DESCRIBEPROT/download.html
- Go to row for Homo sapiens (Human) (9606)
- download raw data 
- save to data/9606_database.json

---------------- IDRome -----------------

Supplementary Table
- https://sid.erda.dk/share_redirect/AVZAJvJnCO/Supplementary_Table_3.xlsx
- save to data/Supplementary_Table_3.xlsx

STARLING/SAXS 
-  https://github.com/holehouse-lab/supportingdata/blob/master/2026/starling_2026/analysis/experimental_comparison/saxs_rg/data_out/all_comparison_data_WITH_STARTING.csv
- save to data/all_comparison_data_WITH_STARTING.csv

'''

import os
import gzip
import shutil
import requests
from tqdm import tqdm
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent 
BASE_PATH = SCRIPT_DIR.parent 
DATA_DIR  = os.path.join(BASE_PATH, "data")


# ── Helpers ───────────────────────────────────────────────────────────────────

def download_file(url, dest_path, desc=None):
    """Stream-download a file with a progress bar."""
    print(f"\n  Downloading: {desc or os.path.basename(dest_path)}")
    print(f"  URL  : {url}")
    print(f"  Dest : {dest_path}")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with open(dest_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, unit_divisor=1024
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))
    print(f"  ✓ Saved to {dest_path}")


def decompress_gz(gz_path, out_path):
    """Decompress a .gz file."""
    print(f"  Decompressing {os.path.basename(gz_path)} ...")
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    print(f"  ✓ Decompressed to {out_path}")


# ── PDB ───────────────────────────────────────────────────────────────────────
# Accessed:  July 16, 2025
# Source:    RCSB PDB
# Derived:   data/protein_input_sets.csv, data/pdb_disprot.fasta
# ⚠️  Updated weekly — will differ from version used in this study

def download_pdb():
    print("\n── PDB ──────────────────────────────────────────────────────────────")
    print("  Accessed: July 16, 2025")
    print("  ⚠️  PDB sequences are updated weekly.")
    print("     File downloaded today will differ from version used in this study.")
    print("     Derived files (data/protein_input_sets.csv, data/pdb_disprot.fasta)")
    print("     were generated from the July 16, 2025 snapshot.")
    url  = "https://files.rcsb.org/pub/pdb/derived_data/pdb_seqres.txt.gz"
    dest = os.path.join(DATA_DIR, "pdb_seqres.txt.gz")
    download_file(url, dest, desc="PDB sequences")


# ── ClinVar ───────────────────────────────────────────────────────────────────
# Accessed:  Feb 3, 2026
# Source:    NCBI ClinVar (public domain)
# Derived:   data/clinvar_sequences.csv
# ⚠️  Updated weekly — will differ from version used in this study

def download_clinvar():
    print("\n── ClinVar ──────────────────────────────────────────────────────────")
    print("  Accessed: Feb 3, 2026")
    print("  ⚠️  ClinVar is updated weekly.")
    print("     File downloaded today will differ from version used in this study.")
    print("     Derived file (data/clinvar_sequences.csv) was generated from")
    print("     the Feb 3, 2026 snapshot.")
    url  = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
    dest = os.path.join(DATA_DIR, "variant_summary.txt.gz")
    download_file(url, dest, desc="ClinVar variant_summary")


# ── gene2refseq ───────────────────────────────────────────────────────────────
# Accessed:  Feb 3, 2026
# Source:    NCBI Gene (public domain)
# Note:      Decompressed to TSV; read with pd.read_csv(..., sep='\t')
# ⚠️  Updated regularly — will differ from version used in this study

def download_gene2refseq():
    print("\n── gene2refseq ──────────────────────────────────────────────────────")
    print("  Accessed: Feb 3, 2026")
    print("  ⚠️  gene2refseq is updated regularly.")
    print("     File downloaded today will differ from version used in this study.")
    url  = "https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2refseq.gz"
    gz   = os.path.join(DATA_DIR, "gene2refseq.gz")
    dest = os.path.join(DATA_DIR, "gene2refseq")
    download_file(url, gz, desc="gene2refseq")
    decompress_gz(gz, dest)


# ── GRCh38 protein ────────────────────────────────────────────────────────────
# Accessed:  Feb 3, 2026
# Source:    NCBI RefSeq (public domain)
# Note:      Decompressed to .faa
# ⚠️  'latest' in URL is not version-pinned — will differ from version used

def download_grch38():
    print("\n── GRCh38 latest protein ────────────────────────────────────────────")
    print("  Accessed: Feb 3, 2026")
    print("  ⚠️  'GRCh38_latest' is not version-pinned.")
    print("     File downloaded today will differ from version used in this study.")
    url  = (
        "https://ftp.ncbi.nlm.nih.gov/refseq/H_sapiens/annotation/"
        "GRCh38_latest/refseq_identifiers/GRCh38_latest_protein.faa.gz"
    )
    gz   = os.path.join(DATA_DIR, "GRCh38_latest_protein.faa.gz")
    dest = os.path.join(DATA_DIR, "GRCh38_latest_protein_copy.faa")
    download_file(url, gz, desc="GRCh38 latest protein")
    decompress_gz(gz, dest)


# ── IDRome supplementary table ────────────────────────────────────────────────
# Static supplementary file — should be stable

def download_idrome_supplementary():
    print("\n── IDRome Supplementary Table 3 ────────────────────────────────────")
    url  = "https://sid.erda.dk/share_redirect/AVZAJvJnCO/Supplementary_Table_3.xlsx"
    dest = os.path.join(DATA_DIR, "Supplementary_Table_3.xlsx")
    download_file(url, dest, desc="IDRome Supplementary Table 3")


# ── STARLING / SAXS ───────────────────────────────────────────────────────────
# Static file on GitHub — stable as long as repo is not reorganised

def download_starling():
    print("\n── STARLING / SAXS comparison data ─────────────────────────────────")
    url  = (
        "https://raw.githubusercontent.com/holehouse-lab/supportingdata/"
        "master/2026/starling_2026/analysis/experimental_comparison/"
        "saxs_rg/data_out/all_comparison_data_WITH_STARTING.csv"
    )
    dest = os.path.join(DATA_DIR, "all_comparison_data_WITH_STARTING.csv")
    download_file(url, dest, desc="STARLING SAXS comparison data")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("Downloading external data sources")
    print("=" * 70)
    print(
        "\n⚠️  Some sources are living databases. Files downloaded today may"
        "\n   differ from those used in this study. \n"
    )

    #download_pdb()
    #download_clinvar()
    #download_gene2refseq()
    #download_grch38()
    download_idrome_supplementary()
    download_starling()

    print("\n" + "=" * 70)
    print(
        "\n📋  MANUAL DOWNLOADS REQUIRED"
        "\n     See the docstring at the top of this file for full instructions."
        "\n"
        "\n  1. data/disprot.fasta"
        "\n     DisProt release 2025_06 with ambiguous evidences — use archived"
        "\n     version from data/disprot.fasta"
        "\n"
        "\n  2. data/9606_database.json"
        "\n     DESCRIBEPROT: https://biomine2.cs.vcu.edu/servers/DESCRIBEPROT/download.html"
        "\n     Download raw data for Homo sapiens (taxon ID 9606)"
    )
    print("\n" + "=" * 70)
    print("\n✓ All automatic downloads complete.")
