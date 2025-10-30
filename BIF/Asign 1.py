# DNA Sequence Analysis

def gc_content(seq):
    """Calculate GC content percentage of DNA sequence"""
    g = seq.count("G")
    c = seq.count("C")
    gc = (g + c) / len(seq) * 100
    return round(gc, 2)

def find_motif(seq, motif):
    """Find all positions of a given motif"""
    positions = []
    for i in range(len(seq) - len(motif) + 1):
        if seq[i:i+len(motif)] == motif:
            positions.append(i+1)   # +1 for 1-based indexing
    return positions

def find_coding_regions(seq):
    """Identify coding regions (start with ATG, end with TAA/TAG/TGA)"""
    start_codon = "ATG"
    stop_codons = ["TAA", "TAG", "TGA"]
    regions = []

    i = 0
    while i < len(seq):
        start = seq.find(start_codon, i)
        if start == -1:
            break
        for j in range(start+3, len(seq), 3):
            codon = seq[j:j+3]
            if codon in stop_codons:
                regions.append((start+1, j+3)) # 1-based indexing
                i = j + 3
                break
        else:
            i = start + 3
    return regions

# ------------------- MAIN PROGRAM -------------------

# Example DNA sequence
dna_seq = "ATGCGTACGTTAGCTAGCGCTAAATGCTAGCTGAATGCCCTAG"

print("DNA Sequence:", dna_seq)
print("\nLength of sequence:", len(dna_seq))

# GC content
gc = gc_content(dna_seq)
print("\nGC Content:", gc, "%")

# Find motif
motif = "CGT"
motif_positions = find_motif(dna_seq, motif)
print(f"\nMotif '{motif}' found at positions:", motif_positions if motif_positions else "Not found")

# Coding regions
coding_regions = find_coding_regions(dna_seq)
print("\nCoding Regions (Start, End positions):", coding_regions if coding_regions else "No coding region found")
