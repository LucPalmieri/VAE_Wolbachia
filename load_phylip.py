import numpy as np

def nucleotide_to_onehot(nuc):
    mapping = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1]
    }
    # For any unknown character (e.g., N, -, etc.) return [0,0,0,0]
    return mapping.get(nuc.upper(), [0, 0, 0, 0])

def load_phylip(filename):
    """
    Reads a PHYLIP file (sequential or interleaved) and returns a dictionary with:
      - "data_set": base filename (without extension)
      - "name": numpy array of taxon names
      - "group": numpy array of default group values ("NA")
      - "one_hot": list of lists of lists (n_taxa x n_sites x 4) with one-hot encoded nucleotides
      - "shape": tuple with (n_taxa, n_sites, 4)
    """
    # Read all non-empty lines
    with open(filename, 'r') as f:
        lines = [line.rstrip() for line in f if line.strip() != ""]

    # Parse header: first line should have two integers: number of taxa and sites
    header = lines[0].split()
    try:
        n_taxa = int(header[0])
        n_sites = int(header[1])
    except (IndexError, ValueError):
        raise ValueError("Header must contain two integers: number of taxa and number of sites.")

    data_lines = lines[1:]
    if len(data_lines) < n_taxa:
        raise ValueError("Not enough data lines for the specified number of taxa.")

    # Initialize lists for taxon names and sequences
    names = []
    sequences = ["" for _ in range(n_taxa)]

    # Process the first block: each of the first n_taxa lines should have taxon name and initial sequence fragment
    for i in range(n_taxa):
        parts = data_lines[i].split()
        if len(parts) < 2:
            raise ValueError(f"Line does not contain both a taxon name and sequence data: {data_lines[i]}")
        names.append(parts[0])
        seq_fragment = "".join(parts[1:])  # In case sequence is split into several parts on the same line
        sequences[i] += seq_fragment

    # Process additional blocks (for interleaved format) until sequences reach the expected length
    current_line = n_taxa
    while len(sequences[0]) < n_sites and current_line < len(data_lines):
        for i in range(n_taxa):
            if current_line >= len(data_lines):
                break
            line = data_lines[current_line].strip()
            if line == "":
                current_line += 1
                continue
            # For interleaved blocks, there is no taxon name so we join all parts together
            seq_fragment = "".join(line.split())
            sequences[i] += seq_fragment
            current_line += 1

    # Warn if any sequence length does not match expected number of sites
    for i, seq in enumerate(sequences):
        if len(seq) != n_sites:
            print(f"Warning: Taxon {names[i]} sequence length ({len(seq)}) does not match expected {n_sites} sites.")

    # One-hot encode each nucleotide in every sequence (keep as a list of lists)
    one_hot_list = []
    for seq in sequences:
        one_hot_seq = [nucleotide_to_onehot(nuc) for nuc in seq]
        one_hot_list.append(one_hot_seq)

    shape = (len(one_hot_list), len(one_hot_list[0]), 4)
    data = {
        "data_set": filename.split('.')[0],
        "name": np.array(names, dtype=str),
        "group": np.array(["NA"] * len(names), dtype=str),
        "one_hot": one_hot_list,  # kept as list of lists of lists for string output
        "shape": shape
    }
    return data

def save_encoded_data(filename, data):
    """
    Saves the encoded data into a text file.
    Each line starts with the taxon name and group, followed by space-separated
    one-hot encoded nucleotide vectors written as strings, e.g. "[1,0,0,0]".
    """
    with open(filename, 'w') as f:
        for i in range(data["shape"][0]):
            line_parts = [data["name"][i], data["group"][i]]
            for j in range(data["shape"][1]):
                vector = data["one_hot"][i][j]
                # Convert each vector to a string like "[1,0,0,0]"
                vector_str = "[" + ",".join(str(x) for x in vector) + "]"
                line_parts.append(vector_str)
            line = " ".join(line_parts)
            f.write(line + "\n")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python load_phylip.py <phylip_file> [<output_file>]")
    else:
        phylip_file = sys.argv[1]
        data = load_phylip(phylip_file)
        print("Data loaded with shape:", data["shape"])
        print("Taxon names:", data["name"])
        if len(sys.argv) >= 3:
            output_file = sys.argv[2]
            save_encoded_data(output_file, data)
            print("Encoded data saved to:", output_file)
