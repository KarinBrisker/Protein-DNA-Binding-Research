from torch.utils.data import Dataset
import torch


"""
class to keep the data for dna and protein - their amino acids and nucleotides ABC etc.
"""


class Dictionary(object):
    def __init__(self):
        self.dna_vocab = ['a', 'c', 'g', 't']
        self.idx2dna = dict(enumerate(self.dna_vocab))
        self.dna2idx = dict(zip(self.idx2dna.values(), self.idx2dna.keys()))

        # uni-grams
        self.amino_acids = ['a', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v',
                            'w', 'y', '#']
        self.idx2amino_acids = dict(enumerate(self.amino_acids))
        self.amino_acids2idx = dict(zip(self.idx2amino_acids.values(), self.idx2amino_acids.keys()))

        self.protein2seq = {}
        self.protein2idx = {}
        i = 0
        lines = open('../DNA_data/protein_seq.txt').readlines()
        for line in lines:
            # line = name <sep> seq
            line = line.split()
            # prot2val:dict. prot name to sequence
            self.protein2seq[line[0]] = line[1].strip().lower()
            self.protein2idx[line[0]] = i
            i = i + 1
        self.proteins = list(self.protein2seq.keys())

    def __len__(self):
        return len(self.idx2amino_acids)

    def len_dna(self):
        return len(self.idx2dna)


"""
this class provides two random samples of: dna, protein, binding score
output for each sample:
        # protein - protein seq
        # dna - dna seq
        # amino acids - 12 features for each amino acid in the protein
        # label - binding score
"""


class ProteinsDataset(Dataset):
    # "protein", "protein_name", "dna1", "score1", "dna2", "score2"]
    def __init__(self, proteins, proteins_names, dna1, score1, dna2, score2, dict, device):
        self.device = device
        self.proteins_ = torch.stack(list(proteins)).to(self.device)

        self.dnas1_ = torch.stack(list(dna1)).to(self.device)
        self.scores1_ = torch.tensor(list(score1))

        self.dnas2_ = torch.stack(list(dna2)).to(self.device)
        self.scores2_ = torch.tensor(list(score2))

        self.prot_idx_ = [dict.protein2idx[x] for x in proteins_names]

        self.labels_ = torch.tensor([1 if (score1[i] - score2[i]) > 0 else 0 for i in range(len(score1))]).to(self.
                                                                                                              device).\
            double()

    def __len__(self):
        return len(self.labels_)

    def __getitem__(self, idx):
        protein = self.proteins_[idx]
        dna = self.dnas1_[idx]
        label = self.labels_[idx]
        prot_idxs = self.prot_idx_[idx]
        dna2 = self.dnas2_[idx]
        return protein, dna, label, prot_idxs, dna2


"""
this class provides two random samples of: dna, protein, binding score
output for each sample:
        # protein - protein seq
        # dna - dna seq
        # amino acids - 12 features for each amino acid in the protein
        # label - binding score
"""


class ProteinsDatasetClassification(Dataset):
    # "protein", "protein_name", "dna", "score"]
    def __init__(self, proteins, proteins_names, dna, score, dict, device):
        self.device = device
        self.proteins_ = torch.stack(list(proteins)).to(self.device)
        self.dnas_ = torch.stack(list(dna)).to(self.device)
        self.scores_ = torch.tensor(list(score))
        self.prot_idx_ = [dict.protein2idx[x] for x in proteins_names]

    def __len__(self):
        return len(self.scores_)

    def __getitem__(self, idx):
        protein = self.proteins_[idx]
        dna = self.dnas_[idx]
        label = self.scores_[idx]
        prot_idxs = self.prot_idx_[idx]
        return protein, dna, label, prot_idxs
