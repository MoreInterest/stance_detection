import torch
import pandas as pd
import numpy as np
from math import ceil
from json import loads
import pickle
import os


class StanceDataset:
    
    """ A data structure to hold information about the examples of a stance
    dataset and to manipulate it. """
    
    def __init__(self, texts, targets, stances=None):
        
        """ Instantiates the dataset given the stance texts, targets
        and labels.
        
        :param texts: a list of strings representing the stance texts;
        :param targets: a list of strings representing the stance targets, such
            that the positions match the ones of the associated texts.
        :param stances: a list of integers representing the labels associated to
            the matching texts and targets; 0 for neutral, 1 for favour and 2 for
            against. It is optional, to allow for unlabelled datasets. """
        
        self.texts = texts
        self.targets = targets
        self.stances = stances
            
    def downsample(self):
        
        """ Returns a downsampled version of this dataset such that for each
        stance target there is an equal number of examples for each label. """
        
        subsets = []
        for target in self.get_unique_targets():
            subset, _ = self.split_by_targets([target])
            counts = {}
            for stance in subset.stances:
                if stance not in counts:
                    counts[stance] = 0
                counts[stance] += 1
            minimum = None
            for stance, count in counts.items():
                minimum = count if minimum is None else min(minimum, count)
            texts, targets, stances = [], [], []
            for text, stance in zip(subset.texts, subset.stances):
                if stances.count(stance) < minimum:
                    texts.append(text)
                    targets.append(target)
                    stances.append(stance)
            subsets.append(StanceDataset(texts, targets, stances))
        z = subsets[0]
        for i in range(1, len(subsets)):
            z = z + subsets[i]
        return z
        
    def replace_target(self, target, replacement):
        
        """ Replaces the textual representation of a target. """
        
        for i, _target in enumerate(self.targets):
            if _target == target:
                self.targets[i] = replacement
        
    def exclude_neutral(self):
        
        """ Returns a filtered version of the dataset such that neutral examples
        are not present. """
        
        texts, targets, stances = [], [], []
        for text, target, stance in zip(self.texts, self.targets, self.stances):
            if stance != 0:
                texts.append(text)
                targets.append(target)
                stances.append(stance - 1)
        return StanceDataset(texts, targets, stances)
        
    def split_by_targets(self, targets):
        
        """ Returns two datasets, one containing the examples for the specified targets
        and one with the remaining ones. """
        
        subset_1, subset_2 = [[], [], []], [[], [], []]
        
        for i in range(len(self)):
            
            subset = subset_1 if self.targets[i] in targets else subset_2
            
            subset[0].append(self.texts[i])
            subset[1].append(self.targets[i])
            subset[2].append(self.stances[i])
            
        return StanceDataset(*subset_1), StanceDataset(*subset_2)
    
    def get_unique_targets(self):
        return sorted(list(set(self.targets)))
    
    def split(self, p):
        
        """ Splits the dataset based on the given percentage split and returns the
        two new datasets; it ensures that in both splits, the label distribution for
        each target is roughly equal. """
        
        targets = self.get_unique_targets()
        subsets = {target: self.split_by_targets([target])[0] for target in targets}
        
        subset_1, subset_2 = StanceDataset([], [], []), StanceDataset([], [], [])
        
        for _, subset in subsets.items():
            a, b = subset._split(p)
            
            subset_1.texts += a.texts
            subset_1.targets += a.targets
            subset_1.stances += a.stances
            
            subset_2.texts += b.texts
            subset_2.targets += b.targets
            subset_2.stances += b.stances
        
        return subset_1, subset_2
    
    def _merge(self, other):
        
        """ Merges two datasets and returns the resulting new dataset. """
        
        if self.stances is not None and other.stances is not None:
            return StanceDataset(self.texts + other.texts, self.targets + other.targets, self.stances + other.stances)
        else:
            return StanceDataset(self.texts + other.texts, self.targets + other.targets)
    
    def _remove(self, other):
        
        """ Returns a dataset in which all the examples in the given dataset are not present
        in this dataset. """
        
        texts, targets, stances = [], [], []
        for text, target, stance in zip(self.texts, self.targets, self.stances):
            if text not in other.texts:
                texts.append(text)
                targets.append(target)
                stances.append(stance)
        return StanceDataset(texts, targets, stances)
    
    def __sub__(self, other):
        return self._remove(other)
    
    def __add__(self, other):
        return self._merge(other)
    
    def __len__(self):
        return len(self.texts)        
        
    def _split(self, p):
        
        delimiter = round(p * len(self))
        distribution = self.get_label_distribution()
        counts = list(delimiter * np.array(distribution))
        
        subset_1, subset_2 = [[], [], []], [[], [], []]
        
        for i in range(len(self)):
            count = counts[int(self.stances[i])]
            subset = subset_1 if count > 0 else subset_2
            counts[int(self.stances[i])] -= count > 0           
            subset[0].append(self.texts[i])
            subset[1].append(self.targets[i])
            subset[2].append(self.stances[i])
            
        return StanceDataset(*subset_1), StanceDataset(*subset_2)
        
    def get_unique_labels(self):
        return sorted(list(set(self.stances)))
    
    def get_label_distribution(self):
        counts = np.array([self.stances.count(i) for i in self.get_unique_labels()])
        return list(counts / np.sum(counts))
        
    def __getitem__(self, key):
        if isinstance(key, slice):
            if self.stances is None:
                return StanceDataset(self.texts[key.start:key.stop], self.targets[key.start:key.stop], None)
            else:
                return StanceDataset(self.texts[key.start:key.stop], self.targets[key.start:key.stop], self.stances[key.start:key.stop])
        

class SemEval(StanceDataset):
    
    def __init__(self, subset, path):
        df = pd.read_csv(path + subset + ".csv")
        texts = list(np.array(df["Tweet"]))
        targets = list(np.array(df["Target"]))
        stances = (torch.tensor((df["Stance"] == "FAVOR") + 2 * (df["Stance"] == "AGAINST"))).tolist()
        StanceDataset.__init__(self, texts, targets, stances)
        
    @staticmethod
    def get_subtask_a_datasets(path):
        return SemEval("train", path=path), SemEval("test", path=path).split_by_targets(["Donald Trump"])[1]
    
    @staticmethod
    def get_subtask_b_datasets(path):
        testing, training = SemEval("test", path=path).split_by_targets(["Donald Trump"])
        training = training + SemEval("train", path=path)
        return training, testing
    
    
class SemEvalSeenUnlabelled(StanceDataset):
    
    def __init__(self, path):
        lines = open(path, "r").readlines()
        raw_texts = [loads(line)["text"] for line in lines]
        texts, targets = [], []        
        for raw_text in raw_texts:
            t = raw_text.lower()
            target = None
            if "hillary" in t or "clinton" in t:
                target = "Hillary Clinton"
            elif "climate" in t:
                target = "Climate Change is a Real Concern"
            elif "femin" in t:
                target = "Feminist Movement"
            elif "abort" in t:
                target = "Legalization of Abortion"
            elif "ath" in t:
                target = "Atheism"
            if target:
                targets.append(target)
                texts.append(raw_text)
        StanceDataset.__init__(self, texts, targets, None)
        
        
class SemEvalUnseenUnlabelled(StanceDataset):
    
    def __init__(self, path):
        lines = open(path, "r").readlines()
        texts = []
        for line in lines:
            text = line[19:]
            if "Not Available" not in text:
                texts.append(text)
        texts = list(set(texts))
        targets = ["Donald Trump"] * len(texts)
        StanceDataset.__init__(self, texts, targets, None)
        
        
class FactmataDataset(StanceDataset):
    
    def __init__(self, path="/media/glacier/matteo/data/factmata_stance.csv", subset=None):
        df = pd.read_csv(path)
        if subset:
            df = df[df["Topic"] == subset]
        texts = list(np.array(df["Text"]))
        targets = list(np.array(df["Topic"]))
        stances = torch.zeros(len(texts), dtype=torch.long)
        stances += torch.tensor(np.array(df["Stance"] == "Favour")).long()
        stances += torch.tensor(np.array(df["Stance"] == "Against") * 2).long()
        StanceDataset.__init__(self, texts, targets, stances.tolist())
        
        
class FactmataUnlabelledDataset(StanceDataset):
    
    def __init__(self, path="/media/glacier/marc/bert-tuning/vaccine-bert/bert_ready_vaccine.txt", limit=None, minimum_length=20, maximum_length=100):
        raw_lines = open(path, "r").readlines()
        lines = []
        for line in raw_lines:
            line = str(line)
            if minimum_length < len(line) < maximum_length:
                lines.append(line.replace("\n", ""))
        if limit is not None:
            lines = lines[:limit]
        StanceDataset.__init__(self, lines, ["Vaccination"] * len(lines), None)