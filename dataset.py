import os
import torch
import numpy as np
from midi.midi_utils import midiread
from scamp import Session, wait
from tqdm import tqdm


dataset_path = "/Users/aymeric/Desktop/AI/DiffusionModels/data/Nottingham"


def pre_encode(note: np.ndarray):
    byte_array = np.packbits(note)
    return int.from_bytes(byte_array, byteorder='big')


def pre_encode_partition(piano_roll: np.ndarray):
    return np.array([pre_encode(note) for note in piano_roll])


def decode(pre_encoded: np.int64):
    byte_array = np.frombuffer(pre_encoded.tobytes()[::-1], dtype=np.uint8)
    return np.unpackbits(byte_array)


def decode_partition(pre_encoded_partition: np.ndarray):
    return np.array([decode(pre_encoded) for pre_encoded in pre_encoded_partition])


def get_pre_encoded_partition(file_path: str, dt: float = 0.25, r: tuple = (31, 95)):
    midiIn = midiread(file_path)
    piano_roll = midiIn.create_piano_roll(r, dt)
    
    return pre_encode_partition(piano_roll)


def get_pre_encoded_partitions(folder_path: str, min_length: int = 1):
    partitions = []
    
    for partition_name in tqdm(os.listdir(folder_path)):
        partition_path = os.path.join(folder_path, partition_name)
        pre_encoded_partition = get_pre_encoded_partition(partition_path)
        if pre_encoded_partition.size > min_length:
            partitions.append(pre_encoded_partition)
    
    return partitions


def get_encoded_partitions(device: torch.device, min_length: int = 1):
    test_partitions = get_pre_encoded_partitions("/Users/aymeric/Desktop/AI/DiffusionModels/data/Nottingham/train", min_length)
    train_partitions = get_pre_encoded_partitions("/Users/aymeric/Desktop/AI/MusicAttention/midis", min_length)
    
    sorted_notes = sorted(set(np.concatenate(train_partitions + test_partitions)))
    
    notes_id = {float(encoded_note): i + 1 for i, encoded_note in enumerate(sorted_notes)}
    notes_id[0.0] = 0

    train_partitions_id = [torch.tensor([notes_id[float(encoded_note)] for encoded_note in partition], dtype=torch.long, device=device) for partition in train_partitions]
    test_partitions_id = [torch.tensor([notes_id[float(encoded_note)] for encoded_note in partition], dtype=torch.long, device=device) for partition in test_partitions]
    
    return train_partitions_id, test_partitions_id, sorted_notes



def play(partition: np.ndarray, volume: float = 1.0):
    s = Session()
    instrument = s.new_part("piano")
    
    notes = []
    
    for note, values in enumerate(partition.T):
        values = np.concatenate(([0], values, [0]))
        indicies = np.where(np.diff(values) != 0)
        notes += [(start, end - start, note) for start, end in zip(indicies[0][::2], indicies[0][1::2])]
    
    time = 0
    for start, duration, note in sorted(notes):
        if start > time:
            wait(0.25 * (start - time))
            time = start
        
        instrument.play_note(note + 31, volume, 0.25 * duration, blocking=False)


class Dataset:
    def __init__(self, device: torch.device, min_partition_length: int = 1):
        self.device = device
        self.train_encoded_partitions, self.test_encoded_partitions, self.sorted_notes = get_encoded_partitions(device, min_partition_length)
        self.vocab_size = len(self.sorted_notes)
    
    def __len__(self):
        return len(self.train_encoded_partitions) + len(self.test_encoded_partitions)
    
    
    def __getitem__(self, idx: int):
        return self.partitions_id[idx]
    
    
    def sample(self, n: int, context_size: int, train: bool = True):
        partitions_idx = np.random.randint(0, len(self.train_encoded_partitions) if train else len(self.test_encoded_partitions), n)
        
        contexts = torch.zeros((n, context_size + 1), dtype=torch.long, device=self.device)
        for i, partition_idx in enumerate(partitions_idx):
            partition: torch.Tensor = self.train_encoded_partitions[partition_idx] if train else self.test_encoded_partitions[partition_idx]
            if partition.size(0) <= context_size + 1:
                contexts[i, :partition.size(0)] = partition
                continue
            
            start = np.random.randint(0, partition.size(0) - context_size - 1)
            contexts[i, :] = partition[start:start + context_size + 1]
        
        return contexts
    
    
    def decode_partition(self, encoded_partition: np.ndarray):
        return np.array([decode(self.sorted_notes[encoded_note]) for encoded_note in encoded_partition])


# dataset = Dataset(torch.device("cpu"), min_partition_length=32)
# random_encoded_partition = dataset.sample(5, 100, train=True)[2]
# random_partition = dataset.decode_partition(random_encoded_partition)
# play(random_partition)