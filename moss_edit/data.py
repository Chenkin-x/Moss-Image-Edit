"""Random-access editing pairs: Lance column projection and TAR byte offsets."""
import bisect
import io
import json
import math
import os
from pathlib import Path
import sqlite3

import lance
import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset, Sampler


def resize_pair(source, target, max_edge):
    source = ImageOps.exif_transpose(source).convert('RGB')
    target = ImageOps.exif_transpose(target).convert('RGB')
    scale = min(1., max_edge / max(source.size))
    size = tuple(max(16, min(max_edge, int(math.floor(x*scale/16 + .5))*16)) for x in source.size)
    # Single-image editing: source and target share exactly the same H x W.
    return [image.resize(size, Image.Resampling.LANCZOS) for image in (source, target)]


def as_tensor(image):
    return torch.from_numpy(np.array(image, dtype=np.uint8, copy=True)).permute(2, 0, 1).contiguous()


class MixedEdits(Dataset):
    def __init__(self, config):
        self.config = config
        self.manifest = json.loads((Path(config['data_index'])/'sources.json').read_text())
        self.manifest['noob_count'] = len(np.load(config['noob_pair_index'], mmap_mode='r'))
        self.shards = self.manifest['conceptedit']
        self.ends = np.cumsum([s['count'] for s in self.shards]).tolist()
        self.connections, self.files = {}, {}
        self.pairs = self.images = self.image_rows = self.pair_rows = None

    def __len__(self):
        return self.manifest['noob_count'] + self.ends[-1]

    def _open_noob(self):
        if self.pairs is None:
            root = Path(self.manifest['noob_root'])
            self.pairs = lance.dataset(str(root/'pairs.lance'), version=self.manifest['noob_pairs_version'])
            self.images = lance.dataset(str(root/'images.lance'), version=self.manifest['noob_images_version'])
            self.image_rows = np.load(self.manifest['noob_image_map'], mmap_mode='r')
            self.pair_rows = np.load(self.config['noob_pair_index'], mmap_mode='r')

    def raw_noob_batch(self, indices):
        self._open_noob()
        offsets = self.pair_rows[indices].tolist()
        pairs = self.pairs.take(offsets, columns=['pair_id','description','img_a_id','img_b_id','rev']).to_pylist()
        assert all(p['rev'] == 0 and p['description'].strip() for p in pairs)
        image_ids = np.array([i for p in pairs for i in (p['img_a_id'], p['img_b_id'])], dtype=np.int64)
        rows = self.image_rows[image_ids]
        assert (rows >= 0).all(), 'Unknown NOOB2 image ID'
        images = self.images.take(rows.tolist(), columns=['img_id','data']).to_pylist()
        assert [r['img_id'] for r in images] == image_ids.tolist()
        return [dict(id=f"noob2:v{self.manifest['noob_pairs_version']}:{offsets[i]}", instruction=p['description'],
                     source=images[2*i]['data'], target=images[2*i+1]['data'], dataset='noob2')
                for i, p in enumerate(pairs)]

    def raw_edit(self, index):
        shard_id = bisect.bisect_right(self.ends, index)
        local = index - (self.ends[shard_id-1] if shard_id else 0)
        shard = self.shards[shard_id]
        if shard_id not in self.connections:
            self.connections[shard_id] = sqlite3.connect(f"file:{shard['index']}?mode=ro", uri=True)
            self.files[shard_id] = os.open(shard['path'], os.O_RDONLY)
        row = self.connections[shard_id].execute('select id,instruction,so,sn,toff,tn from samples where i=?', (local,)).fetchone()
        sample_id, prompt, so, sn, toff, tn = row
        fd = self.files[shard_id]
        return dict(id='conceptedit:'+sample_id, instruction=prompt,
                    source=os.pread(fd,sn,so), target=os.pread(fd,tn,toff), dataset='conceptedit')

    def decode(self, row):
        with Image.open(io.BytesIO(row['source'])) as source, Image.open(io.BytesIO(row['target'])) as target:
            source, target = resize_pair(source, target, self.config['max_edge'])
        return {k:v for k,v in row.items() if k not in ('source','target')} | {
            'source': as_tensor(source), 'target': as_tensor(target)}

    def __getitems__(self, keys):
        # DataLoader gives the whole microbatch: one batched Lance take for its
        # pairs and one for its images, with only the columns needed by training.
        noob_positions = [i for i, (kind, _) in enumerate(keys) if kind == 0]
        noob = self.raw_noob_batch([keys[i][1] for i in noob_positions]) if noob_positions else []
        result = [None]*len(keys)
        for i, row in zip(noob_positions, noob): result[i] = row
        for i, (kind,index) in enumerate(keys):
            if kind == 1: result[i] = self.raw_edit(index)
        return [self.decode(row) for row in result]

    def __getitem__(self, key):
        return self.__getitems__([key])[0]


class MixedBatchSampler(Sampler):
    def __init__(self, data, *, rank, world, batch_size, steps, accumulation, seed, start_step=0):
        self.data, self.rank, self.world = data, rank, world
        self.batch_size, self.steps, self.accumulation = batch_size, steps, accumulation
        self.seed, self.start_step = seed, start_step

    def __iter__(self):
        # Step-local deterministic RNG makes resume independent of worker prefetch.
        excluded = set(tuple(x) for x in self.data.config.get('validation_keys', []))
        for step in range(self.start_step, self.steps):
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, step, self.rank]))
            for _ in range(self.accumulation):
                keys = []
                while len(keys) < self.batch_size:
                    kind = int(rng.random() < self.data.config.get('conceptedit_probability', .5))
                    count = self.data.ends[-1] if kind else self.data.manifest['noob_count']
                    key = (kind, int(rng.integers(count)))
                    if key not in excluded: keys.append(key)
                yield keys

    def __len__(self):
        return (self.steps-self.start_step)*self.accumulation


def keep_list(rows):
    return rows


def init_worker(_):
    torch.set_num_threads(1)
