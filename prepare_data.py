"""Index completed ConceptEdit TARs and the explicit NOOB2 image IDs once."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import sqlite3
import tarfile

import lance
import numpy as np

from moss_edit.components import read_config


def index_tar(job):
    archive, output = map(Path, job)
    target = output / (archive.stem + '.sqlite')
    if target.exists():
        with sqlite3.connect(target) as db:
            return dict(path=str(archive), index=str(target), count=db.execute('select count(*) from samples').fetchone()[0])
    tmp = target.with_suffix('.partial')
    tmp.unlink(missing_ok=True)
    with sqlite3.connect(tmp) as db, tarfile.open(archive, 'r:') as tf:
        db.execute('create table samples (i integer primary key, id text, instruction text, so integer, sn integer, toff integer, tn integer)')
        images, records = {}, []
        for member in tf:
            if not member.isfile():
                continue
            if member.name.endswith('.json'):
                records.append(json.load(tf.extractfile(member)))
            else:
                images[member.name] = (member.offset_data, member.size)
        rows = []
        for row in records:
            # Pair by the JSON's exact filenames, never TAR member order.
            so, sn = images[row['images']['source']]
            toff, tn = images[row['images']['edited']]
            rows.append((len(rows), row['id'], row['instruction']['detailed_en'], so, sn, toff, tn))
        db.executemany('insert into samples values (?,?,?,?,?,?,?)', rows)
        db.commit()
    os.replace(tmp, target)
    result = dict(path=str(archive), index=str(target), count=len(rows))
    print(json.dumps({'event': 'tar_indexed', **result}), flush=True)
    return result


def index_noob_pairs(pairs, config):
    target = Path(config['noob_pair_index'])
    if target.exists():
        return len(np.load(target, mmap_mode='r'))
    # Scan without filtering so offsets match Lance take() row indices.
    eligible, offset = [], 0
    scanner = pairs.scanner(columns=['rev', 'description'], scan_in_order=True)
    for batch in scanner.to_batches():
        rows = batch.to_pylist()
        eligible.extend(offset + i for i, row in enumerate(rows)
                        if row['rev'] == 0 and (row['description'] or '').strip())
        offset += len(rows)
    rng = np.random.default_rng(config['noob_subset_seed'])
    selected = rng.choice(eligible, size=config['noob_subset_samples'], replace=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_suffix('.partial').open('wb') as file:
        np.save(file, selected)
    os.replace(target.with_suffix('.partial'), target)
    return len(selected)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    args = p.parse_args()
    config = read_config(args.config)
    output = Path(config['data_index']); output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'sources.json'
    # Freeze the completed shard list on the first invocation.
    selected_path = output / 'selected_shards.json'
    if selected_path.exists():
        selected = json.loads(selected_path.read_text())
    else:
        root = Path(config['conceptedit_root'])
        selected = [str(root/'enhanced_prompt_random_resolution'/json.loads(p.read_text())['shard'])
                    for p in sorted((root/'.pipeline/completed').glob('*.json'))]
        selected_path.write_text(json.dumps(selected, indent=2))
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        shards = list(pool.map(index_tar, [(x, str(output)) for x in selected]))

    noob = Path(config['noob_root'])
    images = lance.dataset(str(noob/'images.lance'))
    mapping_path = output / 'noob_image_id_to_row.npy'
    if not mapping_path.exists():
        # img_id is a dataset key, not an assumed physical row number.
        ids = images.to_table(columns=['img_id'])['img_id'].combine_chunks().to_numpy()
        mapping = np.lib.format.open_memmap(str(mapping_path)+'.partial', mode='w+', dtype=np.int64, shape=(int(ids.max())+1,))
        mapping[:] = -1
        mapping[ids] = np.arange(len(ids), dtype=np.int64)
        mapping.flush(); del mapping
        os.replace(str(mapping_path)+'.partial', mapping_path)
    pairs = lance.dataset(str(noob/'pairs.lance'))
    noob_count = index_noob_pairs(pairs, config)
    manifest = {'conceptedit': shards, 'conceptedit_count': sum(s['count'] for s in shards),
                'noob_root': str(noob), 'noob_count': noob_count,
                'noob_image_map': str(mapping_path), 'noob_pairs_version': pairs.version,
                'noob_images_version': images.version}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'event': 'index_ready', 'noob_samples': manifest['noob_count'],
                      'edit_samples': manifest['conceptedit_count'], 'shards': len(shards)}), flush=True)


if __name__ == '__main__':
    main()
