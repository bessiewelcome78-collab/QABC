#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, hashlib
from pathlib import Path
import yaml

IMG_EXT={'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}

def count_files(p: Path):
    return sum(1 for x in p.iterdir() if x.is_file() and x.suffix.lower() in IMG_EXT)

def sha256(path: Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def resolve_target_root(data_dir: Path, target: str):
    names=[target]
    if target=='UWaterlooSkinCancer': names += ['UWaterloo']
    for n in names:
        root=data_dir/n
        for cand in [root/'Test_Folder', root/'test', root/'Test']:
            if (cand/'img').is_dir() and (cand/'label').is_dir():
                return root, cand
    raise FileNotFoundError(f'{target}: no Test_Folder/img,label under {data_dir}')

def resolve_prompt_dir(root: Path, target: str):
    cands=[root/'Prompts_Folder']
    if target=='UWaterlooSkinCancer':
        cands += [root.parent/'UWaterlooSkinCancer'/'Prompts_Folder', root.parent/'UWaterloo'/'Prompts_Folder']
    for p in cands:
        if p.is_dir() and (p/'Test_text_original.xlsx').is_file(): return p
    raise FileNotFoundError(f'{target}: Test_text_original.xlsx not found in prompt folder candidates')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--project',required=True)
    ap.add_argument('--data-dir',required=True)
    ap.add_argument('--study-id',required=True)
    ap.add_argument('--manifest',default='configs/domain_generalization_qabr/JBTL15_QABR_TABLE2.json')
    ap.add_argument('--write-resolved',default='')
    a=ap.parse_args()
    project=Path(a.project).resolve(); data_dir=Path(a.data_dir).resolve()
    manifest=json.loads((project/a.manifest).read_text())
    resolved={'study_id':a.study_id,'sources':{},'targets':{}}
    errors=[]
    for src,spec in manifest['sources'].items():
        cfgp=project/spec['config']
        if not cfgp.is_file(): errors.append(f'{src}: source config missing {cfgp}'); continue
        cfg=yaml.safe_load(cfgp.read_text())
        checks={
            'DATASET.NAME': cfg.get('DATASET',{}).get('NAME')==src,
            'MODEL.QABR.ENABLED': cfg.get('MODEL',{}).get('QABR',{}).get('ENABLED') is True,
            'M1.ENABLED': cfg.get('M1',{}).get('ENABLED') is False,
            'TRAIN.NUM_EPOCHS': int(cfg.get('TRAIN',{}).get('NUM_EPOCHS',-1))==100,
            'TRAIN.BATCH_SIZE': int(cfg.get('TRAIN',{}).get('BATCH_SIZE',-1))==24,
            'TRAIN.LEARNING_RATE': abs(float(cfg.get('TRAIN',{}).get('LEARNING_RATE',-1))-3e-4)<1e-12,
            'TRAIN.VAL_NUM_SAMPLES': int(cfg.get('TRAIN',{}).get('VAL_NUM_SAMPLES',-1))==10,
            'TEST.NUM_SAMPLES': int(cfg.get('TEST',{}).get('NUM_SAMPLES',-1))==30,
            'TRAIN.RBAL_EDGE_WEIGHT': float(cfg.get('TRAIN',{}).get('RBAL_EDGE_WEIGHT',999))==0.0,
            'TRAIN.RBAL_NORMAL_WEIGHT': float(cfg.get('TRAIN',{}).get('RBAL_NORMAL_WEIGHT',999))==0.0,
        }
        for k,ok in checks.items():
            if not ok: errors.append(f'{src}: formal source contract failed: {k}')
        lock=project/spec['lock_file'].format(study_id=a.study_id)
        if not lock.is_file(): errors.append(f'{src}: LOCKED_CHECKPOINT missing {lock}'); continue
        cp=Path(lock.read_text().strip())
        if not cp.is_file() or cp.stat().st_size==0: errors.append(f'{src}: locked checkpoint missing/nonempty: {cp}'); continue
        if not cp.name.endswith('_best_val.pth'): errors.append(f'{src}: locked checkpoint is not best_val: {cp.name}')
        resolved['sources'][src]={'config':str(cfgp),'lock_file':str(lock),'checkpoint':str(cp),'checkpoint_name':cp.name,'checkpoint_bytes':cp.stat().st_size}
        print(f'[PASS] SOURCE {src}: {cp}')
        for tgt in spec['targets']:
            try:
                root,test=resolve_target_root(data_dir,tgt)
                prompt=resolve_prompt_dir(root,tgt)
                ni=count_files(test/'img'); nl=count_files(test/'label'); exp=int(manifest['expected_cases'][tgt])
                if ni!=exp or nl<exp:
                    raise RuntimeError(f'cases img={ni} label={nl}, expected img={exp}, label>={exp}')
                resolved['targets'][tgt]={'dataset_root':str(root),'test_path':str(test)+'/','prompt_path':str(prompt)+'/', 'images':ni,'labels':nl,'expected':exp}
                print(f'[PASS] TARGET {tgt}: images={ni} labels={nl} prompt=yes')
            except Exception as e: errors.append(f'{src}->{tgt}: {e}')
    if errors:
        print('\n[FAIL] preflight errors:')
        for e in errors: print(' -',e)
        raise SystemExit(20)
    out=Path(a.write_resolved) if a.write_resolved else project/'qabr_table2_resolved.json'
    out.write_text(json.dumps(resolved,indent=2),encoding='utf-8')
    print(f'[PASS] QABR Table-2 DG preflight complete; resolved={out}')
if __name__=='__main__': main()
