"""Render the first ten usable validation episodes with frozen V4 parameters."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import hashlib
import cv2
import numpy as np
import sim_core as sc
from params import load_params
from render_replay import render_episode

HERE=Path(__file__).resolve().parent


def render_one(path):
    path=Path(path)
    out=HERE/'videos'/f'{path.stem}_mujoco.mp4'
    if out.exists(): raise FileExistsError(out)
    p=load_params(HERE/'params.yaml')
    ep=sc.load_episode(path)
    result=sc.simulate_episode(ep,p)
    np.savetxt(out.with_suffix('.csv'),np.column_stack([ep.start_norm,ep.executed,ep.real_norm,result.pred_norm]),delimiter=',',
               header='start_x_norm,start_y_norm,m16,m17,m18,m19,real_end_x_norm,real_end_y_norm,sim_end_x_norm,sim_end_y_norm',comments='')
    render_episode(path,out,p,30,960,720,True,0)
    cap=cv2.VideoCapture(str(out)); expected=len(ep.real_norm)*round(p.row_dt*30)
    actual=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=cap.get(cv2.CAP_PROP_FPS)
    ok,frame=cap.read()
    assert ok and actual==expected and fps==30 and frame.shape[:2]==(720,960)
    cv2.imwrite(str(out.with_suffix('.png')),frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES,actual-1)
    assert cap.read()[0], 'Last frame cannot be decoded'
    cap.release()
    meta=dict(episode=path.name,rows=len(ep.real_norm),frames=actual,fps=fps,seconds=actual/fps,
              width=960,height=720,open_loop=True,real_smoothing=0,
              visualization='MuJoCo 3D; linear interpolation between simulated row endpoints',
              model='nonelastic unchanged V4 parameters; slipping/stalling spool',
              mean_normalized_error=float(np.linalg.norm(result.pred_norm-ep.real_norm,axis=1).mean()),
              off_board_rows=result.off_board_rows,input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
              video_bytes=out.stat().st_size,verified=True)
    out.with_suffix('.json').write_text(json.dumps(meta,indent=2))
    return meta


def main():
    paths=[]; skipped=[]
    for path in sorted((HERE.parent/'data/val').glob('*.csv')):
        try: sc.load_episode(path)
        except ValueError as exc:
            if 'No valid rows' not in str(exc): raise
            skipped.append(path.name)
            continue
        paths.append(path)
        if len(paths)==10: break
    assert len(paths)==10
    (HERE/'videos').mkdir(exist_ok=True)
    manifest=dict(selection='First ten sorted usable validation CSVs; no performance-based selection',
                  episodes=[p.name for p in paths],skipped=skipped,
                  source_hashes={f:hashlib.sha256((HERE/f).read_bytes()).hexdigest() for f in ('params.yaml','params.py','sim_core.py','render_replay.py')})
    (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2))
    with ProcessPoolExecutor(max_workers=2) as pool:
        results=[]
        for r in pool.map(render_one,paths):
            results.append(r)
            print(f"VERIFIED {len(results)}/10: {r['episode']} ({r['seconds']:.1f} seconds)",flush=True)
    manifest['videos']=results
    (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('All ten videos rendered and verified.',flush=True)


if __name__=='__main__': main()
