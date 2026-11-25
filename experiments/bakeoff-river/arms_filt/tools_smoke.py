import sys, time; sys.path.insert(0,'/home/claude/bakeoff/river/harness'); sys.path.insert(0,'/home/claude/bakeoff/river/arms_filt/enkf_watersim')
import numpy as np, bake, score as S
t0=time.process_time(); import predict as P; print('import', time.process_time()-t0)
B=bake.load_bank('val',['F','qo','attrs','st','lat','lon','start'])
for L in (1,3,7):
  for i in (0,1):
    b=bake.sim_window(B,i,0,365,'gauged',L); y=bake.sim_target(B,i,b)
    t0=time.process_time(); o=P.predict(b); dt=time.process_time()-t0
    r=S.simval_scores(y,o['mean'],np.stack([o[k] for k in ('q05','q25','q50','q75','q95')],1))
    print('sim',i,'L',L,'nse %.3f cov90 %.2f'%(r['nse'],r['cov90']),'%.2fs'%dt)
bid=bake.dev_basins()[0]
for L in (1,3,7):
  b=bake.dev_batch(bid,'gauged',L); t0=time.process_time(); o=P.predict(b); dt=time.process_time()-t0
  obs=bake._obs(bid,'dev'); r=S.score_real(obs,o['mean'],np.stack([o[k] for k in ('q05','q25','q50','q75','q95')],1))
  print('dev',bid,'L',L,'nse %.3f kge %.3f cov90 %.2f'%(r['nse'],r['kge'],r['cov90']),'%.2fs'%dt)
