import h5py
import numpy as np
import os

in_name = 'SPICE-1.1.3.hdf5'
out_dir = 'data0.25/'
train_name = 'spice_train.hdf5'
val_name = 'spice_val.hdf5'
test_name = 'spice_test.hdf5'
F_cutoff = 0.25
val_ratio = 0.05
test_ratio = 0.05

if not os.path.exists(out_dir):
    os.mkdir(out_dir)
fin = h5py.File(in_name, 'r')
ft = h5py.File(out_dir+train_name, 'w')
fv = h5py.File(out_dir+val_name, 'w')
fe = h5py.File(out_dir+test_name, 'w')
log = open(out_dir+'log.txt', 'w')

tottrain = 0
totval = 0
tottest = 0

for mm in fin:
    raw_num = fin[mm]['dft_total_energy'].shape[0]
    inds = [i for i,f in enumerate(fin[mm]['dft_total_gradient']) if (f<F_cutoff).all()]
    num = len(inds)
    nval = int(np.floor(num*val_ratio))
    nval += int(val_ratio*num-nval>np.random.rand())
    ntest = int(np.floor(num*test_ratio))
    ntest += int(test_ratio*num-ntest>np.random.rand())
    val_inds = np.random.choice(inds, size=nval, replace=False)
    val_inds.sort()
    inds = [i for i in inds if i not in val_inds]
    test_inds = np.random.choice(inds, size=ntest, replace=False)
    test_inds.sort()
    inds = [i for i in inds if i not in test_inds]
    tottrain += len(inds)
    totval += nval
    tottest += ntest
    # print(inds,val_inds,test_inds)
    # print(len(inds),nval,ntest)
    log.write(mm+'\t'+'\t'.join([str(i) for i in [len(inds),nval,ntest]])+'\n')
    log.flush()
    for f,i in zip([ft,fv,fe],[inds,val_inds,test_inds]):
        if len(i)>0:
            f.create_group(mm)
            f[mm]['atomic_numbers'] = list(fin[mm]['atomic_numbers'])
            f[mm]['dft_total_energy'] = fin[mm]['dft_total_energy'][i]
            f[mm]['formation_energy'] = fin[mm]['formation_energy'][i]
            f[mm]['conformations'] = fin[mm]['conformations'][i]
            f[mm]['dft_total_gradient'] = fin[mm]['dft_total_gradient'][i]

log.write('-------------\n')
log.write(' '.join(str(i) for i in [tottrain,totval,tottest])+'\n')
tot = tottrain+totval+tottest
log.write(' '.join(str(i) for i in [100*tottrain/tot,100*totval/tot,100*tottest/tot])+'\n')
print('Done')
ft.close()
fv.close()
fe.close()
log.close()
