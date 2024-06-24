from M_validation_analysis import *
import os
import sys

#p = sys.argv[0]
p = os.getcwd()
print('reading from : ', p,flush  = True)
dirs = ["id/", "ood_ads/", "ood_cat/", "ood_both/"]
sub_dirs = [str(i)+'/' for i in range(1,5)]
epoch = 405
inst = Read_domains_validation(p+'/', dirs, sub_dirs, epoch)
model_validation = inst.get_error()
