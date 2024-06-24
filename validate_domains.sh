#!/bin/bash

Train_dir="$(pwd)"
mkdir validation
cp sample.job train.ini  *pkl validation
cd validation  

d='#SBATCH -A'
p='#SBATCH -A CNHPC_1491920'
sed -i "s%$d.*%$p%g"  sample.job

j='#SBATCH --time'
k='#SBATCH --time 4:49:00'
sed -i "s%$j.*%$k%g"  sample.job
x='#SBATCH --job-name='
y='#SBATCH --job-name=val_domains'
sed -i "s%$x.*%$y%g"  sample.job



a='train_dir = .'
b='train_dir = '
sed -i "s%$a%$b$Train_dir%g"  train.ini 

c='val_data_dir'
v='val_data_dir = /leonardo_scratch/large/userexternal/mtaleblo/Dataset/2M/all_validation_data/'
m='all_npy/'
sed -i "s%$c.*%$v%g" train.ini 
sed -i '/^.*DATA_INFORMATION.*/i task=val\nval_dir=./val\n'  train.ini     

dirs="id/   ood_ads/  ood_cat/  ood_both/" 

for dir in $dirs 
do
       	mkdir $dir 
       	cp sample.job train.ini *pkl  $dir 
       	cd $dir 
       	sed -i "s%$v%$v$dir$m%g"  train.ini
        echo $dir
        for i in {1..4}
        do
                mkdir $i
                cp sample.job train.ini *pkl  $i
                cd $i
                sed -i "s%$v$dir$m%$v$dir$m$i%g"  train.ini
	        echo $(pwd)
                sbatch sample.job
                cd ../
        done
        
       	cd ../  
done


# for dir in $dirs 
# do
#        	cd $dir
# 	echo $dir
#       	for i in {1..4}
#        	do
# 	       	mkdir $i
# 	      	cp sample.job train.ini  $i
# 	       	cd $i
# 	       	sed -i "s%$v$dir$m%$v$dir$m$i%g"  train.ini
# 	        sbatch sample.job	
# 	       	echo $i
# 	        cd ../	
#        	done 
#        	cd ../   
# done
# 
