#!/bin/bash 

PRE=s 
VLEN=16 
SDIM=$VLEN 
EDIM=128 
IB=64
NTHDS=
RBLK=
KRUNTIME=
BESTK=64
#commandline argument 
usage="Usage: $0 [OPTION] ... 
Options: 
-v [val] 	value of vector width (vlen), see simd.h to find system vlen
-s [val]	starting value of dimension, must be multiple of vlen
-e [val]	ending value of dimension, must be multiple of vlen
-p [s,d]	precision of floating point, s for single precision, d for double precision
-i [32,64]	precision of int 
-t [val]	number of threads 
-r [crb,acrb,bacrb]	register blocking  
-k [0,1]	is kruntime ? 1 or 0 
-b [val]        best K (DIM) value, needed when kruntime=1, -s & -e skipped then
--help 		display help and exit 
"

while getopts "v:i:s:e:p:t:r:k:b:" opt
do
   case $opt in 
      v) 
         VLEN=$OPTARG
         ;; 
      i) 
         IB=$OPTARG
         ;; 
      s) 
         SDIM=$OPTARG
         ;; 
      e) 
         EDIM=$OPTARG
         ;; 
      p) 
         PRE=$OPTARG
         ;;
      t) 
         NTHDS=$OPTARG
         ;;
      r) 
         RBLK=$OPTARG
         ;;
      k) 
         KRUNTIME=$OPTARG
         ;;
      b) 
         BESTK=$OPTARG
         ;;
      \?)
         echo "$usage"
         exit 1 
         ;;
   esac
done


mkdir -p bin 
mkdir -p generated 
mkdir -p generated/src 
mkdir -p generated/include 

make clean 

#
#  When kruntime=1, we will generate kernel upto the bestK with a interval of 
#  vlen. When any runtime K (dim) is greater than bestK, we use register blocking 
#  as bestK and keep the remianing k-iteration rolled. 
#
if [ $KRUNTIME -eq 1 ]
then
   SDIM=$VLEN
   EDIM=$BESTK
fi

#
#  Assumption: SDIM, EDIM and BESTK all are multiple of VLEN.. need an assert
#

#generate header 
make header pre=$PRE vlen=$VLEN mdim=$EDIM ibit=$IB kruntime=$KRUNTIME bestK=$BESTK 

#generate Makefile 
make gmakefile pre=$PRE vlen=$VLEN mdim=$EDIM ibit=$IB nthds=$NTHDS 

# generate all kernels, but last one 
echo "Generating kernels in directory: " $GENdir 
echo "===========================================" 
for (( d=$SDIM; d < $EDIM; d=$d+$VLEN ))
{
   make srcfile pre=$PRE vlen=$VLEN dim=$d ibit=$IB regblk=$RBLK kruntime=0
}

#
#  generate last kernel with kruntime=1 
#
if [ $KRUNTIME -eq 1 ]
then
   make srcfile pre=$PRE vlen=$VLEN dim=$BESTK ibit=$IB regblk=$RBLK kruntime=1
else
   make srcfile pre=$PRE vlen=$VLEN dim=$EDIM ibit=$IB regblk=$RBLK kruntime=0
fi

# build the static library 
make staticlibs 
