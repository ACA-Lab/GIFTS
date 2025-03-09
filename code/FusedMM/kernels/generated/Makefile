include ../Make.inc 
ibit=64
IFLAGS = -DINDEXTYPE=int$(ibit)_t
OMPFLAGS = -fopenmp
PTFLAGS = $(OMPFLAGS) -DPTTIME -DLDB -DNTHREADS=32
SFLAGS = 
INC=$(INCSdir)/kernels.h 
#generated headers 
gcnGINC=$(GENINCdir)/sgkernels_gcn.h  
spmmGINC=$(GENINCdir)/sgkernels_spmm.h  
sigmoidGINC=$(GENINCdir)/sgkernels_sigmoid.h  
tdistGINC=$(GENINCdir)/sgkernels_tdist.h  
ARCHIVER = ar 
ARFLAGS = r
RANLIB = echo 
SDMMlib = $(LIBdir)/slibgfusedMM_sequential.a 
SDMMlib_pt = $(LIBdir)/slibgfusedMM_pt.a 

all: slib slib_pt 

sobj = $(BINdir)/skernels.o $(GENSRCdir)/sgfusedMM_K16_gcn_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K16_gcn_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K32_gcn_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K32_gcn_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K48_gcn_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K48_gcn_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K64_gcn_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K64_gcn_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K80_gcn_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K80_gcn_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K96_gcn_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K96_gcn_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K112_gcn_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K112_gcn_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K128_gcn_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K128_gcn_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K144_gcn_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K144_gcn_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K160_gcn_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K160_gcn_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K176_gcn_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K176_gcn_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K192_gcn_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K192_gcn_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K208_gcn_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K208_gcn_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K224_gcn_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K224_gcn_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K240_gcn_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K240_gcn_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K256_gcn_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K256_gcn_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K272_gcn_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K272_gcn_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K288_gcn_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K288_gcn_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K304_gcn_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K304_gcn_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K320_gcn_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K320_gcn_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K336_gcn_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K336_gcn_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K352_gcn_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K352_gcn_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K368_gcn_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K368_gcn_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K384_gcn_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K384_gcn_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K400_gcn_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K400_gcn_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K416_gcn_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K416_gcn_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K432_gcn_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K432_gcn_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K448_gcn_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K448_gcn_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K464_gcn_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K464_gcn_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K480_gcn_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K480_gcn_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K496_gcn_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K496_gcn_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K512_gcn_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K512_gcn_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K16_spmm_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K16_spmm_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K32_spmm_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K32_spmm_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K48_spmm_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K48_spmm_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K64_spmm_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K64_spmm_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K80_spmm_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K80_spmm_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K96_spmm_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K96_spmm_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K112_spmm_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K112_spmm_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K128_spmm_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K128_spmm_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K144_spmm_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K144_spmm_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K160_spmm_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K160_spmm_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K176_spmm_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K176_spmm_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K192_spmm_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K192_spmm_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K208_spmm_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K208_spmm_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K224_spmm_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K224_spmm_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K240_spmm_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K240_spmm_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K256_spmm_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K256_spmm_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K272_spmm_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K272_spmm_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K288_spmm_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K288_spmm_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K304_spmm_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K304_spmm_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K320_spmm_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K320_spmm_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K336_spmm_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K336_spmm_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K352_spmm_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K352_spmm_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K368_spmm_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K368_spmm_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K384_spmm_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K384_spmm_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K400_spmm_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K400_spmm_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K416_spmm_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K416_spmm_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K432_spmm_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K432_spmm_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K448_spmm_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K448_spmm_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K464_spmm_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K464_spmm_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K480_spmm_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K480_spmm_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K496_spmm_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K496_spmm_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K512_spmm_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K512_spmm_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K16_sigmoid_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K16_sigmoid_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K32_sigmoid_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K32_sigmoid_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K48_sigmoid_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K48_sigmoid_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K64_sigmoid_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K64_sigmoid_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K80_sigmoid_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K80_sigmoid_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K96_sigmoid_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K96_sigmoid_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K112_sigmoid_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K112_sigmoid_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K128_sigmoid_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K128_sigmoid_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K144_sigmoid_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K144_sigmoid_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K160_sigmoid_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K160_sigmoid_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K176_sigmoid_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K176_sigmoid_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K192_sigmoid_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K192_sigmoid_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K208_sigmoid_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K208_sigmoid_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K224_sigmoid_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K224_sigmoid_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K240_sigmoid_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K240_sigmoid_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K256_sigmoid_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K256_sigmoid_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K272_sigmoid_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K272_sigmoid_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K288_sigmoid_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K288_sigmoid_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K304_sigmoid_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K304_sigmoid_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K320_sigmoid_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K320_sigmoid_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K336_sigmoid_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K336_sigmoid_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K352_sigmoid_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K352_sigmoid_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K368_sigmoid_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K368_sigmoid_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K384_sigmoid_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K384_sigmoid_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K400_sigmoid_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K400_sigmoid_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K416_sigmoid_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K416_sigmoid_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K432_sigmoid_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K432_sigmoid_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K448_sigmoid_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K448_sigmoid_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K464_sigmoid_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K464_sigmoid_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K480_sigmoid_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K480_sigmoid_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K496_sigmoid_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K496_sigmoid_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K512_sigmoid_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K512_sigmoid_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K16_tdist_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K16_tdist_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K32_tdist_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K32_tdist_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K48_tdist_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K48_tdist_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K64_tdist_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K64_tdist_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K80_tdist_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K80_tdist_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K96_tdist_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K96_tdist_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K112_tdist_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K112_tdist_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K128_tdist_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K128_tdist_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K144_tdist_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K144_tdist_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K160_tdist_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K160_tdist_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K176_tdist_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K176_tdist_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K192_tdist_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K192_tdist_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K208_tdist_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K208_tdist_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K224_tdist_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K224_tdist_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K240_tdist_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K240_tdist_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K256_tdist_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K256_tdist_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K272_tdist_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K272_tdist_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K288_tdist_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K288_tdist_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K304_tdist_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K304_tdist_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K320_tdist_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K320_tdist_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K336_tdist_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K336_tdist_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K352_tdist_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K352_tdist_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K368_tdist_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K368_tdist_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K384_tdist_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K384_tdist_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K400_tdist_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K400_tdist_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K416_tdist_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K416_tdist_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K432_tdist_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K432_tdist_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K448_tdist_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K448_tdist_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K464_tdist_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K464_tdist_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K480_tdist_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K480_tdist_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K496_tdist_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K496_tdist_b1_csr.o \
       $(GENSRCdir)/sgfusedMM_K512_tdist_b0_csr.o \
       $(GENSRCdir)/sgfusedMM_K512_tdist_b1_csr.o

slib: $(LIBdir)/slib.grd 

$(LIBdir)/slib.grd: $(sobj)
	$(ARCHIVER) $(ARFLAGS) $(SDMMlib) $(sobj) 
	$(RANLIB) $(SDMMlib)  
	touch $(LIBdir)/slib.grd 

$(BINdir)/skernels.o : $(SRCdir)/kernels.c 
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -I$(GENINCdir) -o $@ -c $(SRCdir)/kernels.c 

$(GENSRCdir)/sgfusedMM_K16_gcn_b0_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K16_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K16_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K16_gcn_b1_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K16_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K16_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K32_gcn_b0_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K32_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K32_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K32_gcn_b1_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K32_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K32_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K48_gcn_b0_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K48_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K48_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K48_gcn_b1_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K48_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K48_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K64_gcn_b0_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K64_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K64_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K64_gcn_b1_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K64_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K64_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K80_gcn_b0_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K80_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K80_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K80_gcn_b1_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K80_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K80_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K96_gcn_b0_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K96_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K96_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K96_gcn_b1_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K96_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K96_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K112_gcn_b0_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K112_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K112_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K112_gcn_b1_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K112_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K112_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K128_gcn_b0_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K128_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K128_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K128_gcn_b1_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K128_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K128_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K144_gcn_b0_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K144_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K144_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K144_gcn_b1_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K144_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K144_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K160_gcn_b0_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K160_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K160_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K160_gcn_b1_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K160_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K160_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K176_gcn_b0_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K176_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K176_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K176_gcn_b1_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K176_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K176_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K192_gcn_b0_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K192_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K192_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K192_gcn_b1_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K192_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K192_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K208_gcn_b0_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K208_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K208_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K208_gcn_b1_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K208_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K208_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K224_gcn_b0_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K224_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K224_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K224_gcn_b1_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K224_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K224_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K240_gcn_b0_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K240_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K240_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K240_gcn_b1_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K240_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K240_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K256_gcn_b0_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K256_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K256_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K256_gcn_b1_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K256_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K256_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K272_gcn_b0_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K272_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K272_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K272_gcn_b1_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K272_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K272_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K288_gcn_b0_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K288_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K288_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K288_gcn_b1_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K288_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K288_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K304_gcn_b0_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K304_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K304_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K304_gcn_b1_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K304_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K304_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K320_gcn_b0_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K320_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K320_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K320_gcn_b1_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K320_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K320_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K336_gcn_b0_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K336_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K336_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K336_gcn_b1_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K336_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K336_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K352_gcn_b0_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K352_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K352_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K352_gcn_b1_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K352_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K352_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K368_gcn_b0_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K368_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K368_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K368_gcn_b1_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K368_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K368_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K384_gcn_b0_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K384_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K384_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K384_gcn_b1_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K384_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K384_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K400_gcn_b0_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K400_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K400_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K400_gcn_b1_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K400_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K400_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K416_gcn_b0_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K416_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K416_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K416_gcn_b1_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K416_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K416_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K432_gcn_b0_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K432_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K432_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K432_gcn_b1_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K432_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K432_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K448_gcn_b0_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K448_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K448_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K448_gcn_b1_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K448_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K448_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K464_gcn_b0_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K464_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K464_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K464_gcn_b1_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K464_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K464_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K480_gcn_b0_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K480_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K480_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K480_gcn_b1_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K480_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K480_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K496_gcn_b0_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K496_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K496_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K496_gcn_b1_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K496_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K496_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K512_gcn_b0_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K512_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K512_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K512_gcn_b1_csr.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K512_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K512_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K16_spmm_b0_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K16_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K16_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K16_spmm_b1_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K16_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K16_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K32_spmm_b0_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K32_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K32_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K32_spmm_b1_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K32_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K32_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K48_spmm_b0_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K48_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K48_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K48_spmm_b1_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K48_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K48_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K64_spmm_b0_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K64_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K64_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K64_spmm_b1_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K64_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K64_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K80_spmm_b0_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K80_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K80_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K80_spmm_b1_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K80_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K80_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K96_spmm_b0_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K96_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K96_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K96_spmm_b1_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K96_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K96_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K112_spmm_b0_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K112_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K112_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K112_spmm_b1_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K112_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K112_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K128_spmm_b0_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K128_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K128_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K128_spmm_b1_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K128_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K128_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K144_spmm_b0_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K144_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K144_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K144_spmm_b1_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K144_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K144_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K160_spmm_b0_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K160_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K160_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K160_spmm_b1_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K160_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K160_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K176_spmm_b0_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K176_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K176_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K176_spmm_b1_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K176_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K176_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K192_spmm_b0_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K192_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K192_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K192_spmm_b1_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K192_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K192_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K208_spmm_b0_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K208_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K208_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K208_spmm_b1_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K208_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K208_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K224_spmm_b0_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K224_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K224_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K224_spmm_b1_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K224_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K224_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K240_spmm_b0_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K240_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K240_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K240_spmm_b1_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K240_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K240_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K256_spmm_b0_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K256_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K256_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K256_spmm_b1_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K256_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K256_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K272_spmm_b0_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K272_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K272_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K272_spmm_b1_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K272_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K272_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K288_spmm_b0_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K288_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K288_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K288_spmm_b1_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K288_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K288_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K304_spmm_b0_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K304_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K304_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K304_spmm_b1_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K304_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K304_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K320_spmm_b0_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K320_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K320_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K320_spmm_b1_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K320_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K320_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K336_spmm_b0_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K336_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K336_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K336_spmm_b1_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K336_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K336_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K352_spmm_b0_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K352_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K352_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K352_spmm_b1_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K352_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K352_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K368_spmm_b0_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K368_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K368_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K368_spmm_b1_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K368_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K368_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K384_spmm_b0_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K384_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K384_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K384_spmm_b1_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K384_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K384_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K400_spmm_b0_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K400_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K400_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K400_spmm_b1_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K400_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K400_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K416_spmm_b0_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K416_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K416_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K416_spmm_b1_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K416_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K416_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K432_spmm_b0_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K432_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K432_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K432_spmm_b1_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K432_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K432_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K448_spmm_b0_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K448_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K448_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K448_spmm_b1_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K448_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K448_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K464_spmm_b0_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K464_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K464_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K464_spmm_b1_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K464_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K464_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K480_spmm_b0_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K480_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K480_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K480_spmm_b1_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K480_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K480_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K496_spmm_b0_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K496_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K496_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K496_spmm_b1_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K496_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K496_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K512_spmm_b0_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K512_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K512_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K512_spmm_b1_csr.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K512_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K512_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K16_sigmoid_b0_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K16_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K16_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K16_sigmoid_b1_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K16_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K16_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K32_sigmoid_b0_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K32_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K32_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K32_sigmoid_b1_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K32_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K32_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K48_sigmoid_b0_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K48_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K48_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K48_sigmoid_b1_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K48_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K48_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K64_sigmoid_b0_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K64_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K64_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K64_sigmoid_b1_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K64_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K64_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K80_sigmoid_b0_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K80_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K80_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K80_sigmoid_b1_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K80_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K80_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K96_sigmoid_b0_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K96_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K96_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K96_sigmoid_b1_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K96_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K96_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K112_sigmoid_b0_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K112_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K112_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K112_sigmoid_b1_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K112_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K112_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K128_sigmoid_b0_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K128_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K128_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K128_sigmoid_b1_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K128_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K128_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K144_sigmoid_b0_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K144_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K144_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K144_sigmoid_b1_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K144_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K144_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K160_sigmoid_b0_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K160_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K160_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K160_sigmoid_b1_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K160_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K160_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K176_sigmoid_b0_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K176_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K176_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K176_sigmoid_b1_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K176_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K176_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K192_sigmoid_b0_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K192_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K192_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K192_sigmoid_b1_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K192_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K192_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K208_sigmoid_b0_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K208_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K208_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K208_sigmoid_b1_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K208_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K208_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K224_sigmoid_b0_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K224_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K224_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K224_sigmoid_b1_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K224_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K224_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K240_sigmoid_b0_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K240_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K240_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K240_sigmoid_b1_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K240_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K240_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K256_sigmoid_b0_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K256_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K256_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K256_sigmoid_b1_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K256_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K256_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K272_sigmoid_b0_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K272_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K272_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K272_sigmoid_b1_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K272_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K272_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K288_sigmoid_b0_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K288_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K288_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K288_sigmoid_b1_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K288_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K288_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K304_sigmoid_b0_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K304_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K304_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K304_sigmoid_b1_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K304_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K304_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K320_sigmoid_b0_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K320_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K320_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K320_sigmoid_b1_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K320_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K320_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K336_sigmoid_b0_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K336_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K336_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K336_sigmoid_b1_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K336_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K336_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K352_sigmoid_b0_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K352_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K352_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K352_sigmoid_b1_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K352_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K352_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K368_sigmoid_b0_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K368_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K368_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K368_sigmoid_b1_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K368_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K368_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K384_sigmoid_b0_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K384_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K384_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K384_sigmoid_b1_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K384_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K384_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K400_sigmoid_b0_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K400_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K400_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K400_sigmoid_b1_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K400_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K400_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K416_sigmoid_b0_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K416_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K416_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K416_sigmoid_b1_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K416_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K416_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K432_sigmoid_b0_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K432_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K432_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K432_sigmoid_b1_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K432_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K432_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K448_sigmoid_b0_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K448_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K448_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K448_sigmoid_b1_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K448_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K448_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K464_sigmoid_b0_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K464_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K464_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K464_sigmoid_b1_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K464_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K464_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K480_sigmoid_b0_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K480_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K480_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K480_sigmoid_b1_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K480_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K480_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K496_sigmoid_b0_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K496_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K496_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K496_sigmoid_b1_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K496_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K496_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K512_sigmoid_b0_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K512_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K512_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K512_sigmoid_b1_csr.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K512_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K512_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K16_tdist_b0_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K16_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K16_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K16_tdist_b1_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K16_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K16_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K32_tdist_b0_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K32_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K32_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K32_tdist_b1_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K32_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K32_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K48_tdist_b0_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K48_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K48_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K48_tdist_b1_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K48_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K48_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K64_tdist_b0_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K64_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K64_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K64_tdist_b1_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K64_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K64_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K80_tdist_b0_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K80_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K80_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K80_tdist_b1_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K80_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K80_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K96_tdist_b0_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K96_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K96_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K96_tdist_b1_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K96_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K96_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K112_tdist_b0_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K112_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K112_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K112_tdist_b1_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K112_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K112_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K128_tdist_b0_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K128_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K128_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K128_tdist_b1_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K128_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K128_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K144_tdist_b0_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K144_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K144_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K144_tdist_b1_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K144_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K144_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K160_tdist_b0_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K160_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K160_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K160_tdist_b1_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K160_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K160_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K176_tdist_b0_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K176_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K176_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K176_tdist_b1_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K176_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K176_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K192_tdist_b0_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K192_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K192_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K192_tdist_b1_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K192_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K192_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K208_tdist_b0_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K208_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K208_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K208_tdist_b1_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K208_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K208_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K224_tdist_b0_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K224_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K224_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K224_tdist_b1_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K224_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K224_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K240_tdist_b0_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K240_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K240_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K240_tdist_b1_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K240_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K240_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K256_tdist_b0_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K256_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K256_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K256_tdist_b1_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K256_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K256_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K272_tdist_b0_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K272_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K272_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K272_tdist_b1_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K272_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K272_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K288_tdist_b0_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K288_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K288_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K288_tdist_b1_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K288_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K288_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K304_tdist_b0_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K304_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K304_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K304_tdist_b1_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K304_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K304_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K320_tdist_b0_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K320_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K320_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K320_tdist_b1_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K320_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K320_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K336_tdist_b0_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K336_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K336_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K336_tdist_b1_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K336_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K336_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K352_tdist_b0_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K352_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K352_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K352_tdist_b1_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K352_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K352_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K368_tdist_b0_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K368_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K368_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K368_tdist_b1_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K368_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K368_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K384_tdist_b0_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K384_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K384_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K384_tdist_b1_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K384_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K384_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K400_tdist_b0_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K400_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K400_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K400_tdist_b1_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K400_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K400_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K416_tdist_b0_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K416_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K416_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K416_tdist_b1_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K416_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K416_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K432_tdist_b0_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K432_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K432_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K432_tdist_b1_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K432_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K432_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K448_tdist_b0_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K448_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K448_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K448_tdist_b1_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K448_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K448_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K464_tdist_b0_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K464_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K464_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K464_tdist_b1_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K464_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K464_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K480_tdist_b0_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K480_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K480_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K480_tdist_b1_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K480_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K480_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K496_tdist_b0_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K496_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K496_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K496_tdist_b1_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K496_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K496_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K512_tdist_b0_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K512_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K512_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K512_tdist_b1_csr.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K512_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(SFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K512_tdist_csr.c

sobj_pt = $(BINdir)/skernels_pt.o $(GENSRCdir)/sgfusedMM_K16_gcn_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K16_gcn_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K32_gcn_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K32_gcn_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K48_gcn_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K48_gcn_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K64_gcn_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K64_gcn_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K80_gcn_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K80_gcn_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K96_gcn_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K96_gcn_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K112_gcn_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K112_gcn_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K128_gcn_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K128_gcn_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K144_gcn_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K144_gcn_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K160_gcn_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K160_gcn_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K176_gcn_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K176_gcn_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K192_gcn_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K192_gcn_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K208_gcn_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K208_gcn_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K224_gcn_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K224_gcn_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K240_gcn_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K240_gcn_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K256_gcn_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K256_gcn_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K272_gcn_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K272_gcn_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K288_gcn_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K288_gcn_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K304_gcn_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K304_gcn_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K320_gcn_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K320_gcn_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K336_gcn_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K336_gcn_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K352_gcn_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K352_gcn_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K368_gcn_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K368_gcn_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K384_gcn_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K384_gcn_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K400_gcn_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K400_gcn_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K416_gcn_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K416_gcn_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K432_gcn_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K432_gcn_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K448_gcn_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K448_gcn_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K464_gcn_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K464_gcn_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K480_gcn_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K480_gcn_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K496_gcn_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K496_gcn_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K512_gcn_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K512_gcn_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K16_spmm_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K16_spmm_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K32_spmm_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K32_spmm_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K48_spmm_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K48_spmm_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K64_spmm_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K64_spmm_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K80_spmm_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K80_spmm_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K96_spmm_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K96_spmm_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K112_spmm_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K112_spmm_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K128_spmm_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K128_spmm_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K144_spmm_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K144_spmm_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K160_spmm_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K160_spmm_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K176_spmm_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K176_spmm_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K192_spmm_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K192_spmm_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K208_spmm_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K208_spmm_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K224_spmm_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K224_spmm_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K240_spmm_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K240_spmm_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K256_spmm_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K256_spmm_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K272_spmm_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K272_spmm_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K288_spmm_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K288_spmm_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K304_spmm_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K304_spmm_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K320_spmm_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K320_spmm_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K336_spmm_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K336_spmm_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K352_spmm_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K352_spmm_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K368_spmm_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K368_spmm_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K384_spmm_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K384_spmm_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K400_spmm_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K400_spmm_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K416_spmm_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K416_spmm_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K432_spmm_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K432_spmm_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K448_spmm_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K448_spmm_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K464_spmm_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K464_spmm_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K480_spmm_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K480_spmm_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K496_spmm_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K496_spmm_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K512_spmm_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K512_spmm_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K16_sigmoid_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K16_sigmoid_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K32_sigmoid_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K32_sigmoid_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K48_sigmoid_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K48_sigmoid_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K64_sigmoid_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K64_sigmoid_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K80_sigmoid_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K80_sigmoid_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K96_sigmoid_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K96_sigmoid_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K112_sigmoid_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K112_sigmoid_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K128_sigmoid_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K128_sigmoid_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K144_sigmoid_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K144_sigmoid_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K160_sigmoid_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K160_sigmoid_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K176_sigmoid_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K176_sigmoid_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K192_sigmoid_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K192_sigmoid_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K208_sigmoid_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K208_sigmoid_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K224_sigmoid_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K224_sigmoid_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K240_sigmoid_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K240_sigmoid_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K256_sigmoid_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K256_sigmoid_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K272_sigmoid_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K272_sigmoid_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K288_sigmoid_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K288_sigmoid_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K304_sigmoid_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K304_sigmoid_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K320_sigmoid_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K320_sigmoid_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K336_sigmoid_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K336_sigmoid_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K352_sigmoid_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K352_sigmoid_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K368_sigmoid_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K368_sigmoid_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K384_sigmoid_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K384_sigmoid_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K400_sigmoid_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K400_sigmoid_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K416_sigmoid_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K416_sigmoid_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K432_sigmoid_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K432_sigmoid_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K448_sigmoid_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K448_sigmoid_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K464_sigmoid_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K464_sigmoid_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K480_sigmoid_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K480_sigmoid_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K496_sigmoid_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K496_sigmoid_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K512_sigmoid_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K512_sigmoid_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K16_tdist_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K16_tdist_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K32_tdist_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K32_tdist_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K48_tdist_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K48_tdist_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K64_tdist_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K64_tdist_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K80_tdist_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K80_tdist_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K96_tdist_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K96_tdist_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K112_tdist_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K112_tdist_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K128_tdist_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K128_tdist_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K144_tdist_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K144_tdist_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K160_tdist_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K160_tdist_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K176_tdist_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K176_tdist_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K192_tdist_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K192_tdist_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K208_tdist_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K208_tdist_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K224_tdist_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K224_tdist_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K240_tdist_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K240_tdist_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K256_tdist_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K256_tdist_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K272_tdist_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K272_tdist_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K288_tdist_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K288_tdist_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K304_tdist_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K304_tdist_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K320_tdist_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K320_tdist_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K336_tdist_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K336_tdist_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K352_tdist_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K352_tdist_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K368_tdist_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K368_tdist_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K384_tdist_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K384_tdist_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K400_tdist_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K400_tdist_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K416_tdist_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K416_tdist_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K432_tdist_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K432_tdist_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K448_tdist_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K448_tdist_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K464_tdist_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K464_tdist_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K480_tdist_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K480_tdist_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K496_tdist_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K496_tdist_b1_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K512_tdist_b0_csr_pt.o \
          $(GENSRCdir)/sgfusedMM_K512_tdist_b1_csr_pt.o

slib_pt: $(LIBdir)/slib_pt.grd 

$(LIBdir)/slib_pt.grd: $(sobj_pt)
	$(ARCHIVER) $(ARFLAGS) $(SDMMlib_pt) $(sobj_pt) 
	$(RANLIB) $(SDMMlib_pt)  
	touch $(LIBdir)/slib_pt.grd 

$(BINdir)/skernels_pt.o : $(SRCdir)/kernels.c 
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -I$(GENINCdir) -o $@ -c $(SRCdir)/kernels.c 

$(GENSRCdir)/sgfusedMM_K16_gcn_b0_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K16_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K16_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K16_gcn_b1_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K16_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K16_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K32_gcn_b0_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K32_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K32_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K32_gcn_b1_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K32_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K32_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K48_gcn_b0_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K48_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K48_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K48_gcn_b1_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K48_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K48_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K64_gcn_b0_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K64_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K64_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K64_gcn_b1_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K64_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K64_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K80_gcn_b0_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K80_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K80_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K80_gcn_b1_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K80_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K80_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K96_gcn_b0_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K96_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K96_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K96_gcn_b1_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K96_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K96_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K112_gcn_b0_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K112_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K112_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K112_gcn_b1_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K112_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K112_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K128_gcn_b0_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K128_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K128_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K128_gcn_b1_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K128_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K128_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K144_gcn_b0_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K144_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K144_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K144_gcn_b1_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K144_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K144_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K160_gcn_b0_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K160_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K160_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K160_gcn_b1_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K160_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K160_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K176_gcn_b0_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K176_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K176_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K176_gcn_b1_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K176_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K176_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K192_gcn_b0_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K192_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K192_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K192_gcn_b1_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K192_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K192_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K208_gcn_b0_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K208_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K208_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K208_gcn_b1_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K208_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K208_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K224_gcn_b0_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K224_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K224_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K224_gcn_b1_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K224_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K224_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K240_gcn_b0_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K240_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K240_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K240_gcn_b1_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K240_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K240_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K256_gcn_b0_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K256_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K256_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K256_gcn_b1_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K256_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K256_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K272_gcn_b0_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K272_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K272_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K272_gcn_b1_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K272_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K272_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K288_gcn_b0_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K288_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K288_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K288_gcn_b1_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K288_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K288_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K304_gcn_b0_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K304_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K304_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K304_gcn_b1_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K304_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K304_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K320_gcn_b0_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K320_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K320_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K320_gcn_b1_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K320_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K320_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K336_gcn_b0_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K336_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K336_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K336_gcn_b1_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K336_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K336_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K352_gcn_b0_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K352_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K352_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K352_gcn_b1_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K352_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K352_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K368_gcn_b0_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K368_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K368_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K368_gcn_b1_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K368_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K368_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K384_gcn_b0_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K384_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K384_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K384_gcn_b1_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K384_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K384_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K400_gcn_b0_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K400_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K400_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K400_gcn_b1_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K400_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K400_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K416_gcn_b0_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K416_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K416_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K416_gcn_b1_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K416_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K416_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K432_gcn_b0_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K432_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K432_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K432_gcn_b1_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K432_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K432_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K448_gcn_b0_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K448_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K448_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K448_gcn_b1_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K448_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K448_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K464_gcn_b0_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K464_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K464_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K464_gcn_b1_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K464_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K464_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K480_gcn_b0_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K480_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K480_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K480_gcn_b1_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K480_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K480_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K496_gcn_b0_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K496_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K496_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K496_gcn_b1_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K496_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K496_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K512_gcn_b0_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K512_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K512_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K512_gcn_b1_csr_pt.o : $(gcnGINC) \
   $(GENSRCdir)/sgfusedMM_K512_gcn_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K512_gcn_csr.c
$(GENSRCdir)/sgfusedMM_K16_spmm_b0_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K16_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K16_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K16_spmm_b1_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K16_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K16_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K32_spmm_b0_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K32_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K32_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K32_spmm_b1_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K32_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K32_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K48_spmm_b0_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K48_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K48_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K48_spmm_b1_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K48_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K48_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K64_spmm_b0_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K64_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K64_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K64_spmm_b1_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K64_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K64_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K80_spmm_b0_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K80_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K80_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K80_spmm_b1_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K80_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K80_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K96_spmm_b0_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K96_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K96_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K96_spmm_b1_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K96_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K96_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K112_spmm_b0_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K112_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K112_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K112_spmm_b1_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K112_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K112_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K128_spmm_b0_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K128_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K128_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K128_spmm_b1_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K128_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K128_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K144_spmm_b0_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K144_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K144_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K144_spmm_b1_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K144_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K144_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K160_spmm_b0_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K160_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K160_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K160_spmm_b1_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K160_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K160_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K176_spmm_b0_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K176_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K176_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K176_spmm_b1_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K176_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K176_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K192_spmm_b0_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K192_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K192_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K192_spmm_b1_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K192_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K192_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K208_spmm_b0_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K208_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K208_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K208_spmm_b1_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K208_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K208_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K224_spmm_b0_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K224_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K224_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K224_spmm_b1_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K224_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K224_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K240_spmm_b0_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K240_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K240_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K240_spmm_b1_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K240_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K240_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K256_spmm_b0_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K256_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K256_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K256_spmm_b1_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K256_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K256_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K272_spmm_b0_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K272_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K272_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K272_spmm_b1_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K272_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K272_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K288_spmm_b0_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K288_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K288_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K288_spmm_b1_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K288_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K288_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K304_spmm_b0_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K304_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K304_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K304_spmm_b1_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K304_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K304_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K320_spmm_b0_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K320_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K320_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K320_spmm_b1_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K320_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K320_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K336_spmm_b0_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K336_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K336_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K336_spmm_b1_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K336_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K336_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K352_spmm_b0_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K352_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K352_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K352_spmm_b1_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K352_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K352_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K368_spmm_b0_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K368_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K368_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K368_spmm_b1_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K368_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K368_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K384_spmm_b0_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K384_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K384_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K384_spmm_b1_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K384_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K384_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K400_spmm_b0_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K400_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K400_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K400_spmm_b1_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K400_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K400_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K416_spmm_b0_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K416_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K416_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K416_spmm_b1_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K416_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K416_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K432_spmm_b0_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K432_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K432_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K432_spmm_b1_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K432_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K432_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K448_spmm_b0_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K448_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K448_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K448_spmm_b1_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K448_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K448_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K464_spmm_b0_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K464_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K464_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K464_spmm_b1_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K464_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K464_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K480_spmm_b0_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K480_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K480_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K480_spmm_b1_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K480_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K480_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K496_spmm_b0_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K496_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K496_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K496_spmm_b1_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K496_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K496_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K512_spmm_b0_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K512_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K512_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K512_spmm_b1_csr_pt.o : $(spmmGINC) \
   $(GENSRCdir)/sgfusedMM_K512_spmm_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K512_spmm_csr.c
$(GENSRCdir)/sgfusedMM_K16_sigmoid_b0_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K16_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K16_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K16_sigmoid_b1_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K16_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K16_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K32_sigmoid_b0_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K32_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K32_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K32_sigmoid_b1_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K32_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K32_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K48_sigmoid_b0_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K48_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K48_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K48_sigmoid_b1_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K48_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K48_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K64_sigmoid_b0_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K64_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K64_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K64_sigmoid_b1_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K64_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K64_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K80_sigmoid_b0_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K80_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K80_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K80_sigmoid_b1_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K80_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K80_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K96_sigmoid_b0_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K96_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K96_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K96_sigmoid_b1_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K96_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K96_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K112_sigmoid_b0_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K112_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K112_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K112_sigmoid_b1_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K112_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K112_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K128_sigmoid_b0_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K128_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K128_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K128_sigmoid_b1_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K128_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K128_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K144_sigmoid_b0_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K144_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K144_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K144_sigmoid_b1_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K144_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K144_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K160_sigmoid_b0_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K160_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K160_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K160_sigmoid_b1_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K160_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K160_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K176_sigmoid_b0_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K176_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K176_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K176_sigmoid_b1_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K176_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K176_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K192_sigmoid_b0_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K192_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K192_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K192_sigmoid_b1_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K192_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K192_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K208_sigmoid_b0_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K208_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K208_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K208_sigmoid_b1_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K208_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K208_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K224_sigmoid_b0_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K224_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K224_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K224_sigmoid_b1_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K224_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K224_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K240_sigmoid_b0_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K240_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K240_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K240_sigmoid_b1_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K240_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K240_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K256_sigmoid_b0_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K256_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K256_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K256_sigmoid_b1_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K256_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K256_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K272_sigmoid_b0_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K272_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K272_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K272_sigmoid_b1_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K272_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K272_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K288_sigmoid_b0_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K288_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K288_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K288_sigmoid_b1_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K288_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K288_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K304_sigmoid_b0_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K304_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K304_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K304_sigmoid_b1_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K304_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K304_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K320_sigmoid_b0_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K320_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K320_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K320_sigmoid_b1_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K320_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K320_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K336_sigmoid_b0_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K336_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K336_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K336_sigmoid_b1_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K336_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K336_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K352_sigmoid_b0_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K352_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K352_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K352_sigmoid_b1_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K352_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K352_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K368_sigmoid_b0_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K368_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K368_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K368_sigmoid_b1_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K368_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K368_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K384_sigmoid_b0_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K384_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K384_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K384_sigmoid_b1_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K384_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K384_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K400_sigmoid_b0_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K400_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K400_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K400_sigmoid_b1_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K400_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K400_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K416_sigmoid_b0_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K416_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K416_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K416_sigmoid_b1_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K416_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K416_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K432_sigmoid_b0_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K432_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K432_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K432_sigmoid_b1_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K432_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K432_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K448_sigmoid_b0_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K448_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K448_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K448_sigmoid_b1_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K448_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K448_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K464_sigmoid_b0_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K464_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K464_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K464_sigmoid_b1_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K464_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K464_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K480_sigmoid_b0_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K480_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K480_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K480_sigmoid_b1_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K480_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K480_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K496_sigmoid_b0_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K496_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K496_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K496_sigmoid_b1_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K496_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K496_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K512_sigmoid_b0_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K512_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K512_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K512_sigmoid_b1_csr_pt.o : $(sigmoidGINC) \
   $(GENSRCdir)/sgfusedMM_K512_sigmoid_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K512_sigmoid_csr.c
$(GENSRCdir)/sgfusedMM_K16_tdist_b0_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K16_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K16_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K16_tdist_b1_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K16_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K16_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K32_tdist_b0_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K32_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K32_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K32_tdist_b1_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K32_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K32_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K48_tdist_b0_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K48_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K48_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K48_tdist_b1_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K48_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K48_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K64_tdist_b0_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K64_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K64_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K64_tdist_b1_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K64_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K64_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K80_tdist_b0_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K80_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K80_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K80_tdist_b1_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K80_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K80_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K96_tdist_b0_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K96_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K96_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K96_tdist_b1_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K96_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K96_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K112_tdist_b0_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K112_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K112_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K112_tdist_b1_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K112_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K112_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K128_tdist_b0_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K128_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K128_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K128_tdist_b1_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K128_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K128_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K144_tdist_b0_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K144_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K144_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K144_tdist_b1_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K144_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K144_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K160_tdist_b0_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K160_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K160_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K160_tdist_b1_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K160_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K160_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K176_tdist_b0_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K176_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K176_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K176_tdist_b1_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K176_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K176_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K192_tdist_b0_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K192_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K192_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K192_tdist_b1_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K192_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K192_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K208_tdist_b0_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K208_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K208_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K208_tdist_b1_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K208_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K208_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K224_tdist_b0_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K224_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K224_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K224_tdist_b1_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K224_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K224_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K240_tdist_b0_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K240_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K240_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K240_tdist_b1_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K240_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K240_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K256_tdist_b0_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K256_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K256_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K256_tdist_b1_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K256_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K256_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K272_tdist_b0_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K272_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K272_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K272_tdist_b1_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K272_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K272_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K288_tdist_b0_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K288_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K288_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K288_tdist_b1_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K288_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K288_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K304_tdist_b0_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K304_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K304_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K304_tdist_b1_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K304_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K304_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K320_tdist_b0_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K320_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K320_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K320_tdist_b1_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K320_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K320_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K336_tdist_b0_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K336_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K336_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K336_tdist_b1_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K336_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K336_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K352_tdist_b0_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K352_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K352_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K352_tdist_b1_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K352_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K352_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K368_tdist_b0_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K368_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K368_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K368_tdist_b1_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K368_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K368_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K384_tdist_b0_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K384_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K384_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K384_tdist_b1_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K384_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K384_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K400_tdist_b0_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K400_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K400_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K400_tdist_b1_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K400_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K400_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K416_tdist_b0_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K416_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K416_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K416_tdist_b1_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K416_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K416_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K432_tdist_b0_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K432_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K432_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K432_tdist_b1_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K432_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K432_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K448_tdist_b0_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K448_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K448_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K448_tdist_b1_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K448_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K448_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K464_tdist_b0_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K464_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K464_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K464_tdist_b1_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K464_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K464_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K480_tdist_b0_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K480_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K480_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K480_tdist_b1_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K480_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K480_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K496_tdist_b0_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K496_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K496_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K496_tdist_b1_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K496_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K496_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K512_tdist_b0_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K512_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA0 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K512_tdist_csr.c
$(GENSRCdir)/sgfusedMM_K512_tdist_b1_csr_pt.o : $(tdistGINC) \
   $(GENSRCdir)/sgfusedMM_K512_tdist_csr.c
	$(KCC) $(KCCFLAGS) $(ARCHFLAGS) $(IFLAGS) $(PTFLAGS) -DSREAL \
        -DBETA1 -I$(SIMDdir) -o $@ -c $(GENSRCdir)/sgfusedMM_K512_tdist_csr.c



clean : 
	rm -f $(GENSRCdir)/s*.o $(LIBdir)/s*.a  $(LIBdir)/s*.grd 


