#
#  Top Makefile to run, test and time FUSEDMM 
#
BIN = ./bin
Kdir = ./kernels
Tdir = ./test
KLIBdir = $(Kdir)/lib
KINCdir = $(Kdir)/include

# indextype : int64_t or int32_t
# NOTE: when comparing with MKL, use ibit=64 since we are using MKL_ILP64
ibit=64
#ibit=32

# valuetype precision : double single 
pre=s

#
# SIMD width on system: 
#    Depend on ARCH, configure step sets SIMD variable in kernels/make.inc 
#    See kernels/include/simd.h for different width on different system  
#
vlen=16

#
#  Register blocking strategies: 
#  	bacrb = all three dense matrix register blocked
#  	acrb = X(A) and Z(C) register blocked
#  	crb = only Z(C) is register blocked 
#
regblk=bacrb 
#regblk=acrb 
#regblk=crb 

#
#   Two different phases: 
#   	K-compile time: kernel generated upto mdim but fully unrolled  
#   	K-runtime: kernel generated upto bestK unrolled and rolled beyond
#   Note: K-compile time mode only support kernels upto mdim, for arbitrary 
#   K, we should use K-runtime but after tuning the bestK 
#
#kruntime=0
mdim=1024 

kruntime=1   # 0 means K compile time, used in tuning phase  
bestK=512    # needed when kruntime=1, normally got from tuning step  

kern=s   # t = tdist/fr, s = sigmoid, m = spmm, g = gcn 
data=dataset/harvard.mtx      
d=128 
# =============================================================================
#  General Flags 
# ============================================================================
#setup flags based on type 
ifeq ($(pre), d)
   dtyp=-DDREAL
else
   dtyp=-DSREAL
endif
TYPFLAGS = -DINDEXTYPE=int$(ibit)_t -DINT$(ibit) $(dtyp)

# Library info  
sLIBS=$(KLIBdir)/$(pre)libgfusedMM_sequential.a 
ptLIBS=$(KLIBdir)/$(pre)libgfusedMM_pt.a 

KINCS=$(KINCdir)/kernels.h 

#
# tester/timer's compiler 
#
CC = gcc
CCFLAGS = -fopenmp -O3 -march=native 

CPP = g++
CPPFLAGS = -fopenmp -O3 -march=native -std=c++11

#
# My parallel flags 
#
ldb=l
NTHREADS=32
#NTHREADS=6
LDB=LOAD_BALANCE 
MYPT_FLAG = -DPTTIME -DNTHREADS=$(NTHREADS) -D$(LDB)  
#MYPT_FLAG = -DPTTIME -DNTHREADS=$(NTHREADS) -DSTATIC  

# =============================================================================
#	Flags for MKL 
# =============================================================================

MKLROOT = /opt/intel/mkl
#
#parallel version of MKL 
#
PT_CC_MKL_FLAG = -DMKL_ILP64 -m64 -I${MKLROOT}/include
PT_LD_MKL_FLAG =  -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a \
	       ${MKLROOT}/lib/intel64/libmkl_gnu_thread.a \
	       ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lgomp \
	       -lpthread -lm -ldl  

#serial version of MKL 
CC_MKL_FLAG =  -DMKL_ILP64 -m64 -I${MKLROOT}/include
LD_MKL_FLAG =  -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a \
	       ${MKLROOT}/lib/intel64/libmkl_sequential.a \
	       ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread \
	       -lm -ldl 

# =============================================================================
#  Default Target  
# ==============================================================================
all: $(BIN)/x$(pre)FusedMMtime_gcn_pt $(BIN)/x$(pre)FusedMMtime_spmm_pt \
     $(BIN)/x$(pre)FusedMMtime_fr_pt $(BIN)/x$(pre)FusedMMtime_tdist_pt \
     $(BIN)/x$(pre)FusedMMtime_sigmoid_pt $(BIN)/x$(pre)OptFusedMMtime_gcn_pt \
     $(BIN)/x$(pre)OptFusedMMtime_spmm_pt $(BIN)/x$(pre)OptFusedMMtime_fr_pt \
     $(BIN)/x$(pre)OptFusedMMtime_tdist_pt \
     $(BIN)/x$(pre)OptFusedMMtime_sigmoid_pt

test: $(BIN)/x$(pre)OptFusedMMtime_gcn_pt \
      $(BIN)/x$(pre)OptFusedMMtime_spmm_pt $(BIN)/x$(pre)OptFusedMMtime_fr_pt \
      $(BIN)/x$(pre)OptFusedMMtime_tdist_pt \
      $(BIN)/x$(pre)OptFusedMMtime_sigmoid_pt
	$(BIN)/x$(pre)OptFusedMMtime_gcn_pt -input $(data) -T 1 -K $(d)  
	$(BIN)/x$(pre)OptFusedMMtime_spmm_pt -input $(data) -T 1 -K $(d)  
	$(BIN)/x$(pre)OptFusedMMtime_fr_pt -input $(data) -T 1 -K $(d)  
	$(BIN)/x$(pre)OptFusedMMtime_tdist_pt -input $(data) -T 1 -K $(d)  
	$(BIN)/x$(pre)OptFusedMMtime_sigmoid_pt -input $(data) -T 1 -K $(d)  


# =============================================================================
# Build with MKL to compare results for SPMM 
# =============================================================================

mkl: $(BIN)/x$(pre)OptFusedMMtime_spmm_MKL_pt 
	$(BIN)/x$(pre)OptFusedMMtime_spmm_MKL_pt -input $(data) -T 1 -K $(d)  

$(BIN)/$(pre)OptFusedMMtime_spmm_MKL_pt.o: $(Tdir)/fusedMMtime.cpp fusedMM.h \
   fusedMM_internal.h   
	mkdir -p $(BIN)
	$(CPP) $(CPPFLAGS) $(TYPFLAGS) -DTIME_MKL -I$(KINCdir) -DSPMM_UDEF \
	   -DCPP $(PT_CC_MKL_FLAG) $(MYPT_FLAG) -c $(Tdir)/fusedMMtime.cpp -o $@   
$(BIN)/x$(pre)OptFusedMMtime_spmm_MKL_pt: $(BIN)/$(pre)OptFusedMMtime_spmm_MKL_pt.o \
   $(BIN)/$(pre)OptFusedMM_pt.o $(ptLIBS)  
	$(CPP) $(CPPFLAGS) -o $@ $^ $(ptLIBS) -lm $(PT_LD_MKL_FLAG)

# ===========================================================================
# To generate FusedMM kernels 
# ===========================================================================

$(sLIBS)  : $(ptLIBS)
$(ptLIBS) : $(Kdir)/rungen.sh  
	cd $(Kdir) ; ./rungen.sh -p $(pre) -i $(ibit) -s $(vlen) -e $(mdim) \
	   -v $(vlen) -t $(NTHREADS) -r $(regblk) -k $(kruntime) -b $(bestK)

# =============================================================================
#  Target for executable 
# ==============================================================================

# ==============================================================================
#  parallel version 
# ==============================================================================

#
#  Compiling Fusedmm  
#
$(BIN)/$(pre)OptFusedMM_pt.o: fusedMM.c fusedMM.h fusedMM_internal.h
	mkdir -p $(BIN)
	$(CC) $(CCFLAGS) $(TYPFLAGS) -I$(KINCdir) $(MYPT_FLAG) -DENABLE_OPT_FUSEDMM \
           -c fusedMM.c -o $@   
$(BIN)/$(pre)FusedMM_pt.o: fusedMM.c fusedMM.h fusedMM_internal.h
	mkdir -p $(BIN)
	$(CC) $(CCFLAGS) $(TYPFLAGS) -I$(KINCdir) $(MYPT_FLAG)  \
           -c fusedMM.c -o $@   
#
#  Compiling FusedMMTime  
#
$(BIN)/$(pre)FusedMMtime_gcn_pt.o: $(Tdir)/fusedMMtime.cpp fusedMM.h \
   $(KINCdir)/kernels.h  
	mkdir -p $(BIN)
	$(CPP) $(CPPFLAGS) $(TYPFLAGS) -I$(KINCdir) -DGCN_UDEF \
	   -DCPP $(MYPT_FLAG) -c $(Tdir)/fusedMMtime.cpp -o $@   
$(BIN)/$(pre)FusedMMtime_spmm_pt.o: $(Tdir)/fusedMMtime.cpp fusedMM.h \
   $(KINCdir)/kernels.h  
	mkdir -p $(BIN)
	$(CPP) $(CPPFLAGS) $(TYPFLAGS) -I$(KINCdir) -DSPMM_UDEF \
	   -DCPP $(MYPT_FLAG) -c $(Tdir)/fusedMMtime.cpp -o $@   
$(BIN)/$(pre)FusedMMtime_fr_pt.o: $(Tdir)/fusedMMtime.cpp fusedMM.h \
   $(KINCdir)/kernels.h  
	mkdir -p $(BIN)
	$(CPP) $(CPPFLAGS) $(TYPFLAGS) -I$(KINCdir) -DFR_UDEF \
	   -DCPP $(MYPT_FLAG) -c $(Tdir)/fusedMMtime.cpp -o $@   
$(BIN)/$(pre)FusedMMtime_tdist_pt.o: $(Tdir)/fusedMMtime.cpp fusedMM.h \
   $(KINCdir)/kernels.h  
	mkdir -p $(BIN)
	$(CPP) $(CPPFLAGS) $(TYPFLAGS) -I$(KINCdir) -DTDIST_UDEF \
	   -DCPP $(MYPT_FLAG) -c $(Tdir)/fusedMMtime.cpp -o $@   
$(BIN)/$(pre)FusedMMtime_sigmoid_pt.o: $(Tdir)/fusedMMtime.cpp fusedMM.h \
   $(KINCdir)/kernels.h  
	mkdir -p $(BIN)
	$(CPP) $(CPPFLAGS) $(TYPFLAGS) -I$(KINCdir) -DSIGMOID_UDEF \
	   -DCPP $(MYPT_FLAG) -c $(Tdir)/fusedMMtime.cpp -o $@   
#
#  Executables 
#
$(BIN)/x$(pre)OptFusedMMtime_gcn_pt: $(BIN)/$(pre)FusedMMtime_gcn_pt.o \
   $(BIN)/$(pre)OptFusedMM_pt.o $(ptLIBS)  
	$(CPP) $(CPPFLAGS) -o $@ $^ $(ptLIBS) -lm
$(BIN)/x$(pre)OptFusedMMtime_spmm_pt: $(BIN)/$(pre)FusedMMtime_spmm_pt.o \
   $(BIN)/$(pre)OptFusedMM_pt.o $(ptLIBS)  
	$(CPP) $(CPPFLAGS) -o $@ $^ $(ptLIBS) -lm
$(BIN)/x$(pre)OptFusedMMtime_fr_pt: $(BIN)/$(pre)FusedMMtime_fr_pt.o \
   $(BIN)/$(pre)OptFusedMM_pt.o $(ptLIBS)  
	$(CPP) $(CPPFLAGS) -o $@ $^ $(ptLIBS) -lm
$(BIN)/x$(pre)OptFusedMMtime_tdist_pt: $(BIN)/$(pre)FusedMMtime_tdist_pt.o \
   $(BIN)/$(pre)OptFusedMM_pt.o $(ptLIBS)  
	$(CPP) $(CPPFLAGS) -o $@ $^ $(ptLIBS) -lm
$(BIN)/x$(pre)OptFusedMMtime_sigmoid_pt: $(BIN)/$(pre)FusedMMtime_sigmoid_pt.o \
   $(BIN)/$(pre)OptFusedMM_pt.o $(ptLIBS)  
	$(CPP) $(CPPFLAGS) -o $@ $^ $(ptLIBS) -lm
$(BIN)/x$(pre)FusedMMtime_gcn_pt: $(BIN)/$(pre)FusedMMtime_gcn_pt.o \
   $(BIN)/$(pre)FusedMM_pt.o $(ptLIBS)  
	$(CPP) $(CPPFLAGS) -o $@ $^ $(ptLIBS) -lm
$(BIN)/x$(pre)FusedMMtime_spmm_pt: $(BIN)/$(pre)FusedMMtime_spmm_pt.o \
   $(BIN)/$(pre)FusedMM_pt.o $(ptLIBS)  
	$(CPP) $(CPPFLAGS) -o $@ $^ $(ptLIBS) -lm
$(BIN)/x$(pre)FusedMMtime_fr_pt: $(BIN)/$(pre)FusedMMtime_fr_pt.o \
   $(BIN)/$(pre)FusedMM_pt.o $(ptLIBS)  
	$(CPP) $(CPPFLAGS) -o $@ $^ $(ptLIBS) -lm
$(BIN)/x$(pre)FusedMMtime_tdist_pt: $(BIN)/$(pre)FusedMMtime_tdist_pt.o \
   $(BIN)/$(pre)FusedMM_pt.o $(ptLIBS)  
	$(CPP) $(CPPFLAGS) -o $@ $^ $(ptLIBS) -lm
$(BIN)/x$(pre)FusedMMtime_sigmoid_pt: $(BIN)/$(pre)FusedMMtime_sigmoid_pt.o \
   $(BIN)/$(pre)FusedMM_pt.o $(ptLIBS)  
	$(CPP) $(CPPFLAGS) -o $@ $^ $(ptLIBS) -lm
   
# ==============================================================================
#  serial version 
# ==============================================================================

#
#  Compiling Fusedmm  
#
$(BIN)/$(pre)OptFusedMM.o: fusedMM.c fusedMM.h fusedMM_internal.h
	mkdir -p $(BIN)
	$(CC) $(CCFLAGS) $(TYPFLAGS) -I$(KINCdir)  -DENABLE_OPT_FUSEDMM \
           -c fusedMM.c -o $@   
$(BIN)/$(pre)FusedMM.o: fusedMM.c fusedMM.h fusedMM_internal.h
	mkdir -p $(BIN)
	$(CC) $(CCFLAGS) $(TYPFLAGS) -I$(KINCdir)   \
           -c fusedMM.c -o $@   
#
#  Compiling FusedMMTime  
#
$(BIN)/$(pre)FusedMMtime_gcn.o: $(Tdir)/fusedMMtime.cpp fusedMM.h \
   $(KINCdir)/kernels.h  
	mkdir -p $(BIN)
	$(CPP) $(CPPFLAGS) $(TYPFLAGS) -I$(KINCdir) -DGCN_UDEF \
	   -DCPP  -c $(Tdir)/fusedMMtime.cpp -o $@   
$(BIN)/$(pre)FusedMMtime_spmm.o: $(Tdir)/fusedMMtime.cpp fusedMM.h \
   $(KINCdir)/kernels.h  
	mkdir -p $(BIN)
	$(CPP) $(CPPFLAGS) $(TYPFLAGS) -I$(KINCdir) -DSPMM_UDEF \
	   -DCPP  -c $(Tdir)/fusedMMtime.cpp -o $@   
$(BIN)/$(pre)FusedMMtime_fr.o: $(Tdir)/fusedMMtime.cpp fusedMM.h \
   $(KINCdir)/kernels.h  
	mkdir -p $(BIN)
	$(CPP) $(CPPFLAGS) $(TYPFLAGS) -I$(KINCdir) -DFR_UDEF \
	   -DCPP  -c $(Tdir)/fusedMMtime.cpp -o $@   
$(BIN)/$(pre)FusedMMtime_tdist.o: $(Tdir)/fusedMMtime.cpp fusedMM.h \
   $(KINCdir)/kernels.h  
	mkdir -p $(BIN)
	$(CPP) $(CPPFLAGS) $(TYPFLAGS) -I$(KINCdir) -DTDIST_UDEF \
	   -DCPP  -c $(Tdir)/fusedMMtime.cpp -o $@   
$(BIN)/$(pre)FusedMMtime_sigmoid.o: $(Tdir)/fusedMMtime.cpp fusedMM.h \
   $(KINCdir)/kernels.h  
	mkdir -p $(BIN)
	$(CPP) $(CPPFLAGS) $(TYPFLAGS) -I$(KINCdir) -DSIGMOID_UDEF \
	   -DCPP  -c $(Tdir)/fusedMMtime.cpp -o $@   
#
#  Executables 
#
$(BIN)/x$(pre)OptFusedMMtime_gcn: $(BIN)/$(pre)FusedMMtime_gcn.o \
   $(BIN)/$(pre)OptFusedMM.o $(sLIBs)  
	$(CPP) $(CPPFLAGS) -o $@ $^ $(sLIBs) -lm
$(BIN)/x$(pre)OptFusedMMtime_spmm: $(BIN)/$(pre)FusedMMtime_spmm.o \
   $(BIN)/$(pre)OptFusedMM.o $(sLIBs)  
	$(CPP) $(CPPFLAGS) -o $@ $^ $(sLIBs) -lm
$(BIN)/x$(pre)OptFusedMMtime_fr: $(BIN)/$(pre)FusedMMtime_fr.o \
   $(BIN)/$(pre)OptFusedMM.o $(sLIBs)  
	$(CPP) $(CPPFLAGS) -o $@ $^ $(sLIBs) -lm
$(BIN)/x$(pre)OptFusedMMtime_tdist: $(BIN)/$(pre)FusedMMtime_tdist.o \
   $(BIN)/$(pre)OptFusedMM.o $(sLIBs)  
	$(CPP) $(CPPFLAGS) -o $@ $^ $(sLIBs) -lm
$(BIN)/x$(pre)OptFusedMMtime_sigmoid: $(BIN)/$(pre)FusedMMtime_sigmoid.o \
   $(BIN)/$(pre)OptFusedMM.o $(sLIBs)  
	$(CPP) $(CPPFLAGS) -o $@ $^ $(sLIBs) -lm
$(BIN)/x$(pre)FusedMMtime_gcn: $(BIN)/$(pre)FusedMMtime_gcn.o \
   $(BIN)/$(pre)FusedMM.o $(sLIBs)  
	$(CPP) $(CPPFLAGS) -o $@ $^ $(sLIBs) -lm
$(BIN)/x$(pre)FusedMMtime_spmm: $(BIN)/$(pre)FusedMMtime_spmm.o \
   $(BIN)/$(pre)FusedMM.o $(sLIBs)  
	$(CPP) $(CPPFLAGS) -o $@ $^ $(sLIBs) -lm
$(BIN)/x$(pre)FusedMMtime_fr: $(BIN)/$(pre)FusedMMtime_fr.o \
   $(BIN)/$(pre)FusedMM.o $(sLIBs)  
	$(CPP) $(CPPFLAGS) -o $@ $^ $(sLIBs) -lm
$(BIN)/x$(pre)FusedMMtime_tdist: $(BIN)/$(pre)FusedMMtime_tdist.o \
   $(BIN)/$(pre)FusedMM.o $(sLIBs)  
	$(CPP) $(CPPFLAGS) -o $@ $^ $(sLIBs) -lm
$(BIN)/x$(pre)FusedMMtime_sigmoid: $(BIN)/$(pre)FusedMMtime_sigmoid.o \
   $(BIN)/$(pre)FusedMM.o $(sLIBs)  
	$(CPP) $(CPPFLAGS) -o $@ $^ $(sLIBs) -lm
   
#
# cleanup 
#
clean:
	rm -rf ./bin/*

killlib: 
	cd $(Kdir) ; make clean pre=s
