#
# inc file for the Makefile  of the Kernel generator 
#
   TOP = /home/chenruiyang/Code/GNN/GNN_ICPP/Official_Code/FusedMM/kernels
#  ---------------------------------------
#  Make.inc for GSDDMM kernel generator 
#  ---------------------------------------

   SHELL = /bin/sh 

#  ----------------------------------------
#  Paths to various directories 
#  ---------------------------------------
   
   BINdir = $(TOP)/bin
   BLDdir = $(TOP)/obj
   SRCdir = $(TOP)/src
   CGENdir = $(TOP)/codegen
   INCSdir = $(TOP)/include
   SIMDdir = $(TOP)/simd
   LIBdir = $(TOP)/lib
   RUNdir= $(TOP)/script
   GENdir = $(TOP)/generated
   GENSRCdir = $(GENdir)/src
   GENINCdir = $(GENdir)/include

#  ----------------------------------------
#  General purpose compiler 
#  ---------------------------------------
   CC = gcc
   CFLAGS = -O3 -w 

#  ----------------------------------------
#  Kernel compiler (replaced by)
#     gcc 
#  KFLAGS (replaced by) 
#     -O3 -mavx -mfma
#     -O3 -mavx512f -mavx512dq 
#     -O3 -march=armv8-a
#  ---------------------------------------
   KCC = gcc 
   KCCFLAGS = -O3 -mavx512f -mavx512dq 
#  -----------------------------------------------------
#  Architecture related, we will probe the system later 
#  See $(INCSdir)/simd.h for details
#  
#  ARCH
#     BLC_X86 
#     BLC_ARM64
#     --BLC_OpenPOWER
#  SIMD 
#     BLC_AVXZ
#     BLC_AVX2
#     BLC_AVX
#     BLC_ARM_ASIMD
#     BLC_POWER_VSX
#        
#  -----------------------------------------------------
   ARCH = BLC_X86 
   SIMD = BLC_AVXZ  
   ARCHFLAGS = -DBLC_ARCH -D$(ARCH) -D$(SIMD)  
