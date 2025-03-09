#include<stdint.h>
#ifdef PTTIME
   #include<omp.h>
#endif
#define SREAL 1
#include"../../simd/simd.h"
#if VLEN != 16
   #error "ARCH VLEN doesn't match with generator's VLEN, see simd.h " 
#endif

/*
 * Register block  A,C and B(innermost loop) will require most registers, works 
 * better on small value of k
 */
extern int SOP_UDEF_FUNC(float val, float *out);  
/*extern INDEXTYPE MAXBOUND ;*/
#ifdef BETA0
void sgfusedMM_K176_tdist_b0_csr
#else /* BETA1 version */
void sgfusedMM_K176_tdist_b1_csr
#endif
(
   const char tkern,  	   // 's' 't' 'm'
   const INDEXTYPE m,      // rows of dense A matrix 
   const INDEXTYPE n,      // rows of dense B matrix
   const INDEXTYPE k,      // cols of A or dimension. not used since K compile time   
   const float alpha,     // const to scale, not use yet  
   const INDEXTYPE nnz,    // nonzeros of the sparse matrix 
   const INDEXTYPE rows,   // number of rows of the sparse matrix  
   const INDEXTYPE cols,   // number of columns of the sparse matrix 
   const float *val,       // value of  the sparse matrix 
   const INDEXTYPE *indx,  // colids -> column indices of sparse matrix 
   const INDEXTYPE *pntrb, // starting index for rowptr of csr of sparse matrix
   const INDEXTYPE *pntre, // ending index for rowptr of csr of sparse matrix 
   const float *a,        // Dense A matrix
   const INDEXTYPE lda,    // leading dimension of a (col size since row-major)  
   const float *b,        // Dense B matrix
   const INDEXTYPE ldb,    // leading dimension of b (col size since row-major)  
   const float beta,      // beta value, compile time not used  
   float *c,              // Dense matrix c
   const INDEXTYPE ldc     // leading dimension size of c (col size since roa-major) 
)
{
#if 0
   const float maxbound = 5.0;
#endif
#if defined(PTTIME) && defined(LDB)
   omp_set_num_threads(NTHREADS);
   #pragma omp parallel
   {
      INDEXTYPE RowPerThd, tt;
      INDEXTYPE i, rowb, rowe;
      INDEXTYPE Mnnz = 0; /* non-zero count in M rows  */
      INDEXTYPE deg, cumRow, curRow;
      INDEXTYPE id = omp_get_thread_num();
      INDEXTYPE nthreads = omp_get_num_threads(); 
      
      for (i=0; i < m; i++)
         Mnnz += (pntre[i] - pntrb[i]); 
      RowPerThd = Mnnz / nthreads; 
      
      curRow = cumRow = 0; 
      tt = 1; 
      rowe = -1;  /* init */
      /* set rowstart for 1st thread */ 
      if (id == 0) 
         rowb = 0;
      for (i=0; i < m; i++)
      {
         deg = pntre[i] - pntrb[i]; 
         cumRow += deg;
         curRow += deg;
         if (curRow > RowPerThd)
         {
            if (tt == id)
               rowb = i; 
            else if (tt == id+1)
               rowe = i; 
            curRow = 0;
            RowPerThd = (Mnnz - cumRow) / (nthreads - tt);
            tt += 1; 
         }
      }
      if (tt == id+1)
         rowe = m; 

      for (i=rowb; i < rowe; i++)
#else /* not LBD or not PTTIME */
   #ifdef PTTIME
      #ifdef NTHREADS
      omp_set_num_threads(NTHREADS);
      #endif
      #ifdef DYNAMIC 
         #pragma omp parallel for schedule(dynamic)
      #else
         #pragma omp parallel for schedule(static)
      #endif
   #endif
   for (INDEXTYPE i = 0; i < m; i++)
#endif
   {
      register VTYPE Va0, Vc0, Va1, Vc1, Va2, Vc2, Va3, Vc3, Va4, Vc4, Va5, Vc5,
                     Va6, Vc6, Va7, Vc7, Va8, Vc8, Va9, Vc9, Va10, Vc10;
      INDEXTYPE iindex = i * 176; 
      const float *Ai = a + iindex; 
      float *Ci = c + iindex; 
      VTYPE VMAXBOUND, VMINBOUND; 
#if 0
      BCL_vset1(VMAXBOUND, maxbound); 
      BCL_vset1(VMINBOUND, -maxbound); 
#endif
#ifdef BETA0
/*
 * NO need to load C, just zerod Vector register    
 */
      BCL_vzero(Vc0); 
      BCL_vzero(Vc1); 
      BCL_vzero(Vc2); 
      BCL_vzero(Vc3); 
      BCL_vzero(Vc4); 
      BCL_vzero(Vc5); 
      BCL_vzero(Vc6); 
      BCL_vzero(Vc7); 
      BCL_vzero(Vc8); 
      BCL_vzero(Vc9); 
      BCL_vzero(Vc10); 

#else /* beta1 */
      // load Vc 
      BCL_vldu(Vc0, Ci+VLEN*0); 
      BCL_vldu(Vc1, Ci+VLEN*1); 
      BCL_vldu(Vc2, Ci+VLEN*2); 
      BCL_vldu(Vc3, Ci+VLEN*3); 
      BCL_vldu(Vc4, Ci+VLEN*4); 
      BCL_vldu(Vc5, Ci+VLEN*5); 
      BCL_vldu(Vc6, Ci+VLEN*6); 
      BCL_vldu(Vc7, Ci+VLEN*7); 
      BCL_vldu(Vc8, Ci+VLEN*8); 
      BCL_vldu(Vc9, Ci+VLEN*9); 
      BCL_vldu(Vc10, Ci+VLEN*10); 
#endif
      // load Va 
      BCL_vldu(Va0, Ai+VLEN*0); 
      BCL_vldu(Va1, Ai+VLEN*1); 
      BCL_vldu(Va2, Ai+VLEN*2); 
      BCL_vldu(Va3, Ai+VLEN*3); 
      BCL_vldu(Va4, Ai+VLEN*4); 
      BCL_vldu(Va5, Ai+VLEN*5); 
      BCL_vldu(Va6, Ai+VLEN*6); 
      BCL_vldu(Va7, Ai+VLEN*7); 
      BCL_vldu(Va8, Ai+VLEN*8); 
      BCL_vldu(Va9, Ai+VLEN*9); 
      BCL_vldu(Va10, Ai+VLEN*10); 

      for (INDEXTYPE j = pntrb[i]; j < pntre[i]; j++)
      {
         VTYPE Vb0, Vb1, Vb2, Vb3, Vb4, Vb5, Vb6, Vb7, Vb8, Vb9, Vb10;
         VTYPE Vd0, Vd1, Vd2, Vd3, Vd4, Vd5, Vd6, Vd7, Vd8, Vd9, Vd10, Vt;
         VTYPE Vatt0, Vatt1, Vatt2, Vatt3, Vatt4, Vatt5, Vatt6, Vatt7, Vatt8,
               Vatt9, Vatt10;
         float attrc = 0;
         INDEXTYPE colidj = indx[j];
         INDEXTYPE jindex = colidj*176;
         const float *Bj = b + jindex; 
         // load Vxj 
         BCL_vldu(Vb0, Bj+VLEN*0); 
         BCL_vldu(Vb1, Bj+VLEN*1); 
         BCL_vldu(Vb2, Bj+VLEN*2); 
         BCL_vldu(Vb3, Bj+VLEN*3); 
         BCL_vldu(Vb4, Bj+VLEN*4); 
         BCL_vldu(Vb5, Bj+VLEN*5); 
         BCL_vldu(Vb6, Bj+VLEN*6); 
         BCL_vldu(Vb7, Bj+VLEN*7); 
         BCL_vldu(Vb8, Bj+VLEN*8); 
         BCL_vldu(Vb9, Bj+VLEN*9); 
         BCL_vldu(Vb10, Bj+VLEN*10); 
      // init Vatt  
         BCL_vzero(Vatt0);
         BCL_vzero(Vatt1);
         BCL_vzero(Vatt2);
         BCL_vzero(Vatt3);
         BCL_vzero(Vatt4);
         BCL_vzero(Vatt5);
         BCL_vzero(Vatt6);
         BCL_vzero(Vatt7);
         BCL_vzero(Vatt8);
         BCL_vzero(Vatt9);
         BCL_vzero(Vatt10);

      // vsub
         BCL_vsub(Vd0, Va0, Vb0);
         BCL_vsub(Vd1, Va1, Vb1);
         BCL_vsub(Vd2, Va2, Vb2);
         BCL_vsub(Vd3, Va3, Vb3);
         BCL_vsub(Vd4, Va4, Vb4);
         BCL_vsub(Vd5, Va5, Vb5);
         BCL_vsub(Vd6, Va6, Vb6);
         BCL_vsub(Vd7, Va7, Vb7);
         BCL_vsub(Vd8, Va8, Vb8);
         BCL_vsub(Vd9, Va9, Vb9);
         BCL_vsub(Vd10, Va10, Vb10);
      // vmac 
         BCL_vmac(Vatt0, Vd0, Vd0);
         BCL_vmac(Vatt1, Vd1, Vd1);
         BCL_vmac(Vatt2, Vd2, Vd2);
         BCL_vmac(Vatt3, Vd3, Vd3);
         BCL_vmac(Vatt4, Vd4, Vd4);
         BCL_vmac(Vatt5, Vd5, Vd5);
         BCL_vmac(Vatt6, Vd6, Vd6);
         BCL_vmac(Vatt7, Vd7, Vd7);
         BCL_vmac(Vatt8, Vd8, Vd8);
         BCL_vmac(Vatt9, Vd9, Vd9);
         BCL_vmac(Vatt10, Vd10, Vd10);
         // binary tree reduction 
         BCL_vadd(Vatt0, Vatt0, Vatt1);
         BCL_vadd(Vatt2, Vatt2, Vatt3);
         BCL_vadd(Vatt4, Vatt4, Vatt5);
         BCL_vadd(Vatt6, Vatt6, Vatt7);
         BCL_vadd(Vatt8, Vatt8, Vatt9);
         BCL_vadd(Vatt0, Vatt0, Vatt2);
         BCL_vadd(Vatt4, Vatt4, Vatt6);
         BCL_vadd(Vatt8, Vatt8, Vatt10);
         BCL_vadd(Vatt0, Vatt0, Vatt4);
         BCL_vadd(Vatt0, Vatt0, Vatt8);

         BCL_vrsum1(attrc, Vatt0);
#if 0
         BCL_vset1(Vatt0, attrc); // a = a
         BCL_vset1(Vt, 1.0f); // t = 1.0
         BCL_vadd(Vatt0, Vatt0, Vt); // a = 1.0 + a
         BCL_vrcp(Vatt0, Vatt0); // a = 1/a
         BCL_vset1(Vt, -2.0f); // t = -2
         BCL_vmul(Vatt0, Vatt0, Vt); // a = -2 * a
         BCL_vmax(Vatt0, Vatt0, VMINBOUND);
         BCL_vmin(Vatt0, Vatt0, VMAXBOUND); 
#else
         SOP_UDEF_FUNC(attrc, &attrc);
         BCL_vset1(Vatt0, attrc); 
#endif
      // vmul 
         BCL_vmul(Vd0, Vatt0, Vd0);
         BCL_vmul(Vd1, Vatt0, Vd1);
         BCL_vmul(Vd2, Vatt0, Vd2);
         BCL_vmul(Vd3, Vatt0, Vd3);
         BCL_vmul(Vd4, Vatt0, Vd4);
         BCL_vmul(Vd5, Vatt0, Vd5);
         BCL_vmul(Vd6, Vatt0, Vd6);
         BCL_vmul(Vd7, Vatt0, Vd7);
         BCL_vmul(Vd8, Vatt0, Vd8);
         BCL_vmul(Vd9, Vatt0, Vd9);
         BCL_vmul(Vd10, Vatt0, Vd10);
      // vadd 
         BCL_vadd(Vc0, Vc0, Vd0);
         BCL_vadd(Vc1, Vc1, Vd1);
         BCL_vadd(Vc2, Vc2, Vd2);
         BCL_vadd(Vc3, Vc3, Vd3);
         BCL_vadd(Vc4, Vc4, Vd4);
         BCL_vadd(Vc5, Vc5, Vd5);
         BCL_vadd(Vc6, Vc6, Vd6);
         BCL_vadd(Vc7, Vc7, Vd7);
         BCL_vadd(Vc8, Vc8, Vd8);
         BCL_vadd(Vc9, Vc9, Vd9);
         BCL_vadd(Vc10, Vc10, Vd10);
      }
      BCL_vstu(Ci + VLEN*0, Vc0); 
      BCL_vstu(Ci + VLEN*1, Vc1); 
      BCL_vstu(Ci + VLEN*2, Vc2); 
      BCL_vstu(Ci + VLEN*3, Vc3); 
      BCL_vstu(Ci + VLEN*4, Vc4); 
      BCL_vstu(Ci + VLEN*5, Vc5); 
      BCL_vstu(Ci + VLEN*6, Vc6); 
      BCL_vstu(Ci + VLEN*7, Vc7); 
      BCL_vstu(Ci + VLEN*8, Vc8); 
      BCL_vstu(Ci + VLEN*9, Vc9); 
      BCL_vstu(Ci + VLEN*10, Vc10); 
   }
#if defined(PTTIME) && defined(LDB)
   }
#endif
}
