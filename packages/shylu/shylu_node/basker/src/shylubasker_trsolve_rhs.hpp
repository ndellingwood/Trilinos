#ifndef SHYLUBASKER_SOLVE_RHS_HPP
#define SHYLUBASKER_SOLVE_RHS_HPP

/*Basker Includes*/
//#include "shylubasker_decl.hpp"
#include "shylubasker_matrix_decl.hpp"
#include "shylubasker_matrix_view_decl.hpp"
#include "shylubasker_types.hpp"
#include "shylubasker_util.hpp"

/*Kokkos Includes*/
#ifdef BASKER_KOKKOS
#include <Kokkos_Core.hpp>
#include <impl/Kokkos_Timer.hpp>
#else
#include <omp.h>
#endif

/*System Includes*/
#include <iostream>
#include <string>

//#define BASKER_DEBUG_SOLVE_RHS

using namespace std;

namespace BaskerNS
{

  //Note: we will want to come back and make
  //a much better multivector solve interface
  template <class Int, class Entry, class Exe_Space>
  BASKER_INLINE
  int Basker<Int,Entry,Exe_Space>::solve_interfacetr
  (
   Int nrhs,
   Entry *_x, // Solution
   Entry *_y  // rhs
  )
  {
    for(Int r = 0; r < nrhs; r++)
    {
      solve_interfacetr(&(_x[r*gm]), &(_y[r*gm]));
    }

    return 0;
  }//end solve_interfacetr(nrhs,x,y);

// Transpose solve items:
// - Mark "TODO" for functions needing to be reformulated
// - Append "tr" to each routine
// - Add new function declares to shylubasker_decl.hpp
//
// Level 1: Modify permutation order
//            (PAQ) * Q^T*x = P*b     original
//          (PAQ)^T * P*x   = Q^T*b   transpose solve
//          - Swap order: apply inverse permutation Q^T to b; apply permutation P to x
// Level 2: Modify solve order of partitions:
//           ND "big block"; update off-diag partition; BTF, from top-left down
// Level 3: ND "big-block" block access reformulation
//          - serial_forward_solvetr: change order of accessing blocks; update component solve; update the spmv rhs for "transpose" interpretation
//          - serial_backward_solvetr
// Level 4: Off-diagonal update reformulation - spmv w/ rhs updates, reinterpreted via "transpose" matrices
// Level 5: BTF - upper left to lower right, skip "off-diagonal" update of the first block; off-diagonal updates at each block-level

  // _x will be solution (properly permuted)
  // _y is originally the rhs
  // In this routine, first the rhs _y is copied and permuted to x_view_ptr_copy, y_view_ptr_copy set to 0's
  //  In subsequent solver calls, x_view_ptr_copy (initially the permuted rhs)
  //  is updated/modified during block solves
  //  y_view_ptr_copy stores the solution pivots
  // After solve is complete, the {x,y}_view_ptr_copy results are permuted to original ordering
  // and copied back to the raw pointer _x; x stores small btf block results, y stores nd partition results
  //
  template <class Int, class Entry, class Exe_Space>
  BASKER_INLINE
  int Basker<Int,Entry,Exe_Space>::solve_interfacetr
  (
   Entry *_x, // Solution (len = gn)
   Entry *_y  // rhs
  )
  {
    //printf( "\n -- solve_interfacetr --\n" );
    //for (Int i = 0; i < gn; i++) printf( " input: x(%d) = %e\n",i,_x[i] );
    //printf( "\n" );
#if 1
    // FIXME Skipped blk_matching, scaling and pivoting for now, reintroduce...
    // Transpose: Swap permutation order
    permute_and_init_for_solve(_y, x_view_ptr_copy, y_view_ptr_copy, perm_comp_array, gn);

    solve_interfacetr(x_view_ptr_copy, y_view_ptr_copy); //x is now permuted rhs; y is 0 

    permute_inv_and_finalcopy_after_solve(_x, x_view_ptr_copy, y_view_ptr_copy, perm_inv_comp_array, gn);
#elif 0
    // FIXME remove final copy, add in the "init" logic to pre-solve permutation
    // Pre-solve permutations
    if (Options.blk_matching != 0) {
        // apply amd col-permutation from numeric
        permute_and_finalcopy_after_solve(&(y_view_ptr_scale(0)), x_view_ptr_copy, y_view_ptr_copy, numeric_col_iperm_array, gn);
        //for (Int i = 0; i < gn; i++) printf( " > %d:%d: %.16e %.16e -> %.16e\n",i,numeric_col_iperm_array(i),x_view_ptr_copy(i),y_view_ptr_copy(i), y_view_ptr_scale(i));

        const Int poffset = btf_tabs(btf_tabs_offset);
        for (Int i = 0; i < poffset; i++) {
            x_view_ptr_copy(i) = y_view_ptr_scale(i);
        }
        for (Int i = poffset; i < gn; i++) {
            y_view_ptr_copy(i) = y_view_ptr_scale(i);
        }
    }
    permute_and_finalcopy_after_solve(_x, x_view_ptr_copy, y_view_ptr_copy, perm_comp_array, gn);
    //for (Int i = 0; i < gn; i++) printf( " %d:%d; %e %e, %e %e\n",i,perm_comp_array(i),x_view_ptr_copy(i),y_view_ptr_copy(i), _x[i],scale_col_array(i));

    if (Options.blk_matching != 0) {
      for(Int i = 0; i < gn; i++) {
        Int col = symbolic_col_iperm_array(i);
        _x[i] = scale_col_array(col) * _x[i];
      }
    }

    // Solve
    solve_interfacetr(x_view_ptr_copy, y_view_ptr_copy); //x is now permuted rhs; y is 0 

    // FIXME remove init, add in the "final copy" logic to post-solve permutation
    // Post-solve permutations - ensure final result in the original order
    if (Options.blk_matching != 0) {
        // apply mwm+amd row scaling from numeric
        for(Int i = 0; i < gn; i++) {
            Int row = order_blk_mwm_array(symbolic_row_iperm_array(i));
            y_view_ptr_scale(i) = scale_row_array(row) * _y[i];
            //printf( " symbolic_row_iperm(%d) = %d\n",i,symbolic_row_iperm_array(i) );
            //printf( " scale_row(%d) = %e\n",row,scale_row_array(row) );
        }
        //printf( " > after scale:\n" );
        //for (Int i = 0; i < gn; i++) printf( " > y(%d) = %.16e\n",i,y_view_ptr_scale(i) );

        // apply mwm row-perm from nummeric
        permute_inv_and_init_for_solve(&(y_view_ptr_scale(0)), x_view_ptr_copy, y_view_ptr_copy, perm_inv_comp_array, gn);
        //printf( " > after symbolic-perm:\n" );
        //for (Int i = 0; i < gn; i++) printf( " > y(%d) = %.16e, x(%d) = %.16e\n",i,y_view_ptr_scale(i), i,x_view_ptr_copy(i) );

        // apply row-perm from symbolic
        permute_with_workspace(x_view_ptr_copy, numeric_row_iperm_array, gn);
    } else {
        permute_inv_and_init_for_solve(_y, x_view_ptr_copy, y_view_ptr_copy, perm_inv_comp_array, gn);
    }
    //printf( " > after perm:\n" );
    //for (Int i = 0; i < gn; i++) printf( " %d %.16e %.16e\n",i, x_view_ptr_copy(i),y_view_ptr_copy(i) );
    //printf( "\n" );

    if (Options.no_pivot == BASKER_FALSE) {
        // apply partial pivoting from numeric
        //for (Int i = 0; i < gn; i++) printf( " gperm(%d) = %d\n",i,gperm(i) );
        permute_inv_with_workspace(x_view_ptr_copy, gperm, gn);
        //printf( " > after partial-pivot:\n" );
        //for (Int i = 0; i < gn; i++) printf( " %d %.16e %.16e\n",i, x_view_ptr_copy(i),y_view_ptr_copy(i) );
        //printf( "\n" );
    }

#else
    // Non-transpose solve option
    if (Options.blk_matching != 0) {
        // apply mwm+amd row scaling from numeric
        for(Int i = 0; i < gn; i++) {
            Int row = order_blk_mwm_array(symbolic_row_iperm_array(i));
            y_view_ptr_scale(i) = scale_row_array(row) * _y[i];
            //printf( " symbolic_row_iperm(%d) = %d\n",i,symbolic_row_iperm_array(i) );
            //printf( " scale_row(%d) = %e\n",row,scale_row_array(row) );
        }
        //printf( " > after scale:\n" );
        //for (Int i = 0; i < gn; i++) printf( " > y(%d) = %.16e\n",i,y_view_ptr_scale(i) );

        // apply mwm row-perm from nummeric
        permute_inv_and_init_for_solve(&(y_view_ptr_scale(0)), x_view_ptr_copy, y_view_ptr_copy, perm_inv_comp_array, gn);
        //printf( " > after symbolic-perm:\n" );
        //for (Int i = 0; i < gn; i++) printf( " > y(%d) = %.16e, x(%d) = %.16e\n",i,y_view_ptr_scale(i), i,x_view_ptr_copy(i) );

        // apply row-perm from symbolic
        permute_with_workspace(x_view_ptr_copy, numeric_row_iperm_array, gn);
    } else {
        permute_inv_and_init_for_solve(_y, x_view_ptr_copy, y_view_ptr_copy, perm_inv_comp_array, gn);
    }
    //printf( " > after perm:\n" );
    //for (Int i = 0; i < gn; i++) printf( " %d %.16e %.16e\n",i, x_view_ptr_copy(i),y_view_ptr_copy(i) );
    //printf( "\n" );

    if (Options.no_pivot == BASKER_FALSE) {
        // apply partial pivoting from numeric
        //for (Int i = 0; i < gn; i++) printf( " gperm(%d) = %d\n",i,gperm(i) );
        permute_inv_with_workspace(x_view_ptr_copy, gperm, gn);
        //printf( " > after partial-pivot:\n" );
        //for (Int i = 0; i < gn; i++) printf( " %d %.16e %.16e\n",i, x_view_ptr_copy(i),y_view_ptr_copy(i) );
        //printf( "\n" );
    }

    // solve
    //for (Int i = 0; i < gn; i++) printf( " %d %.16e\n",i,x_view_ptr_copy(i) );
    //printf( "\n" );
    solve_interfacetr(x_view_ptr_copy, y_view_ptr_copy); //x is now permuted rhs; y is 0 
    //for (Int i = 0; i < gn; i++) printf( " %d %.16e %.16e\n",i,x_view_ptr_copy(i),y_view_ptr_copy(i) );
    //printf( "\n" );

    if (Options.blk_matching != 0) {
        // apply amd col-permutation from numeric
        permute_and_finalcopy_after_solve(&(y_view_ptr_scale(0)), x_view_ptr_copy, y_view_ptr_copy, numeric_col_iperm_array, gn);
        //for (Int i = 0; i < gn; i++) printf( " > %d:%d: %.16e %.16e -> %.16e\n",i,numeric_col_iperm_array(i),x_view_ptr_copy(i),y_view_ptr_copy(i), y_view_ptr_scale(i));

        const Int poffset = btf_tabs(btf_tabs_offset);
        for (Int i = 0; i < poffset; i++) {
            x_view_ptr_copy(i) = y_view_ptr_scale(i);
        }
        for (Int i = poffset; i < gn; i++) {
            y_view_ptr_copy(i) = y_view_ptr_scale(i);
        }
    }
    permute_and_finalcopy_after_solve(_x, x_view_ptr_copy, y_view_ptr_copy, perm_comp_array, gn);
    //for (Int i = 0; i < gn; i++) printf( " %d:%d; %e %e, %e %e\n",i,perm_comp_array(i),x_view_ptr_copy(i),y_view_ptr_copy(i), _x[i],scale_col_array(i));

    if (Options.blk_matching != 0) {
      for(Int i = 0; i < gn; i++) {
        Int col = symbolic_col_iperm_array(i);
        _x[i] = scale_col_array(col) * _x[i];
      }
    }
#endif
    return 0;
  }


  template <class Int, class Entry, class Exe_Space>
  BASKER_INLINE
  int Basker<Int,Entry,Exe_Space>::solve_interfacetr
  ( 
   ENTRY_1DARRAY & x, // x is permuted rhs at input
   ENTRY_1DARRAY & y  // y is 0 at input 
  )
  {
    #ifdef BASKER_DEBUG_SOLVE_RHS
    printf("\n\n");
    printf("X: \n");
    for(Int i = 0; i < gn; i++)
    {
      printf("%f, " , x(i));
    }
    printf("\n\n");
    printf("RHS: \n");
    for(Int i =0; i < gm; i++)
    {
      printf("%f, ", y(i)); 
    }
    printf("\n\n");
    #endif

    if(Options.btf == BASKER_FALSE)
    {
      // ND Solve option ONLY
      if(btf_tabs_offset != 0)
      {
        serial_solvetr(x,y);
      }
    }
    else
    {
      // BTF partition present; ND partition also solved within routine if "big block" present
      //A\y -> y
      serial_btf_solvetr(x,y);
    }

    #ifdef BASKER_DEBUG_SOLVE_RHS
    printf("\n\n");
    printf("X: \n");
    for(Int i = 0; i < gn; i++)
    {
      printf("%lf, " , x(i));
    }
    printf("\n\n");
    printf("RHS: \n");
    for(Int i = 0; i < gm; i++)
    {
      printf("%f, ", y(i)); 
    }
    printf("\n\n");
    #endif

    return 0;
  }
  

  template <class Int, class Entry, class Exe_Space>
  BASKER_INLINE
  int Basker<Int,Entry,Exe_Space>::serial_solvetr
  (
   ENTRY_1DARRAY & x, // Permuted rhs at input
   ENTRY_1DARRAY & y  // 0 at input
  )
  {
    //UT\x -> y
    // x <- was overwritten by permuted b
    // TODO transpose interpretation
    serial_forward_solvetr(x,y);

    //printVec(y,gn);

    for(Int i =0; i<gn; ++i)
    {
      x(i) = 0;
    }
    //LT\y -> x
    // y is intermediate solution; x is final permuted solution
    // TODO transpose interpretation
    serial_backward_solvetr(y,x);

    return 0;
  }//end serial solve()


  template <class Int, class Entry, class Exe_Space>
  BASKER_INLINE
  int Basker<Int,Entry,Exe_Space>::serial_btf_solvetr
  (
   ENTRY_1DARRAY & x, // Permuted rhs at input
   ENTRY_1DARRAY & y  // 0 at input
  )
  {

    // 1. Solve BTF_A ND region
    //now do the forward backward solve
    //UT\x ->y
    // TODO internal transpose interpretation
    serial_forward_solvetr(x,y);
    #ifdef BASKER_DEBUG_SOLVE_RHS
    printf("Done, serial_forward \n");
    printf("\n x \n");
    printVec(x, gn);
    printf("\n y \n");
    printVec(y, gn);
    printf("\n\n");
    #endif
    //LT\y->x
    // TODO internal transpose interpretation
    serial_backward_solvetr(y,x);
    #ifdef BASKER_DEBUG_SOLVE_RHS
    printf("Done, serial_backward \n");
    printf("\n x \n");
    printVec(x, gn);
    printf("\n y \n");
    printVec(y, gn);
    printf("\n\n");
    #endif

    //copy lower part down
    #ifdef BASKER_DEBUG_SOLVE_RHS
    printf("copying lower starting: %ld \n", (long)btf_tabs[btf_tabs_offset]);
    #endif

    // 2. Update B (if present)
    //BTF_B*y -> x
    if(btf_tabs_offset !=  0)
    {
    // TODO internal transpose interpretation
      neg_spmv_perm(BTF_B, y, x);
    }

    #ifdef BASKER_DEBUG_SOLVE_RHS
    printf("Done, SPMV BTF_B UPDATE \n");
    printf("\n x \n");
    printVec(x, gn);
    printf("\n y \n");
    printVec(y, gn);
    printf("\n\n");
    #endif

    // 3. BTF: Start in C upper-left block and move down the diag
    // OLD In first level, only do U\L\x->y
    // TODO In *last* level, only do LT\UT\x->y (no off-diag update needed)
    // NOTE: nblocks == btf_nblks - btf_tabs_offset
    for(Int b = 0; b < (btf_nblks-btf_tabs_offset); b++)
    {

      //---Lower solve (transpose)
      // TODO replace with UT(C) = UBTF(b) x
      BASKER_MATRIX &UTC = UBTF(b);
    #ifdef BASKER_DEBUG_SOLVE_RHS
      printf("\n\n btf b: %ld (%d x %d)\n", (long)b, (int)UTC.nrow, (int)UTC.ncol);
    #endif

      // TODO internal transpose interpretation
      //L(C)\x -> y 
      // TODO: replace with UT(C)\x -> y x
      lower_tri_solve(UTC,x,y);

      //printVec(y,gn);

      // TODO replace with LT(C) = LBTF(b) x
      BASKER_MATRIX &LTC = LBTF(b);
      // TODO internal transpose interpretation
      //U(C)\x -> y
      // TODO: replace with LT(C)\x -> y x
      upper_tri_solve(LTC,x,y);

      #ifdef BASKER_DEBUG_SOLVE_RHS
      printf("Before spmv\n");
      printf("Inner Vector y print\n");
      printVec(y, gn);
      printf("Inner Vector x print\n");
      printVec(x, gn);
      printf("\n");
      #endif

      //-----Update BTF block off-diag
      //if(b > btf_tabs_offset)
      // TODO Add check, do not need update on final lower-right block?
      {
        //x = x - BTF_C*y;
        // TODO internal transpose interpretation
        spmv_BTF(b+btf_tabs_offset, BTF_C, x, y);
      }

      #ifdef BASKER_DEBUG_SOLVE_RHS
      printf("After spmv\n");
      printf("Inner Vector y print\n");
      printVec(y, gn);
      printf("Inner Vector x print\n");
      printVec(x, gn);
      #endif

      //BASKER_MATRIX &LTC = LBTF[b];
      //LT\x -> y
      //upper_tri_solve(LTC,x,y);
    }

    #ifdef BASKER_DEBUG_SOLVE_RHS
    printf("Done, BTF-C Solve \n");
    printf("\n x \n");
    printVec(x, gn);
    printf("\n y \n");
    printVec(y, gn);
    printf("\n\n");
    #endif

    return 0;
  }//end serial_btf_solvetr


  template <class Int, class Entry, class Exe_Space>
  BASKER_INLINE
  int Basker<Int,Entry,Exe_Space>::serial_forward_solvetr
  (
   ENTRY_1DARRAY & x, // modified rhs
   ENTRY_1DARRAY & y  // partial solution
  )
  {
    #ifdef BASKER_DEBUG_SOLVE_RHS
    printf("Called serial forward solve \n");
    #endif

#ifdef TRANSPOSE_SOLVER
    //Forward solve on AT
    // Track offset into UT_{first,second}
    Int soffset = 0;
    // Iterate over the blocks by block-column
    for(Int b = 0; b < tree.nblks; ++b)
    {
      // LL_size(col) also acts as size for UT row
      auto col_nnzs = LL_size(b);
      auto diag_first = UT_first(soffset);
      auto diag_second = UT_second(soffset);
    #ifdef BASKER_DEBUG_SOLVE_RHS
      printf("Upper Transpose Solve blk: %d size=%d\n",b,(int)LU_size(b));
    #endif

      // This will be UT, which will require changing indexing pattern for U()(), but accessed in transpose fashion; then interpret U as CRS for UT
      BASKER_MATRIX &UTD = LU(diag_first)(diag_second);
      //UT\x -> y 
    // TODO transpose interpretation
      lower_tri_solve(UTD, x, y);

      //Update offdiag
      // TODO - interpret blocks as falling along a row rather than column for updates
      // Iterate over the row via new counter from shylubasker_tree.hpp: matrix_to_views_2D
      for(Int bb = 1; bb < col_nnzs; ++bb)
      {
        auto first = UT_first(soffset+bb);
        auto second = UT_second(soffset+bb);
        #ifdef BASKER_DEBUG_SOLVE_RHS
        printf("UT Solver Update blk: %d %d \n", first, second);
        #endif

        BASKER_MATRIX &UTO = LU(first)(second);
        //x = x - UTD*y;
        // TODO spmv perm needs matrix-transpose interpretation - UTO as input
        neg_spmv_perm(UTO, y, x);
      }
      soffset += col_nnzs;
    }
#else
    //Forward solve on A
    for(Int b = 0; b < tree.nblks; ++b)
    {
    #ifdef BASKER_DEBUG_SOLVE_RHS
      printf("Upper Transpose Solve blk: %d size=%d\n",b,(int)LU_size(b));
    #endif

      // This will be UT, which will require changing indexing pattern for U()(), but accessed in transpose fashion; then interpret U as CRS for UT
      //
      BASKER_MATRIX &L = LL(b)(0);
      //L\x -> y 
    // TODO transpose interpretation
      lower_tri_solve(L, x, y);

      //Update offdiag
      // TODO - interpret blocks as falling along a row rather than column for updates
      // Iterate over the row via new counter from shylubasker_tree.hpp: matrix_to_views_2D
      for(Int bb = 1; bb < LL_size(b); ++bb)
      {
        #ifdef BASKER_DEBUG_SOLVE_RHS
        printf("Lower Solver Update blk: %d %d \n",
            b, bb);
        #endif

        BASKER_MATRIX &LD = LL(b)(bb);
        //x = x - LD*y;
    // TODO transpose interpretation
        neg_spmv_perm(LD, y, x);
      }
      //printVec(y,gn);
    }
#endif

    #ifdef BASKER_DEBUG_SOLVE_RHS
    printf("Done forward solve A \n");
    printVec(y, gn);
    #endif

    return 0;
  }//end serial_forward_solvetr()

  template<class Int, class Entry, class Exe_Space>
  BASKER_INLINE
  int Basker<Int,Entry,Exe_Space>::serial_backward_solvetr
  (
   ENTRY_1DARRAY & y,
   ENTRY_1DARRAY & x
  )
  {
    #ifdef BASKER_DEBUG_SOLVE_RHS
    printf("called serial backward solve \n");
    #endif

#ifdef TRANSPOSE_SOLVER
    //Backward solve on AT
    // Track offset into LT_{first,second} - should point at final entry of each "chunk" of pointer offsets
    Int eoffset = LT_first.extent(0)-1;
    // Iterate over the blocks by block-column from final column backwards
    for(Int b = tree.nblks-1; b >=0; b--)
    {
      #ifdef BASKER_DEBUG_SOLVE_RHS
      printf("LT solve blk: %d \n", b);
      #endif

      // LU_size(col) also acts as size for LT row
      //   final entry is the diag in each offset chunk
      //   FIXME - need "end" eoffset, traversing backwards through pointer offset
      auto col_nnzs = LU_size(b);  // Will subtract this off the end...
      auto diag_first = LT_first(eoffset);
      auto diag_second = LT_second(eoffset);

      //LT\y -> x
      BASKER_MATRIX &LTD = LL(diag_first)(diag_second);
      // TODO LT\y -> x
      upper_tri_solve(LTD,y,x); // NDE: y , x positions swapped...
                              //      seems role of x and y changed...

      // TODO: Order may need to change for transpose
      //for(Int bb = LU_size(b)-2; bb >= 0; bb--) // - 2 to skip the diag; the - 1 to bump from size to actual index
      for(Int bb = 1; bb < col_nnzs; bb++) // - 2 to skip the diag; the - 1 to bump from size to actual index
      {
        auto first = LT_first(eoffset-bb);
        auto second = LT_second(eoffset-bb);
        #ifdef BASKER_DEBUG_SOLVE_RHS
        printf("Upper solver spmv: %d %d \n", b, bb);
        #endif

        // y = y - LBT*x;
        BASKER_MATRIX &LBT = LL(first)(second);
        // TODO spmv needs transpose interpretation of matrix
        neg_spmv(LBT,x,y);
      }
      eoffset -= col_nnzs;
    }//end over all blks
#else
    // TODO: Order may need to change for transpose
    for(Int b = tree.nblks-1; b >=0; b--)
    {
      #ifdef BASKER_DEBUG_SOLVE_RHS
      printf("Upper solve blk: %d \n", b);
      #endif

      //U\y -> x
      BASKER_MATRIX &U = LU(b)(LU_size(b)-1);
      // TODO LT\y -> x
      upper_tri_solve(U,y,x); // NDE: y , x positions swapped...
                              //      seems role of x and y changed...

      // TODO: Order may need to change for transpose
      for(Int bb = LU_size(b)-2; bb >= 0; bb--)
      {
        #ifdef BASKER_DEBUG_SOLVE_RHS
        printf("Upper solver spmv: %d %d \n",
            b, bb);
        #endif

        //y = y - UB*x;
        // TODO y = y - LBT*x
        BASKER_MATRIX &UB = LU(b)(bb);
        neg_spmv(UB,x,y);
      }
    }//end over all blks
#endif

    #ifdef BASKER_DEBUG_SOLVE_RHS
    printf("Done with Upper Solve: \n");
    printVec(x, gn);
    #endif

    return 0;
  }//end serial_backward_solvetr()


  //Horrible, cheap spmv
  //y = M*x
  template <class Int, class Entry, class Exe_Space>
  BASKER_INLINE
  int Basker<Int,Entry,Exe_Space>::spmv
  (
    BASKER_MATRIX &M,
    ENTRY_1DARRAY x,
    ENTRY_1DARRAY y
  )
  {
    //Add checks
    #ifdef BASKER_DEBUG_SOLVE_RHS
    printf("SPMV. scol: %d ncol: %d nnz: %d \n",
        M.scol, M.ncol, M.nnz);
    M.info();
    #endif

    const Int bcol = M.scol;
    const Int brow = M.srow;
    //for(Int k=M.scol; k < (M.scol+M.ncol); k++)
    for(Int k = 0; k < M.ncol; ++k)
    {
      const auto xkbcol = x(k+bcol);
      const Int istart = M.col_ptr(k);
      const Int iend = M.col_ptr(k+1);

      //for(Int i = M.col_ptr(k); i<M.col_ptr(k+1); ++i)
      for(Int i = istart; i<iend; ++i)
      {
        const Int j = M.row_idx(i);

        //y(j+brow) += M.val(i)*x(k+bcol);
        y(j+brow) += M.val(i)*xkbcol;

      }
    }
    return 0;
  }//spmv


  //y = y - M*x
  template <class Int, class Entry, class Exe_Space>
  BASKER_INLINE
  int Basker<Int,Entry,Exe_Space>::neg_spmv
  (
   BASKER_MATRIX &M,
   ENTRY_1DARRAY x, 
   ENTRY_1DARRAY y  
  )
  {
    //Add checks
    #ifdef BASKER_DEBUG_SOLVE_RHS
    printf("SPMV. scol: %d ncol: %d \n", M.scol, M.ncol);
    #endif

    const Int bcol = M.scol;
    const Int msrow = M.srow;
    //const Int brow = M.srow;
    for(Int k=0; k < M.ncol; ++k)
    {
      const auto xkbcol = x(k+bcol);
      const Int istart = M.col_ptr(k);
      const Int iend = M.col_ptr(k+1);
      //for(Int i = M.col_ptr(k); i < M.col_ptr(k+1); ++i)
      for(Int i = istart; i < iend; ++i)
      {
        const Int j = M.row_idx(i) + msrow;

        //y(j) -= M.val(i)*x(k+bcol);
        y(j) -= M.val(i)*xkbcol;
      }
    }

    return 0;
  }//neg_spmv


  // x <- x - M*y , gperm applied to row_id writing to x
  template <class Int, class Entry, class Exe_Space>
  BASKER_INLINE
  int Basker<Int,Entry,Exe_Space>::neg_spmv_perm
  (
   BASKER_MATRIX &M,
   ENTRY_1DARRAY &y, 
   ENTRY_1DARRAY &x  
  )
  {
    //Add checks
    #ifdef BASKER_DEBUG_SOLVE_RHS
    printf("SPMV. scol: %d ncol: %d \n", M.scol, M.ncol);
    #endif

    const Int bcol = M.scol;
    const Int msrow = M.srow;

    //for(Int k=M.scol; k < (M.scol+M.ncol); k++)
    for(Int k=0; k < M.ncol; ++k)
    {
      const Int istart = M.col_ptr(k);
      const Int iend   = M.col_ptr(k+1);
      const auto ykbcol = y(k+bcol);

      //for(Int i = M.col_ptr(k); i < M.col_ptr(k+1); ++i) 
      for(Int i = istart; i < iend; ++i) //NDE retest with const vars, scope tightly
      {
        // const Int j = M.row_idx(i) + msrow;

        const Int j = (Options.no_pivot == BASKER_FALSE) ? 
                       gperm(M.row_idx(i) + msrow) :
                       (M.row_idx(i) + msrow) ;

        x(j) -= M.val(i)*ykbcol;
      }
    }

    return 0;
  }//neg_spmv


  //M\x = y
  template <class Int, class Entry, class Exe_Space>
  BASKER_INLINE
  int Basker<Int,Entry,Exe_Space>::lower_tri_solve
  (
   BASKER_MATRIX &M,
   ENTRY_1DARRAY &x, 
   ENTRY_1DARRAY &y  
  )
  {
    const Int bcol = M.scol;
    const Int brow = M.scol; // FIXME Should be srow?

    //M.info();

    for(Int k = 0; k < M.ncol; ++k)
    {
      //Test if zero pivot value
      #ifdef BASKER_DEBUG_SOLVE_RHS
      BASKER_ASSERT(M.val[M.col_ptr[k]]!=0.0, "LOWER PIVOT 0");
      //printf("Lower tri.  k: %d out: %f in: %f piv: %f \n",
      //   k+bcol, y[k+bcol], x[k+bcol], M.val[M.col_ptr[k]]);
      #endif

      // TODO NDE: Need to make sure this is properly checked in numeric factorization
      /*
      if(M.val[M.col_ptr[k]] == 0.0) 
      {
        printf("Lower Pivot: %d %f \n", 
            M.row_idx[M.col_ptr[k]],
            M.val[M.col_ptr[k]]);
        return -1;
      }
      */

      //Replace with Entry divide in future
      const Int istart = M.col_ptr(k);
      const Int iend = M.col_ptr(k+1);

      //printf( " %d %d %e\n",M.row_idx(M.col_ptr(k)),k,M.val(M.col_ptr(k)));
      //printf( " -> %e %e (%d, %d,%d)\n",y(k+brow),x(k+bcol),k,brow,bcol );
      y(k+brow) = x(k+bcol) / M.val(M.col_ptr(k));

      const auto ykbcol = y(k+bcol);
      //for(Int i = M.col_ptr(k)+1; i < M.col_ptr(k+1); ++i)
      for(Int i = istart+1; i < iend; ++i)
      {
        const Int j = (Options.no_pivot == BASKER_FALSE) ? 
                        gperm(M.row_idx(i)+brow) :
                             (M.row_idx(i)+brow) ;

        #ifdef BASKER_DEBUG_SOLVE_RHS
        BASKER_ASSERT(j != BASKER_MAX_IDX,"Using nonperm\n");
        #endif

        //x(j) -= M.val(i)*y(k+bcol);
        x(j) -= M.val(i)*ykbcol;
      } //over all nnz in a column

    } //over each column

    return 0;
  } //end lower_tri_solve


  //U\x = y
  // Note: In serial_backward_solve usage, the vars do not match up
  template <class Int, class Entry, class Exe_Space>
  BASKER_INLINE
  int Basker<Int, Entry, Exe_Space>::upper_tri_solve
  (
   BASKER_MATRIX &M,
   ENTRY_1DARRAY &x,
   ENTRY_1DARRAY &y 
  )
  {
    const Int bcol = M.scol;
    const Int brow = M.srow;

    for(Int k = M.ncol; k >= 1; k--)
    {

      #ifdef BASKER_DEBUG_SOLVE_RHS
      BASKER_ASSERT(M.val[M.col_ptr[k]-1]!=0.0,"UpperPivot\n");
      printf("Upper Tri Solve, scol: %d ncol: %d \n",
        M.scol, M.ncol);

      #endif

      // TODO NDE: Need to make sure this is properly checked in numeric factorization
      /*
      if(M.val(M.col_ptr(k)-1)==0)
      {
        printf("Upper pivot: %d %f \n",
            M.row_idx[M.col_ptr[k]-1],
            M.val[M.col_ptr[k]-1]);
        return -1;
      }
      */

      //Comeback and do with and entry divide
      const Int istart = M.col_ptr(k);
      const Int iend = M.col_ptr(k-1);

      y(k+brow-1)  =  x(k+bcol-1) / M.val(M.col_ptr(k)-1);

      const auto ykbcol = y(k+bcol-1);
      //for(Int i = M.col_ptr(k)-2; i >= M.col_ptr(k-1); --i) 
      for(Int i = istart-2; i >= iend; --i)
      {
        const Int j = M.row_idx(i) + brow; //NDE: why isn't gperm here like above?

        //x(j) -= M.val(i) * y(k+bcol-1);
        x(j) -= M.val(i) * ykbcol;
      }

    }//end over all columns

    return 0;
  } //end upper_tri_solve


  // x <- x - M*y , gperm applied to row_id writing to x
  //   M is locally indexed within its block
  //   x, y require extra offset of the block for global indexing
  template <class Int, class Entry, class Exe_Space>
  BASKER_INLINE
  int Basker<Int,Entry,Exe_Space>::spmv_BTF
  (
   Int tab,
   BASKER_MATRIX &M,
   ENTRY_1DARRAY &x, // modified rhs
   ENTRY_1DARRAY &y  // intermediate solution
  )
  {
    //Tab = block in    
    const Int bcol = btf_tabs(tab)- M.scol;
    const Int mscol = M.scol;
    const Int brow = M.srow;
    const Int ecol = btf_tabs(tab+1) - M.scol;

    #ifdef BASKER_DEBUG_SOLVE_RHS
    Int erow = 0;
    if(tab > 0)
    {
      erow = btf_tabs(tab);
    }
    else
    {
      erow = brow-1;
    }

    printf("BTF_UPDATE, TAB: %d [%d %d] [%d %d] \n",
        tab, brow, erow, bcol, ecol);
    #endif

    //loop over each column
    for(Int k = bcol; k < ecol; ++k)
    {
      //const Int kcol = k+M.scol;
      const Int istart = M.col_ptr(k);
      const Int iend = M.col_ptr(k+1);

      const auto ykmcol = y(k+mscol);
      //for(Int i = M.col_ptr(k); i < M.col_ptr(k+1); ++i) 
      for(Int i = istart; i < iend; ++i)
      {
        // j == rowid; gperm is partial pivoted row, ccs
        //   TODO interpreting as crs transpose, this would be done at the outer loop level (column)?
        const Int j = (Options.no_pivot == BASKER_FALSE) ? 
                        gperm(M.row_idx(i)+brow) :
                        (M.row_idx(i)+brow) ;

       #ifdef BASKER_DEBUG_SOLVE_RHS
        printf("BTF_UPDATE-val, j: %d x: %f y: %f, val: %f \n",
            j, x[j], y[k+M.scol], M.val[i]);
       #endif

        //x(j) -= M.val(i)*y(k+M.scol);
        x(j) -= M.val(i)*ykmcol;
      } //over all nnz in row

    } // end for over col

    return 0;
  } //end spmv_BTF();
  
} //end namespace BaskerNS
#endif //end ifndef basker_solver_rhs
