#ifndef SHYLUBASKER_SOLVE_RHS_TR_HPP
#define SHYLUBASKER_SOLVE_RHS_TR_HPP

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


namespace BaskerNS
{

  template <class Int, class Entry, class Exe_Space>
  BASKER_INLINE
  int Basker<Int,Entry,Exe_Space>::solve_interfacetr
  (
   Int _nrhs,
   Entry *_x, // Solution
   Entry *_y  // rhs
  )
  {
    for(Int r = 0; r < _nrhs; r++)
    {
      solve_interface(&(_x[r*gm]), &(_y[r*gm]));
    }

    return 0;
  }//end solve_interface(_nrhs,x,y);

  template <class Int, class Entry, class Exe_Space>
  BASKER_INLINE
  int Basker<Int,Entry,Exe_Space>::solve_interfacetr
  (
   Entry *_x, // Solution (len = gn)
   Entry *_y  // rhs
  )
  {
    // TODO: Add other permutation options
    printf("---- Pre-Solve printOptions: ----\n");
    Options.printOptions();
    printf( " >> gperm; print non-identity gperm output\n" );
    for (Int i = 0; i < gn; i++) {
      if (gperm(i) != i) {
        printf( "  >> gperm(%d) = %d\n", i, gperm(i) );
      }
    }
    printf( " >> gpermi; print non-identity gpermi output\n" );
    for (Int i = 0; i < gn; i++) {
      if (gpermi(i) != i) {
        printf( "  >> gpermi(%d) = %d\n", i, gpermi(i) );
      }
    }

    // Transpose: Swap permutation order - only handles case without 
  // TODO: determine which case+options the below routine supports - i.e. WHAT to enable and WHAT to disable
    std::cout << "  Permute 1a: permute_and_init_for_solve" << std::endl;
    for (Int i = 0; i < gn; i++) printf( "  perm_comp_array(%d) = %d\n",i,perm_comp_array[i] );
    for (Int i = 0; i < gn; i++) printf( "  (unused) perm_inv_comp_array(%d) = %d\n",i,perm_inv_comp_array[i] );
    permute_and_init_for_solve(_y, x_view_ptr_copy, y_view_ptr_copy, perm_comp_array, gn);
      // rhs content from _y has been permuted and copied to x_view_ptr_copy which will act as the rhs-to-update during solve calls; y_view_ptr_copy will store the pivots (i.e. solutions)


    // TODO: Add this interface, which will dispatch to the necessary component routines
    solve_interfacetr(x_view_ptr_copy, y_view_ptr_copy); //input: x is permuted rhs; y is 0  |  output: solution "spread" between x and y??


    // TODO: What/where does the gperm(i) need to be applied in the transpose solve context????
    if (Options.no_pivot == BASKER_FALSE) {
      // apply partial pivoting from numeric
      //for (Int i = 0; i < gn; i++) printf( " gperm(%d) = %d\n",i,gperm(i) );
      std::cout << "  Permute no_pivot == false: permute_with_workspace" << std::endl;
      permute_with_workspace(x_view_ptr_copy, gperm, gn); // TODO Should this be with gpermi????
    }

    std::cout << "  Permute 2b: permute_inv_and_finalcopy_after_solve" << std::endl;
    for (Int i = 0; i < gn; i++) printf( "  perm_inv_comp_array(%d) = %d\n",i,perm_inv_comp_array[i] );
    permute_inv_and_finalcopy_after_solve(_x, x_view_ptr_copy, y_view_ptr_copy, perm_inv_comp_array, gn);
      // final solution is permuted back to original ordering and copied to _x

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

    // FIXME Add other options
    serial_btf_solve_tr(x,y);

    return 0;
  }

  // This results in the "solution" of the diagonal block transpose solve - input Matrix is "upper triangular" via transpose
  // Called after upper_tri_solve_tr
  // x should be mod. rhs, though x == y at input in range of M
  template <class Int, class Entry, class Exe_Space>
  BASKER_INLINE
  int Basker<Int,Entry,Exe_Space>::lower_tri_solve_tr
  (
   BASKER_MATRIX &M,
   ENTRY_1DARRAY &x, // mod rhs at input; to be garbage
   ENTRY_1DARRAY &y, // y != x at input likely (U^T likely not unit diag); to be soln
   Int offset
  )
  {
    // block diagonal offset stays the same as non-transpose
    const Int bcol = M.scol + offset;
    const Int brow = M.scol + offset; // FIXME to M.srow? srow == scol should be true to diagonal blocks...

    std::cout << "  L^T begin: bcol = " << bcol << "  brow = " << brow << "  offset = " << offset << std::endl;
    M.print();
    // k is a col of L CCS; for transpose solve, treat k as a row of L^T
    for(Int k = M.ncol-1; k >= 0; --k)
    {
      //Test if zero pivot value
      #ifdef BASKER_DEBUG_SOLVE_RHS
      BASKER_ASSERT(M.val[M.col_ptr[k]]!=0.0, "LOWER PIVOT 0");
      #endif
      std::cout << "  LT: k = " << k << "  bcol = " << bcol << "  brow = " << brow << std::endl;

      //const Int istart = M.col_ptr(k)+1; // start one after diagonal; pivot solved after removing all non-diag contributions from rhs
      // these will be the offsets for col k, which we are treating as transposed matrix row k
      const Int istart = M.col_ptr(k);
      const Int iend = M.col_ptr(k+1);

      // Start 1 past the diagonal, skip for row updates before final pivot; 
      // For L, the diagonal is the first entry of the column (or transposed row), as opposed to being final entry with U
      // For the first iteration, this loop should not be entered because the only entry is the diagonal
      for(Int i = istart+1; i < iend; ++i)
      {
        // global row id for dense vectors
        // TODO gperm: does writing to y requires indirection via partial pivot gperm location? Or do we only do this when accessing row/col from L?
        // TODO gperm ilke so?
        //const Int j = M.row_idx(i)+brow;
        const Int j = (Options.no_pivot == BASKER_FALSE) ? 
                        gperm(M.row_idx(i)+brow) :
                             (M.row_idx(i)+brow) ;

        #ifdef BASKER_DEBUG_SOLVE_RHS
        BASKER_ASSERT(j != BASKER_MAX_IDX,"Using nonperm\n");
        #endif

        //x(k+brow) -= M.val(i)*x(j); // FIXME x(j) will be overwritten as loop iterates??? should be y(j)?
        std::cout << "    inner loop: i = " << i << "  j = " << j << "  brow = " << brow << std::endl;
        std::cout << "    Before update to k+brow: x(k+brow) = " << x(k+brow) << "  y(j) = " << y(j) << "  M.val(i) = " << M.val(i) << std::endl;
        x(k+brow) -= M.val(i)*y(j);
        std::cout << "    After update to k+brow: x(k+brow) = " << x(k+brow) << std::endl;
      } //over all nnz in a column
      // Complete solution and store in rhs x 
      //x(k+brow) = x(k+bcol) / M.val(M.col_ptr(k)); // FIXME should be writing to y...
      std::cout << "  LT Pre-row k solve: y(k+brow) = " << y(k+brow) << " x(k+bcol) = " << x(k+bcol) << " M.val(istart) (diag entry) = " << M.val(istart) << std::endl;
      y(k+brow) = x(k+bcol) / M.val(M.col_ptr(k));
      std::cout << "  After row k solve: y(k+brow) = " << y(k+brow) << std::endl;
      // TODO gperm: Apply gperm when writing to y for transpose?
      // store solution in y with indirection via partial pivot gperm location ?
      //y(k+brow) = x(k+bcol);
      
    } //over each column

    return 0;
  } //end lower_tri_solve_tr


  // Input matrix is "lower tri" matrix via transpose
  template <class Int, class Entry, class Exe_Space>
  BASKER_INLINE
  int Basker<Int, Entry, Exe_Space>::upper_tri_solve_tr
  (
   BASKER_MATRIX &M, // U^T i.e. lower tri matrix with diagonal as last entry or indices pointer
   ENTRY_1DARRAY &x,
   ENTRY_1DARRAY &y,
   Int offset
  )
  {
    // block diagonal offset stays the same as non-transpose
    const Int bcol = M.scol + offset;
    const Int brow = M.srow + offset;

    std::cout << "  U^T begin: bcol = " << bcol << "  brow = " << brow << "  offset = " << offset << std::endl;
    M.print();
    // Solve initial row 0 transpose (i.e. col 0) before inner loop
    std::cout << "  UT Pre-row 0 solve: y(brow) = " << y(brow) << " x(bcol) = " << x(bcol) << " M.val(M.col_ptr(1)-1) (diag entry) = " << M.val(M.col_ptr(1)-1) << std::endl;
    y(brow) = x(bcol) / M.val(M.col_ptr(1)-1);
    std::cout << "  After row 0 solve: y(brow) = " << y(brow) << std::endl;

    // k is a col of U CCS; for transpose solve, treat k as a row of U^T
    for(Int k = 1; k < M.ncol; k++) // k == 0 already handled above
    {
      #ifdef BASKER_DEBUG_SOLVE_RHS
      BASKER_ASSERT(M.val[M.col_ptr[k]-1]!=0.0, "UPPER PIVOT == 0\n");
        /*
        printf("Upper Tri Solve, scol: %d ncol: %d \n",
          M.scol, M.ncol);
        */
      #endif
      std::cout << "  UT: k = " << k << "  bcol = " << bcol << "  brow = " << brow << std::endl;

      const Int istart = M.col_ptr(k);
      const Int iend = M.col_ptr(k+1);

      // skip the diagonal during row updates
      // for U, the diagonal should be stored as last entry of a column (or row for U^T) (not first, like L)
      for(Int i = istart; i < iend-1; ++i)
      {
        const Int j = M.row_idx(i) + brow;
        std::cout << "    inner loop: i = " << i << "  j = " << j << "  brow = " << brow << std::endl;

        std::cout << "    Before update to k+brow: x(k+brow) = " << x(k+brow) << "  y(j) = " << y(j) << "  M.val(i) = " << M.val(i) << std::endl;
        x(k+brow) -= M.val(i)*y(j); // FIXME possible broken update here???
        std::cout << "    After update to k+brow: x(k+brow) = " << x(k+brow) << std::endl;
      }
      // finish the diag 
      std::cout << "  UT Pre-row k solve: y(k+brow) = " << y(k+brow) << " x(k+bcol) = " << x(k+bcol) << " M.val(iend-1) (diag entry) = " << M.val(iend-1) << std::endl;
      y(k+brow) = x(k+bcol) / M.val(iend-1); // y == x in M range assumed true at end of this routine, but not automatic as with non-transpose lower_tri_solve since U^T diagonal is not necessarily 1's
      std::cout << "  After row k solve: y(k+brow) = " << y(k+brow) << std::endl;
      std::cout << "    about to update x: k+bcol = " << k+bcol << " k+brow = " << k+brow << std::endl;
      x(k+bcol) = y(k+brow); // enforce x == y at end to avoid issues with lower_tri_solve_tr
      std::cout << "    x update: x(k+bcol) = " << x(k+bcol) << "  y(k+brow) = " << y(k+brow) << std::endl;

    }//end over all columns
    x(bcol) = y(brow);  // set k == 0 values equal after updates complete

    return 0;
  } //end upper_tri_solve_tr


  // non-transpose A*x=b CCS
  // x1*col1(A) + ... + xn*coln(A) = b

  // transpose spmv
  // row1(AT) CRS == col1(A) CCS
  // transpose A^T*x=b, interpretting A^T as CRS to reuse A pointers since A is not transposed;
  // row1(A^T).*x + row2(A^T).*x ... = b, i.e.
  // col1(A).*x + col2(A).*x ... = b
  template <class Int, class Entry, class Exe_Space>
  BASKER_INLINE
  int Basker<Int,Entry,Exe_Space>::spmv_BTF_tr
  (
   Int tab, // treat this as info to determine where the spmv terminates (i.e. final col of transpose matrix)
   BASKER_MATRIX &M,
   ENTRY_1DARRAY &x, // modified rhs
   ENTRY_1DARRAY &y, // intermediate solution
   bool full
  )
  {
    // What identifying block info needed before starting transpose spmv rhs update?
    //Tab = block in    
    M.print();
    Int bcol = btf_tabs(tab)- M.scol; // FIXME no offset correction needed for transpose case; or, is offset correction needed for BTF_D ??
    Int ecol = btf_tabs(tab+1) - M.scol; // FIXME ditto
    //Int bcol = btf_tabs(tab+1) - M.scol;
    //Int ecol = btf_tabs(tab+2) - M.scol; // FIXME tab+2 will eventually hit out of bounds
    Int brow = M.srow;
    if (ecol > M.ncol) {
      // for D block, btf_tabs(tab+1) > ncol.
      ecol = M.ncol;
    }
    std::cout << "  tab = " << tab << "  M.scol = " << M.scol << "  bcol = " << bcol << "  ecol = " << ecol << std::endl;

    // "Genuine" begins and ends for sanity, not within optimization trick reuse - FIXME is above correct, or should I use below?
    // TODO Confirm above is correct, or determine if changes needed for below...
    // brow == bcol of non-transpose
    // bcol == brow of non-transpose

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

    //loop over each row of transpose (column of original)
    for(Int k = bcol; k < ecol; ++k) {
      const Int istart = M.col_ptr(k);
      const Int iend = M.col_ptr(k+1);
      std::cout << "  spmv_BTF_tr: k = " << k << "  istart = " << istart << "  iend = " << iend << std::endl;

      const Int gk = k+M.scol;

      for(Int i = istart; i < iend; ++i) {
        const auto j = M.row_idx(i);
        //const Int gj = j+M.srow;
        // TODO gperm ilke so?
        const Int gj = (Options.no_pivot == BASKER_FALSE) ? 
                       gperm(M.row_idx(i) + M.srow) :
                       (M.row_idx(i) + M.srow) ;

       #ifdef BASKER_DEBUG_SOLVE_RHS
        printf("BTF_UPDATE-val, j: %d x: %f y: %f, val: %f \n",
            gj, x[gj], y[k+M.scol], M.val[i]);
       #endif
        std::cout << "    inner loop : gk = " << gk << "  j = " << j << "  gj = " << gj << "  bcol = " << bcol << std::endl;

        //if (full || gj < bcol) // FIXME: Clarify that "full" is needed for spmv tr; what is boundary/filter for indices for to allow through (and is this for gperm???); add restriction for transpose columns i.e. row ids to not contribute if row id earlier than a cut-off or lower bound
        if (full || gj < bcol) // bcol will act as a "starting bcol offset" for what portion of the matrix to skip in the spmv^T update; we use the local col id j (row id of non-transpose)
        //if (full || gj < M.scol) // this is right in BTF_C region, but I think breaks in BTF_D???
        {
          std::cout << "     Pre-update j < bcol: x(gk) = " << x(gk) << "  y(gj) = " << y(gj) << "  M.val(i) = " << M.val(i) << std::endl;
          x(gk) -= M.val(i)*y(gj);
          std::cout << "     After update j < bcol: x(gk) = " << x(gk) << std::endl;
        }
      } //over all nnz in row
    } // end for over col

    return 0;
  } //end spmv_BTF_tr;

  template <class Int, class Entry, class Exe_Space>
  BASKER_INLINE
  int Basker<Int,Entry,Exe_Space>::neg_spmv_tr
  (
   BASKER_MATRIX &M,
   ENTRY_1DARRAY x, 
   ENTRY_1DARRAY y,
   Int offset
  )
  {
    //Add checks
    #ifdef BASKER_DEBUG_SOLVE_RHS
    printf("SPMV. scol: %d ncol: %d \n", M.scol, M.ncol);
    #endif

    const Int bcol  = M.scol + offset;
    const Int msrow = M.srow + offset;
    for(Int k=0; k < M.ncol; ++k)
    {
      const Int istart = M.col_ptr(k);
      const Int iend = M.col_ptr(k+1);

      const Int gk = k+bcol;
      for(Int i = istart; i < iend; ++i)
      {
        const Int gj = M.row_idx(i) + msrow;
        //y(gk) -= M.val(i)*x(gj);
        x(gk) -= M.val(i)*y(gj);
      }
    }

    return 0;
  } //neg_spmv_tr

  template <class Int, class Entry, class Exe_Space>
  BASKER_INLINE
  int Basker<Int,Entry,Exe_Space>::neg_spmv_perm_tr
  (
   BASKER_MATRIX &M,
   ENTRY_1DARRAY &y, 
   ENTRY_1DARRAY &x,
   Int offset
  )
  {
    const Int bcol  = M.scol + offset;
    const Int msrow = M.srow + offset;
    #ifdef BASKER_DEBUG_SOLVE_RHS
    printf("SPMV. scol: %d ncol: %d, srow: %d nrow: %d \n", M.scol, M.ncol, M.srow, M.nrow);
    if (Options.no_pivot == BASKER_FALSE) {
      printf("P=[\n");
      for(Int k=0; k < M.nrow; ++k) printf("%d+%d, %d\n",msrow,k,gperm(k+msrow));
      printf("];\n");
    }
    printf("M=[\n");
    for(Int k=0; k < M.ncol; ++k) {
      for(Int i = M.col_ptr(k); i < M.col_ptr(k+1); ++i)
        printf( "%d %d %d %.16e\n",gperm(M.row_idx(i) + msrow)-msrow, M.row_idx(i),k,M.val(i) );
    }
    printf("];\n");
    #endif

    for(Int k=0; k < M.ncol; ++k)
    {
      const Int istart = M.col_ptr(k);
      const Int iend   = M.col_ptr(k+1);

      const Int gk = k+bcol;
      for(Int i = istart; i < iend; ++i) //NDE retest with const vars, scope tightly
      {
        //const Int gj = M.row_idx(i)+msrow;
        // TODO gperm ilke so?
        const Int gj = (Options.no_pivot == BASKER_FALSE) ? 
                       gperm(M.row_idx(i) + msrow) :
                       (M.row_idx(i) + msrow) ;
        x(gk) -= M.val(i)*y(gj);
      }
    }

    return 0;
  } //neg_spmv_perm_tr


  // solve L^T*x=y - transpose means pattern of "backward" solve
  // this is the final stage of the solve
  template <class Int, class Entry, class Exe_Space>
  BASKER_INLINE
  int Basker<Int,Entry,Exe_Space>::l_tran_brfa_solve
  (
   ENTRY_1DARRAY & y, // in: partial solution 
   ENTRY_1DARRAY & x  // out: solution in btfa range
  )
  {
    #ifdef BASKER_DEBUG_SOLVE_RHS
    printf("Called serial forward solve \n");
    #endif

        printf(" Vector y before l_tran_btfa_solve \n");
        printVec(y, gn);
        printf(" Vector x before l_tran_btfa_solve \n");
        printVec(x, gn);

    Int scol_top = btf_tabs[btf_top_tabs_offset]; // the first column index of A
    for(int b = tree.nblks-1; b >= 0; b--)
    {
      std::cout << "\nLT: Block Row b = " << b << "\n\n" << std::endl;
      std::cout << "    LL_size(b) = " << LL_size(b) << std::endl;
      // Update off-diag in the block-row before the diag solve
      for(int bb = LL_size(b)-1; bb > 0; bb--)
      {
        std::cout << "    LT rhs update (BTF_A). bb = " << bb << std::endl;
        BASKER_MATRIX &LD = LL(b)(bb);
        printf("LT update blk (%d, %d): size=(%dx%d) srow=%d, scol=%d\n",b,bb, (int)LD.nrow,(int)LD.ncol, (int)LD.srow,(int)LD.scol);
        LD.print();
        //neg_spmv_perm_tr(LD, y, x, scol_top);
        neg_spmv_perm_tr(LD, x, y, scol_top); // update y as mod. rhs, x as solution
        printf(" Vector y after neg_spmv_perm_tr \n");
        printVec(y, gn);
        printf(" Vector x after neg_spmv_perm_tr \n");
        printVec(x, gn);
      }
      BASKER_MATRIX &L = LL(b)(0);
      std::cout << "  LT solve (BTF_A). b = " << b << std::endl;
      if (L.nrow != 0 && L.ncol != 0) // Avoid degenerate case e.g. empty block following nd-partitioning
        lower_tri_solve_tr(L, y, x, scol_top); // x and y should be equal after in M range...
        printf("LT diag blk (%d, %d): size=(%dx%d) srow=%d, scol=%d\n",b,0, (int)L.nrow,(int)L.ncol, (int)L.srow,(int)L.scol);
        printf(" Vector y after lower_tri_solve_tr \n");
        printVec(y, gn);
        printf(" Vector x after lower_tri_solve_tr \n");
        printVec(x, gn);
    }

        printf(" Vector y after l_tran_btfa_solve \n");
        printVec(y, gn);
        printf(" Vector x after l_tran_btfa_solve \n");
        printVec(x, gn);



#ifdef TRANSPOSE_SOLVER_TRY1_BROKEN
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
      lower_tri_solve_tr(UTD, x, y);

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
        neg_spmv_perm_tr(UTO, y, x);
      }
      soffset += col_nnzs;
    }

    #ifdef BASKER_DEBUG_SOLVE_RHS
    printf("Done forward solve A \n");
    printVec(y, gn);
    #endif
#endif
    return 0;
  }//end l_tran_brfa_solve()


  // U^T*y = x, transpose implies "forward" solve pattern
  template<class Int, class Entry, class Exe_Space>
  BASKER_INLINE
  int Basker<Int,Entry,Exe_Space>::u_tran_btfa_solve
  (
   ENTRY_1DARRAY & x,
   ENTRY_1DARRAY & y
  )
  {
    #ifdef BASKER_DEBUG_SOLVE_RHS
    printf("called serial backward solve \n");
    #endif

        printf(" Vector y before u_tran_btfa_solve \n");
        printVec(y, gn);
        printf(" Vector x before u_tran_btfa_solve \n");
        printVec(x, gn);

    Int scol_top = btf_tabs[btf_top_tabs_offset]; // the first column index of A
    for(Int b = 0; b < tree.nblks; b++)
    {
      std::cout << "\nUT: Block Row b = " << b << "\n\n" << std::endl;
      std::cout << "      LU_size(b) = " << LU_size(b) << std::endl;
      for(Int bb = 0; bb <  LU_size(b)-1; bb++)
      {
        std::cout << "    UT update rhs (BTF_A). bb = " << bb << std::endl;
        // update offdiag corresponding to the block-row
        BASKER_MATRIX &UB = LU(b)(bb);
        printf("UT update blk (%d, %d): size=(%dx%d) srow=%d, scol=%d\n",b,bb, (int)UB.nrow,(int)UB.ncol, (int)UB.srow,(int)UB.scol);
        UB.print();
        neg_spmv_tr(UB, x, y, scol_top);
        printf(" Vector y after neg_spmv_tr \n");
        printVec(y, gn);
        printf(" Vector x after neg_spmv_tr \n");
        printVec(x, gn);
      }
      BASKER_MATRIX &U = LU(b)(LU_size(b)-1);
      std::cout << "    UT solve (BTF_A). b = " << b << std::endl;
      if (U.nrow != 0 && U.ncol != 0) // Avoid degenerate case
        upper_tri_solve_tr(U, x, y, scol_top); // TODO FIXME : order of x y args to this routine
        printf("UT diag blk (%d, %d): size=(%dx%d) srow=%d, scol=%d\n",b,LU_size(b)-1, (int)U.nrow,(int)U.ncol, (int)U.srow,(int)U.scol);
        printf("  Right after upper_tri_solve_tr call\n");
        printf(" Vector y after upper_tri_solve_tr \n");
        printVec(y, gn);
        printf(" Vector x after upper_tri_solve_tr \n");
        printVec(x, gn);
    }


    // TODO possible FIXME: Make sure x == y in the range of BTF_A after this routine, before handoff to L^T*x=y solve
    // assign y vals to x, or vice versa???
    if (BTF_A.ncol > 0) {
      for (Int i = 0; i < BTF_A.ncol; i++) {
        x(scol_top+i) = y(scol_top+i); // TODO Check if this is correct assignment...
      }
    }

        printf(" Vector y after u_tran_btfa_solve \n");
        printVec(y, gn);
        printf(" Vector x after u_tran_btfa_solve \n");
        printVec(x, gn);

#ifdef TRANSPOSE_SOLVER_TRY1_BROKEN
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
      upper_tri_solve_tr(LTD,y,x); // NDE: y , x positions swapped...
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
        neg_spmv_tr(LBT,x,y);
      }
      eoffset -= col_nnzs;
    }//end over all blks

    #ifdef BASKER_DEBUG_SOLVE_RHS
    printf("Done with Upper Solve: \n");
    printVec(x, gn);
    #endif
#endif
    return 0;
  }//end u_tran_btfa_solve()



  template <class Int, class Entry, class Exe_Space>
  BASKER_INLINE
  int Basker<Int,Entry,Exe_Space>::serial_btf_solve_tr
  (
   ENTRY_1DARRAY & x, // Permuted rhs at input
   ENTRY_1DARRAY & y  // 0 at input
  )
  {
    std::cout << " BTF Stats:\n";
    std::cout << "    btf_nblks = " << btf_nblks << std::endl;
    std::cout << "    btf_tabs_offset = " << btf_tabs_offset << std::endl;
    std::cout << "    btf_top_tabs_offset = " << btf_top_tabs_offset << std::endl;
    std::cout << "    btf_top_nblks = " << btf_top_nblks << std::endl;

      printf(" Vector y (before solves)\n");
      printVec(y, gn);
      printf(" Vector x (before solves)\n");
      printVec(x, gn);

    // P1 T
    // BTF_D T solve
    if(btf_top_tabs_offset >  0)
    {
      std::cout << "BTF_D^T begin:" << std::endl;
      for(Int b = 0; b < btf_top_tabs_offset; b++)
      {
        std::cout << "  diag block b = " << b << std::endl;
        // FIXME possibly need to update b input to "b+1" to allow for offset to next rows update? Or maybe need to revise this impl if iterating by col rather than tranpose-row impl?
        if ( b > 0 )
          spmv_BTF_tr(b, BTF_D, x, y, false); // TODO use x and y (not just x,x) to update rhs in x but continue with y as solution
        printf(" Vector y after spmv_BTF_tr\n");
        printVec(y, gn);
        printf(" Vector x after spmv_BTF_tr\n");
        printVec(x, gn);

        BASKER_MATRIX &UC = U_D(b);
        if (UC.nrow != 0 && UC.ncol != 0) // Avoid degenerate case
          upper_tri_solve_tr(UC, x, y); // TODO Want x == y after this op...

        printf( "\n After UT solve (b=%d): x(i),y(i)\n",b ); fflush(stdout);
        for (Int i = 0; i < gn; i++) printf( " %e %e\n",x(i),y(i));
        printf( "\n");

        BASKER_MATRIX &LC = L_D(b);
        if (LC.nrow != 0 && LC.ncol != 0) // Avoid degenerate case
          lower_tri_solve_tr(LC, x, y); // TODO Want x == mod. rhs, y == soln  after this op

        printf( "\n After LT solve (b=%d): x(i),y(i)\n",b ); fflush(stdout);
        for (Int i = 0; i < gn; i++) printf( " %e %e\n",x(i),y(i));
        printf( "\n");

      }
      // ASSUMPTION: x is mod. rhs (and has some garbage in BTF_D range), y stores solution from BTF_D range

    // Update for offdiag BTF_E T
      std::cout << "BTF_E^T update begin:" << std::endl;
      neg_spmv_perm_tr(BTF_E, y, x); // TODO use y as soln, x stores rhs updates
        printf(" Vector y after neg_spmv_perm_tr\n");
        printVec(y, gn);
        printf(" Vector x after neg_spmv_perm_tr\n");
        printVec(x, gn);
      // ASSUMPTION: x is mod. rhs (and has some garbage in BTF_D range), y stores solution from BTF_D range

      Int srow_d = BTF_D.srow;
      Int erow_d = BTF_D.nrow + srow_d;
      for (Int i = srow_d; i < erow_d; i++) {
        x(i) = y(i);
      }
    }


    // P2 T
    // BTF_A T solves
    //std::cout << "BTF_A matrix" << std::endl;
    //BTF_A.print(); // TODO needs guards, print if BTF_A null results in silent runtime fail
    std::cout << "BTF_A serial_backward_solve_tr" << std::endl;
    u_tran_btfa_solve(x,y); // TODO Note: U^T*y=x
    std::cout << "BTF_A serial_forward_solve_tr" << std::endl;
    l_tran_brfa_solve(y,x); // TODO Note: L^T*x=y
    // ASSUMPTION: in BTF_A range, y is mod. rhs (and has some garbage in BTF_A range), x stores solution
        printf(" Vector y after BTF_A solve \n");
        printVec(y, gn);
        printf(" Vector x after BTF_A solve \n");
        printVec(x, gn);

    // Update for offdiag BTF_B T
    if(btf_tabs_offset !=  0)
    {
      std::cout << "BTF_B^T update begin:" << std::endl;
      neg_spmv_perm_tr(BTF_B,x,x); // x is updated rhs in BTF_C range, solution in BTF_D and BTF_A range
        printf(" Vector x after BTF_B update \n");
        printVec(x, gn);
        printf(" Vector y after BTF_B update \n");
        printVec(y, gn);
    }

    // P3 T
    Int nblks_c = btf_nblks-btf_tabs_offset;
    std::cout << "BTF_C^T region:" << std::endl;
    for(Int b = 0;  b < nblks_c; b++) {
      std::cout << "  diag block b = " << b << std::endl;
        // FIXME start with off-diag block update for previous solve
      // Update off-diag
        // FIXME possibly need to update b input to "b+1" to allow for offset to next rows update? Or maybe need to revise this impl if iterating by col rather than tranpose-row impl?
      if ( b > 0 )
        spmv_BTF_tr(b+btf_tabs_offset, BTF_C, x, y, true); // FIXME the offset should be for the termination of the spmv, rather than the start...; should this be true or false???

        printf(" Vector y print (after spmv_BTF_tr update )\n");
        printVec(y, gn);
        printf(" Vector x print (after spmv_BTF_tr update )\n");
        printVec(x, gn);
#if 1
        printf(" Vector y (before upper_tri_tr solves)\n");
        printVec(y, gn);
        printf(" Vector x (before upper_tri_tr solves)\n");
        printVec(x, gn);
#endif
      BASKER_MATRIX &UC = UBTF(b);
        std::cout << "UC  b: " << b << std::endl;
        UC.print();
      if (UC.nrow != 0 && UC.ncol != 0) // Avoid degenerate case
        upper_tri_solve_tr(UC,x,y);
#if 1
        printf(" Vector y (after upper_tri_tr solves)\n");
        printVec(y, gn);
        printf(" Vector x (after upper_tri_tr solves)\n");
        printVec(x, gn);
#endif

      BASKER_MATRIX &LC = LBTF(b);
        std::cout << "LC  b: " << b << std::endl;
        LC.print();
      if (LC.nrow != 0 && LC.ncol != 0) // Avoid degenerate case
        lower_tri_solve_tr(LC,x,y);
#if 1
        printf(" Vector y print (after lower_tri_tr solves)\n");
        printVec(y, gn);
        printf(" Vector x print (after lower_tri_tr solves)\n");
        printVec(x, gn);
#endif
    }
      // copy the solution for C from y to x
      Int srow_c = BTF_C.srow;
      Int erow_c = BTF_C.nrow + srow_c;
      for (Int i = srow_c; i < erow_c; i++) {
        x(i) = y(i);
      }


    return 0;
  }

#if 0
// FIXME - fix components + collective...
// No idea if below is correct, but provides the "roadmap" for the planned impl
  template <class Int, class Entry, class Exe_Space>
  BASKER_INLINE
  int Basker<Int,Entry,Exe_Space>::serial_btf_solve_tr
  (
   ENTRY_1DARRAY & x, // Permuted rhs at input
   ENTRY_1DARRAY & y  // 0 at input
  )
  {
    // 1. Solve small diag blocks (via btf) BTF_D; update offdiags in BTF_D
#if 1 // For output checking
    std::cout << "  BTF Stats:\n";
    std::cout << "    btf_nblks = " << btf_nblks << std::endl;
    std::cout << "    btf_tabs_offset = " << btf_tabs_offset << std::endl;
    std::cout << "    btf_top_tabs_offset = " << btf_top_tabs_offset << std::endl;
    std::cout << "    btf_top_nblks = " << btf_top_nblks << std::endl;

#endif
    if(btf_top_tabs_offset >  0)
    {
      // FIXME: This is not needed here in transpose solve - no solution from BTF_C...
      // copy the solution for C from y to x - needed for off-diag BTF_E update
      /*
      Int srow_c = BTF_C.srow;
      Int erow_c = BTF_C.nrow + srow_c;
      for (Int i = srow_c; i < erow_c; i++) {
        x(i) = y(i);
      }
      */
//
      for(Int b = 0; b < btf_top_tabs_offset; b++)
      {
#if 1
        std::cout << "BTF_D  b: " << b << std::endl;
        printf(" Vector y print (before solves)\n");
        printVec(y, gn);
        printf(" Vector x (before solves)print\n");
        printVec(x, gn);
#endif
        BASKER_MATRIX &UC = U_D(b);
        std::cout << "    UC upper_tri_solve_tr" << std::endl;
        UC.print();
        upper_tri_solve_tr(UC, y, x);
        printf(" Vector y after uppertri \n");
        printVec(y, gn);
        printf(" Vector x after uppertri \n");
        printVec(x, gn);

        BASKER_MATRIX &LC = L_D(b);
        std::cout << "    LC lower_tri_solve_tr" << std::endl;
        LC.print();
        lower_tri_solve_tr(LC, x, y);
        printf(" Vector y after lowertri \n");
        printVec(y, gn);
        printf(" Vector x after lowertri \n");
        printVec(x, gn);
        // offdiag
        spmv_BTF_tr(b, BTF_D, x, x, false);
        printf(" Vector y after spmv_BTF_tr\n");
        printVec(y, gn);
        printf(" Vector x after spmv_BTF_tr\n");
        printVec(x, gn);
      }

    // 2. Update off-diag BTF_E; Solve Large block ND BTF_A
    // Update offdiag BTF_E
    // FIXME CHECK THIS!!!!
    //   Does this need "x,x" inputs to work i.e. copy y soln to x before off-diag spmv updated?
        std::cout << "BTF_E update neg_spmv_perm_tr" << std::endl;
        BTF_E.print();
      neg_spmv_perm_tr(BTF_E, x, x);
        printf(" Vector y after neg_spmv_perm_tr\n");
        printVec(y, gn);
        printf(" Vector x after neg_spmv_perm_tr\n");
        printVec(x, gn);
    }

    // ND BTF_A:
    std::cout << "BTF_A serial_forward_solve_tr" << std::endl;
    serial_forward_solve_tr(x,y);
    std::cout << "BTF_A serial_backward_solve_tr" << std::endl;
    serial_backward_solve_tr(y,x);

    // 3. Update off-diag BTF_B; Solve small blocks (via small separator) BTF_C and update offdiags
    // Update offdiag BTF_B
    //BTF_B*y -> x
    if(btf_tabs_offset !=  0)
    {
      std::cout << "BTF_B update neg_spmv_perm_tr" << std::endl;
        BTF_B.print();
        printf(" Vector y before neg_spmv_perm_tr\n");
        printVec(y, gn);
        printf(" Vector x before neg_spmv_perm_tr\n");
        printVec(x, gn);
      neg_spmv_perm_tr(BTF_B,y,x);
        printf(" Vector y after neg_spmv_perm_tr\n");
        printVec(y, gn);
        printf(" Vector x after neg_spmv_perm_tr\n");
        printVec(x, gn);
    }

    // FIXME: Check this, does this need y soln from BTF_C copied over to x before solving these blocks??
      Int srow_c = BTF_C.srow;
      Int erow_c = BTF_C.nrow + srow_c;
      for (Int i = srow_c; i < erow_c; i++) {
        x(i) = y(i);
      }

    // BTF_C: "upper-left to lower-right" block iteration order i.e. [0,1,...,nblks_c)
    Int nblks_c = btf_nblks-btf_tabs_offset;
    printf("  nblks_c=%ld  btf_nblks=%ld  btf_tabs_offset=%ld\n", (long)nblks_c, (long)btf_nblks, (long)btf_tabs_offset);
    for(Int b = 0; b < nblks_c; b++) {
#if 1
        printf(" Vector y print (before solves)\n");
        printVec(y, gn);
        printf(" Vector x (before solves)print\n");
        printVec(x, gn);
#endif
      BASKER_MATRIX &UC = UBTF(b);
        std::cout << "UC  b: " << b << std::endl;
        UC.print();
      upper_tri_solve_tr(UC,x,y);
#if 1
        printf(" Vector y print (after solves)\n");
        printVec(y, gn);
        printf(" Vector x print (after solves)\n");
        printVec(x, gn);
#endif

      BASKER_MATRIX &LC = LBTF(b);
        std::cout << "LC  b: " << b << std::endl;
        LC.print();
      lower_tri_solve_tr(LC,x,y);
#if 1
        printf(" Vector y print (after solves)\n");
        printVec(y, gn);
        printf(" Vector x print (after solves)\n");
        printVec(x, gn);
#endif

      spmv_BTF_tr(b+btf_tabs_offset, BTF_C, x, y);
#if 1
        printf(" Vector y print (after spmv_BTF_tr)\n");
        printVec(y, gn);
        printf(" Vector x print (after spmv_BTF_tr)\n");
        printVec(x, gn);
#endif

    }
    return 0;
  }
#endif

} // namespace BaskerNX

#endif
