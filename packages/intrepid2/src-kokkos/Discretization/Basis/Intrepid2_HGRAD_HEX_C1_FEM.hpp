// @HEADER
// ************************************************************************
//
//                           Intrepid2 Package
//                 Copyright (2007) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Kyungjoo Kim  (kyukim@sandia.gov), or
//                    Mauro Perego  (mperego@sandia.gov)
//
// ************************************************************************
// @HEADER

/** \file   Intrepid_G_HEX_C1_FEM.hpp
    \brief  Header file for the Intrepid2::G_HEX_C1_FEM class.
    \author Created by P. Bochev and D. Ridzal.
            Kokkorized by Kyungjoo Kim
 */

#ifndef __INTREPID2_HGRAD_HEX_C1_FEM_HPP__
#define __INTREPID2_HGRAD_HEX_C1_FEM_HPP__

#include "Intrepid2_Basis.hpp"

namespace Intrepid2 {
  
  /** \class  Intrepid2::Basis_HGRAD_HEX_C1_FEM
      \brief  Implementation of the default H(grad)-compatible FEM basis of degree 1 on Hexahedron cell 
      
      Implements Lagrangian basis of degree 1 on the reference Hexahedron cell. The basis has
      cardinality 8 and spans a COMPLETE tri-linear polynomial space. Basis functions are dual 
      to a unisolvent set of degrees-of-freedom (DoF) defined and enumerated as follows:
      
      \verbatim
      =================================================================================================
      |         |           degree-of-freedom-tag table                    |                           |
      |   DoF   |----------------------------------------------------------|      DoF definition       |
      | ordinal |  subc dim    | subc ordinal | subc DoF ord |subc num DoF |                           |
      |=========|==============|==============|==============|=============|===========================|
      |    0    |       0      |       0      |       0      |      1      |   L_0(u) = u(-1,-1,-1)    |
      |---------|--------------|--------------|--------------|-------------|---------------------------|
      |    1    |       0      |       1      |       0      |      1      |   L_1(u) = u( 1,-1,-1)    |
      |---------|--------------|--------------|--------------|-------------|---------------------------|
      |    2    |       0      |       2      |       0      |      1      |   L_2(u) = u( 1, 1,-1)    |
      |---------|--------------|--------------|--------------|-------------|---------------------------|
      |    3    |       0      |       3      |       0      |      1      |   L_3(u) = u(-1, 1,-1)    |
      |---------|--------------|--------------|--------------|-------------|---------------------------|
      |    4    |       0      |       4      |       0      |      1      |   L_4(u) = u(-1,-1, 1)    |
      |---------|--------------|--------------|--------------|-------------|---------------------------|
      |    5    |       0      |       5      |       0      |      1      |   L_5(u) = u( 1,-1, 1)    |
      |---------|--------------|--------------|--------------|-------------|---------------------------|
      |    6    |       0      |       6      |       0      |      1      |   L_6(u) = u( 1, 1, 1)    |
      |---------|--------------|--------------|--------------|-------------|---------------------------|
      |    7    |       0      |       7      |       0      |      1      |   L_7(u) = u(-1, 1, 1)    |
      |=========|==============|==============|==============|=============|===========================|
      |   MAX   |  maxScDim=0  |  maxScOrd=7  |  maxDfOrd=0  |      -      |                           |
      |=========|==============|==============|==============|=============|===========================|
      \endverbatim
  
  */
  
  template<typename ExecSpaceType = void>
  class Basis_HGRAD_HEX_C1_FEM : public Basis<ExecSpaceType> {
  public:
    
    template<EOperator opType>
    struct Serial {
      template<typename outputValueValueType, class ...outputValueProperties,
               typename inputPointValueType,  class ...inputPointProperties>
      KOKKOS_INLINE_FUNCTION
      static void
      getValues( /**/  Kokkos::DynRankView<outputValueValueType,outputValueProperties...> outputValues,
                 const Kokkos::DynRankView<inputPointValueType, inputPointProperties...>  inputPoints );
      
    };
    
    template<typename outputValueViewType,
             typename inputPointViewType,
             EOperator opType>
    struct Functor {
      /**/  outputValueViewType _outputValues;
      const inputPointViewType  _inputPoints;
      
      KOKKOS_INLINE_FUNCTION
      Functor( /**/  outputValueViewType outputValues_,
               /**/  inputPointViewType  inputPoints_ )
        : _outputValues(outputValues_), _inputPoints(inputPoints_) {}
      
      KOKKOS_INLINE_FUNCTION
      void operator()(const ordinal_type pt) const {
        switch (opType) {
        case OPERATOR_VALUE : {
          auto       output = Kokkos::subdynrankview( _outputValues, Kokkos::ALL(), pt );
          const auto input  = Kokkos::subdynrankview( _inputPoints,                 pt, Kokkos::ALL() );
          Serial<opType>::getValues( output, input );
          break;
        }
        case OPERATOR_GRAD :
        case OPERATOR_CURL :
        case OPERATOR_D2 :
        case OPERATOR_MAX : {
          auto       output = Kokkos::subdynrankview( _outputValues, Kokkos::ALL(), pt, Kokkos::ALL() );
          const auto input  = Kokkos::subdynrankview( _inputPoints,                 pt, Kokkos::ALL() );
          Serial<opType>::getValues( output, input );
          break;
        }
        default: {
          INTREPID2_TEST_FOR_ABORT( opType != OPERATOR_VALUE &&
                                    opType != OPERATOR_GRAD &&
                                    opType != OPERATOR_CURL &&
                                    opType != OPERATOR_D2 &&
                                    opType != OPERATOR_MAX,
                                    ">>> ERROR: (Intrepid2::Basis_HGRAD_HEX_C1_FEM::Serial::getValues) operator is not supported");
        }
        }
      }
    };

    
    
    /** \brief  Constructor.
     */
    Basis_HGRAD_HEX_C1_FEM();
  
    
    /** \brief  Evaluation of a FEM basis on a <strong>reference Hexahedron</strong> cell. 
    
        Returns values of <var>operatorType</var> acting on FEM basis functions for a set of
        points in the <strong>reference Hexahedron</strong> cell. For rank and dimensions of
        I/O array arguments see Section \ref basis_md_array_sec.
  
        \param  outputValues      [out] - rank-2 or 3 array with the computed basis values
        \param  inputPoints       [in]  - rank-2 array with dimensions (P,D) containing reference points  
        \param  operatorType      [in]  - operator applied to basis functions    
    */
    template<typename outputValueValueType, class ...outputValueProperties,
             typename inputPointValueType,  class ...inputPointProperties>
    void
    getValues( /**/  Kokkos::DynRankView<outputValueValueType,outputValueProperties...> outputValues,
               const Kokkos::DynRankView<inputPointValueType, inputPointProperties...>  inputPoints,
               const EOperator operatorType  = OPERATOR_VALUE ) const;

  
    /** \brief  Returns spatial locations (coordinates) of degrees of freedom on a
        <strong>reference Quadrilateral</strong>.

        \param  DofCoords      [out] - array with the coordinates of degrees of freedom,
        dimensioned (F,D)
    */
    template<typename dofCoordValueType, class ...dofCoordProperties>
    void
    getDofCoords( Kokkos::DynRankView<dofCoordValueType,dofCoordProperties...> dofCoords ) const;

  };
}// namespace Intrepid2

#include "Intrepid2_HGRAD_HEX_C1_FEMDef.hpp"

#endif