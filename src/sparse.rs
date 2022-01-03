use std::sync::Arc;

pub use graphblas_sparse_linear_algebra as graphblas;

pub use graphblas::context::{Context, Mode};

pub use graphblas::operators as ops;

pub use graphblas::error::*;
pub use graphblas::util::*;
pub use graphblas::value_types::sparse_matrix::*;
pub use graphblas::value_types::sparse_scalar::*;
pub use graphblas::value_types::sparse_vector::*;
pub use graphblas::value_types::value_type::*;

pub use ops::binary_operator::BinaryOperator as BinOp;
pub use ops::binary_operator::{Divide, First, Minus, Plus, Second, Times};

pub use ops::apply::*;

pub use ops::element_wise_addition::{
    ElementWiseMatrixAdditionBinaryOperator as EwiseMatAddBinOp,
    ElementWiseMatrixAdditionMonoidOperator as EwiseMatAddMonOp,
    ElementWiseMatrixAdditionSemiring as EwiseMatAddSemiring,
    ElementWiseVectorAdditionBinaryOperator as EwiseVecAddBinOp,
    ElementWiseVectorAdditionMonoidOperator as EwiseVecAddMonOp,
    ElementWiseVectorAdditionSemiring as EwiseVecAddSemiring,
};

pub use ops::element_wise_multiplication::{
    ElementWiseMatrixMultiplicationBinaryOperator as EwiseMatMulBinOp,
    ElementWiseMatrixMultiplicationMonoidOperator as EwiseMatMulMonOp,
    ElementWiseMatrixMultiplicationSemiring as EwiseMatMulSemiring,
    ElementWiseVectorMultiplicationBinaryOperator as EwiseVecMulBinOp,
    ElementWiseVectorMultiplicationMonoidOperator as EwiseVecMulMonOp,
    ElementWiseVectorMultiplicationSemiring as EwiseVecMulSemiring,
};

pub use ops::extract::*;
pub use ops::insert::*;

pub use ops::kronecker_product::{
    BinaryOperatorKroneckerProductOperator as BinOpKronProdOp,
    MonoidKroneckerProduct as MonoidKronProd,
    SemiringKroneckerProduct as SemiringKronProd,
};

pub use ops::mask::*;

pub use ops::monoid::*;
pub use ops::multiplication::{
    MatrixMultiplicationOperator as MatMulOp,
    MatrixVectorMultiplicationOperator as MatVecMulOp,
    VectorMatrixMultiplicationOperator as VecMatMulOp,
};

pub use ops::options::OperatorOptions;

pub use ops::reduce::*;

pub use ops::select::*;

pub use ops::semiring::*;

pub use ops::subinsert::*;

pub use ops::transpose::MatrixTranspose;
pub use ops::unary_operator::*;

// pub use ops::binary_operator::BinaryOperator as BinOp;
// pub use ops::binary_operator::{Divide, First, Minus, Plus, Second, Times};
