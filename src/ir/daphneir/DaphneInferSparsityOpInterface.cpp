/*
 * Copyright 2021 The DAPHNE Consortium
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <compiler/utils/CompilerUtils.h>
#include <ir/daphneir/Daphne.h>

#include <mlir/IR/Value.h>

#include <parser/metadata/MetaDataParser.h>

#include <vector>
#include <stdexcept>
#include <utility>

namespace mlir::daphne {
#include <ir/daphneir/DaphneInferSparsityOpInterface.cpp.inc>
}

using namespace mlir;
using namespace mlir::OpTrait;

// ****************************************************************************
// Utilities
// ****************************************************************************

double getSparsityOrUnknownFromType(Value v) {
    Type t = v.getType();
    if(auto mt = t.dyn_cast<daphne::MatrixType>())
        return mt.getSparsity();
    else // scalar or frame
        // TODO: read scalar value (if 0 -> sparsity 0.0)
        return -1.0;
}

// ****************************************************************************
// Sparsity inference interface implementations
// ****************************************************************************

std::vector<double> daphne::DiagMatrixOp::inferSparsity() {
    auto argTy = getArg().getType().dyn_cast<daphne::MatrixType>();
    auto k = argTy.getNumRows();
    auto sparsity = argTy.getSparsity();

    if(argTy.getSparsity() == -1.0) {
        sparsity = 1;
    }

    return {sparsity / k};
}

std::vector<double> daphne::MatMulOp::inferSparsity() {
    auto lhsTy = getLhs().getType().dyn_cast<daphne::MatrixType>();
    auto rhsTy = getRhs().getType().dyn_cast<daphne::MatrixType>();
    if(lhsTy.getSparsity() == -1.0 || rhsTy.getSparsity() == -1.0) {
        return {-1.0};
    }
    auto k = lhsTy.getNumCols();
    if(k == -1) {
        k = rhsTy.getNumRows();
    }
    if(k == -1)
        return {-1.0};
    else
        // unbiased estimate
        return {1.0 - std::pow(1.0 - lhsTy.getSparsity() * rhsTy.getSparsity(), k)};
}

std::vector<double> daphne::TriOp::inferSparsity() {
    auto argTy = getArg().getType().dyn_cast<daphne::MatrixType>();
    if(argTy.getSparsity() == -1.0) {
        return {-1.0};
    }
    // TODO: remove diagonal
    return {argTy.getSparsity() / 2.0};
}

std::vector<double> daphne::ReadOp::inferSparsity() {
    std::pair<bool, std::string> p = CompilerUtils::isConstant<std::string>(getFileName());
    if(p.first) {
        FileMetaData fmd = MetaDataParser::readMetaData(p.second);
        if (fmd.numNonZeros == -1)
            return {-1.0};
        // TODO: maybe use type shape info instead of file? (would require correct order of optimization passes)
        return {(static_cast<double>(fmd.numNonZeros) / fmd.numRows) / fmd.numCols};
    }
    else
        return {-1.0};
}

// --------------------------------------------------------------------
// Unary
// --------------------------------------------------------------------

std::vector<double> daphne::SliceColOp::inferSparsity() {
    auto argTy = getSource().getType().dyn_cast<daphne::MatrixType>();
    if(argTy.getSparsity() == -1.0) {
        return {-1.0};
        }
        return {argTy.getSparsity()};
}

std::vector<double> daphne::SliceRowOp::inferSparsity() {
    auto argTy = getSource().getType().dyn_cast<daphne::MatrixType>();
    if(argTy.getSparsity() == -1.0) {
        return {-1.0};
        }
        return {argTy.getSparsity()};
}

// --------------------------------------------------------------------
// Data Generation
// --------------------------------------------------------------------

std::vector<double> daphne::FillOp::inferSparsity() {
    auto co = CompilerUtils::constantOfAnyType(getArg());
    if (!co) {
        return {-1.0};
    }

    double v = 0.0;

    auto valueAttr = co->getAttr("value");
    if (auto floatAttr = valueAttr.dyn_cast<mlir::FloatAttr>()) {
        v = floatAttr.getValueAsDouble();
    } else if (auto intAttr = valueAttr.dyn_cast<mlir::IntegerAttr>()) {
        if (intAttr.getType().isSignlessInteger()) {
            v = static_cast<double>(intAttr.getInt());
        } else if (intAttr.getType().isSignedInteger()) {
            v = static_cast<double>(intAttr.getSInt());
        }
    } else {
        throw std::runtime_error("Unsupported type for FillOp sparsity inference");
    }

    if (v == -1.0) {
        return {-1.0};
    } else if (v == 0.0) {
        return {0.0};
    } else {
        return {1.0};
    }
}

std::vector<double> daphne::SeqOp::inferSparsity() {
    auto fromCo = CompilerUtils::constantOfAnyType(getFrom());
    auto toCo = CompilerUtils::constantOfAnyType(getTo());
    auto incCo = CompilerUtils::constantOfAnyType(getInc());

    if (!fromCo || !toCo || !incCo) {
        return {-1.0};
    }
    // helper lamdba function to extract the values out of the constantOperations
    auto getDoubleValue = [](mlir::Operation *co) -> double {
        auto valueAttr = co->getAttr("value");
        if (auto floatAttr = valueAttr.dyn_cast<mlir::FloatAttr>()) {
            return floatAttr.getValueAsDouble();
        } else if (auto intAttr = valueAttr.dyn_cast<mlir::IntegerAttr>()) {
            if (intAttr.getType().isSignlessInteger()) {
                return static_cast<double>(intAttr.getInt());
            } else if (intAttr.getType().isSignedInteger()) {
                return  static_cast<double>(intAttr.getSInt());
            }
        }
        throw std::runtime_error("Unsupported type for SeqOp sparsity inference");
    };

    double from = getDoubleValue(fromCo);
    double to = getDoubleValue(toCo);
    double inc = getDoubleValue(incCo);

    if ((from < 0 && inc < 0) || (from > 0 && inc > 0) || (from < 0 && to < 0) || (from > 0 && to > 0)) {
        return {1.0};
    } else if (fmod(from, inc) == 0) {
        int numRows = abs((to - from) / inc) + 1;
        return {1.0 / (double)numRows};
    } else {
        return {1.0};
    }
}

// --------------------------------------------------------------------
// Elementwise Unary
// --------------------------------------------------------------------

std::vector<double> daphne::EwAbsOp::inferSparsity() {
    auto argTy = getArg().getType().dyn_cast<daphne::MatrixType>();
    return {argTy.getSparsity()};
}

std::vector<double> daphne::EwSignOp::inferSparsity() {
    auto argTy = getArg().getType().dyn_cast<daphne::MatrixType>();
    return {argTy.getSparsity()};
}

std::vector<double> daphne::EwSqrtOp::inferSparsity() {
    auto argTy = getArg().getType().dyn_cast<daphne::MatrixType>();
    return {argTy.getSparsity()};
}

std::vector<double> daphne::EwExpOp::inferSparsity() {
    return {1.0};
}

std::vector<double> daphne::EwLnOp::inferSparsity() {
    return {1.0};
}

std::vector<double> daphne::EwSinOp::inferSparsity() {
    return {-1.0};
}

std::vector<double> daphne::EwCosOp::inferSparsity() {
    return {-1.0};
}

std::vector<double> daphne::EwTanOp::inferSparsity() {
    return {-1.0};
}

std::vector<double> daphne::EwSinhOp::inferSparsity() {
    auto argTy = getArg().getType().dyn_cast<daphne::MatrixType>();
    return {argTy.getSparsity()};
}

std::vector<double> daphne::EwCoshOp::inferSparsity() {
    return {1.0};
}

std::vector<double> daphne::EwTanhOp::inferSparsity() {
    auto argTy = getArg().getType().dyn_cast<daphne::MatrixType>();
    return {argTy.getSparsity()};
}

std::vector<double> daphne::EwAsinOp::inferSparsity() {
    auto argTy = getArg().getType().dyn_cast<daphne::MatrixType>();
    return {argTy.getSparsity()};
}

std::vector<double> daphne::EwAcosOp::inferSparsity() {
    return {-1.0};
}

std::vector<double> daphne::EwAtanOp::inferSparsity() {
    auto argTy = getArg().getType().dyn_cast<daphne::MatrixType>();
    return {argTy.getSparsity()};
}


// ****************************************************************************
// Sparsity inference trait implementations
// ****************************************************************************

// TODO This is also used in DaphneInferShapeOpInterface.cpp, make it a central
// utility.
/**
 * @brief Utility for trying a parametric trait for all values of the parameter
 * from 0 to some upper bound.
 */
template<size_t upper, template<size_t> class tryParametricTrait>
struct tryParamTraitUntil {
    static void apply(double &sparsity, Operation *op) {
        tryParametricTrait<upper>::apply(sparsity, op);
        tryParamTraitUntil<upper - 1, tryParametricTrait>::apply(sparsity, op);
    }
};
template<template<size_t> class tryParametricTrait>
struct tryParamTraitUntil<0, tryParametricTrait> {
    static void apply(double &sparsity, Operation *op) {
        tryParametricTrait<0>::apply(sparsity, op);
    }
};

template<size_t i>
struct trySparsityFromIthScalar {
    static void apply(double &sparsity, Operation *op) {
        if(op->hasTrait<SparsityFromIthScalar<i>::template Impl>())
            sparsity = CompilerUtils::constantOrDefault<double>(op->getOperand(i), -1);
    }
};

template<size_t i>
struct trySparsityFromIthArg {
    static void apply(double &sparsity, Operation *op) {
        if(op->hasTrait<SparsityFromIthArg<i>::template Impl>())
            sparsity = getSparsityOrUnknownFromType(op->getOperand(i));
    }
};

// ****************************************************************************
// Sparsity inference function
// ****************************************************************************

std::vector<double> daphne::tryInferSparsity(Operation *op) {
    if(auto inferSparsityOp = llvm::dyn_cast<daphne::InferSparsity>(op))
        // If the operation implements the sparsity inference interface,
        // we apply that.
        return inferSparsityOp.inferSparsity();
    else if(op->getNumResults() == 1) {
        // If the operation does not implement the sparsity inference interface
        // and has exactly one result, we utilize its sparsity inference traits.
        double sparsity = -1.0;

        if(op->hasTrait<CompletelyDense>()) {
            sparsity = 1.0;
        }

        if(op->hasTrait<EwSparseIfBoth>()) {
            auto spLhs = getSparsityOrUnknownFromType(op->getOperand(0));
            auto spRhs = getSparsityOrUnknownFromType(op->getOperand(1));
            if(spLhs != -1.0 && spRhs != -1.0)
                // unbiased estimate
                sparsity = spLhs + spRhs - spLhs * spRhs;
        }

        if(op->hasTrait<EwSparseIfEither>()) {
            auto spLhs = getSparsityOrUnknownFromType(op->getOperand(0));
            auto spRhs = getSparsityOrUnknownFromType(op->getOperand(1));
            if(spLhs != -1.0 && spRhs != -1.0)
                // unbiased estimate
                sparsity = spLhs * spRhs;
            else if (spLhs != -1.0)
                sparsity = spLhs;
            else if (spRhs != -1.0)
                sparsity = spRhs;
        }

        // Our parametric traits addressing a certain argument are supported
        // for up to 10 arguments (this can easily be changed here).
        // There does not seem to be a way in MLIR do it more generically,
        // since the parameters of parametric traits are template parameters.
        const size_t u = 9;
        tryParamTraitUntil<u, trySparsityFromIthScalar>::apply(sparsity, op);
        tryParamTraitUntil<u, trySparsityFromIthArg>::apply(sparsity, op);

        return {sparsity};
    }
    else {
        // If the operation does not implement the sparsity inference interface
        // and has zero or more than one results, we return unknown.
        std::vector<double> sparsities;
        for(size_t i = 0; i < op->getNumResults(); i++)
            sparsities.push_back(-1);
        return sparsities;
    }
}
