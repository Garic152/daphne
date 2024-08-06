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
    Type fromTy = getFrom().getType();
    if(fromTy.isF64()) {
        try {
            double vFrom = CompilerUtils::constantOrThrow<double>(getFrom());
            double vTo = CompilerUtils::constantOrThrow<double>(getTo());
            double vInc = CompilerUtils::constantOrThrow<double>(getInc());

            if ((vFrom < 0.0 && vInc < 0.0) || (vFrom > 0.0 && vInc > 0.0) || (vFrom < 0.0 && vTo < 0.0) || (vFrom > 0.0 && vTo > 0.0)) {
                return {1.0};
            } else if (fmod(vFrom, vInc) == 0.0) {
                int numRows = abs((vTo - vFrom) / vInc) + 1.0;
                return {1 - (1.0 / (double)numRows)};
            } 
            return {1.0};
        }
        catch(const std::runtime_error & e) {
            return {-1.0};
        }
    }
    if(fromTy.isF32()) {
        try {
            float vFrom = CompilerUtils::constantOrThrow<float>(getFrom());
            float vTo = CompilerUtils::constantOrThrow<float>(getTo());
            float vInc = CompilerUtils::constantOrThrow<float>(getInc());

            if ((vFrom < 0.0 && vInc < 0.0) || (vFrom > 0.0 && vInc > 0.0) || (vFrom < 0.0 && vTo < 0.0) || (vFrom > 0.0 && vTo > 0.0)) {
                return {1.0};
            } else if (fmod(vFrom, vInc) == 0.0) {
                int numRows = abs((vTo - vFrom) / vInc) + 1.0;
                return {1 - (1.0 / (double)numRows)};
            } 
            return {1.0};
        }
        catch(const std::runtime_error & e) {
            return {-1.0};
        }
    }
    else if(fromTy.isSignedInteger(64)) {
        try {
            int64_t vFrom = CompilerUtils::constantOrThrow<int64_t>(getFrom());
            int64_t vTo = CompilerUtils::constantOrThrow<int64_t>(getTo());
            int64_t vInc = CompilerUtils::constantOrThrow<int64_t>(getInc());
            
            if ((vFrom < 0 && vInc < 0) || (vFrom > 0 && vInc > 0) || (vFrom < 0 && vTo < 0) || (vFrom > 0 && vTo > 0)) {
                return {1.0};
            } else if (fmod(vFrom, vInc) == 0) {
                int numRows = abs((vTo - vFrom) / vInc) + 1;
                return {1 - (1.0 / (double)numRows)};
            } 
            return {1.0};
        }
        catch(const std::runtime_error & e) {
            return {-1.0};
        }
    }
    throw ErrorHandler::compilerError(
        getLoc(), "InferSparsityOpInterface (daphne::SeqOp::inferSparsity)",
        "at the moment, sparsity inference for SeqOp supports only F64/F32 and "
        "SI64 value types");
}

// --------------------------------------------------------------------
// Elementwise Binary
// --------------------------------------------------------------------

std::vector<double> daphne::EwAddOp::inferSparsity() {
    /**
     * Uses the probability P(A || B) to estimate the output sparsity.
     */
    auto lhs = getLhs().getType().dyn_cast<daphne::MatrixType>();
    auto rhs = getRhs().getType().dyn_cast<daphne::MatrixType>();

    double lhsSparsity = lhs.getSparsity();
    double rhsSparsity = rhs.getSparsity();

    if (lhsSparsity == -1.0 || rhsSparsity == -1.0) {
        return {-1.0};
    } else if (lhsSparsity == 0.0) {
        return {rhsSparsity};
    } else if (rhsSparsity == 0.0) {
        return {lhsSparsity};
    }
    return {lhsSparsity + rhsSparsity - lhsSparsity * rhsSparsity};
}

std::vector<double> daphne::EwSubOp::inferSparsity() {
    /**
     * Uses the probability P(A || B) to estimate the output sparsity.
     */
    auto lhs = getLhs().getType().dyn_cast<daphne::MatrixType>();
    auto rhs = getRhs().getType().dyn_cast<daphne::MatrixType>();

    double lhsSparsity = lhs.getSparsity();
    double rhsSparsity = rhs.getSparsity();

    if (lhsSparsity == -1.0 || rhsSparsity == -1.0) {
        return {-1.0};
    } else if (lhsSparsity == 0.0) {
        return {rhsSparsity};
    } else if (rhsSparsity == 0.0) {
        return {lhsSparsity};
    }
    return {lhsSparsity + rhsSparsity - lhsSparsity * rhsSparsity};
}

std::vector<double> daphne::EwMulOp::inferSparsity() {
    /**
     * Uses the probability P(A && B) to estimate the output sparsity.
     */
    auto lhs = getLhs().getType().dyn_cast<daphne::MatrixType>();
    auto rhs = getRhs().getType().dyn_cast<daphne::MatrixType>();

    double lhsSparsity = lhs.getSparsity();
    double rhsSparsity = rhs.getSparsity();

    if ((lhsSparsity == -1.0 && rhsSparsity == -1.0) || (lhsSparsity == -1.0 && rhsSparsity != 0.0) || (lhsSparsity != 0.0 && rhsSparsity == -1.0)) {
        return {-1.0};
    }
    return {lhsSparsity * rhsSparsity};
}

std::vector<double> daphne::EwDivOp::inferSparsity() {
    auto lhs = getLhs().getType().dyn_cast<daphne::MatrixType>();
    auto rhs = getRhs().getType().dyn_cast<daphne::MatrixType>();

    double lhsSparsity = lhs.getSparsity();
    double rhsSparsity = rhs.getSparsity();

    if (lhsSparsity == -1.0 || (lhsSparsity == -1.0 && rhsSparsity == -1.0)) {
        return {-1.0};
    }
    return {lhsSparsity};
}

std::vector<double> daphne::EwPowOp::inferSparsity() {
    /**
     * First handles trivial cases and then uses the probability of P(A && !B).
     */
    auto lhs = getLhs().getType().dyn_cast<daphne::MatrixType>();
    auto rhs = getRhs().getType().dyn_cast<daphne::MatrixType>();

    double lhsSparsity = lhs.getSparsity();
    double rhsSparsity = rhs.getSparsity();

    if (lhsSparsity == -1.0 && rhsSparsity == -1.0) {
        return {-1.0};
    } else if (lhsSparsity == -1.0) {
        if (rhsSparsity == 0.0) {
            return {1.0};
        } else {
            return {-1.0};
        }
    } else if (rhsSparsity == -1.0) {
        if (lhsSparsity == 1.0) {
            return {1.0};
        } else {
            return {-1.0};
        }
    }

    if (lhsSparsity == 1.0 || rhsSparsity == 0.0) {
        return {1.0};
    } else if (lhsSparsity == 0.0 && rhsSparsity == 1.0) {
        return {0.0};
    }

    return {lhsSparsity + (1 - rhsSparsity) - lhsSparsity * (1 - rhsSparsity)};
}

std::vector<double> daphne::EwModOp::inferSparsity() {
    auto lhs = getLhs().getType().dyn_cast<daphne::MatrixType>();

    double lhsSparsity = lhs.getSparsity();

    if (lhsSparsity == 0) {
        return {0.0};
    }
    return {-1.0};
}

std::vector<double> daphne::EwAndOp::inferSparsity() {
    /**
     * Uses the probability P(A && B) to estimate the output sparsity.
     */
    auto lhs = getLhs().getType().dyn_cast<daphne::MatrixType>();
    auto rhs = getRhs().getType().dyn_cast<daphne::MatrixType>();

    double lhsSparsity = lhs.getSparsity();
    double rhsSparsity = rhs.getSparsity();

    if ((lhsSparsity == -1.0 && rhsSparsity == -1.0) || (lhsSparsity != 0.0 && rhsSparsity == -1.0) || (lhsSparsity == -1.0 && rhsSparsity != 0.0)) {
        return {-1.0};
    }
    return {lhsSparsity * rhsSparsity};
}

std::vector<double> daphne::EwOrOp::inferSparsity() {
    /**
     * Uses the probability P(A || B) to estimate the output sparsity.
     */
    auto lhs = getLhs().getType().dyn_cast<daphne::MatrixType>();
    auto rhs = getRhs().getType().dyn_cast<daphne::MatrixType>();

    double lhsSparsity = lhs.getSparsity();
    double rhsSparsity = rhs.getSparsity();

    if ((lhsSparsity == -1.0 && rhsSparsity == -1.0) || (lhsSparsity != 1.0 && rhsSparsity == -1.0) || (lhsSparsity == -1.0 && rhsSparsity != 1.0)) {
        return {-1.0};
    } else if (lhsSparsity == 1.0 || rhsSparsity == 1.0) {
        return {1.0};
    } else if (lhsSparsity == 0.0) {
        return {rhsSparsity};
    } else if (rhsSparsity == 0.0) {
        return {lhsSparsity};
    }
    return {lhsSparsity + rhsSparsity - lhsSparsity * rhsSparsity};
}

std::vector<double> daphne::EwXorOp::inferSparsity() {
    /**
     * Uses the probability P(A ⊕ B) to estimate the output sparsity.
     */
    auto lhs = getLhs().getType().dyn_cast<daphne::MatrixType>();
    auto rhs = getRhs().getType().dyn_cast<daphne::MatrixType>();

    double lhsSparsity = lhs.getSparsity();
    double rhsSparsity = rhs.getSparsity();

    if (lhsSparsity == -1.0 || rhsSparsity == -1.0) {
        return {-1.0};
    } else if ((lhsSparsity == 1.0 && rhsSparsity == 1.0) || (lhsSparsity == 0.0 && rhsSparsity == 0.0)) {
        return {0.0};
    } else if ((lhsSparsity == 1.0 && rhsSparsity == 0.0) || (lhsSparsity == 0.0 && rhsSparsity == 1.0)) {
        return {1.0};
    }
    // TODO This function requires testing after #551 is implemented.
    return {lhsSparsity + rhsSparsity - 2 * (lhsSparsity * rhsSparsity)};
}

std::vector<double> daphne::EwEqOp::inferSparsity() {
    auto lhs = getLhs().getType().dyn_cast<daphne::MatrixType>();
    auto rhs = getRhs().getType().dyn_cast<daphne::MatrixType>();

    double lhsSparsity = lhs.getSparsity();
    double rhsSparsity = rhs.getSparsity();

    if (lhsSparsity == -1.0 || rhsSparsity == -1.0) {
        return {-1.0};
    } else if (lhsSparsity == 0.0 && rhsSparsity == 0.0) {
        return {1.0};
    } else if ((lhsSparsity == 1.0 && rhsSparsity == 0.0) || (lhsSparsity == 0.0 && rhsSparsity == 1.0)) {
        return {0.0};
    }
    return {-1.0};
}

std::vector<double> daphne::EwNeqOp::inferSparsity() {
    auto lhs = getLhs().getType().dyn_cast<daphne::MatrixType>();
    auto rhs = getRhs().getType().dyn_cast<daphne::MatrixType>();

    double lhsSparsity = lhs.getSparsity();
    double rhsSparsity = rhs.getSparsity();

    if (lhsSparsity == -1.0 || rhsSparsity == -1.0) {
        return {-1.0};
    } else if (lhsSparsity == 0.0 && rhsSparsity == 0.0) {
        return {0.0};
    } else if ((lhsSparsity == 1.0 && rhsSparsity == 0.0) || (lhsSparsity == 0.0 && rhsSparsity == 1.0)) {
        return {1.0};
    }
    return {-1.0};
}

std::vector<double> daphne::EwLeOp::inferSparsity() {
    auto lhs = getLhs().getType().dyn_cast<daphne::MatrixType>();
    auto rhs = getRhs().getType().dyn_cast<daphne::MatrixType>();

    double lhsSparsity = lhs.getSparsity();
    double rhsSparsity = rhs.getSparsity();

    if (lhsSparsity == -1.0 || rhsSparsity == -1.0) {
        return {-1.0};
    } else if (lhsSparsity == 0.0 && rhsSparsity == 0.0) {
        return {1.0};
    }
    return {-1.0};
}

std::vector<double> daphne::EwGeOp::inferSparsity() {
    auto lhs = getLhs().getType().dyn_cast<daphne::MatrixType>();
    auto rhs = getRhs().getType().dyn_cast<daphne::MatrixType>();

    double lhsSparsity = lhs.getSparsity();
    double rhsSparsity = rhs.getSparsity();

    if (lhsSparsity == -1.0 || rhsSparsity == -1.0) {
        return {-1.0};
    } else if (lhsSparsity == 0.0 && rhsSparsity == 0.0) {
        return {1.0};
    }
    return {-1.0};
}


// --------------------------------------------------------------------
// Outer Binary
// --------------------------------------------------------------------

std::vector<double> daphne::OuterAddOp::inferSparsity() {
    /**
     * Uses the probability P(A || B) to estimate the output sparsity.
     */
    auto lhs = getLhs().getType().dyn_cast<daphne::MatrixType>();
    auto rhs = getRhs().getType().dyn_cast<daphne::MatrixType>();

    double lhsSparsity = lhs.getSparsity();
    double rhsSparsity = rhs.getSparsity();

    if (lhsSparsity == -1.0 || rhsSparsity == -1.0) {
        return {-1.0};
    } else if (lhsSparsity == 0.0) {
        return {rhsSparsity};
    } else if (rhsSparsity == 0.0) {
        return {lhsSparsity};
    }
    return {lhsSparsity + rhsSparsity - lhsSparsity * rhsSparsity};
}

std::vector<double> daphne::OuterSubOp::inferSparsity() {
    /**
     * Uses the probability P(A || B) to estimate the output sparsity.
     */
    auto lhs = getLhs().getType().dyn_cast<daphne::MatrixType>();
    auto rhs = getRhs().getType().dyn_cast<daphne::MatrixType>();

    double lhsSparsity = lhs.getSparsity();
    double rhsSparsity = rhs.getSparsity();

    if (lhsSparsity == -1.0 || rhsSparsity == -1.0) {
        return {-1.0};
    } else if (lhsSparsity == 0.0) {
        return {rhsSparsity};
    } else if (rhsSparsity == 0.0) {
        return {lhsSparsity};
    }
    return {lhsSparsity + rhsSparsity - lhsSparsity * rhsSparsity};
}

std::vector<double> daphne::OuterMulOp::inferSparsity() {
    /**
     * Uses the probability P(A && B) to estimate the output sparsity.
     */
    auto lhs = getLhs().getType().dyn_cast<daphne::MatrixType>();
    auto rhs = getRhs().getType().dyn_cast<daphne::MatrixType>();

    double lhsSparsity = lhs.getSparsity();
    double rhsSparsity = rhs.getSparsity();

    if ((lhsSparsity == -1.0 && rhsSparsity == -1.0) || (lhsSparsity == -1.0 && rhsSparsity != 0.0) || (lhsSparsity != 0.0 && rhsSparsity == -1.0)) {
        return {-1.0};
    }
    return {lhsSparsity * rhsSparsity};
}

std::vector<double> daphne::OuterDivOp::inferSparsity() {
    auto lhs = getLhs().getType().dyn_cast<daphne::MatrixType>();
    auto rhs = getRhs().getType().dyn_cast<daphne::MatrixType>();

    double lhsSparsity = lhs.getSparsity();
    double rhsSparsity = rhs.getSparsity();

    if (lhsSparsity == -1.0 || (lhsSparsity == -1.0 && rhsSparsity == -1.0)) {
        return {-1.0};
    }
    return {lhsSparsity};
}

std::vector<double> daphne::OuterPowOp::inferSparsity() {
    /**
     * First handles trivial cases and then uses the probability of P(A && !B).
     */
    auto lhs = getLhs().getType().dyn_cast<daphne::MatrixType>();
    auto rhs = getRhs().getType().dyn_cast<daphne::MatrixType>();

    double lhsSparsity = lhs.getSparsity();
    double rhsSparsity = rhs.getSparsity();

    if (lhsSparsity == -1.0 && rhsSparsity == -1.0) {
        return {-1.0};
    } else if (lhsSparsity == -1.0) {
        if (rhsSparsity == 0.0) {
            return {1.0};
        } else {
            return {-1.0};
        }
    } else if (rhsSparsity == -1.0) {
        if (lhsSparsity == 1.0) {
            return {1.0};
        } else {
            return {-1.0};
        }
    }

    if (lhsSparsity == 1.0 || rhsSparsity == 0.0) {
        return {1.0};
    } else if (lhsSparsity == 0.0 && rhsSparsity == 1.0) {
        return {0.0};
    }

    return {lhsSparsity + (1 - rhsSparsity) - lhsSparsity * (1 - rhsSparsity)};
}

std::vector<double> daphne::OuterModOp::inferSparsity() {
    auto lhs = getLhs().getType().dyn_cast<daphne::MatrixType>();

    double lhsSparsity = lhs.getSparsity();

    if (lhsSparsity == 0) {
        return {0.0};
    }
    return {-1.0};
}

std::vector<double> daphne::OuterAndOp::inferSparsity() {
    /**
     * Uses the probability P(A && B) to estimate the output sparsity.
     */
    auto lhs = getLhs().getType().dyn_cast<daphne::MatrixType>();
    auto rhs = getRhs().getType().dyn_cast<daphne::MatrixType>();

    double lhsSparsity = lhs.getSparsity();
    double rhsSparsity = rhs.getSparsity();

    if ((lhsSparsity == -1.0 && rhsSparsity == -1.0) || (lhsSparsity != 0.0 && rhsSparsity == -1.0) || (lhsSparsity == -1.0 && rhsSparsity != 0.0)) {
        return {-1.0};
    }
    return {lhsSparsity * rhsSparsity};
}

std::vector<double> daphne::OuterOrOp::inferSparsity() {
    /**
     * Uses the probability P(A || B) to estimate the output sparsity.
     */
    auto lhs = getLhs().getType().dyn_cast<daphne::MatrixType>();
    auto rhs = getRhs().getType().dyn_cast<daphne::MatrixType>();

    double lhsSparsity = lhs.getSparsity();
    double rhsSparsity = rhs.getSparsity();

    if ((lhsSparsity == -1.0 && rhsSparsity == -1.0) || (lhsSparsity != 1.0 && rhsSparsity == -1.0) || (lhsSparsity == -1.0 && rhsSparsity != 1.0)) {
        return {-1.0};
    } else if (lhsSparsity == 1.0 || rhsSparsity == 1.0) {
        return {1.0};
    } else if (lhsSparsity == 0.0) {
        return {rhsSparsity};
    } else if (rhsSparsity == 0.0) {
        return {lhsSparsity};
    }
    return {lhsSparsity + rhsSparsity - lhsSparsity * rhsSparsity};
}

std::vector<double> daphne::OuterXorOp::inferSparsity() {
    /**
     * Uses the probability P(A ⊕ B) to estimate the output sparsity.
     */
    auto lhs = getLhs().getType().dyn_cast<daphne::MatrixType>();
    auto rhs = getRhs().getType().dyn_cast<daphne::MatrixType>();

    double lhsSparsity = lhs.getSparsity();
    double rhsSparsity = rhs.getSparsity();

    if (lhsSparsity == -1.0 || rhsSparsity == -1.0) {
        return {-1.0};
    } else if ((lhsSparsity == 1.0 && rhsSparsity == 1.0) || (lhsSparsity == 0.0 && rhsSparsity == 0.0)) {
        return {0.0};
    } else if ((lhsSparsity == 1.0 && rhsSparsity == 0.0) || (lhsSparsity == 0.0 && rhsSparsity == 1.0)) {
        return {1.0};
    }
    // TODO This function requires testing after #551 is implemented.
    return {lhsSparsity + rhsSparsity - 2 * (lhsSparsity * rhsSparsity)};
}

std::vector<double> daphne::OuterEqOp::inferSparsity() {
    auto lhs = getLhs().getType().dyn_cast<daphne::MatrixType>();
    auto rhs = getRhs().getType().dyn_cast<daphne::MatrixType>();

    double lhsSparsity = lhs.getSparsity();
    double rhsSparsity = rhs.getSparsity();

    if (lhsSparsity == -1.0 || rhsSparsity == -1.0) {
        return {-1.0};
    } else if (lhsSparsity == 0.0 && rhsSparsity == 0.0) {
        return {1.0};
    } else if ((lhsSparsity == 1.0 && rhsSparsity == 0.0) || (lhsSparsity == 0.0 && rhsSparsity == 1.0)) {
        return {0.0};
    }
    return {-1.0};
}

std::vector<double> daphne::OuterNeqOp::inferSparsity() {
    auto lhs = getLhs().getType().dyn_cast<daphne::MatrixType>();
    auto rhs = getRhs().getType().dyn_cast<daphne::MatrixType>();

    double lhsSparsity = lhs.getSparsity();
    double rhsSparsity = rhs.getSparsity();

    if (lhsSparsity == -1.0 || rhsSparsity == -1.0) {
        return {-1.0};
    } else if (lhsSparsity == 0.0 && rhsSparsity == 0.0) {
        return {0.0};
    } else if ((lhsSparsity == 1.0 && rhsSparsity == 0.0) || (lhsSparsity == 0.0 && rhsSparsity == 1.0)) {
        return {1.0};
    }
    return {-1.0};
}

std::vector<double> daphne::OuterLeOp::inferSparsity() {
    auto lhs = getLhs().getType().dyn_cast<daphne::MatrixType>();
    auto rhs = getRhs().getType().dyn_cast<daphne::MatrixType>();

    double lhsSparsity = lhs.getSparsity();
    double rhsSparsity = rhs.getSparsity();

    if (lhsSparsity == -1.0 || rhsSparsity == -1.0) {
        return {-1.0};
    } else if (lhsSparsity == 0.0 && rhsSparsity == 0.0) {
        return {1.0};
    }
    return {-1.0};
}

std::vector<double> daphne::OuterGeOp::inferSparsity() {
    auto lhs = getLhs().getType().dyn_cast<daphne::MatrixType>();
    auto rhs = getRhs().getType().dyn_cast<daphne::MatrixType>();

    double lhsSparsity = lhs.getSparsity();
    double rhsSparsity = rhs.getSparsity();

    if (lhsSparsity == -1.0 || rhsSparsity == -1.0) {
        return {-1.0};
    } else if (lhsSparsity == 0.0 && rhsSparsity == 0.0) {
        return {1.0};
    }
    return {-1.0};
}

// --------------------------------------------------------------------
// Aggregation
// --------------------------------------------------------------------




// --------------------------------------------------------------------
// Reorganization
// --------------------------------------------------------------------

std::vector<double> daphne::ColBindOp::inferSparsity() {
    /**
     * Sparsity is estimated by finding the ratio of non-zero cells in the input matrices and the output matrix.
     */
    auto lhs = getLhs().getType().dyn_cast<daphne::MatrixType>();
    auto rhs = getRhs().getType().dyn_cast<daphne::MatrixType>();

    double lhsSparsity = lhs.getSparsity();
    double rhsSparsity = rhs.getSparsity();

    if (lhsSparsity == -1.0 || rhsSparsity == -1.0) {
        return {-1.0};
    }

    int64_t lhsCols = lhs.getNumCols();
    int64_t lhsRows = lhs.getNumRows();
    int64_t rhsCols = rhs.getNumCols();
    int64_t rhsRows = rhs.getNumRows();

    int64_t lhsCells = lhsCols * lhsRows;
    int64_t rhsCells = rhsCols * rhsRows;

    return {(lhsSparsity * lhsCells + rhsSparsity * rhsCells) / (lhsCells + rhsCells)};
}

std::vector<double> daphne::RowBindOp::inferSparsity() {
    /**
    * Sparsity is estimated by finding the ratio of non-zero cells in the input matrices and the output matrix.
    */
    auto lhs = getLhs().getType().dyn_cast<daphne::MatrixType>();
    auto rhs = getRhs().getType().dyn_cast<daphne::MatrixType>();

    double lhsSparsity = lhs.getSparsity();
    double rhsSparsity = rhs.getSparsity();

    if (lhsSparsity == -1.0 || rhsSparsity == -1.0) {
        return {-1.0};
    }

    int64_t lhsCols = lhs.getNumCols();
    int64_t lhsRows = lhs.getNumRows();
    int64_t rhsCols = rhs.getNumCols();
    int64_t rhsRows = rhs.getNumRows();

    int64_t lhsCells = lhsCols * lhsRows;
    int64_t rhsCells = rhsCols * rhsRows;

    return {(lhsSparsity * lhsCells + rhsSparsity * rhsCells) / (lhsCells + rhsCells)};
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

        if(op->hasTrait<SparsityRemains>()) {
            sparsity = getSparsityOrUnknownFromType(op->getOperand(0));
        }

        if(op->hasTrait<SparsityUnknown>()) {
            sparsity = -1.0;
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

        if(op->hasTrait<SparseIfInputSparse>()) {
            auto spLhs = getSparsityOrUnknownFromType(op->getOperand(0));
            if (op->getNumOperands() > 1) {
                auto spRhs = getSparsityOrUnknownFromType(op->getOperand(1));
                if (spLhs == 0.0 && spRhs == 0.0) {
                    sparsity = 0.0;
                } else {
                    sparsity = -1.0;
                }
            } else {
                if (spLhs == 0.0) {
                    sparsity = 0.0;
                } else {
                    sparsity = -1.0;
                }
            }
        }

        if(op->hasTrait<DenseIfInputDense>()) {
            auto spLhs = getSparsityOrUnknownFromType(op->getOperand(0));
            if (op->getNumOperands() > 1) {
                auto spRhs = getSparsityOrUnknownFromType(op->getOperand(1));
                if (spLhs == 1.0 && spRhs == 1.0) {
                    sparsity = 1.0;
                } else {
                    sparsity = -1.0;
                }
            } else {
                if (spLhs == 1.0) {
                    sparsity = 1.0;
                } else {
                    sparsity = -1.0;
                }
            }
        }

        if(op->hasTrait<SparsityRemainsIfInputOneOrZero>()) {
            auto spLhs = getSparsityOrUnknownFromType(op->getOperand(0));
            if (op->getNumOperands() > 1) {
                auto spRhs = getSparsityOrUnknownFromType(op->getOperand(1));
                if (spLhs == 0.0 && spRhs == 0.0) {
                    sparsity = 0.0;
                } else if (spLhs == 1.0 && spRhs == 1.0) {
                    sparsity = 1.0;
                } else {
                    sparsity = -1.0;
                }
            } else {
                if (spLhs == 0.0) {
                    sparsity = 0.0;
                } else if (spLhs == 1.0) {
                    sparsity = 1.0;
                } else {
                    sparsity = -1.0;
                }
            }
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