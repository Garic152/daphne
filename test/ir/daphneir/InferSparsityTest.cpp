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

#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/Seq.h>
#include <runtime/local/kernels/Fill.h>


#include <tags.h>

#include <catch.hpp>

// --------------------------------------------------------------------
// Data Generation
// --------------------------------------------------------------------

TEMPLATE_PRODUCT_TEST_CASE("Sparsity Inference - FillOp", TAG_KERNELS, (DenseMatrix, Matrix), (int64_t, double)) {
    using DTRes = TestType;
    using VT = typename DTRes::VT;
    const size_t numRows = 2;
    const size_t numCols = 2;

    for (const VT arg : {VT(0), VT(1)}) {
        DYNAMIC_SECTION("Filling matrix with " << +arg) {
            DTRes *res = nullptr;
            fill<DTRes, typename DTRes::VT>(res, arg, numRows, numCols, nullptr);

            size_t numNonZeros = 0;
            for (size_t r = 0; r < numRows; r++) {
                for (size_t c = 0; c < numCols; c++) {
                    const typename DTRes::VT v = res->get(r, c);
                    if (v) {
                        CHECK(v != static_cast<VT>(0));
                        numNonZeros++;
                    }
                }
            }
            const double sparsity = numNonZeros / static_cast<double>(numRows * numCols);
            // std::cout << std::fixed << std::setprecision(1);
            // std::cout << "arg = " << +arg << " of type " << typeid(VT).name() << std::endl;
            // std::cout << "sparsity = " << sparsity << std::endl;
            CHECK(sparsity == arg);
            DataObjectFactory::destroy(res);
        }
    }
}


TEMPLATE_PRODUCT_TEST_CASE("Sparsity Inference - SeqOp(Integers)", TAG_KERNELS, (DenseMatrix), (int8_t, int32_t, int64_t)) {
    using DTRes = TestType;
    using VT = typename DTRes::VT;

    // Creating test bundles to use for SeqOp
    const VT bundles[][3] = {
        {0, 1, 1},
        {0, 7, 1},
        {1, 3, 1},
        {2, -1, -1},
        {-2, -1, 1}
    };

    // Solutions to compare generated sparsity agains
    const double solutions[] = {0.5, 0.875, 1.0, 0.75, 1.0};

    const size_t numBundles = sizeof(bundles) / sizeof(bundles[0]);

    for (size_t i = 0; i < numBundles; ++i) {
        const auto& bundle = bundles[i];
        const double solution = solutions[i];

        const VT from = bundle[0];
        const VT to = bundle[1];
        const VT inc = bundle[2];

        DYNAMIC_SECTION("Generating Matrix from " << +from << " to " << +to << " with " << +inc << " increments.") {
            DTRes *res = nullptr;
            seq<DTRes>(res, from, to, inc, nullptr);

            size_t numRows = res->getNumRows();

            size_t numNonZeros = 0;
            for (size_t c = 0; c < numRows; c++) {
                const typename DTRes::VT v = res->get(c, 0);
                if (v) {
                    CHECK(v != static_cast<VT>(0));
                    numNonZeros++;
                }
            }
            const double sparsity = numNonZeros / static_cast<double>(numRows);
            CHECK(sparsity == solution);
            DataObjectFactory::destroy(res);
        }
    }
}

TEMPLATE_PRODUCT_TEST_CASE("Sparsity Inference - SeqOp(Floats)", TAG_KERNELS, (DenseMatrix), (float, double)) {
    using DTRes = TestType;
    using VT = typename DTRes::VT;

    // Creating test bundles to use for SeqOp
    const VT bundles[][3] = {
        {0.0, 0.01, 0.01},
        {0.0, 0.07, 0.01},
        {0.01, 0.03, 0.01},
        {0.02, -0.01, -0.01},
        {-0.02, -0.01, 0.01}
    };

    // Solutions to compare generated sparsity agains
    const double solutions[] = {0.5, 0.875, 1.0, 0.75, 1.0};

    const size_t numBundles = sizeof(bundles) / sizeof(bundles[0]);

    for (size_t i = 0; i < numBundles; ++i) {
        const auto& bundle = bundles[i];
        const double solution = solutions[i];

        const VT from = bundle[0];
        const VT to = bundle[1];
        const VT inc = bundle[2];

        DYNAMIC_SECTION("Generating Matrix from " << +from << " to " << +to << " with " << +inc << " increments.") {
            DTRes *res = nullptr;
            seq<DTRes>(res, from, to, inc, nullptr);

            size_t numRows = res->getNumRows();

            size_t numNonZeros = 0;
            for (size_t c = 0; c < numRows; c++) {
                const typename DTRes::VT v = res->get(c, 0);
                if (v) {
                    CHECK(v != static_cast<VT>(0));
                    numNonZeros++;
                }
            }
            const double sparsity = numNonZeros / static_cast<double>(numRows);
            CHECK(sparsity == solution);
            DataObjectFactory::destroy(res);
        }
    }
}