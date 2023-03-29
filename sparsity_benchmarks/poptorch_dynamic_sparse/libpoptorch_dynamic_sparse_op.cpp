// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <iostream>
#include <memory>
#include <vector>

#include <popops/ElementWise.hpp>
#include <popsparse/MatMul.hpp>
#include <popsparse/MatMulParams.hpp>
#include <popsparse/SparsePartitioner.hpp>
#include <popsparse/codelets.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <popart/op.hpp>
#include <popart/opmanager.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>
#pragma GCC diagnostic pop

///////////////////////////////////////////////////////////////////////////////
// Core

namespace {
/**
 * A simple wrapper for a COO (coordinate) format sparse matrix.
 *
 * Note that we don't use popsparse:COOMatrix since it is missing the fields
 * {nRows, nCols, blockSize}.
 */
struct StaticCooMatrix {
    unsigned nRows;
    unsigned nCols;
    unsigned blockSize;
    std::vector<size_t> rows;
    std::vector<size_t> cols;
    std::vector<float> values;

    StaticCooMatrix(unsigned nRows,
                    unsigned nCols,
                    unsigned blockSize,
                    std::vector<size_t> rows,
                    std::vector<size_t> cols,
                    std::vector<float> values)
        : nRows(nRows),
          nCols(nCols),
          blockSize(blockSize),
          rows(std::move(rows)),
          cols(std::move(cols)),
          values(std::move(values)) {}
    StaticCooMatrix(const StaticCooMatrix&) = default;
    StaticCooMatrix(StaticCooMatrix&&) = default;

    popsparse::COOMatrix<float> toPopSparse() const {
        // Index conversion, as popsparse::COOMatrix expects element indices, whereas
        // StaticCooMatrix provides block indices
        std::vector<size_t> colIndices(cols.size());
        std::transform(cols.begin(), cols.end(), colIndices.begin(),
                       [this](auto idx) { return blockSize * idx; });
        std::vector<size_t> rowIndices(rows.size());
        std::transform(rows.begin(), rows.end(), rowIndices.begin(),
                       [this](auto idx) { return blockSize * idx; });
        return popsparse::COOMatrix<float>(values, colIndices, rowIndices, {blockSize, blockSize});
    }
};

poplar::Tensor staticSparseDenseMatMul(poplar::Graph& graph,
                                       const StaticCooMatrix& matrix,
                                       const poplar::Tensor& rhs,
                                       poplar::program::Sequence& program,
                                       const poplar::DebugContext& debugContext) {
    static popsparse::dynamic::PlanningCache cache;

    auto batchSize = rhs.dim(1);
    auto dtype = rhs.elementType();
    poplar::OptionFlags options{{"partialsType", "half"}};

    // Sparsity setup / configuration
    auto sparsityParams = popsparse::dynamic::SparsityParams(
        matrix.blockSize == 1 ? popsparse::dynamic::SparsityType::Element
                              : popsparse::dynamic::SparsityType::Block,
        popsparse::dynamic::SparsityStructure::Unstructured, {matrix.blockSize, matrix.blockSize});

    auto params = popsparse::dynamic::MatMulParams::createWithNumNonZeroValues(
        sparsityParams, matrix.values.size(), 1, matrix.nRows, matrix.nCols, batchSize);

    auto dummyLhs = popsparse::dynamic::createSparseDenseMatMulLHS(
        graph, dtype, params, {debugContext, "dummyLhs"}, options, &cache);
    auto dummyMetaInfo = dummyLhs.getMetaInfoTensor();
    auto dummyNzValues = dummyLhs.getNzValuesTensor();

    // Create constant inputs
    popsparse::dynamic::Partitioner<float> partitioner(params, dtype, graph.getTarget(), options,
                                                       &cache);
    auto weights = partitioner.createSparsityDataImpl(matrix.toPopSparse());
    popsparse::dynamic::SparseTensor lhs(
        graph.addConstant<uint64_t>(dummyMetaInfo.elementType(), dummyMetaInfo.shape(),
                                    weights.metaInfo, {debugContext, "lhs_metaInfo"}),
        graph.addConstant<float>(dummyNzValues.elementType(), dummyNzValues.shape(),
                                 weights.nzValues, {debugContext, "lhs_nzValues"}),
        dummyLhs.getOpMetaData());
    graph.setTileMapping(lhs.getMetaInfoTensor(), graph.getTileMapping(dummyMetaInfo));
    graph.setTileMapping(lhs.getNzValuesTensor(), graph.getTileMapping(dummyNzValues));

    // Run
    return popsparse::dynamic::sparseDenseMatMul(
        graph, lhs, rhs.expand({0}), program, /*transposeLHS=*/false,
        /*transposeRHS=*/false, debugContext, options, &cache);
}
}  // namespace

///////////////////////////////////////////////////////////////////////////////
// Custom op wrapping

namespace Onnx::CustomOperators {
const popart::OperatorIdentifier StaticDynSparse = {"ai.graphcore", "StaticDynSparse", 1};
}  // namespace Onnx::CustomOperators

namespace {
struct CustomOp : popart::Op {
    StaticCooMatrix matrix;

    CustomOp(const popart::OperatorIdentifier& _opid,
             StaticCooMatrix matrix,
             const popart::Op::Settings& settings_)
        : popart::Op(_opid, settings_), matrix(std::move(matrix)) {}

    std::unique_ptr<Op> clone() const final { return std::make_unique<CustomOp>(*this); }

    float getSubgraphValue() const final { return getLowSubgraphValue(); }

    // Shape inference
    void setup() {
        auto input = inInfo(0);
        outInfo(0) = {input.dataType(), {matrix.nRows, input.dim(1)}};
    }
};

struct CustomOpx : popart::popx::Opx {
    CustomOpx(popart::Op* op, popart::popx::Devicex* devicex) : popart::popx::Opx(op, devicex) {
        verifyOp<CustomOp>(op, Onnx::CustomOperators::StaticDynSparse);
    }

    void grow(poplar::program::Sequence& prog) const final {
        popsparse::addCodelets(graph());  // Note: this might not belong here
        auto& op = getOp<CustomOp>();
        auto input = get(inId(0));
        auto output = staticSparseDenseMatMul(graph(), op.matrix, input, prog,
                                              debugContext("staticDynSparse"));
        insert(outId(0), output);
    }
};

popart::OpDefinition::DataTypes T = {popart::DataType::FLOAT16, popart::DataType::FLOAT};
popart::OpCreator<CustomOp> opCreator(
    {{Onnx::CustomOperators::StaticDynSparse,
      {popart::OpDefinition::Inputs({{"input", T}}), popart::OpDefinition::Outputs({{"output", T}}),
       popart::OpDefinition::Attributes({{"n_rows", {"int"}},
                                         {"n_cols", {"int"}},
                                         {"block_size", {"int"}},
                                         {"rows", {"*"}},
                                         {"cols", {"*"}},
                                         {"values", {"*"}}})}}},
    [](const popart::OpCreatorInfo& info) {
        auto nRows = info.attributes.getAttribute<popart::Attributes::Int>("n_rows");
        auto nCols = info.attributes.getAttribute<popart::Attributes::Int>("n_cols");
        auto blockSize = info.attributes.getAttribute<popart::Attributes::Int>("block_size");
        auto rows = info.attributes.getAttribute<popart::Attributes::Ints>("rows");
        auto cols = info.attributes.getAttribute<popart::Attributes::Ints>("cols");
        auto values = info.attributes.getAttribute<popart::Attributes::Floats>("values");
        StaticCooMatrix matrix(nRows, nCols, blockSize,
                               std::vector<size_t>(rows.begin(), rows.end()),
                               std::vector<size_t>(cols.begin(), cols.end()), std::move(values));
        return std::make_unique<CustomOp>(info.opid, std::move(matrix), info.settings);
    },
    true);

popart::popx::OpxCreator<CustomOpx> opxCreator(Onnx::CustomOperators::StaticDynSparse);
}  // namespace
