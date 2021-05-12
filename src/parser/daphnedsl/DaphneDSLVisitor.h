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

#ifndef SRC_PARSER_DAPHNEDSL_DAPHNEDSLVISITOR_H
#define SRC_PARSER_DAPHNEDSL_DAPHNEDSLVISITOR_H

#include "antlr4-runtime.h"
#include "DaphneDSLGrammarParser.h"
#include "DaphneDSLGrammarVisitor.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/Value.h>

#include <string>
#include <unordered_map>

class DaphneDSLVisitor : public DaphneDSLGrammarVisitor {
    // By inheriting from DaphneDSLGrammarVisitor (as opposed to
    // DaphneDSLGrammarBaseVisitor), we ensure that any newly added visitor
    // function (e.g. after a change to the grammar file) needs to be
    // considered here. This is to force us not to forget anything.
    
    /**
     * The OpBuilder used to generate DaphneIR operations.
     */
    mlir::OpBuilder builder;
    
    /**
     * Maps a variable name from the input DaphneDSL script to the MLIR SSA
     * value that has been assigned to it most recently.
     */
    std::unordered_map<std::string, mlir::Value> symbolTable;
    
public:
    DaphneDSLVisitor(mlir::OpBuilder & builder) : builder(builder) {
        //
    };
    
    antlrcpp::Any visitScript(DaphneDSLGrammarParser::ScriptContext * ctx) override;

    antlrcpp::Any visitStatement(DaphneDSLGrammarParser::StatementContext * ctx) override;

    antlrcpp::Any visitExprStatement(DaphneDSLGrammarParser::ExprStatementContext * ctx) override;

    antlrcpp::Any visitAssignStatement(DaphneDSLGrammarParser::AssignStatementContext * ctx) override;

    antlrcpp::Any visitLiteralExpr(DaphneDSLGrammarParser::LiteralExprContext * ctx) override;

    antlrcpp::Any visitIdentifierExpr(DaphneDSLGrammarParser::IdentifierExprContext * ctx) override;

    antlrcpp::Any visitParanthesesExpr(DaphneDSLGrammarParser::ParanthesesExprContext * ctx) override;

    antlrcpp::Any visitCallExpr(DaphneDSLGrammarParser::CallExprContext * ctx) override;
    
    antlrcpp::Any visitMatmulExpr(DaphneDSLGrammarParser::MatmulExprContext * ctx) override;
    
    antlrcpp::Any visitMulExpr(DaphneDSLGrammarParser::MulExprContext * ctx) override;
    
    antlrcpp::Any visitAddExpr(DaphneDSLGrammarParser::AddExprContext * ctx) override;
    
    antlrcpp::Any visitCmpExpr(DaphneDSLGrammarParser::CmpExprContext * ctx) override;

    antlrcpp::Any visitLiteral(DaphneDSLGrammarParser::LiteralContext * ctx) override;

};

#endif //SRC_PARSER_DAPHNEDSL_DAPHNEDSLVISITOR_H