/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */
// Generated from PyTypeTypeGrammar.g4 by ANTLR 4.13.1

package org.sonar.python.types.pytype_grammar;

import org.antlr.v4.runtime.tree.ParseTreeListener;

/**
 * This interface defines a complete listener for a parse tree produced by
 * {@link PyTypeTypeGrammarParser}.
 */
public interface PyTypeTypeGrammarListener extends ParseTreeListener {
	/**
	 * Enter a parse tree produced by {@link PyTypeTypeGrammarParser#outer_type}.
	 * @param ctx the parse tree
	 */
	void enterOuter_type(PyTypeTypeGrammarParser.Outer_typeContext ctx);
	/**
	 * Exit a parse tree produced by {@link PyTypeTypeGrammarParser#outer_type}.
	 * @param ctx the parse tree
	 */
	void exitOuter_type(PyTypeTypeGrammarParser.Outer_typeContext ctx);
	/**
	 * Enter a parse tree produced by {@link PyTypeTypeGrammarParser#type}.
	 * @param ctx the parse tree
	 */
	void enterType(PyTypeTypeGrammarParser.TypeContext ctx);
	/**
	 * Exit a parse tree produced by {@link PyTypeTypeGrammarParser#type}.
	 * @param ctx the parse tree
	 */
	void exitType(PyTypeTypeGrammarParser.TypeContext ctx);
	/**
	 * Enter a parse tree produced by {@link PyTypeTypeGrammarParser#union_type}.
	 * @param ctx the parse tree
	 */
	void enterUnion_type(PyTypeTypeGrammarParser.Union_typeContext ctx);
	/**
	 * Exit a parse tree produced by {@link PyTypeTypeGrammarParser#union_type}.
	 * @param ctx the parse tree
	 */
	void exitUnion_type(PyTypeTypeGrammarParser.Union_typeContext ctx);
	/**
	 * Enter a parse tree produced by {@link PyTypeTypeGrammarParser#builtin_type}.
	 * @param ctx the parse tree
	 */
	void enterBuiltin_type(PyTypeTypeGrammarParser.Builtin_typeContext ctx);
	/**
	 * Exit a parse tree produced by {@link PyTypeTypeGrammarParser#builtin_type}.
	 * @param ctx the parse tree
	 */
	void exitBuiltin_type(PyTypeTypeGrammarParser.Builtin_typeContext ctx);
	/**
	 * Enter a parse tree produced by {@link PyTypeTypeGrammarParser#class_type}.
	 * @param ctx the parse tree
	 */
	void enterClass_type(PyTypeTypeGrammarParser.Class_typeContext ctx);
	/**
	 * Exit a parse tree produced by {@link PyTypeTypeGrammarParser#class_type}.
	 * @param ctx the parse tree
	 */
	void exitClass_type(PyTypeTypeGrammarParser.Class_typeContext ctx);
	/**
	 * Enter a parse tree produced by {@link PyTypeTypeGrammarParser#generic_type}.
	 * @param ctx the parse tree
	 */
	void enterGeneric_type(PyTypeTypeGrammarParser.Generic_typeContext ctx);
	/**
	 * Exit a parse tree produced by {@link PyTypeTypeGrammarParser#generic_type}.
	 * @param ctx the parse tree
	 */
	void exitGeneric_type(PyTypeTypeGrammarParser.Generic_typeContext ctx);
	/**
	 * Enter a parse tree produced by {@link PyTypeTypeGrammarParser#anything_type}.
	 * @param ctx the parse tree
	 */
	void enterAnything_type(PyTypeTypeGrammarParser.Anything_typeContext ctx);
	/**
	 * Exit a parse tree produced by {@link PyTypeTypeGrammarParser#anything_type}.
	 * @param ctx the parse tree
	 */
	void exitAnything_type(PyTypeTypeGrammarParser.Anything_typeContext ctx);
	/**
	 * Enter a parse tree produced by {@link PyTypeTypeGrammarParser#qualified_type}.
	 * @param ctx the parse tree
	 */
	void enterQualified_type(PyTypeTypeGrammarParser.Qualified_typeContext ctx);
	/**
	 * Exit a parse tree produced by {@link PyTypeTypeGrammarParser#qualified_type}.
	 * @param ctx the parse tree
	 */
	void exitQualified_type(PyTypeTypeGrammarParser.Qualified_typeContext ctx);
	/**
	 * Enter a parse tree produced by {@link PyTypeTypeGrammarParser#type_list}.
	 * @param ctx the parse tree
	 */
	void enterType_list(PyTypeTypeGrammarParser.Type_listContext ctx);
	/**
	 * Exit a parse tree produced by {@link PyTypeTypeGrammarParser#type_list}.
	 * @param ctx the parse tree
	 */
	void exitType_list(PyTypeTypeGrammarParser.Type_listContext ctx);
	/**
	 * Enter a parse tree produced by {@link PyTypeTypeGrammarParser#builtin}.
	 * @param ctx the parse tree
	 */
	void enterBuiltin(PyTypeTypeGrammarParser.BuiltinContext ctx);
	/**
	 * Exit a parse tree produced by {@link PyTypeTypeGrammarParser#builtin}.
	 * @param ctx the parse tree
	 */
	void exitBuiltin(PyTypeTypeGrammarParser.BuiltinContext ctx);
}