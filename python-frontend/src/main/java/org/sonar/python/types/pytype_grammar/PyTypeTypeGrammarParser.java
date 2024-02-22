/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2023 SonarSource SA
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

import org.antlr.v4.runtime.atn.*;
import org.antlr.v4.runtime.dfa.DFA;
import org.antlr.v4.runtime.*;
import org.antlr.v4.runtime.misc.*;
import org.antlr.v4.runtime.tree.*;
import java.util.List;
import java.util.Iterator;
import java.util.ArrayList;

@SuppressWarnings({"all", "warnings", "unchecked", "unused", "cast", "CheckReturnValue"})
public class PyTypeTypeGrammarParser extends Parser {
	static { RuntimeMetaData.checkVersion("4.13.1", RuntimeMetaData.VERSION); }

	protected static final DFA[] _decisionToDFA;
	protected static final PredictionContextCache _sharedContextCache =
		new PredictionContextCache();
	public static final int
		T__0=1, T__1=2, T__2=3, T__3=4, UNION_PREFIX=5, CLASS_PREFIX=6, GENERIC_CALLABLE_PREFIX=7, 
		ANYTHING_TYPE=8, BUILTINS_PREFIX=9, NONE_TYPE=10, STR=11, BOOL=12, INT=13, 
		FLOAT=14, COMPLEX=15, TUPLE=16, LIST=17, SET=18, DICT=19, PARAMETERS=20, 
		STRING=21, SKIPS=22;
	public static final int
		RULE_outer_type = 0, RULE_type = 1, RULE_union_type = 2, RULE_builtin_type = 3, 
		RULE_class_type = 4, RULE_generic_callable_type = 5, RULE_anything_type = 6, 
		RULE_qualified_type = 7, RULE_type_list = 8, RULE_builtin = 9;
	private static String[] makeRuleNames() {
		return new String[] {
			"outer_type", "type", "union_type", "builtin_type", "class_type", "generic_callable_type", 
			"anything_type", "qualified_type", "type_list", "builtin"
		};
	}
	public static final String[] ruleNames = makeRuleNames();

	private static String[] makeLiteralNames() {
		return new String[] {
			null, "')'", "'.'", "','", "'('", "'UnionType(type_list='", "'ClassType('", 
			"'GenericType(base_type=ClassType(typing.Callable)'", "'AnythingType()'", 
			"'builtins'", "'NoneType'", "'str'", "'bool'", "'int'", "'float'", "'complex'", 
			"'tuple'", "'list'", "'set'", "'dict'", "'parameters='"
		};
	}
	private static final String[] _LITERAL_NAMES = makeLiteralNames();
	private static String[] makeSymbolicNames() {
		return new String[] {
			null, null, null, null, null, "UNION_PREFIX", "CLASS_PREFIX", "GENERIC_CALLABLE_PREFIX", 
			"ANYTHING_TYPE", "BUILTINS_PREFIX", "NONE_TYPE", "STR", "BOOL", "INT", 
			"FLOAT", "COMPLEX", "TUPLE", "LIST", "SET", "DICT", "PARAMETERS", "STRING", 
			"SKIPS"
		};
	}
	private static final String[] _SYMBOLIC_NAMES = makeSymbolicNames();
	public static final Vocabulary VOCABULARY = new VocabularyImpl(_LITERAL_NAMES, _SYMBOLIC_NAMES);

	/**
	 * @deprecated Use {@link #VOCABULARY} instead.
	 */
	@Deprecated
	public static final String[] tokenNames;
	static {
		tokenNames = new String[_SYMBOLIC_NAMES.length];
		for (int i = 0; i < tokenNames.length; i++) {
			tokenNames[i] = VOCABULARY.getLiteralName(i);
			if (tokenNames[i] == null) {
				tokenNames[i] = VOCABULARY.getSymbolicName(i);
			}

			if (tokenNames[i] == null) {
				tokenNames[i] = "<INVALID>";
			}
		}
	}

	@Override
	@Deprecated
	public String[] getTokenNames() {
		return tokenNames;
	}

	@Override

	public Vocabulary getVocabulary() {
		return VOCABULARY;
	}

	@Override
	public String getGrammarFileName() { return "PyTypeTypeGrammar.g4"; }

	@Override
	public String[] getRuleNames() { return ruleNames; }

	@Override
	public String getSerializedATN() { return _serializedATN; }

	@Override
	public ATN getATN() { return _ATN; }

	public PyTypeTypeGrammarParser(TokenStream input) {
		super(input);
		_interp = new ParserATNSimulator(this,_ATN,_decisionToDFA,_sharedContextCache);
	}

	@SuppressWarnings("CheckReturnValue")
	public static class Outer_typeContext extends ParserRuleContext {
		public TypeContext type() {
			return getRuleContext(TypeContext.class,0);
		}
		public TerminalNode EOF() { return getToken(PyTypeTypeGrammarParser.EOF, 0); }
		public Outer_typeContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_outer_type; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof PyTypeTypeGrammarListener ) ((PyTypeTypeGrammarListener)listener).enterOuter_type(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof PyTypeTypeGrammarListener ) ((PyTypeTypeGrammarListener)listener).exitOuter_type(this);
		}
	}

	public final Outer_typeContext outer_type() throws RecognitionException {
		Outer_typeContext _localctx = new Outer_typeContext(_ctx, getState());
		enterRule(_localctx, 0, RULE_outer_type);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(20);
			type();
			setState(21);
			match(EOF);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class TypeContext extends ParserRuleContext {
		public Union_typeContext union_type() {
			return getRuleContext(Union_typeContext.class,0);
		}
		public Builtin_typeContext builtin_type() {
			return getRuleContext(Builtin_typeContext.class,0);
		}
		public Class_typeContext class_type() {
			return getRuleContext(Class_typeContext.class,0);
		}
		public Anything_typeContext anything_type() {
			return getRuleContext(Anything_typeContext.class,0);
		}
		public Generic_callable_typeContext generic_callable_type() {
			return getRuleContext(Generic_callable_typeContext.class,0);
		}
		public Qualified_typeContext qualified_type() {
			return getRuleContext(Qualified_typeContext.class,0);
		}
		public TypeContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_type; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof PyTypeTypeGrammarListener ) ((PyTypeTypeGrammarListener)listener).enterType(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof PyTypeTypeGrammarListener ) ((PyTypeTypeGrammarListener)listener).exitType(this);
		}
	}

	public final TypeContext type() throws RecognitionException {
		TypeContext _localctx = new TypeContext(_ctx, getState());
		enterRule(_localctx, 2, RULE_type);
		try {
			setState(29);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case UNION_PREFIX:
				enterOuterAlt(_localctx, 1);
				{
				setState(23);
				union_type();
				}
				break;
			case BUILTINS_PREFIX:
			case NONE_TYPE:
			case STR:
			case BOOL:
			case INT:
			case FLOAT:
			case COMPLEX:
			case TUPLE:
			case LIST:
			case SET:
			case DICT:
				enterOuterAlt(_localctx, 2);
				{
				setState(24);
				builtin_type();
				}
				break;
			case CLASS_PREFIX:
				enterOuterAlt(_localctx, 3);
				{
				setState(25);
				class_type();
				}
				break;
			case ANYTHING_TYPE:
				enterOuterAlt(_localctx, 4);
				{
				setState(26);
				anything_type();
				}
				break;
			case GENERIC_CALLABLE_PREFIX:
				enterOuterAlt(_localctx, 5);
				{
				setState(27);
				generic_callable_type();
				}
				break;
			case STRING:
				enterOuterAlt(_localctx, 6);
				{
				setState(28);
				qualified_type();
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class Union_typeContext extends ParserRuleContext {
		public TerminalNode UNION_PREFIX() { return getToken(PyTypeTypeGrammarParser.UNION_PREFIX, 0); }
		public Type_listContext type_list() {
			return getRuleContext(Type_listContext.class,0);
		}
		public Union_typeContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_union_type; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof PyTypeTypeGrammarListener ) ((PyTypeTypeGrammarListener)listener).enterUnion_type(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof PyTypeTypeGrammarListener ) ((PyTypeTypeGrammarListener)listener).exitUnion_type(this);
		}
	}

	public final Union_typeContext union_type() throws RecognitionException {
		Union_typeContext _localctx = new Union_typeContext(_ctx, getState());
		enterRule(_localctx, 4, RULE_union_type);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(31);
			match(UNION_PREFIX);
			setState(32);
			type_list();
			setState(33);
			match(T__0);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class Builtin_typeContext extends ParserRuleContext {
		public BuiltinContext builtin() {
			return getRuleContext(BuiltinContext.class,0);
		}
		public TerminalNode BUILTINS_PREFIX() { return getToken(PyTypeTypeGrammarParser.BUILTINS_PREFIX, 0); }
		public Builtin_typeContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_builtin_type; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof PyTypeTypeGrammarListener ) ((PyTypeTypeGrammarListener)listener).enterBuiltin_type(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof PyTypeTypeGrammarListener ) ((PyTypeTypeGrammarListener)listener).exitBuiltin_type(this);
		}
	}

	public final Builtin_typeContext builtin_type() throws RecognitionException {
		Builtin_typeContext _localctx = new Builtin_typeContext(_ctx, getState());
		enterRule(_localctx, 6, RULE_builtin_type);
		try {
			setState(39);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case NONE_TYPE:
			case STR:
			case BOOL:
			case INT:
			case FLOAT:
			case COMPLEX:
			case TUPLE:
			case LIST:
			case SET:
			case DICT:
				enterOuterAlt(_localctx, 1);
				{
				setState(35);
				builtin();
				}
				break;
			case BUILTINS_PREFIX:
				enterOuterAlt(_localctx, 2);
				{
				setState(36);
				match(BUILTINS_PREFIX);
				setState(37);
				match(T__1);
				setState(38);
				builtin();
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class Class_typeContext extends ParserRuleContext {
		public TerminalNode CLASS_PREFIX() { return getToken(PyTypeTypeGrammarParser.CLASS_PREFIX, 0); }
		public Builtin_typeContext builtin_type() {
			return getRuleContext(Builtin_typeContext.class,0);
		}
		public Qualified_typeContext qualified_type() {
			return getRuleContext(Qualified_typeContext.class,0);
		}
		public Class_typeContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_class_type; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof PyTypeTypeGrammarListener ) ((PyTypeTypeGrammarListener)listener).enterClass_type(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof PyTypeTypeGrammarListener ) ((PyTypeTypeGrammarListener)listener).exitClass_type(this);
		}
	}

	public final Class_typeContext class_type() throws RecognitionException {
		Class_typeContext _localctx = new Class_typeContext(_ctx, getState());
		enterRule(_localctx, 8, RULE_class_type);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(41);
			match(CLASS_PREFIX);
			setState(44);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case BUILTINS_PREFIX:
			case NONE_TYPE:
			case STR:
			case BOOL:
			case INT:
			case FLOAT:
			case COMPLEX:
			case TUPLE:
			case LIST:
			case SET:
			case DICT:
				{
				setState(42);
				builtin_type();
				}
				break;
			case STRING:
				{
				setState(43);
				qualified_type();
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
			setState(46);
			match(T__0);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class Generic_callable_typeContext extends ParserRuleContext {
		public TerminalNode GENERIC_CALLABLE_PREFIX() { return getToken(PyTypeTypeGrammarParser.GENERIC_CALLABLE_PREFIX, 0); }
		public TerminalNode PARAMETERS() { return getToken(PyTypeTypeGrammarParser.PARAMETERS, 0); }
		public Type_listContext type_list() {
			return getRuleContext(Type_listContext.class,0);
		}
		public Generic_callable_typeContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_generic_callable_type; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof PyTypeTypeGrammarListener ) ((PyTypeTypeGrammarListener)listener).enterGeneric_callable_type(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof PyTypeTypeGrammarListener ) ((PyTypeTypeGrammarListener)listener).exitGeneric_callable_type(this);
		}
	}

	public final Generic_callable_typeContext generic_callable_type() throws RecognitionException {
		Generic_callable_typeContext _localctx = new Generic_callable_typeContext(_ctx, getState());
		enterRule(_localctx, 10, RULE_generic_callable_type);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(48);
			match(GENERIC_CALLABLE_PREFIX);
			setState(49);
			match(T__2);
			setState(50);
			match(PARAMETERS);
			setState(51);
			type_list();
			setState(52);
			match(T__0);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class Anything_typeContext extends ParserRuleContext {
		public TerminalNode ANYTHING_TYPE() { return getToken(PyTypeTypeGrammarParser.ANYTHING_TYPE, 0); }
		public Anything_typeContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_anything_type; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof PyTypeTypeGrammarListener ) ((PyTypeTypeGrammarListener)listener).enterAnything_type(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof PyTypeTypeGrammarListener ) ((PyTypeTypeGrammarListener)listener).exitAnything_type(this);
		}
	}

	public final Anything_typeContext anything_type() throws RecognitionException {
		Anything_typeContext _localctx = new Anything_typeContext(_ctx, getState());
		enterRule(_localctx, 12, RULE_anything_type);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(54);
			match(ANYTHING_TYPE);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class Qualified_typeContext extends ParserRuleContext {
		public List<TerminalNode> STRING() { return getTokens(PyTypeTypeGrammarParser.STRING); }
		public TerminalNode STRING(int i) {
			return getToken(PyTypeTypeGrammarParser.STRING, i);
		}
		public Qualified_typeContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_qualified_type; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof PyTypeTypeGrammarListener ) ((PyTypeTypeGrammarListener)listener).enterQualified_type(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof PyTypeTypeGrammarListener ) ((PyTypeTypeGrammarListener)listener).exitQualified_type(this);
		}
	}

	public final Qualified_typeContext qualified_type() throws RecognitionException {
		Qualified_typeContext _localctx = new Qualified_typeContext(_ctx, getState());
		enterRule(_localctx, 14, RULE_qualified_type);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(56);
			match(STRING);
			setState(61);
			_errHandler.sync(this);
			_la = _input.LA(1);
			while (_la==T__1) {
				{
				{
				setState(57);
				match(T__1);
				setState(58);
				match(STRING);
				}
				}
				setState(63);
				_errHandler.sync(this);
				_la = _input.LA(1);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class Type_listContext extends ParserRuleContext {
		public List<TypeContext> type() {
			return getRuleContexts(TypeContext.class);
		}
		public TypeContext type(int i) {
			return getRuleContext(TypeContext.class,i);
		}
		public Type_listContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_type_list; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof PyTypeTypeGrammarListener ) ((PyTypeTypeGrammarListener)listener).enterType_list(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof PyTypeTypeGrammarListener ) ((PyTypeTypeGrammarListener)listener).exitType_list(this);
		}
	}

	public final Type_listContext type_list() throws RecognitionException {
		Type_listContext _localctx = new Type_listContext(_ctx, getState());
		enterRule(_localctx, 16, RULE_type_list);
		int _la;
		try {
			setState(79);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,5,_ctx) ) {
			case 1:
				enterOuterAlt(_localctx, 1);
				{
				setState(64);
				match(T__3);
				setState(65);
				type();
				setState(68); 
				_errHandler.sync(this);
				_la = _input.LA(1);
				do {
					{
					{
					setState(66);
					match(T__2);
					setState(67);
					type();
					}
					}
					setState(70); 
					_errHandler.sync(this);
					_la = _input.LA(1);
				} while ( _la==T__2 );
				setState(72);
				match(T__0);
				}
				break;
			case 2:
				enterOuterAlt(_localctx, 2);
				{
				setState(74);
				match(T__3);
				setState(75);
				type();
				setState(76);
				match(T__2);
				setState(77);
				match(T__0);
				}
				break;
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class BuiltinContext extends ParserRuleContext {
		public TerminalNode NONE_TYPE() { return getToken(PyTypeTypeGrammarParser.NONE_TYPE, 0); }
		public TerminalNode BOOL() { return getToken(PyTypeTypeGrammarParser.BOOL, 0); }
		public TerminalNode STR() { return getToken(PyTypeTypeGrammarParser.STR, 0); }
		public TerminalNode INT() { return getToken(PyTypeTypeGrammarParser.INT, 0); }
		public TerminalNode FLOAT() { return getToken(PyTypeTypeGrammarParser.FLOAT, 0); }
		public TerminalNode COMPLEX() { return getToken(PyTypeTypeGrammarParser.COMPLEX, 0); }
		public TerminalNode TUPLE() { return getToken(PyTypeTypeGrammarParser.TUPLE, 0); }
		public TerminalNode LIST() { return getToken(PyTypeTypeGrammarParser.LIST, 0); }
		public TerminalNode SET() { return getToken(PyTypeTypeGrammarParser.SET, 0); }
		public TerminalNode DICT() { return getToken(PyTypeTypeGrammarParser.DICT, 0); }
		public BuiltinContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_builtin; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof PyTypeTypeGrammarListener ) ((PyTypeTypeGrammarListener)listener).enterBuiltin(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof PyTypeTypeGrammarListener ) ((PyTypeTypeGrammarListener)listener).exitBuiltin(this);
		}
	}

	public final BuiltinContext builtin() throws RecognitionException {
		BuiltinContext _localctx = new BuiltinContext(_ctx, getState());
		enterRule(_localctx, 18, RULE_builtin);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(81);
			_la = _input.LA(1);
			if ( !((((_la) & ~0x3f) == 0 && ((1L << _la) & 1047552L) != 0)) ) {
			_errHandler.recoverInline(this);
			}
			else {
				if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
				_errHandler.reportMatch(this);
				consume();
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static final String _serializedATN =
		"\u0004\u0001\u0016T\u0002\u0000\u0007\u0000\u0002\u0001\u0007\u0001\u0002"+
		"\u0002\u0007\u0002\u0002\u0003\u0007\u0003\u0002\u0004\u0007\u0004\u0002"+
		"\u0005\u0007\u0005\u0002\u0006\u0007\u0006\u0002\u0007\u0007\u0007\u0002"+
		"\b\u0007\b\u0002\t\u0007\t\u0001\u0000\u0001\u0000\u0001\u0000\u0001\u0001"+
		"\u0001\u0001\u0001\u0001\u0001\u0001\u0001\u0001\u0001\u0001\u0003\u0001"+
		"\u001e\b\u0001\u0001\u0002\u0001\u0002\u0001\u0002\u0001\u0002\u0001\u0003"+
		"\u0001\u0003\u0001\u0003\u0001\u0003\u0003\u0003(\b\u0003\u0001\u0004"+
		"\u0001\u0004\u0001\u0004\u0003\u0004-\b\u0004\u0001\u0004\u0001\u0004"+
		"\u0001\u0005\u0001\u0005\u0001\u0005\u0001\u0005\u0001\u0005\u0001\u0005"+
		"\u0001\u0006\u0001\u0006\u0001\u0007\u0001\u0007\u0001\u0007\u0005\u0007"+
		"<\b\u0007\n\u0007\f\u0007?\t\u0007\u0001\b\u0001\b\u0001\b\u0001\b\u0004"+
		"\bE\b\b\u000b\b\f\bF\u0001\b\u0001\b\u0001\b\u0001\b\u0001\b\u0001\b\u0001"+
		"\b\u0003\bP\b\b\u0001\t\u0001\t\u0001\t\u0000\u0000\n\u0000\u0002\u0004"+
		"\u0006\b\n\f\u000e\u0010\u0012\u0000\u0001\u0001\u0000\n\u0013S\u0000"+
		"\u0014\u0001\u0000\u0000\u0000\u0002\u001d\u0001\u0000\u0000\u0000\u0004"+
		"\u001f\u0001\u0000\u0000\u0000\u0006\'\u0001\u0000\u0000\u0000\b)\u0001"+
		"\u0000\u0000\u0000\n0\u0001\u0000\u0000\u0000\f6\u0001\u0000\u0000\u0000"+
		"\u000e8\u0001\u0000\u0000\u0000\u0010O\u0001\u0000\u0000\u0000\u0012Q"+
		"\u0001\u0000\u0000\u0000\u0014\u0015\u0003\u0002\u0001\u0000\u0015\u0016"+
		"\u0005\u0000\u0000\u0001\u0016\u0001\u0001\u0000\u0000\u0000\u0017\u001e"+
		"\u0003\u0004\u0002\u0000\u0018\u001e\u0003\u0006\u0003\u0000\u0019\u001e"+
		"\u0003\b\u0004\u0000\u001a\u001e\u0003\f\u0006\u0000\u001b\u001e\u0003"+
		"\n\u0005\u0000\u001c\u001e\u0003\u000e\u0007\u0000\u001d\u0017\u0001\u0000"+
		"\u0000\u0000\u001d\u0018\u0001\u0000\u0000\u0000\u001d\u0019\u0001\u0000"+
		"\u0000\u0000\u001d\u001a\u0001\u0000\u0000\u0000\u001d\u001b\u0001\u0000"+
		"\u0000\u0000\u001d\u001c\u0001\u0000\u0000\u0000\u001e\u0003\u0001\u0000"+
		"\u0000\u0000\u001f \u0005\u0005\u0000\u0000 !\u0003\u0010\b\u0000!\"\u0005"+
		"\u0001\u0000\u0000\"\u0005\u0001\u0000\u0000\u0000#(\u0003\u0012\t\u0000"+
		"$%\u0005\t\u0000\u0000%&\u0005\u0002\u0000\u0000&(\u0003\u0012\t\u0000"+
		"\'#\u0001\u0000\u0000\u0000\'$\u0001\u0000\u0000\u0000(\u0007\u0001\u0000"+
		"\u0000\u0000),\u0005\u0006\u0000\u0000*-\u0003\u0006\u0003\u0000+-\u0003"+
		"\u000e\u0007\u0000,*\u0001\u0000\u0000\u0000,+\u0001\u0000\u0000\u0000"+
		"-.\u0001\u0000\u0000\u0000./\u0005\u0001\u0000\u0000/\t\u0001\u0000\u0000"+
		"\u000001\u0005\u0007\u0000\u000012\u0005\u0003\u0000\u000023\u0005\u0014"+
		"\u0000\u000034\u0003\u0010\b\u000045\u0005\u0001\u0000\u00005\u000b\u0001"+
		"\u0000\u0000\u000067\u0005\b\u0000\u00007\r\u0001\u0000\u0000\u00008="+
		"\u0005\u0015\u0000\u00009:\u0005\u0002\u0000\u0000:<\u0005\u0015\u0000"+
		"\u0000;9\u0001\u0000\u0000\u0000<?\u0001\u0000\u0000\u0000=;\u0001\u0000"+
		"\u0000\u0000=>\u0001\u0000\u0000\u0000>\u000f\u0001\u0000\u0000\u0000"+
		"?=\u0001\u0000\u0000\u0000@A\u0005\u0004\u0000\u0000AD\u0003\u0002\u0001"+
		"\u0000BC\u0005\u0003\u0000\u0000CE\u0003\u0002\u0001\u0000DB\u0001\u0000"+
		"\u0000\u0000EF\u0001\u0000\u0000\u0000FD\u0001\u0000\u0000\u0000FG\u0001"+
		"\u0000\u0000\u0000GH\u0001\u0000\u0000\u0000HI\u0005\u0001\u0000\u0000"+
		"IP\u0001\u0000\u0000\u0000JK\u0005\u0004\u0000\u0000KL\u0003\u0002\u0001"+
		"\u0000LM\u0005\u0003\u0000\u0000MN\u0005\u0001\u0000\u0000NP\u0001\u0000"+
		"\u0000\u0000O@\u0001\u0000\u0000\u0000OJ\u0001\u0000\u0000\u0000P\u0011"+
		"\u0001\u0000\u0000\u0000QR\u0007\u0000\u0000\u0000R\u0013\u0001\u0000"+
		"\u0000\u0000\u0006\u001d\',=FO";
	public static final ATN _ATN =
		new ATNDeserializer().deserialize(_serializedATN.toCharArray());
	static {
		_decisionToDFA = new DFA[_ATN.getNumberOfDecisions()];
		for (int i = 0; i < _ATN.getNumberOfDecisions(); i++) {
			_decisionToDFA[i] = new DFA(_ATN.getDecisionState(i), i);
		}
	}
}