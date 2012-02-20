/*
 * Sonar Python Plugin
 * Copyright (C) 2011 Waleri Enns
 * dev@sonar.codehaus.org
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
 * You should have received a copy of the GNU Lesser General Public
 * License along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02
 */

package org.sonar.plugins.python.antlr;

// $ANTLR 3.2 debian-5 PythonLexer.g 2012-02-19 21:39:01

import org.antlr.runtime.*;
import java.util.Stack;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;
public class PythonLexer extends Lexer {
    public static final int SLASHEQUAL=104;
    public static final int EXPONENT=65;
    public static final int STAR=72;
    public static final int NONZERODIGIT=55;
    public static final int CIRCUMFLEXEQUAL=109;
    public static final int WHILE=47;
    public static final int TRIAPOS=11;
    public static final int GREATEREQUAL=86;
    public static final int LONGBYTES=15;
    public static final int NOT=41;
    public static final int EXCEPT=31;
    public static final int EOF=-1;
    public static final int BREAK=24;
    public static final int PASS=43;
    public static final int LEADING_WS=116;
    public static final int NOTEQUAL=88;
    public static final int MINUSEQUAL=102;
    public static final int VBAR=80;
    public static final int OCTINTEGER=51;
    public static final int RPAREN=90;
    public static final int IMPORT=36;
    public static final int GREATER=84;
    public static final int DOUBLESTAREQUAL=112;
    public static final int RETURN=45;
    public static final int LESS=83;
    public static final int RAISE=44;
    public static final int COMMENT=117;
    public static final int AMPERSANDEQUAL=107;
    public static final int SHORTBYTES=14;
    public static final int RBRACK=92;
    public static final int NONLOCAL=40;
    public static final int ELSE=30;
    public static final int ESCAPESEQ=10;
    public static final int LCURLY=93;
    public static final int RIGHTSHIFT=78;
    public static final int ASSERT=22;
    public static final int TRY=46;
    public static final int DOUBLESLASHEQUAL=105;
    public static final int SHORTSTRING=7;
    public static final int ELIF=29;
    public static final int WS=115;
    public static final int INTPART=63;
    public static final int BINDIGIT=59;
    public static final int VBAREQUAL=108;
    public static final int NONE=18;
    public static final int OR=42;
    public static final int BYTESLITERAL=16;
    public static final int FLOATNUMBER=62;
    public static final int FROM=33;
    public static final int FALSE=17;
    public static final int PERCENTEQUAL=106;
    public static final int LESSEQUAL=85;
    public static final int DOUBLESLASH=75;
    public static final int CLASS=25;
    public static final int EXPONENTFLOAT=61;
    public static final int CONTINUED_LINE=113;
    public static final int LBRACK=91;
    public static final int DEF=27;
    public static final int DOUBLESTAR=73;
    public static final int DEL=28;
    public static final int BYTESPREFIX=13;
    public static final int OCTDIGIT=57;
    public static final int BININTEGER=53;
    public static final int FOR=23;
    public static final int DEDENT=5;
    public static final int RIGHTSHIFTEQUAL=111;
    public static final int AND=20;
    public static final int INDENT=4;
    public static final int POINTFLOAT=60;
    public static final int LPAREN=89;
    public static final int PLUSEQUAL=101;
    public static final int IF=35;
    public static final int STRINGPREFIX=6;
    public static final int AT=99;
    public static final int AS=21;
    public static final int SLASH=74;
    public static final int IN=37;
    public static final int CONTINUE=26;
    public static final int COMMA=95;
    public static final int IS=38;
    public static final int IDENTIFIER=69;
    public static final int EQUAL=87;
    public static final int YIELD=49;
    public static final int TILDE=82;
    public static final int LEFTSHIFTEQUAL=110;
    public static final int PLUS=70;
    public static final int LEFTSHIFT=77;
    public static final int LAMBDA=39;
    public static final int DIGIT=56;
    public static final int DOT=97;
    public static final int IMAGNUMBER=66;
    public static final int WITH=48;
    public static final int INTEGER=54;
    public static final int PERCENT=76;
    public static final int AMPERSAND=79;
    public static final int HEXINTEGER=52;
    public static final int MINUS=71;
    public static final int SEMI=98;
    public static final int TRUE=19;
    public static final int LONGSTRING=8;
    public static final int COLON=96;
    public static final int TRIQUOTE=12;
    public static final int NEWLINE=114;
    public static final int FINALLY=32;
    public static final int STRINGLITERAL=9;
    public static final int RCURLY=94;
    public static final int ASSIGN=100;
    public static final int ID_CONTINUE=68;
    public static final int DECIMALINTEGER=50;
    public static final int GLOBAL=34;
    public static final int FRACTION=64;
    public static final int ID_START=67;
    public static final int STAREQUAL=103;
    public static final int CIRCUMFLEX=81;
    public static final int HEXDIGIT=58;

        /** Handles context-sensitive lexing of implicit line joining such as
         *  the case where newline is ignored in cases like this:
         *  a = [3,
         *       4]
         */
        private int implicitLineJoiningLevel = 0;
        public int startPos=-1;


    // delegates
    // delegators

    public PythonLexer() {;} 
    public PythonLexer(CharStream input) {
        this(input, new RecognizerSharedState());
    }
    public PythonLexer(CharStream input, RecognizerSharedState state) {
        super(input,state);

    }
    public String getGrammarFileName() { return "PythonLexer.g"; }

    public Token nextToken() {
        while (true) {
            if ( input.LA(1)==CharStream.EOF ) {
                return Token.EOF_TOKEN;
            }
            state.token = null;
    	state.channel = Token.DEFAULT_CHANNEL;
            state.tokenStartCharIndex = input.index();
            state.tokenStartCharPositionInLine = input.getCharPositionInLine();
            state.tokenStartLine = input.getLine();
    	state.text = null;
            try {
                int m = input.mark();
                state.backtracking=1; 
                state.failed=false;
                mTokens();
                state.backtracking=0;

                if ( state.failed ) {
                    input.rewind(m);
                    input.consume(); 
                }
                else {
                    emit();
                    return state.token;
                }
            }
            catch (RecognitionException re) {
                // shouldn't happen in backtracking mode, but...
                reportError(re);
                recover(re);
            }
        }
    }

    public void memoize(IntStream input,
    		int ruleIndex,
    		int ruleStartIndex)
    {
    if ( state.backtracking>1 ) super.memoize(input, ruleIndex, ruleStartIndex);
    }

    public boolean alreadyParsedRule(IntStream input, int ruleIndex) {
    if ( state.backtracking>1 ) return super.alreadyParsedRule(input, ruleIndex);
    return false;
    }// $ANTLR start "STRINGLITERAL"
    public final void mSTRINGLITERAL() throws RecognitionException {
        try {
            int _type = STRINGLITERAL;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:78:17: ( ( STRINGPREFIX )? ( SHORTSTRING | LONGSTRING ) )
            // PythonLexer.g:78:19: ( STRINGPREFIX )? ( SHORTSTRING | LONGSTRING )
            {
            // PythonLexer.g:78:19: ( STRINGPREFIX )?
            int alt1=2;
            int LA1_0 = input.LA(1);

            if ( (LA1_0=='R'||LA1_0=='U'||LA1_0=='r'||LA1_0=='u') ) {
                alt1=1;
            }
            switch (alt1) {
                case 1 :
                    // PythonLexer.g:0:0: STRINGPREFIX
                    {
                    mSTRINGPREFIX(); if (state.failed) return ;

                    }
                    break;

            }

            // PythonLexer.g:78:33: ( SHORTSTRING | LONGSTRING )
            int alt2=2;
            int LA2_0 = input.LA(1);

            if ( (LA2_0=='\"') ) {
                int LA2_1 = input.LA(2);

                if ( (LA2_1=='\"') ) {
                    int LA2_3 = input.LA(3);

                    if ( (LA2_3=='\"') ) {
                        alt2=2;
                    }
                    else {
                        alt2=1;}
                }
                else if ( ((LA2_1>='\u0000' && LA2_1<='\t')||(LA2_1>='\u000B' && LA2_1<='!')||(LA2_1>='#' && LA2_1<='\uFFFF')) ) {
                    alt2=1;
                }
                else {
                    if (state.backtracking>0) {state.failed=true; return ;}
                    NoViableAltException nvae =
                        new NoViableAltException("", 2, 1, input);

                    throw nvae;
                }
            }
            else if ( (LA2_0=='\'') ) {
                int LA2_2 = input.LA(2);

                if ( (LA2_2=='\'') ) {
                    int LA2_5 = input.LA(3);

                    if ( (LA2_5=='\'') ) {
                        alt2=2;
                    }
                    else {
                        alt2=1;}
                }
                else if ( ((LA2_2>='\u0000' && LA2_2<='\t')||(LA2_2>='\u000B' && LA2_2<='&')||(LA2_2>='(' && LA2_2<='\uFFFF')) ) {
                    alt2=1;
                }
                else {
                    if (state.backtracking>0) {state.failed=true; return ;}
                    NoViableAltException nvae =
                        new NoViableAltException("", 2, 2, input);

                    throw nvae;
                }
            }
            else {
                if (state.backtracking>0) {state.failed=true; return ;}
                NoViableAltException nvae =
                    new NoViableAltException("", 2, 0, input);

                throw nvae;
            }
            switch (alt2) {
                case 1 :
                    // PythonLexer.g:78:35: SHORTSTRING
                    {
                    mSHORTSTRING(); if (state.failed) return ;

                    }
                    break;
                case 2 :
                    // PythonLexer.g:78:49: LONGSTRING
                    {
                    mLONGSTRING(); if (state.failed) return ;

                    }
                    break;

            }


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "STRINGLITERAL"

    // $ANTLR start "STRINGPREFIX"
    public final void mSTRINGPREFIX() throws RecognitionException {
        try {
            // PythonLexer.g:81:9: ( ( 'r' | 'R' | 'u' | 'U' ) )
            // PythonLexer.g:81:11: ( 'r' | 'R' | 'u' | 'U' )
            {
            if ( input.LA(1)=='R'||input.LA(1)=='U'||input.LA(1)=='r'||input.LA(1)=='u' ) {
                input.consume();
            state.failed=false;
            }
            else {
                if (state.backtracking>0) {state.failed=true; return ;}
                MismatchedSetException mse = new MismatchedSetException(null,input);
                recover(mse);
                throw mse;}


            }

        }
        finally {
        }
    }
    // $ANTLR end "STRINGPREFIX"

    // $ANTLR start "SHORTSTRING"
    public final void mSHORTSTRING() throws RecognitionException {
        try {
            // PythonLexer.g:84:9: ( '\"' ( ESCAPESEQ | ~ ( '\\\\' | '\\n' | '\"' ) )* '\"' | '\\'' ( ESCAPESEQ | ~ ( '\\\\' | '\\n' | '\\'' ) )* '\\'' )
            int alt5=2;
            int LA5_0 = input.LA(1);

            if ( (LA5_0=='\"') ) {
                alt5=1;
            }
            else if ( (LA5_0=='\'') ) {
                alt5=2;
            }
            else {
                if (state.backtracking>0) {state.failed=true; return ;}
                NoViableAltException nvae =
                    new NoViableAltException("", 5, 0, input);

                throw nvae;
            }
            switch (alt5) {
                case 1 :
                    // PythonLexer.g:84:11: '\"' ( ESCAPESEQ | ~ ( '\\\\' | '\\n' | '\"' ) )* '\"'
                    {
                    match('\"'); if (state.failed) return ;
                    // PythonLexer.g:84:15: ( ESCAPESEQ | ~ ( '\\\\' | '\\n' | '\"' ) )*
                    loop3:
                    do {
                        int alt3=3;
                        int LA3_0 = input.LA(1);

                        if ( (LA3_0=='\\') ) {
                            alt3=1;
                        }
                        else if ( ((LA3_0>='\u0000' && LA3_0<='\t')||(LA3_0>='\u000B' && LA3_0<='!')||(LA3_0>='#' && LA3_0<='[')||(LA3_0>=']' && LA3_0<='\uFFFF')) ) {
                            alt3=2;
                        }


                        switch (alt3) {
                    	case 1 :
                    	    // PythonLexer.g:84:17: ESCAPESEQ
                    	    {
                    	    mESCAPESEQ(); if (state.failed) return ;

                    	    }
                    	    break;
                    	case 2 :
                    	    // PythonLexer.g:84:29: ~ ( '\\\\' | '\\n' | '\"' )
                    	    {
                    	    if ( (input.LA(1)>='\u0000' && input.LA(1)<='\t')||(input.LA(1)>='\u000B' && input.LA(1)<='!')||(input.LA(1)>='#' && input.LA(1)<='[')||(input.LA(1)>=']' && input.LA(1)<='\uFFFF') ) {
                    	        input.consume();
                    	    state.failed=false;
                    	    }
                    	    else {
                    	        if (state.backtracking>0) {state.failed=true; return ;}
                    	        MismatchedSetException mse = new MismatchedSetException(null,input);
                    	        recover(mse);
                    	        throw mse;}


                    	    }
                    	    break;

                    	default :
                    	    break loop3;
                        }
                    } while (true);

                    match('\"'); if (state.failed) return ;

                    }
                    break;
                case 2 :
                    // PythonLexer.g:85:11: '\\'' ( ESCAPESEQ | ~ ( '\\\\' | '\\n' | '\\'' ) )* '\\''
                    {
                    match('\''); if (state.failed) return ;
                    // PythonLexer.g:85:16: ( ESCAPESEQ | ~ ( '\\\\' | '\\n' | '\\'' ) )*
                    loop4:
                    do {
                        int alt4=3;
                        int LA4_0 = input.LA(1);

                        if ( (LA4_0=='\\') ) {
                            alt4=1;
                        }
                        else if ( ((LA4_0>='\u0000' && LA4_0<='\t')||(LA4_0>='\u000B' && LA4_0<='&')||(LA4_0>='(' && LA4_0<='[')||(LA4_0>=']' && LA4_0<='\uFFFF')) ) {
                            alt4=2;
                        }


                        switch (alt4) {
                    	case 1 :
                    	    // PythonLexer.g:85:18: ESCAPESEQ
                    	    {
                    	    mESCAPESEQ(); if (state.failed) return ;

                    	    }
                    	    break;
                    	case 2 :
                    	    // PythonLexer.g:85:30: ~ ( '\\\\' | '\\n' | '\\'' )
                    	    {
                    	    if ( (input.LA(1)>='\u0000' && input.LA(1)<='\t')||(input.LA(1)>='\u000B' && input.LA(1)<='&')||(input.LA(1)>='(' && input.LA(1)<='[')||(input.LA(1)>=']' && input.LA(1)<='\uFFFF') ) {
                    	        input.consume();
                    	    state.failed=false;
                    	    }
                    	    else {
                    	        if (state.backtracking>0) {state.failed=true; return ;}
                    	        MismatchedSetException mse = new MismatchedSetException(null,input);
                    	        recover(mse);
                    	        throw mse;}


                    	    }
                    	    break;

                    	default :
                    	    break loop4;
                        }
                    } while (true);

                    match('\''); if (state.failed) return ;

                    }
                    break;

            }
        }
        finally {
        }
    }
    // $ANTLR end "SHORTSTRING"

    // $ANTLR start "LONGSTRING"
    public final void mLONGSTRING() throws RecognitionException {
        try {
            // PythonLexer.g:89:9: ( '\\'\\'\\'' ( options {greedy=false; } : TRIAPOS )* '\\'\\'\\'' | '\"\"\"' ( options {greedy=false; } : TRIQUOTE )* '\"\"\"' )
            int alt8=2;
            int LA8_0 = input.LA(1);

            if ( (LA8_0=='\'') ) {
                alt8=1;
            }
            else if ( (LA8_0=='\"') ) {
                alt8=2;
            }
            else {
                if (state.backtracking>0) {state.failed=true; return ;}
                NoViableAltException nvae =
                    new NoViableAltException("", 8, 0, input);

                throw nvae;
            }
            switch (alt8) {
                case 1 :
                    // PythonLexer.g:89:11: '\\'\\'\\'' ( options {greedy=false; } : TRIAPOS )* '\\'\\'\\''
                    {
                    match("'''"); if (state.failed) return ;

                    // PythonLexer.g:89:20: ( options {greedy=false; } : TRIAPOS )*
                    loop6:
                    do {
                        int alt6=2;
                        int LA6_0 = input.LA(1);

                        if ( (LA6_0=='\'') ) {
                            int LA6_1 = input.LA(2);

                            if ( (LA6_1=='\'') ) {
                                int LA6_3 = input.LA(3);

                                if ( (LA6_3=='\'') ) {
                                    alt6=2;
                                }
                                else if ( ((LA6_3>='\u0000' && LA6_3<='&')||(LA6_3>='(' && LA6_3<='\uFFFF')) ) {
                                    alt6=1;
                                }


                            }
                            else if ( ((LA6_1>='\u0000' && LA6_1<='&')||(LA6_1>='(' && LA6_1<='\uFFFF')) ) {
                                alt6=1;
                            }


                        }
                        else if ( ((LA6_0>='\u0000' && LA6_0<='&')||(LA6_0>='(' && LA6_0<='\uFFFF')) ) {
                            alt6=1;
                        }


                        switch (alt6) {
                    	case 1 :
                    	    // PythonLexer.g:89:46: TRIAPOS
                    	    {
                    	    mTRIAPOS(); if (state.failed) return ;

                    	    }
                    	    break;

                    	default :
                    	    break loop6;
                        }
                    } while (true);

                    match("'''"); if (state.failed) return ;


                    }
                    break;
                case 2 :
                    // PythonLexer.g:90:11: '\"\"\"' ( options {greedy=false; } : TRIQUOTE )* '\"\"\"'
                    {
                    match("\"\"\""); if (state.failed) return ;

                    // PythonLexer.g:90:17: ( options {greedy=false; } : TRIQUOTE )*
                    loop7:
                    do {
                        int alt7=2;
                        int LA7_0 = input.LA(1);

                        if ( (LA7_0=='\"') ) {
                            int LA7_1 = input.LA(2);

                            if ( (LA7_1=='\"') ) {
                                int LA7_3 = input.LA(3);

                                if ( (LA7_3=='\"') ) {
                                    alt7=2;
                                }
                                else if ( ((LA7_3>='\u0000' && LA7_3<='!')||(LA7_3>='#' && LA7_3<='\uFFFF')) ) {
                                    alt7=1;
                                }


                            }
                            else if ( ((LA7_1>='\u0000' && LA7_1<='!')||(LA7_1>='#' && LA7_1<='\uFFFF')) ) {
                                alt7=1;
                            }


                        }
                        else if ( ((LA7_0>='\u0000' && LA7_0<='!')||(LA7_0>='#' && LA7_0<='\uFFFF')) ) {
                            alt7=1;
                        }


                        switch (alt7) {
                    	case 1 :
                    	    // PythonLexer.g:90:43: TRIQUOTE
                    	    {
                    	    mTRIQUOTE(); if (state.failed) return ;

                    	    }
                    	    break;

                    	default :
                    	    break loop7;
                        }
                    } while (true);

                    match("\"\"\""); if (state.failed) return ;


                    }
                    break;

            }
        }
        finally {
        }
    }
    // $ANTLR end "LONGSTRING"

    // $ANTLR start "BYTESLITERAL"
    public final void mBYTESLITERAL() throws RecognitionException {
        try {
            int _type = BYTESLITERAL;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:93:17: ( BYTESPREFIX ( SHORTBYTES | LONGBYTES ) )
            // PythonLexer.g:93:19: BYTESPREFIX ( SHORTBYTES | LONGBYTES )
            {
            mBYTESPREFIX(); if (state.failed) return ;
            // PythonLexer.g:93:31: ( SHORTBYTES | LONGBYTES )
            int alt9=2;
            int LA9_0 = input.LA(1);

            if ( (LA9_0=='\"') ) {
                int LA9_1 = input.LA(2);

                if ( (LA9_1=='\"') ) {
                    int LA9_3 = input.LA(3);

                    if ( (LA9_3=='\"') ) {
                        alt9=2;
                    }
                    else {
                        alt9=1;}
                }
                else if ( ((LA9_1>='\u0000' && LA9_1<='\t')||(LA9_1>='\u000B' && LA9_1<='!')||(LA9_1>='#' && LA9_1<='\uFFFF')) ) {
                    alt9=1;
                }
                else {
                    if (state.backtracking>0) {state.failed=true; return ;}
                    NoViableAltException nvae =
                        new NoViableAltException("", 9, 1, input);

                    throw nvae;
                }
            }
            else if ( (LA9_0=='\'') ) {
                int LA9_2 = input.LA(2);

                if ( (LA9_2=='\'') ) {
                    int LA9_5 = input.LA(3);

                    if ( (LA9_5=='\'') ) {
                        alt9=2;
                    }
                    else {
                        alt9=1;}
                }
                else if ( ((LA9_2>='\u0000' && LA9_2<='\t')||(LA9_2>='\u000B' && LA9_2<='&')||(LA9_2>='(' && LA9_2<='\uFFFF')) ) {
                    alt9=1;
                }
                else {
                    if (state.backtracking>0) {state.failed=true; return ;}
                    NoViableAltException nvae =
                        new NoViableAltException("", 9, 2, input);

                    throw nvae;
                }
            }
            else {
                if (state.backtracking>0) {state.failed=true; return ;}
                NoViableAltException nvae =
                    new NoViableAltException("", 9, 0, input);

                throw nvae;
            }
            switch (alt9) {
                case 1 :
                    // PythonLexer.g:93:33: SHORTBYTES
                    {
                    mSHORTBYTES(); if (state.failed) return ;

                    }
                    break;
                case 2 :
                    // PythonLexer.g:93:46: LONGBYTES
                    {
                    mLONGBYTES(); if (state.failed) return ;

                    }
                    break;

            }


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "BYTESLITERAL"

    // $ANTLR start "BYTESPREFIX"
    public final void mBYTESPREFIX() throws RecognitionException {
        try {
            // PythonLexer.g:96:9: ( ( 'b' | 'B' ) ( 'r' | 'R' )? )
            // PythonLexer.g:96:11: ( 'b' | 'B' ) ( 'r' | 'R' )?
            {
            if ( input.LA(1)=='B'||input.LA(1)=='b' ) {
                input.consume();
            state.failed=false;
            }
            else {
                if (state.backtracking>0) {state.failed=true; return ;}
                MismatchedSetException mse = new MismatchedSetException(null,input);
                recover(mse);
                throw mse;}

            // PythonLexer.g:96:25: ( 'r' | 'R' )?
            int alt10=2;
            int LA10_0 = input.LA(1);

            if ( (LA10_0=='R'||LA10_0=='r') ) {
                alt10=1;
            }
            switch (alt10) {
                case 1 :
                    // PythonLexer.g:
                    {
                    if ( input.LA(1)=='R'||input.LA(1)=='r' ) {
                        input.consume();
                    state.failed=false;
                    }
                    else {
                        if (state.backtracking>0) {state.failed=true; return ;}
                        MismatchedSetException mse = new MismatchedSetException(null,input);
                        recover(mse);
                        throw mse;}


                    }
                    break;

            }


            }

        }
        finally {
        }
    }
    // $ANTLR end "BYTESPREFIX"

    // $ANTLR start "SHORTBYTES"
    public final void mSHORTBYTES() throws RecognitionException {
        try {
            // PythonLexer.g:99:9: ( '\"' ( ESCAPESEQ | ~ ( '\\\\' | '\\n' | '\"' ) )* '\"' | '\\'' ( ESCAPESEQ | ~ ( '\\\\' | '\\n' | '\\'' ) )* '\\'' )
            int alt13=2;
            int LA13_0 = input.LA(1);

            if ( (LA13_0=='\"') ) {
                alt13=1;
            }
            else if ( (LA13_0=='\'') ) {
                alt13=2;
            }
            else {
                if (state.backtracking>0) {state.failed=true; return ;}
                NoViableAltException nvae =
                    new NoViableAltException("", 13, 0, input);

                throw nvae;
            }
            switch (alt13) {
                case 1 :
                    // PythonLexer.g:99:11: '\"' ( ESCAPESEQ | ~ ( '\\\\' | '\\n' | '\"' ) )* '\"'
                    {
                    match('\"'); if (state.failed) return ;
                    // PythonLexer.g:99:15: ( ESCAPESEQ | ~ ( '\\\\' | '\\n' | '\"' ) )*
                    loop11:
                    do {
                        int alt11=3;
                        int LA11_0 = input.LA(1);

                        if ( (LA11_0=='\\') ) {
                            alt11=1;
                        }
                        else if ( ((LA11_0>='\u0000' && LA11_0<='\t')||(LA11_0>='\u000B' && LA11_0<='!')||(LA11_0>='#' && LA11_0<='[')||(LA11_0>=']' && LA11_0<='\uFFFF')) ) {
                            alt11=2;
                        }


                        switch (alt11) {
                    	case 1 :
                    	    // PythonLexer.g:99:17: ESCAPESEQ
                    	    {
                    	    mESCAPESEQ(); if (state.failed) return ;

                    	    }
                    	    break;
                    	case 2 :
                    	    // PythonLexer.g:99:29: ~ ( '\\\\' | '\\n' | '\"' )
                    	    {
                    	    if ( (input.LA(1)>='\u0000' && input.LA(1)<='\t')||(input.LA(1)>='\u000B' && input.LA(1)<='!')||(input.LA(1)>='#' && input.LA(1)<='[')||(input.LA(1)>=']' && input.LA(1)<='\uFFFF') ) {
                    	        input.consume();
                    	    state.failed=false;
                    	    }
                    	    else {
                    	        if (state.backtracking>0) {state.failed=true; return ;}
                    	        MismatchedSetException mse = new MismatchedSetException(null,input);
                    	        recover(mse);
                    	        throw mse;}


                    	    }
                    	    break;

                    	default :
                    	    break loop11;
                        }
                    } while (true);

                    match('\"'); if (state.failed) return ;

                    }
                    break;
                case 2 :
                    // PythonLexer.g:100:11: '\\'' ( ESCAPESEQ | ~ ( '\\\\' | '\\n' | '\\'' ) )* '\\''
                    {
                    match('\''); if (state.failed) return ;
                    // PythonLexer.g:100:16: ( ESCAPESEQ | ~ ( '\\\\' | '\\n' | '\\'' ) )*
                    loop12:
                    do {
                        int alt12=3;
                        int LA12_0 = input.LA(1);

                        if ( (LA12_0=='\\') ) {
                            alt12=1;
                        }
                        else if ( ((LA12_0>='\u0000' && LA12_0<='\t')||(LA12_0>='\u000B' && LA12_0<='&')||(LA12_0>='(' && LA12_0<='[')||(LA12_0>=']' && LA12_0<='\uFFFF')) ) {
                            alt12=2;
                        }


                        switch (alt12) {
                    	case 1 :
                    	    // PythonLexer.g:100:18: ESCAPESEQ
                    	    {
                    	    mESCAPESEQ(); if (state.failed) return ;

                    	    }
                    	    break;
                    	case 2 :
                    	    // PythonLexer.g:100:30: ~ ( '\\\\' | '\\n' | '\\'' )
                    	    {
                    	    if ( (input.LA(1)>='\u0000' && input.LA(1)<='\t')||(input.LA(1)>='\u000B' && input.LA(1)<='&')||(input.LA(1)>='(' && input.LA(1)<='[')||(input.LA(1)>=']' && input.LA(1)<='\uFFFF') ) {
                    	        input.consume();
                    	    state.failed=false;
                    	    }
                    	    else {
                    	        if (state.backtracking>0) {state.failed=true; return ;}
                    	        MismatchedSetException mse = new MismatchedSetException(null,input);
                    	        recover(mse);
                    	        throw mse;}


                    	    }
                    	    break;

                    	default :
                    	    break loop12;
                        }
                    } while (true);

                    match('\''); if (state.failed) return ;

                    }
                    break;

            }
        }
        finally {
        }
    }
    // $ANTLR end "SHORTBYTES"

    // $ANTLR start "LONGBYTES"
    public final void mLONGBYTES() throws RecognitionException {
        try {
            // PythonLexer.g:104:9: ( '\\'\\'\\'' ( options {greedy=false; } : TRIAPOS )* '\\'\\'\\'' | '\"\"\"' ( options {greedy=false; } : TRIQUOTE )* '\"\"\"' )
            int alt16=2;
            int LA16_0 = input.LA(1);

            if ( (LA16_0=='\'') ) {
                alt16=1;
            }
            else if ( (LA16_0=='\"') ) {
                alt16=2;
            }
            else {
                if (state.backtracking>0) {state.failed=true; return ;}
                NoViableAltException nvae =
                    new NoViableAltException("", 16, 0, input);

                throw nvae;
            }
            switch (alt16) {
                case 1 :
                    // PythonLexer.g:104:11: '\\'\\'\\'' ( options {greedy=false; } : TRIAPOS )* '\\'\\'\\''
                    {
                    match("'''"); if (state.failed) return ;

                    // PythonLexer.g:104:20: ( options {greedy=false; } : TRIAPOS )*
                    loop14:
                    do {
                        int alt14=2;
                        int LA14_0 = input.LA(1);

                        if ( (LA14_0=='\'') ) {
                            int LA14_1 = input.LA(2);

                            if ( (LA14_1=='\'') ) {
                                int LA14_3 = input.LA(3);

                                if ( (LA14_3=='\'') ) {
                                    alt14=2;
                                }
                                else if ( ((LA14_3>='\u0000' && LA14_3<='&')||(LA14_3>='(' && LA14_3<='\uFFFF')) ) {
                                    alt14=1;
                                }


                            }
                            else if ( ((LA14_1>='\u0000' && LA14_1<='&')||(LA14_1>='(' && LA14_1<='\uFFFF')) ) {
                                alt14=1;
                            }


                        }
                        else if ( ((LA14_0>='\u0000' && LA14_0<='&')||(LA14_0>='(' && LA14_0<='\uFFFF')) ) {
                            alt14=1;
                        }


                        switch (alt14) {
                    	case 1 :
                    	    // PythonLexer.g:104:46: TRIAPOS
                    	    {
                    	    mTRIAPOS(); if (state.failed) return ;

                    	    }
                    	    break;

                    	default :
                    	    break loop14;
                        }
                    } while (true);

                    match("'''"); if (state.failed) return ;


                    }
                    break;
                case 2 :
                    // PythonLexer.g:105:11: '\"\"\"' ( options {greedy=false; } : TRIQUOTE )* '\"\"\"'
                    {
                    match("\"\"\""); if (state.failed) return ;

                    // PythonLexer.g:105:17: ( options {greedy=false; } : TRIQUOTE )*
                    loop15:
                    do {
                        int alt15=2;
                        int LA15_0 = input.LA(1);

                        if ( (LA15_0=='\"') ) {
                            int LA15_1 = input.LA(2);

                            if ( (LA15_1=='\"') ) {
                                int LA15_3 = input.LA(3);

                                if ( (LA15_3=='\"') ) {
                                    alt15=2;
                                }
                                else if ( ((LA15_3>='\u0000' && LA15_3<='!')||(LA15_3>='#' && LA15_3<='\uFFFF')) ) {
                                    alt15=1;
                                }


                            }
                            else if ( ((LA15_1>='\u0000' && LA15_1<='!')||(LA15_1>='#' && LA15_1<='\uFFFF')) ) {
                                alt15=1;
                            }


                        }
                        else if ( ((LA15_0>='\u0000' && LA15_0<='!')||(LA15_0>='#' && LA15_0<='\uFFFF')) ) {
                            alt15=1;
                        }


                        switch (alt15) {
                    	case 1 :
                    	    // PythonLexer.g:105:43: TRIQUOTE
                    	    {
                    	    mTRIQUOTE(); if (state.failed) return ;

                    	    }
                    	    break;

                    	default :
                    	    break loop15;
                        }
                    } while (true);

                    match("\"\"\""); if (state.failed) return ;


                    }
                    break;

            }
        }
        finally {
        }
    }
    // $ANTLR end "LONGBYTES"

    // $ANTLR start "TRIAPOS"
    public final void mTRIAPOS() throws RecognitionException {
        try {
            // PythonLexer.g:109:9: ( ( '\\'' '\\'' | ( '\\'' )? ) ( ESCAPESEQ | ~ ( '\\\\' | '\\'' ) )+ )
            // PythonLexer.g:109:11: ( '\\'' '\\'' | ( '\\'' )? ) ( ESCAPESEQ | ~ ( '\\\\' | '\\'' ) )+
            {
            // PythonLexer.g:109:11: ( '\\'' '\\'' | ( '\\'' )? )
            int alt18=2;
            int LA18_0 = input.LA(1);

            if ( (LA18_0=='\'') ) {
                int LA18_1 = input.LA(2);

                if ( (LA18_1=='\'') ) {
                    alt18=1;
                }
                else if ( ((LA18_1>='\u0000' && LA18_1<='&')||(LA18_1>='(' && LA18_1<='\uFFFF')) ) {
                    alt18=2;
                }
                else {
                    if (state.backtracking>0) {state.failed=true; return ;}
                    NoViableAltException nvae =
                        new NoViableAltException("", 18, 1, input);

                    throw nvae;
                }
            }
            else if ( ((LA18_0>='\u0000' && LA18_0<='&')||(LA18_0>='(' && LA18_0<='\uFFFF')) ) {
                alt18=2;
            }
            else {
                if (state.backtracking>0) {state.failed=true; return ;}
                NoViableAltException nvae =
                    new NoViableAltException("", 18, 0, input);

                throw nvae;
            }
            switch (alt18) {
                case 1 :
                    // PythonLexer.g:109:13: '\\'' '\\''
                    {
                    match('\''); if (state.failed) return ;
                    match('\''); if (state.failed) return ;

                    }
                    break;
                case 2 :
                    // PythonLexer.g:109:25: ( '\\'' )?
                    {
                    // PythonLexer.g:109:25: ( '\\'' )?
                    int alt17=2;
                    int LA17_0 = input.LA(1);

                    if ( (LA17_0=='\'') ) {
                        alt17=1;
                    }
                    switch (alt17) {
                        case 1 :
                            // PythonLexer.g:0:0: '\\''
                            {
                            match('\''); if (state.failed) return ;

                            }
                            break;

                    }


                    }
                    break;

            }

            // PythonLexer.g:109:33: ( ESCAPESEQ | ~ ( '\\\\' | '\\'' ) )+
            int cnt19=0;
            loop19:
            do {
                int alt19=3;
                int LA19_0 = input.LA(1);

                if ( (LA19_0=='\\') ) {
                    alt19=1;
                }
                else if ( ((LA19_0>='\u0000' && LA19_0<='&')||(LA19_0>='(' && LA19_0<='[')||(LA19_0>=']' && LA19_0<='\uFFFF')) ) {
                    alt19=2;
                }


                switch (alt19) {
            	case 1 :
            	    // PythonLexer.g:109:35: ESCAPESEQ
            	    {
            	    mESCAPESEQ(); if (state.failed) return ;

            	    }
            	    break;
            	case 2 :
            	    // PythonLexer.g:109:47: ~ ( '\\\\' | '\\'' )
            	    {
            	    if ( (input.LA(1)>='\u0000' && input.LA(1)<='&')||(input.LA(1)>='(' && input.LA(1)<='[')||(input.LA(1)>=']' && input.LA(1)<='\uFFFF') ) {
            	        input.consume();
            	    state.failed=false;
            	    }
            	    else {
            	        if (state.backtracking>0) {state.failed=true; return ;}
            	        MismatchedSetException mse = new MismatchedSetException(null,input);
            	        recover(mse);
            	        throw mse;}


            	    }
            	    break;

            	default :
            	    if ( cnt19 >= 1 ) break loop19;
            	    if (state.backtracking>0) {state.failed=true; return ;}
                        EarlyExitException eee =
                            new EarlyExitException(19, input);
                        throw eee;
                }
                cnt19++;
            } while (true);


            }

        }
        finally {
        }
    }
    // $ANTLR end "TRIAPOS"

    // $ANTLR start "TRIQUOTE"
    public final void mTRIQUOTE() throws RecognitionException {
        try {
            // PythonLexer.g:112:9: ( ( '\"' '\"' | ( '\"' )? ) ( ESCAPESEQ | ~ ( '\\\\' | '\"' ) )+ )
            // PythonLexer.g:112:11: ( '\"' '\"' | ( '\"' )? ) ( ESCAPESEQ | ~ ( '\\\\' | '\"' ) )+
            {
            // PythonLexer.g:112:11: ( '\"' '\"' | ( '\"' )? )
            int alt21=2;
            int LA21_0 = input.LA(1);

            if ( (LA21_0=='\"') ) {
                int LA21_1 = input.LA(2);

                if ( (LA21_1=='\"') ) {
                    alt21=1;
                }
                else if ( ((LA21_1>='\u0000' && LA21_1<='!')||(LA21_1>='#' && LA21_1<='\uFFFF')) ) {
                    alt21=2;
                }
                else {
                    if (state.backtracking>0) {state.failed=true; return ;}
                    NoViableAltException nvae =
                        new NoViableAltException("", 21, 1, input);

                    throw nvae;
                }
            }
            else if ( ((LA21_0>='\u0000' && LA21_0<='!')||(LA21_0>='#' && LA21_0<='\uFFFF')) ) {
                alt21=2;
            }
            else {
                if (state.backtracking>0) {state.failed=true; return ;}
                NoViableAltException nvae =
                    new NoViableAltException("", 21, 0, input);

                throw nvae;
            }
            switch (alt21) {
                case 1 :
                    // PythonLexer.g:112:13: '\"' '\"'
                    {
                    match('\"'); if (state.failed) return ;
                    match('\"'); if (state.failed) return ;

                    }
                    break;
                case 2 :
                    // PythonLexer.g:112:23: ( '\"' )?
                    {
                    // PythonLexer.g:112:23: ( '\"' )?
                    int alt20=2;
                    int LA20_0 = input.LA(1);

                    if ( (LA20_0=='\"') ) {
                        alt20=1;
                    }
                    switch (alt20) {
                        case 1 :
                            // PythonLexer.g:0:0: '\"'
                            {
                            match('\"'); if (state.failed) return ;

                            }
                            break;

                    }


                    }
                    break;

            }

            // PythonLexer.g:112:30: ( ESCAPESEQ | ~ ( '\\\\' | '\"' ) )+
            int cnt22=0;
            loop22:
            do {
                int alt22=3;
                int LA22_0 = input.LA(1);

                if ( (LA22_0=='\\') ) {
                    alt22=1;
                }
                else if ( ((LA22_0>='\u0000' && LA22_0<='!')||(LA22_0>='#' && LA22_0<='[')||(LA22_0>=']' && LA22_0<='\uFFFF')) ) {
                    alt22=2;
                }


                switch (alt22) {
            	case 1 :
            	    // PythonLexer.g:112:32: ESCAPESEQ
            	    {
            	    mESCAPESEQ(); if (state.failed) return ;

            	    }
            	    break;
            	case 2 :
            	    // PythonLexer.g:112:44: ~ ( '\\\\' | '\"' )
            	    {
            	    if ( (input.LA(1)>='\u0000' && input.LA(1)<='!')||(input.LA(1)>='#' && input.LA(1)<='[')||(input.LA(1)>=']' && input.LA(1)<='\uFFFF') ) {
            	        input.consume();
            	    state.failed=false;
            	    }
            	    else {
            	        if (state.backtracking>0) {state.failed=true; return ;}
            	        MismatchedSetException mse = new MismatchedSetException(null,input);
            	        recover(mse);
            	        throw mse;}


            	    }
            	    break;

            	default :
            	    if ( cnt22 >= 1 ) break loop22;
            	    if (state.backtracking>0) {state.failed=true; return ;}
                        EarlyExitException eee =
                            new EarlyExitException(22, input);
                        throw eee;
                }
                cnt22++;
            } while (true);


            }

        }
        finally {
        }
    }
    // $ANTLR end "TRIQUOTE"

    // $ANTLR start "ESCAPESEQ"
    public final void mESCAPESEQ() throws RecognitionException {
        try {
            // PythonLexer.g:115:9: ( '\\\\' . )
            // PythonLexer.g:115:11: '\\\\' .
            {
            match('\\'); if (state.failed) return ;
            matchAny(); if (state.failed) return ;

            }

        }
        finally {
        }
    }
    // $ANTLR end "ESCAPESEQ"

    // $ANTLR start "FALSE"
    public final void mFALSE() throws RecognitionException {
        try {
            int _type = FALSE;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:122:13: ( 'False' )
            // PythonLexer.g:122:15: 'False'
            {
            match("False"); if (state.failed) return ;


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "FALSE"

    // $ANTLR start "NONE"
    public final void mNONE() throws RecognitionException {
        try {
            int _type = NONE;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:123:13: ( 'None' )
            // PythonLexer.g:123:15: 'None'
            {
            match("None"); if (state.failed) return ;


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "NONE"

    // $ANTLR start "TRUE"
    public final void mTRUE() throws RecognitionException {
        try {
            int _type = TRUE;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:124:13: ( 'True' )
            // PythonLexer.g:124:15: 'True'
            {
            match("True"); if (state.failed) return ;


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "TRUE"

    // $ANTLR start "AND"
    public final void mAND() throws RecognitionException {
        try {
            int _type = AND;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:125:13: ( 'and' )
            // PythonLexer.g:125:15: 'and'
            {
            match("and"); if (state.failed) return ;


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "AND"

    // $ANTLR start "AS"
    public final void mAS() throws RecognitionException {
        try {
            int _type = AS;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:126:13: ( 'as' )
            // PythonLexer.g:126:15: 'as'
            {
            match("as"); if (state.failed) return ;


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "AS"

    // $ANTLR start "ASSERT"
    public final void mASSERT() throws RecognitionException {
        try {
            int _type = ASSERT;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:127:13: ( 'assert' )
            // PythonLexer.g:127:15: 'assert'
            {
            match("assert"); if (state.failed) return ;


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "ASSERT"

    // $ANTLR start "FOR"
    public final void mFOR() throws RecognitionException {
        try {
            int _type = FOR;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:128:13: ( 'for' )
            // PythonLexer.g:128:15: 'for'
            {
            match("for"); if (state.failed) return ;


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "FOR"

    // $ANTLR start "BREAK"
    public final void mBREAK() throws RecognitionException {
        try {
            int _type = BREAK;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:129:13: ( 'break' )
            // PythonLexer.g:129:15: 'break'
            {
            match("break"); if (state.failed) return ;


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "BREAK"

    // $ANTLR start "CLASS"
    public final void mCLASS() throws RecognitionException {
        try {
            int _type = CLASS;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:130:13: ( 'class' )
            // PythonLexer.g:130:15: 'class'
            {
            match("class"); if (state.failed) return ;


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "CLASS"

    // $ANTLR start "CONTINUE"
    public final void mCONTINUE() throws RecognitionException {
        try {
            int _type = CONTINUE;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:131:13: ( 'continue' )
            // PythonLexer.g:131:15: 'continue'
            {
            match("continue"); if (state.failed) return ;


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "CONTINUE"

    // $ANTLR start "DEF"
    public final void mDEF() throws RecognitionException {
        try {
            int _type = DEF;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:132:13: ( 'def' )
            // PythonLexer.g:132:15: 'def'
            {
            match("def"); if (state.failed) return ;


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "DEF"

    // $ANTLR start "DEL"
    public final void mDEL() throws RecognitionException {
        try {
            int _type = DEL;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:133:13: ( 'del' )
            // PythonLexer.g:133:15: 'del'
            {
            match("del"); if (state.failed) return ;


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "DEL"

    // $ANTLR start "ELIF"
    public final void mELIF() throws RecognitionException {
        try {
            int _type = ELIF;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:134:13: ( 'elif' )
            // PythonLexer.g:134:15: 'elif'
            {
            match("elif"); if (state.failed) return ;


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "ELIF"

    // $ANTLR start "ELSE"
    public final void mELSE() throws RecognitionException {
        try {
            int _type = ELSE;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:135:13: ( 'else' )
            // PythonLexer.g:135:15: 'else'
            {
            match("else"); if (state.failed) return ;


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "ELSE"

    // $ANTLR start "EXCEPT"
    public final void mEXCEPT() throws RecognitionException {
        try {
            int _type = EXCEPT;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:136:13: ( 'except' )
            // PythonLexer.g:136:15: 'except'
            {
            match("except"); if (state.failed) return ;


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "EXCEPT"

    // $ANTLR start "FINALLY"
    public final void mFINALLY() throws RecognitionException {
        try {
            int _type = FINALLY;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:137:13: ( 'finally' )
            // PythonLexer.g:137:15: 'finally'
            {
            match("finally"); if (state.failed) return ;


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "FINALLY"

    // $ANTLR start "FROM"
    public final void mFROM() throws RecognitionException {
        try {
            int _type = FROM;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:138:13: ( 'from' )
            // PythonLexer.g:138:15: 'from'
            {
            match("from"); if (state.failed) return ;


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "FROM"

    // $ANTLR start "GLOBAL"
    public final void mGLOBAL() throws RecognitionException {
        try {
            int _type = GLOBAL;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:139:13: ( 'global' )
            // PythonLexer.g:139:15: 'global'
            {
            match("global"); if (state.failed) return ;


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "GLOBAL"

    // $ANTLR start "IF"
    public final void mIF() throws RecognitionException {
        try {
            int _type = IF;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:140:13: ( 'if' )
            // PythonLexer.g:140:15: 'if'
            {
            match("if"); if (state.failed) return ;


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "IF"

    // $ANTLR start "IMPORT"
    public final void mIMPORT() throws RecognitionException {
        try {
            int _type = IMPORT;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:141:13: ( 'import' )
            // PythonLexer.g:141:15: 'import'
            {
            match("import"); if (state.failed) return ;


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "IMPORT"

    // $ANTLR start "IN"
    public final void mIN() throws RecognitionException {
        try {
            int _type = IN;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:142:13: ( 'in' )
            // PythonLexer.g:142:15: 'in'
            {
            match("in"); if (state.failed) return ;


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "IN"

    // $ANTLR start "IS"
    public final void mIS() throws RecognitionException {
        try {
            int _type = IS;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:143:13: ( 'is' )
            // PythonLexer.g:143:15: 'is'
            {
            match("is"); if (state.failed) return ;


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "IS"

    // $ANTLR start "LAMBDA"
    public final void mLAMBDA() throws RecognitionException {
        try {
            int _type = LAMBDA;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:144:13: ( 'lambda' )
            // PythonLexer.g:144:15: 'lambda'
            {
            match("lambda"); if (state.failed) return ;


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "LAMBDA"

    // $ANTLR start "NONLOCAL"
    public final void mNONLOCAL() throws RecognitionException {
        try {
            int _type = NONLOCAL;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:145:13: ( 'nonlocal' )
            // PythonLexer.g:145:15: 'nonlocal'
            {
            match("nonlocal"); if (state.failed) return ;


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "NONLOCAL"

    // $ANTLR start "NOT"
    public final void mNOT() throws RecognitionException {
        try {
            int _type = NOT;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:146:13: ( 'not' )
            // PythonLexer.g:146:15: 'not'
            {
            match("not"); if (state.failed) return ;


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "NOT"

    // $ANTLR start "OR"
    public final void mOR() throws RecognitionException {
        try {
            int _type = OR;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:147:13: ( 'or' )
            // PythonLexer.g:147:15: 'or'
            {
            match("or"); if (state.failed) return ;


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "OR"

    // $ANTLR start "PASS"
    public final void mPASS() throws RecognitionException {
        try {
            int _type = PASS;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:148:13: ( 'pass' )
            // PythonLexer.g:148:15: 'pass'
            {
            match("pass"); if (state.failed) return ;


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "PASS"

    // $ANTLR start "RAISE"
    public final void mRAISE() throws RecognitionException {
        try {
            int _type = RAISE;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:149:13: ( 'raise' )
            // PythonLexer.g:149:15: 'raise'
            {
            match("raise"); if (state.failed) return ;


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "RAISE"

    // $ANTLR start "RETURN"
    public final void mRETURN() throws RecognitionException {
        try {
            int _type = RETURN;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:150:13: ( 'return' )
            // PythonLexer.g:150:15: 'return'
            {
            match("return"); if (state.failed) return ;


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "RETURN"

    // $ANTLR start "TRY"
    public final void mTRY() throws RecognitionException {
        try {
            int _type = TRY;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:151:13: ( 'try' )
            // PythonLexer.g:151:15: 'try'
            {
            match("try"); if (state.failed) return ;


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "TRY"

    // $ANTLR start "WHILE"
    public final void mWHILE() throws RecognitionException {
        try {
            int _type = WHILE;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:152:13: ( 'while' )
            // PythonLexer.g:152:15: 'while'
            {
            match("while"); if (state.failed) return ;


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "WHILE"

    // $ANTLR start "WITH"
    public final void mWITH() throws RecognitionException {
        try {
            int _type = WITH;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:153:13: ( 'with' )
            // PythonLexer.g:153:15: 'with'
            {
            match("with"); if (state.failed) return ;


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "WITH"

    // $ANTLR start "YIELD"
    public final void mYIELD() throws RecognitionException {
        try {
            int _type = YIELD;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:154:13: ( 'yield' )
            // PythonLexer.g:154:15: 'yield'
            {
            match("yield"); if (state.failed) return ;


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "YIELD"

    // $ANTLR start "INTEGER"
    public final void mINTEGER() throws RecognitionException {
        try {
            int _type = INTEGER;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:161:9: ( DECIMALINTEGER | OCTINTEGER | HEXINTEGER | BININTEGER )
            int alt23=4;
            int LA23_0 = input.LA(1);

            if ( ((LA23_0>='1' && LA23_0<='9')) ) {
                alt23=1;
            }
            else if ( (LA23_0=='0') ) {
                switch ( input.LA(2) ) {
                case 'O':
                case 'o':
                    {
                    alt23=2;
                    }
                    break;
                case 'X':
                case 'x':
                    {
                    alt23=3;
                    }
                    break;
                case 'B':
                case 'b':
                    {
                    alt23=4;
                    }
                    break;
                default:
                    alt23=1;}

            }
            else {
                if (state.backtracking>0) {state.failed=true; return ;}
                NoViableAltException nvae =
                    new NoViableAltException("", 23, 0, input);

                throw nvae;
            }
            switch (alt23) {
                case 1 :
                    // PythonLexer.g:161:11: DECIMALINTEGER
                    {
                    mDECIMALINTEGER(); if (state.failed) return ;

                    }
                    break;
                case 2 :
                    // PythonLexer.g:161:28: OCTINTEGER
                    {
                    mOCTINTEGER(); if (state.failed) return ;

                    }
                    break;
                case 3 :
                    // PythonLexer.g:161:41: HEXINTEGER
                    {
                    mHEXINTEGER(); if (state.failed) return ;

                    }
                    break;
                case 4 :
                    // PythonLexer.g:161:54: BININTEGER
                    {
                    mBININTEGER(); if (state.failed) return ;

                    }
                    break;

            }
            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "INTEGER"

    // $ANTLR start "DECIMALINTEGER"
    public final void mDECIMALINTEGER() throws RecognitionException {
        try {
            // PythonLexer.g:163:9: ( NONZERODIGIT ( DIGIT )* ( 'l' | 'L' )? | ( '0' )+ ( 'l' | 'L' )? )
            int alt28=2;
            int LA28_0 = input.LA(1);

            if ( ((LA28_0>='1' && LA28_0<='9')) ) {
                alt28=1;
            }
            else if ( (LA28_0=='0') ) {
                alt28=2;
            }
            else {
                if (state.backtracking>0) {state.failed=true; return ;}
                NoViableAltException nvae =
                    new NoViableAltException("", 28, 0, input);

                throw nvae;
            }
            switch (alt28) {
                case 1 :
                    // PythonLexer.g:163:11: NONZERODIGIT ( DIGIT )* ( 'l' | 'L' )?
                    {
                    mNONZERODIGIT(); if (state.failed) return ;
                    // PythonLexer.g:163:24: ( DIGIT )*
                    loop24:
                    do {
                        int alt24=2;
                        int LA24_0 = input.LA(1);

                        if ( ((LA24_0>='0' && LA24_0<='9')) ) {
                            alt24=1;
                        }


                        switch (alt24) {
                    	case 1 :
                    	    // PythonLexer.g:0:0: DIGIT
                    	    {
                    	    mDIGIT(); if (state.failed) return ;

                    	    }
                    	    break;

                    	default :
                    	    break loop24;
                        }
                    } while (true);

                    // PythonLexer.g:163:31: ( 'l' | 'L' )?
                    int alt25=2;
                    int LA25_0 = input.LA(1);

                    if ( (LA25_0=='L'||LA25_0=='l') ) {
                        alt25=1;
                    }
                    switch (alt25) {
                        case 1 :
                            // PythonLexer.g:
                            {
                            if ( input.LA(1)=='L'||input.LA(1)=='l' ) {
                                input.consume();
                            state.failed=false;
                            }
                            else {
                                if (state.backtracking>0) {state.failed=true; return ;}
                                MismatchedSetException mse = new MismatchedSetException(null,input);
                                recover(mse);
                                throw mse;}


                            }
                            break;

                    }


                    }
                    break;
                case 2 :
                    // PythonLexer.g:163:48: ( '0' )+ ( 'l' | 'L' )?
                    {
                    // PythonLexer.g:163:48: ( '0' )+
                    int cnt26=0;
                    loop26:
                    do {
                        int alt26=2;
                        int LA26_0 = input.LA(1);

                        if ( (LA26_0=='0') ) {
                            alt26=1;
                        }


                        switch (alt26) {
                    	case 1 :
                    	    // PythonLexer.g:0:0: '0'
                    	    {
                    	    match('0'); if (state.failed) return ;

                    	    }
                    	    break;

                    	default :
                    	    if ( cnt26 >= 1 ) break loop26;
                    	    if (state.backtracking>0) {state.failed=true; return ;}
                                EarlyExitException eee =
                                    new EarlyExitException(26, input);
                                throw eee;
                        }
                        cnt26++;
                    } while (true);

                    // PythonLexer.g:163:53: ( 'l' | 'L' )?
                    int alt27=2;
                    int LA27_0 = input.LA(1);

                    if ( (LA27_0=='L'||LA27_0=='l') ) {
                        alt27=1;
                    }
                    switch (alt27) {
                        case 1 :
                            // PythonLexer.g:
                            {
                            if ( input.LA(1)=='L'||input.LA(1)=='l' ) {
                                input.consume();
                            state.failed=false;
                            }
                            else {
                                if (state.backtracking>0) {state.failed=true; return ;}
                                MismatchedSetException mse = new MismatchedSetException(null,input);
                                recover(mse);
                                throw mse;}


                            }
                            break;

                    }


                    }
                    break;

            }
        }
        finally {
        }
    }
    // $ANTLR end "DECIMALINTEGER"

    // $ANTLR start "NONZERODIGIT"
    public final void mNONZERODIGIT() throws RecognitionException {
        try {
            // PythonLexer.g:165:9: ( '1' .. '9' )
            // PythonLexer.g:165:11: '1' .. '9'
            {
            matchRange('1','9'); if (state.failed) return ;

            }

        }
        finally {
        }
    }
    // $ANTLR end "NONZERODIGIT"

    // $ANTLR start "DIGIT"
    public final void mDIGIT() throws RecognitionException {
        try {
            // PythonLexer.g:167:9: ( '0' .. '9' )
            // PythonLexer.g:167:11: '0' .. '9'
            {
            matchRange('0','9'); if (state.failed) return ;

            }

        }
        finally {
        }
    }
    // $ANTLR end "DIGIT"

    // $ANTLR start "OCTINTEGER"
    public final void mOCTINTEGER() throws RecognitionException {
        try {
            // PythonLexer.g:169:9: ( '0' ( 'o' | 'O' ) ( OCTDIGIT )+ ( 'l' | 'L' )? )
            // PythonLexer.g:169:11: '0' ( 'o' | 'O' ) ( OCTDIGIT )+ ( 'l' | 'L' )?
            {
            match('0'); if (state.failed) return ;
            if ( input.LA(1)=='O'||input.LA(1)=='o' ) {
                input.consume();
            state.failed=false;
            }
            else {
                if (state.backtracking>0) {state.failed=true; return ;}
                MismatchedSetException mse = new MismatchedSetException(null,input);
                recover(mse);
                throw mse;}

            // PythonLexer.g:169:29: ( OCTDIGIT )+
            int cnt29=0;
            loop29:
            do {
                int alt29=2;
                int LA29_0 = input.LA(1);

                if ( ((LA29_0>='0' && LA29_0<='7')) ) {
                    alt29=1;
                }


                switch (alt29) {
            	case 1 :
            	    // PythonLexer.g:0:0: OCTDIGIT
            	    {
            	    mOCTDIGIT(); if (state.failed) return ;

            	    }
            	    break;

            	default :
            	    if ( cnt29 >= 1 ) break loop29;
            	    if (state.backtracking>0) {state.failed=true; return ;}
                        EarlyExitException eee =
                            new EarlyExitException(29, input);
                        throw eee;
                }
                cnt29++;
            } while (true);

            // PythonLexer.g:169:39: ( 'l' | 'L' )?
            int alt30=2;
            int LA30_0 = input.LA(1);

            if ( (LA30_0=='L'||LA30_0=='l') ) {
                alt30=1;
            }
            switch (alt30) {
                case 1 :
                    // PythonLexer.g:
                    {
                    if ( input.LA(1)=='L'||input.LA(1)=='l' ) {
                        input.consume();
                    state.failed=false;
                    }
                    else {
                        if (state.backtracking>0) {state.failed=true; return ;}
                        MismatchedSetException mse = new MismatchedSetException(null,input);
                        recover(mse);
                        throw mse;}


                    }
                    break;

            }


            }

        }
        finally {
        }
    }
    // $ANTLR end "OCTINTEGER"

    // $ANTLR start "HEXINTEGER"
    public final void mHEXINTEGER() throws RecognitionException {
        try {
            // PythonLexer.g:171:9: ( '0' ( 'x' | 'X' ) ( HEXDIGIT )+ ( 'l' | 'L' )? )
            // PythonLexer.g:171:11: '0' ( 'x' | 'X' ) ( HEXDIGIT )+ ( 'l' | 'L' )?
            {
            match('0'); if (state.failed) return ;
            if ( input.LA(1)=='X'||input.LA(1)=='x' ) {
                input.consume();
            state.failed=false;
            }
            else {
                if (state.backtracking>0) {state.failed=true; return ;}
                MismatchedSetException mse = new MismatchedSetException(null,input);
                recover(mse);
                throw mse;}

            // PythonLexer.g:171:29: ( HEXDIGIT )+
            int cnt31=0;
            loop31:
            do {
                int alt31=2;
                int LA31_0 = input.LA(1);

                if ( ((LA31_0>='0' && LA31_0<='9')||(LA31_0>='A' && LA31_0<='F')||(LA31_0>='a' && LA31_0<='f')) ) {
                    alt31=1;
                }


                switch (alt31) {
            	case 1 :
            	    // PythonLexer.g:0:0: HEXDIGIT
            	    {
            	    mHEXDIGIT(); if (state.failed) return ;

            	    }
            	    break;

            	default :
            	    if ( cnt31 >= 1 ) break loop31;
            	    if (state.backtracking>0) {state.failed=true; return ;}
                        EarlyExitException eee =
                            new EarlyExitException(31, input);
                        throw eee;
                }
                cnt31++;
            } while (true);

            // PythonLexer.g:171:39: ( 'l' | 'L' )?
            int alt32=2;
            int LA32_0 = input.LA(1);

            if ( (LA32_0=='L'||LA32_0=='l') ) {
                alt32=1;
            }
            switch (alt32) {
                case 1 :
                    // PythonLexer.g:
                    {
                    if ( input.LA(1)=='L'||input.LA(1)=='l' ) {
                        input.consume();
                    state.failed=false;
                    }
                    else {
                        if (state.backtracking>0) {state.failed=true; return ;}
                        MismatchedSetException mse = new MismatchedSetException(null,input);
                        recover(mse);
                        throw mse;}


                    }
                    break;

            }


            }

        }
        finally {
        }
    }
    // $ANTLR end "HEXINTEGER"

    // $ANTLR start "BININTEGER"
    public final void mBININTEGER() throws RecognitionException {
        try {
            // PythonLexer.g:173:9: ( '0' ( 'b' | 'B' ) ( BINDIGIT )+ ( 'l' | 'L' )? )
            // PythonLexer.g:173:11: '0' ( 'b' | 'B' ) ( BINDIGIT )+ ( 'l' | 'L' )?
            {
            match('0'); if (state.failed) return ;
            if ( input.LA(1)=='B'||input.LA(1)=='b' ) {
                input.consume();
            state.failed=false;
            }
            else {
                if (state.backtracking>0) {state.failed=true; return ;}
                MismatchedSetException mse = new MismatchedSetException(null,input);
                recover(mse);
                throw mse;}

            // PythonLexer.g:173:29: ( BINDIGIT )+
            int cnt33=0;
            loop33:
            do {
                int alt33=2;
                int LA33_0 = input.LA(1);

                if ( ((LA33_0>='0' && LA33_0<='1')) ) {
                    alt33=1;
                }


                switch (alt33) {
            	case 1 :
            	    // PythonLexer.g:0:0: BINDIGIT
            	    {
            	    mBINDIGIT(); if (state.failed) return ;

            	    }
            	    break;

            	default :
            	    if ( cnt33 >= 1 ) break loop33;
            	    if (state.backtracking>0) {state.failed=true; return ;}
                        EarlyExitException eee =
                            new EarlyExitException(33, input);
                        throw eee;
                }
                cnt33++;
            } while (true);

            // PythonLexer.g:173:39: ( 'l' | 'L' )?
            int alt34=2;
            int LA34_0 = input.LA(1);

            if ( (LA34_0=='L'||LA34_0=='l') ) {
                alt34=1;
            }
            switch (alt34) {
                case 1 :
                    // PythonLexer.g:
                    {
                    if ( input.LA(1)=='L'||input.LA(1)=='l' ) {
                        input.consume();
                    state.failed=false;
                    }
                    else {
                        if (state.backtracking>0) {state.failed=true; return ;}
                        MismatchedSetException mse = new MismatchedSetException(null,input);
                        recover(mse);
                        throw mse;}


                    }
                    break;

            }


            }

        }
        finally {
        }
    }
    // $ANTLR end "BININTEGER"

    // $ANTLR start "OCTDIGIT"
    public final void mOCTDIGIT() throws RecognitionException {
        try {
            // PythonLexer.g:175:9: ( '0' .. '7' )
            // PythonLexer.g:175:11: '0' .. '7'
            {
            matchRange('0','7'); if (state.failed) return ;

            }

        }
        finally {
        }
    }
    // $ANTLR end "OCTDIGIT"

    // $ANTLR start "HEXDIGIT"
    public final void mHEXDIGIT() throws RecognitionException {
        try {
            // PythonLexer.g:177:9: ( DIGIT | 'a' .. 'f' | 'A' .. 'F' )
            // PythonLexer.g:
            {
            if ( (input.LA(1)>='0' && input.LA(1)<='9')||(input.LA(1)>='A' && input.LA(1)<='F')||(input.LA(1)>='a' && input.LA(1)<='f') ) {
                input.consume();
            state.failed=false;
            }
            else {
                if (state.backtracking>0) {state.failed=true; return ;}
                MismatchedSetException mse = new MismatchedSetException(null,input);
                recover(mse);
                throw mse;}


            }

        }
        finally {
        }
    }
    // $ANTLR end "HEXDIGIT"

    // $ANTLR start "BINDIGIT"
    public final void mBINDIGIT() throws RecognitionException {
        try {
            // PythonLexer.g:179:9: ( '0' | '1' )
            // PythonLexer.g:
            {
            if ( (input.LA(1)>='0' && input.LA(1)<='1') ) {
                input.consume();
            state.failed=false;
            }
            else {
                if (state.backtracking>0) {state.failed=true; return ;}
                MismatchedSetException mse = new MismatchedSetException(null,input);
                recover(mse);
                throw mse;}


            }

        }
        finally {
        }
    }
    // $ANTLR end "BINDIGIT"

    // $ANTLR start "FLOATNUMBER"
    public final void mFLOATNUMBER() throws RecognitionException {
        try {
            int _type = FLOATNUMBER;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:186:13: ( POINTFLOAT | EXPONENTFLOAT )
            int alt35=2;
            alt35 = dfa35.predict(input);
            switch (alt35) {
                case 1 :
                    // PythonLexer.g:186:15: POINTFLOAT
                    {
                    mPOINTFLOAT(); if (state.failed) return ;

                    }
                    break;
                case 2 :
                    // PythonLexer.g:186:28: EXPONENTFLOAT
                    {
                    mEXPONENTFLOAT(); if (state.failed) return ;

                    }
                    break;

            }
            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "FLOATNUMBER"

    // $ANTLR start "POINTFLOAT"
    public final void mPOINTFLOAT() throws RecognitionException {
        try {
            // PythonLexer.g:188:9: ( ( ( INTPART )? FRACTION ) | ( INTPART '.' ) )
            int alt37=2;
            alt37 = dfa37.predict(input);
            switch (alt37) {
                case 1 :
                    // PythonLexer.g:188:11: ( ( INTPART )? FRACTION )
                    {
                    // PythonLexer.g:188:11: ( ( INTPART )? FRACTION )
                    // PythonLexer.g:188:13: ( INTPART )? FRACTION
                    {
                    // PythonLexer.g:188:13: ( INTPART )?
                    int alt36=2;
                    int LA36_0 = input.LA(1);

                    if ( ((LA36_0>='0' && LA36_0<='9')) ) {
                        alt36=1;
                    }
                    switch (alt36) {
                        case 1 :
                            // PythonLexer.g:0:0: INTPART
                            {
                            mINTPART(); if (state.failed) return ;

                            }
                            break;

                    }

                    mFRACTION(); if (state.failed) return ;

                    }


                    }
                    break;
                case 2 :
                    // PythonLexer.g:189:11: ( INTPART '.' )
                    {
                    // PythonLexer.g:189:11: ( INTPART '.' )
                    // PythonLexer.g:189:13: INTPART '.'
                    {
                    mINTPART(); if (state.failed) return ;
                    match('.'); if (state.failed) return ;

                    }


                    }
                    break;

            }
        }
        finally {
        }
    }
    // $ANTLR end "POINTFLOAT"

    // $ANTLR start "EXPONENTFLOAT"
    public final void mEXPONENTFLOAT() throws RecognitionException {
        try {
            // PythonLexer.g:192:9: ( ( INTPART | POINTFLOAT ) EXPONENT )
            // PythonLexer.g:192:11: ( INTPART | POINTFLOAT ) EXPONENT
            {
            // PythonLexer.g:192:11: ( INTPART | POINTFLOAT )
            int alt38=2;
            alt38 = dfa38.predict(input);
            switch (alt38) {
                case 1 :
                    // PythonLexer.g:192:13: INTPART
                    {
                    mINTPART(); if (state.failed) return ;

                    }
                    break;
                case 2 :
                    // PythonLexer.g:192:23: POINTFLOAT
                    {
                    mPOINTFLOAT(); if (state.failed) return ;

                    }
                    break;

            }

            mEXPONENT(); if (state.failed) return ;

            }

        }
        finally {
        }
    }
    // $ANTLR end "EXPONENTFLOAT"

    // $ANTLR start "INTPART"
    public final void mINTPART() throws RecognitionException {
        try {
            // PythonLexer.g:194:9: ( ( DIGIT )+ )
            // PythonLexer.g:194:11: ( DIGIT )+
            {
            // PythonLexer.g:194:11: ( DIGIT )+
            int cnt39=0;
            loop39:
            do {
                int alt39=2;
                int LA39_0 = input.LA(1);

                if ( ((LA39_0>='0' && LA39_0<='9')) ) {
                    alt39=1;
                }


                switch (alt39) {
            	case 1 :
            	    // PythonLexer.g:0:0: DIGIT
            	    {
            	    mDIGIT(); if (state.failed) return ;

            	    }
            	    break;

            	default :
            	    if ( cnt39 >= 1 ) break loop39;
            	    if (state.backtracking>0) {state.failed=true; return ;}
                        EarlyExitException eee =
                            new EarlyExitException(39, input);
                        throw eee;
                }
                cnt39++;
            } while (true);


            }

        }
        finally {
        }
    }
    // $ANTLR end "INTPART"

    // $ANTLR start "FRACTION"
    public final void mFRACTION() throws RecognitionException {
        try {
            // PythonLexer.g:196:9: ( '.' ( DIGIT )+ )
            // PythonLexer.g:196:11: '.' ( DIGIT )+
            {
            match('.'); if (state.failed) return ;
            // PythonLexer.g:196:15: ( DIGIT )+
            int cnt40=0;
            loop40:
            do {
                int alt40=2;
                int LA40_0 = input.LA(1);

                if ( ((LA40_0>='0' && LA40_0<='9')) ) {
                    alt40=1;
                }


                switch (alt40) {
            	case 1 :
            	    // PythonLexer.g:0:0: DIGIT
            	    {
            	    mDIGIT(); if (state.failed) return ;

            	    }
            	    break;

            	default :
            	    if ( cnt40 >= 1 ) break loop40;
            	    if (state.backtracking>0) {state.failed=true; return ;}
                        EarlyExitException eee =
                            new EarlyExitException(40, input);
                        throw eee;
                }
                cnt40++;
            } while (true);


            }

        }
        finally {
        }
    }
    // $ANTLR end "FRACTION"

    // $ANTLR start "EXPONENT"
    public final void mEXPONENT() throws RecognitionException {
        try {
            // PythonLexer.g:198:9: ( ( 'e' | 'E' ) ( '+' | '-' )? ( DIGIT )+ )
            // PythonLexer.g:198:11: ( 'e' | 'E' ) ( '+' | '-' )? ( DIGIT )+
            {
            if ( input.LA(1)=='E'||input.LA(1)=='e' ) {
                input.consume();
            state.failed=false;
            }
            else {
                if (state.backtracking>0) {state.failed=true; return ;}
                MismatchedSetException mse = new MismatchedSetException(null,input);
                recover(mse);
                throw mse;}

            // PythonLexer.g:198:25: ( '+' | '-' )?
            int alt41=2;
            int LA41_0 = input.LA(1);

            if ( (LA41_0=='+'||LA41_0=='-') ) {
                alt41=1;
            }
            switch (alt41) {
                case 1 :
                    // PythonLexer.g:
                    {
                    if ( input.LA(1)=='+'||input.LA(1)=='-' ) {
                        input.consume();
                    state.failed=false;
                    }
                    else {
                        if (state.backtracking>0) {state.failed=true; return ;}
                        MismatchedSetException mse = new MismatchedSetException(null,input);
                        recover(mse);
                        throw mse;}


                    }
                    break;

            }

            // PythonLexer.g:198:40: ( DIGIT )+
            int cnt42=0;
            loop42:
            do {
                int alt42=2;
                int LA42_0 = input.LA(1);

                if ( ((LA42_0>='0' && LA42_0<='9')) ) {
                    alt42=1;
                }


                switch (alt42) {
            	case 1 :
            	    // PythonLexer.g:0:0: DIGIT
            	    {
            	    mDIGIT(); if (state.failed) return ;

            	    }
            	    break;

            	default :
            	    if ( cnt42 >= 1 ) break loop42;
            	    if (state.backtracking>0) {state.failed=true; return ;}
                        EarlyExitException eee =
                            new EarlyExitException(42, input);
                        throw eee;
                }
                cnt42++;
            } while (true);


            }

        }
        finally {
        }
    }
    // $ANTLR end "EXPONENT"

    // $ANTLR start "IMAGNUMBER"
    public final void mIMAGNUMBER() throws RecognitionException {
        try {
            int _type = IMAGNUMBER;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:205:13: ( ( FLOATNUMBER | INTPART ) ( 'j' | 'J' ) )
            // PythonLexer.g:205:15: ( FLOATNUMBER | INTPART ) ( 'j' | 'J' )
            {
            // PythonLexer.g:205:15: ( FLOATNUMBER | INTPART )
            int alt43=2;
            alt43 = dfa43.predict(input);
            switch (alt43) {
                case 1 :
                    // PythonLexer.g:205:17: FLOATNUMBER
                    {
                    mFLOATNUMBER(); if (state.failed) return ;

                    }
                    break;
                case 2 :
                    // PythonLexer.g:205:31: INTPART
                    {
                    mINTPART(); if (state.failed) return ;

                    }
                    break;

            }

            if ( input.LA(1)=='J'||input.LA(1)=='j' ) {
                input.consume();
            state.failed=false;
            }
            else {
                if (state.backtracking>0) {state.failed=true; return ;}
                MismatchedSetException mse = new MismatchedSetException(null,input);
                recover(mse);
                throw mse;}


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "IMAGNUMBER"

    // $ANTLR start "IDENTIFIER"
    public final void mIDENTIFIER() throws RecognitionException {
        try {
            int _type = IDENTIFIER;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:212:13: ( ID_START ( ID_CONTINUE )* )
            // PythonLexer.g:212:15: ID_START ( ID_CONTINUE )*
            {
            mID_START(); if (state.failed) return ;
            // PythonLexer.g:212:24: ( ID_CONTINUE )*
            loop44:
            do {
                int alt44=2;
                int LA44_0 = input.LA(1);

                if ( ((LA44_0>='0' && LA44_0<='9')||(LA44_0>='A' && LA44_0<='Z')||LA44_0=='_'||(LA44_0>='a' && LA44_0<='z')) ) {
                    alt44=1;
                }


                switch (alt44) {
            	case 1 :
            	    // PythonLexer.g:0:0: ID_CONTINUE
            	    {
            	    mID_CONTINUE(); if (state.failed) return ;

            	    }
            	    break;

            	default :
            	    break loop44;
                }
            } while (true);


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "IDENTIFIER"

    // $ANTLR start "ID_START"
    public final void mID_START() throws RecognitionException {
        try {
            // PythonLexer.g:216:9: ( '_' | 'A' .. 'Z' | 'a' .. 'z' )
            // PythonLexer.g:
            {
            if ( (input.LA(1)>='A' && input.LA(1)<='Z')||input.LA(1)=='_'||(input.LA(1)>='a' && input.LA(1)<='z') ) {
                input.consume();
            state.failed=false;
            }
            else {
                if (state.backtracking>0) {state.failed=true; return ;}
                MismatchedSetException mse = new MismatchedSetException(null,input);
                recover(mse);
                throw mse;}


            }

        }
        finally {
        }
    }
    // $ANTLR end "ID_START"

    // $ANTLR start "ID_CONTINUE"
    public final void mID_CONTINUE() throws RecognitionException {
        try {
            // PythonLexer.g:223:9: ( '_' | 'A' .. 'Z' | 'a' .. 'z' | '0' .. '9' )
            // PythonLexer.g:
            {
            if ( (input.LA(1)>='0' && input.LA(1)<='9')||(input.LA(1)>='A' && input.LA(1)<='Z')||input.LA(1)=='_'||(input.LA(1)>='a' && input.LA(1)<='z') ) {
                input.consume();
            state.failed=false;
            }
            else {
                if (state.backtracking>0) {state.failed=true; return ;}
                MismatchedSetException mse = new MismatchedSetException(null,input);
                recover(mse);
                throw mse;}


            }

        }
        finally {
        }
    }
    // $ANTLR end "ID_CONTINUE"

    // $ANTLR start "PLUS"
    public final void mPLUS() throws RecognitionException {
        try {
            int _type = PLUS;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:234:17: ( '+' )
            // PythonLexer.g:234:19: '+'
            {
            match('+'); if (state.failed) return ;

            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "PLUS"

    // $ANTLR start "MINUS"
    public final void mMINUS() throws RecognitionException {
        try {
            int _type = MINUS;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:235:17: ( '-' )
            // PythonLexer.g:235:19: '-'
            {
            match('-'); if (state.failed) return ;

            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "MINUS"

    // $ANTLR start "STAR"
    public final void mSTAR() throws RecognitionException {
        try {
            int _type = STAR;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:236:17: ( '*' )
            // PythonLexer.g:236:19: '*'
            {
            match('*'); if (state.failed) return ;

            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "STAR"

    // $ANTLR start "DOUBLESTAR"
    public final void mDOUBLESTAR() throws RecognitionException {
        try {
            int _type = DOUBLESTAR;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:237:17: ( '**' )
            // PythonLexer.g:237:19: '**'
            {
            match("**"); if (state.failed) return ;


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "DOUBLESTAR"

    // $ANTLR start "SLASH"
    public final void mSLASH() throws RecognitionException {
        try {
            int _type = SLASH;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:238:17: ( '/' )
            // PythonLexer.g:238:19: '/'
            {
            match('/'); if (state.failed) return ;

            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "SLASH"

    // $ANTLR start "DOUBLESLASH"
    public final void mDOUBLESLASH() throws RecognitionException {
        try {
            int _type = DOUBLESLASH;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:239:17: ( '//' )
            // PythonLexer.g:239:19: '//'
            {
            match("//"); if (state.failed) return ;


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "DOUBLESLASH"

    // $ANTLR start "PERCENT"
    public final void mPERCENT() throws RecognitionException {
        try {
            int _type = PERCENT;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:240:17: ( '%' )
            // PythonLexer.g:240:19: '%'
            {
            match('%'); if (state.failed) return ;

            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "PERCENT"

    // $ANTLR start "LEFTSHIFT"
    public final void mLEFTSHIFT() throws RecognitionException {
        try {
            int _type = LEFTSHIFT;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:241:17: ( '<<' )
            // PythonLexer.g:241:19: '<<'
            {
            match("<<"); if (state.failed) return ;


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "LEFTSHIFT"

    // $ANTLR start "RIGHTSHIFT"
    public final void mRIGHTSHIFT() throws RecognitionException {
        try {
            int _type = RIGHTSHIFT;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:242:17: ( '>>' )
            // PythonLexer.g:242:19: '>>'
            {
            match(">>"); if (state.failed) return ;


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "RIGHTSHIFT"

    // $ANTLR start "AMPERSAND"
    public final void mAMPERSAND() throws RecognitionException {
        try {
            int _type = AMPERSAND;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:243:17: ( '&' )
            // PythonLexer.g:243:19: '&'
            {
            match('&'); if (state.failed) return ;

            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "AMPERSAND"

    // $ANTLR start "VBAR"
    public final void mVBAR() throws RecognitionException {
        try {
            int _type = VBAR;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:244:17: ( '|' )
            // PythonLexer.g:244:19: '|'
            {
            match('|'); if (state.failed) return ;

            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "VBAR"

    // $ANTLR start "CIRCUMFLEX"
    public final void mCIRCUMFLEX() throws RecognitionException {
        try {
            int _type = CIRCUMFLEX;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:245:17: ( '^' )
            // PythonLexer.g:245:19: '^'
            {
            match('^'); if (state.failed) return ;

            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "CIRCUMFLEX"

    // $ANTLR start "TILDE"
    public final void mTILDE() throws RecognitionException {
        try {
            int _type = TILDE;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:246:17: ( '~' )
            // PythonLexer.g:246:19: '~'
            {
            match('~'); if (state.failed) return ;

            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "TILDE"

    // $ANTLR start "LESS"
    public final void mLESS() throws RecognitionException {
        try {
            int _type = LESS;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:247:17: ( '<' )
            // PythonLexer.g:247:19: '<'
            {
            match('<'); if (state.failed) return ;

            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "LESS"

    // $ANTLR start "GREATER"
    public final void mGREATER() throws RecognitionException {
        try {
            int _type = GREATER;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:248:17: ( '>' )
            // PythonLexer.g:248:19: '>'
            {
            match('>'); if (state.failed) return ;

            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "GREATER"

    // $ANTLR start "LESSEQUAL"
    public final void mLESSEQUAL() throws RecognitionException {
        try {
            int _type = LESSEQUAL;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:249:17: ( '<=' )
            // PythonLexer.g:249:19: '<='
            {
            match("<="); if (state.failed) return ;


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "LESSEQUAL"

    // $ANTLR start "GREATEREQUAL"
    public final void mGREATEREQUAL() throws RecognitionException {
        try {
            int _type = GREATEREQUAL;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:250:17: ( '>=' )
            // PythonLexer.g:250:19: '>='
            {
            match(">="); if (state.failed) return ;


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "GREATEREQUAL"

    // $ANTLR start "EQUAL"
    public final void mEQUAL() throws RecognitionException {
        try {
            int _type = EQUAL;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:251:17: ( '==' )
            // PythonLexer.g:251:19: '=='
            {
            match("=="); if (state.failed) return ;


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "EQUAL"

    // $ANTLR start "NOTEQUAL"
    public final void mNOTEQUAL() throws RecognitionException {
        try {
            int _type = NOTEQUAL;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:252:17: ( '!=' )
            // PythonLexer.g:252:19: '!='
            {
            match("!="); if (state.failed) return ;


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "NOTEQUAL"

    // $ANTLR start "LPAREN"
    public final void mLPAREN() throws RecognitionException {
        try {
            int _type = LPAREN;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:260:13: ( '(' )
            // PythonLexer.g:260:15: '('
            {
            match('('); if (state.failed) return ;
            if ( state.backtracking==1 ) {
              implicitLineJoiningLevel += 1;
            }

            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "LPAREN"

    // $ANTLR start "RPAREN"
    public final void mRPAREN() throws RecognitionException {
        try {
            int _type = RPAREN;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:261:13: ( ')' )
            // PythonLexer.g:261:15: ')'
            {
            match(')'); if (state.failed) return ;
            if ( state.backtracking==1 ) {
              implicitLineJoiningLevel -= 1;
            }

            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "RPAREN"

    // $ANTLR start "LBRACK"
    public final void mLBRACK() throws RecognitionException {
        try {
            int _type = LBRACK;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:262:13: ( '[' )
            // PythonLexer.g:262:15: '['
            {
            match('['); if (state.failed) return ;
            if ( state.backtracking==1 ) {
              implicitLineJoiningLevel += 1;
            }

            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "LBRACK"

    // $ANTLR start "RBRACK"
    public final void mRBRACK() throws RecognitionException {
        try {
            int _type = RBRACK;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:263:13: ( ']' )
            // PythonLexer.g:263:15: ']'
            {
            match(']'); if (state.failed) return ;
            if ( state.backtracking==1 ) {
              implicitLineJoiningLevel -= 1;
            }

            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "RBRACK"

    // $ANTLR start "LCURLY"
    public final void mLCURLY() throws RecognitionException {
        try {
            int _type = LCURLY;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:264:13: ( '{' )
            // PythonLexer.g:264:15: '{'
            {
            match('{'); if (state.failed) return ;
            if ( state.backtracking==1 ) {
              implicitLineJoiningLevel += 1;
            }

            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "LCURLY"

    // $ANTLR start "RCURLY"
    public final void mRCURLY() throws RecognitionException {
        try {
            int _type = RCURLY;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:265:13: ( '}' )
            // PythonLexer.g:265:15: '}'
            {
            match('}'); if (state.failed) return ;
            if ( state.backtracking==1 ) {
              implicitLineJoiningLevel -= 1;
            }

            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "RCURLY"

    // $ANTLR start "COMMA"
    public final void mCOMMA() throws RecognitionException {
        try {
            int _type = COMMA;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:267:13: ( ',' )
            // PythonLexer.g:267:15: ','
            {
            match(','); if (state.failed) return ;

            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "COMMA"

    // $ANTLR start "COLON"
    public final void mCOLON() throws RecognitionException {
        try {
            int _type = COLON;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:268:13: ( ':' )
            // PythonLexer.g:268:15: ':'
            {
            match(':'); if (state.failed) return ;

            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "COLON"

    // $ANTLR start "DOT"
    public final void mDOT() throws RecognitionException {
        try {
            int _type = DOT;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:269:13: ( '.' )
            // PythonLexer.g:269:15: '.'
            {
            match('.'); if (state.failed) return ;

            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "DOT"

    // $ANTLR start "SEMI"
    public final void mSEMI() throws RecognitionException {
        try {
            int _type = SEMI;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:270:13: ( ';' )
            // PythonLexer.g:270:15: ';'
            {
            match(';'); if (state.failed) return ;

            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "SEMI"

    // $ANTLR start "AT"
    public final void mAT() throws RecognitionException {
        try {
            int _type = AT;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:271:13: ( '@' )
            // PythonLexer.g:271:15: '@'
            {
            match('@'); if (state.failed) return ;

            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "AT"

    // $ANTLR start "ASSIGN"
    public final void mASSIGN() throws RecognitionException {
        try {
            int _type = ASSIGN;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:272:13: ( '=' )
            // PythonLexer.g:272:15: '='
            {
            match('='); if (state.failed) return ;

            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "ASSIGN"

    // $ANTLR start "PLUSEQUAL"
    public final void mPLUSEQUAL() throws RecognitionException {
        try {
            int _type = PLUSEQUAL;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:275:18: ( '+=' )
            // PythonLexer.g:275:20: '+='
            {
            match("+="); if (state.failed) return ;


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "PLUSEQUAL"

    // $ANTLR start "MINUSEQUAL"
    public final void mMINUSEQUAL() throws RecognitionException {
        try {
            int _type = MINUSEQUAL;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:276:18: ( '-=' )
            // PythonLexer.g:276:20: '-='
            {
            match("-="); if (state.failed) return ;


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "MINUSEQUAL"

    // $ANTLR start "STAREQUAL"
    public final void mSTAREQUAL() throws RecognitionException {
        try {
            int _type = STAREQUAL;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:277:18: ( '*=' )
            // PythonLexer.g:277:20: '*='
            {
            match("*="); if (state.failed) return ;


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "STAREQUAL"

    // $ANTLR start "SLASHEQUAL"
    public final void mSLASHEQUAL() throws RecognitionException {
        try {
            int _type = SLASHEQUAL;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:278:18: ( '/=' )
            // PythonLexer.g:278:20: '/='
            {
            match("/="); if (state.failed) return ;


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "SLASHEQUAL"

    // $ANTLR start "DOUBLESLASHEQUAL"
    public final void mDOUBLESLASHEQUAL() throws RecognitionException {
        try {
            int _type = DOUBLESLASHEQUAL;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:279:18: ( '//=' )
            // PythonLexer.g:279:20: '//='
            {
            match("//="); if (state.failed) return ;


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "DOUBLESLASHEQUAL"

    // $ANTLR start "PERCENTEQUAL"
    public final void mPERCENTEQUAL() throws RecognitionException {
        try {
            int _type = PERCENTEQUAL;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:280:18: ( '%=' )
            // PythonLexer.g:280:20: '%='
            {
            match("%="); if (state.failed) return ;


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "PERCENTEQUAL"

    // $ANTLR start "AMPERSANDEQUAL"
    public final void mAMPERSANDEQUAL() throws RecognitionException {
        try {
            int _type = AMPERSANDEQUAL;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:281:18: ( '&=' )
            // PythonLexer.g:281:20: '&='
            {
            match("&="); if (state.failed) return ;


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "AMPERSANDEQUAL"

    // $ANTLR start "VBAREQUAL"
    public final void mVBAREQUAL() throws RecognitionException {
        try {
            int _type = VBAREQUAL;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:282:18: ( '|=' )
            // PythonLexer.g:282:20: '|='
            {
            match("|="); if (state.failed) return ;


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "VBAREQUAL"

    // $ANTLR start "CIRCUMFLEXEQUAL"
    public final void mCIRCUMFLEXEQUAL() throws RecognitionException {
        try {
            int _type = CIRCUMFLEXEQUAL;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:283:18: ( '^=' )
            // PythonLexer.g:283:20: '^='
            {
            match("^="); if (state.failed) return ;


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "CIRCUMFLEXEQUAL"

    // $ANTLR start "LEFTSHIFTEQUAL"
    public final void mLEFTSHIFTEQUAL() throws RecognitionException {
        try {
            int _type = LEFTSHIFTEQUAL;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:284:18: ( '<<=' )
            // PythonLexer.g:284:20: '<<='
            {
            match("<<="); if (state.failed) return ;


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "LEFTSHIFTEQUAL"

    // $ANTLR start "RIGHTSHIFTEQUAL"
    public final void mRIGHTSHIFTEQUAL() throws RecognitionException {
        try {
            int _type = RIGHTSHIFTEQUAL;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:285:18: ( '>>=' )
            // PythonLexer.g:285:20: '>>='
            {
            match(">>="); if (state.failed) return ;


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "RIGHTSHIFTEQUAL"

    // $ANTLR start "DOUBLESTAREQUAL"
    public final void mDOUBLESTAREQUAL() throws RecognitionException {
        try {
            int _type = DOUBLESTAREQUAL;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:286:18: ( '**=' )
            // PythonLexer.g:286:20: '**='
            {
            match("**="); if (state.failed) return ;


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "DOUBLESTAREQUAL"

    // $ANTLR start "CONTINUED_LINE"
    public final void mCONTINUED_LINE() throws RecognitionException {
        try {
            int _type = CONTINUED_LINE;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:298:9: ( '\\\\' ( '\\r' )? '\\n' ( ' ' | '\\t' )* )
            // PythonLexer.g:298:11: '\\\\' ( '\\r' )? '\\n' ( ' ' | '\\t' )*
            {
            match('\\'); if (state.failed) return ;
            // PythonLexer.g:298:16: ( '\\r' )?
            int alt45=2;
            int LA45_0 = input.LA(1);

            if ( (LA45_0=='\r') ) {
                alt45=1;
            }
            switch (alt45) {
                case 1 :
                    // PythonLexer.g:298:17: '\\r'
                    {
                    match('\r'); if (state.failed) return ;

                    }
                    break;

            }

            match('\n'); if (state.failed) return ;
            // PythonLexer.g:298:29: ( ' ' | '\\t' )*
            loop46:
            do {
                int alt46=2;
                int LA46_0 = input.LA(1);

                if ( (LA46_0=='\t'||LA46_0==' ') ) {
                    alt46=1;
                }


                switch (alt46) {
            	case 1 :
            	    // PythonLexer.g:
            	    {
            	    if ( input.LA(1)=='\t'||input.LA(1)==' ' ) {
            	        input.consume();
            	    state.failed=false;
            	    }
            	    else {
            	        if (state.backtracking>0) {state.failed=true; return ;}
            	        MismatchedSetException mse = new MismatchedSetException(null,input);
            	        recover(mse);
            	        throw mse;}


            	    }
            	    break;

            	default :
            	    break loop46;
                }
            } while (true);

            if ( state.backtracking==1 ) {
               _channel=HIDDEN; 
            }

            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "CONTINUED_LINE"

    // $ANTLR start "NEWLINE"
    public final void mNEWLINE() throws RecognitionException {
        try {
            int _type = NEWLINE;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:308:9: ( ( ( '\\u000C' )? ( '\\r' )? '\\n' )+ )
            // PythonLexer.g:308:11: ( ( '\\u000C' )? ( '\\r' )? '\\n' )+
            {
            // PythonLexer.g:308:11: ( ( '\\u000C' )? ( '\\r' )? '\\n' )+
            int cnt49=0;
            loop49:
            do {
                int alt49=2;
                int LA49_0 = input.LA(1);

                if ( (LA49_0=='\n'||(LA49_0>='\f' && LA49_0<='\r')) ) {
                    alt49=1;
                }


                switch (alt49) {
            	case 1 :
            	    // PythonLexer.g:308:13: ( '\\u000C' )? ( '\\r' )? '\\n'
            	    {
            	    // PythonLexer.g:308:13: ( '\\u000C' )?
            	    int alt47=2;
            	    int LA47_0 = input.LA(1);

            	    if ( (LA47_0=='\f') ) {
            	        alt47=1;
            	    }
            	    switch (alt47) {
            	        case 1 :
            	            // PythonLexer.g:0:0: '\\u000C'
            	            {
            	            match('\f'); if (state.failed) return ;

            	            }
            	            break;

            	    }

            	    // PythonLexer.g:308:23: ( '\\r' )?
            	    int alt48=2;
            	    int LA48_0 = input.LA(1);

            	    if ( (LA48_0=='\r') ) {
            	        alt48=1;
            	    }
            	    switch (alt48) {
            	        case 1 :
            	            // PythonLexer.g:0:0: '\\r'
            	            {
            	            match('\r'); if (state.failed) return ;

            	            }
            	            break;

            	    }

            	    match('\n'); if (state.failed) return ;

            	    }
            	    break;

            	default :
            	    if ( cnt49 >= 1 ) break loop49;
            	    if (state.backtracking>0) {state.failed=true; return ;}
                        EarlyExitException eee =
                            new EarlyExitException(49, input);
                        throw eee;
                }
                cnt49++;
            } while (true);

            if ( state.backtracking==1 ) {

                          if (startPos==0 || implicitLineJoiningLevel>0)
                          {
                              _channel=HIDDEN;
                          }
                      
            }

            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "NEWLINE"

    // $ANTLR start "WS"
    public final void mWS() throws RecognitionException {
        try {
            int _type = WS;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:322:9: ({...}? => ( ' ' | '\\t' )+ )
            // PythonLexer.g:322:11: {...}? => ( ' ' | '\\t' )+
            {
            if ( !((startPos>0)) ) {
                if (state.backtracking>0) {state.failed=true; return ;}
                throw new FailedPredicateException(input, "WS", "startPos>0");
            }
            // PythonLexer.g:322:27: ( ' ' | '\\t' )+
            int cnt50=0;
            loop50:
            do {
                int alt50=2;
                int LA50_0 = input.LA(1);

                if ( (LA50_0=='\t'||LA50_0==' ') ) {
                    alt50=1;
                }


                switch (alt50) {
            	case 1 :
            	    // PythonLexer.g:
            	    {
            	    if ( input.LA(1)=='\t'||input.LA(1)==' ' ) {
            	        input.consume();
            	    state.failed=false;
            	    }
            	    else {
            	        if (state.backtracking>0) {state.failed=true; return ;}
            	        MismatchedSetException mse = new MismatchedSetException(null,input);
            	        recover(mse);
            	        throw mse;}


            	    }
            	    break;

            	default :
            	    if ( cnt50 >= 1 ) break loop50;
            	    if (state.backtracking>0) {state.failed=true; return ;}
                        EarlyExitException eee =
                            new EarlyExitException(50, input);
                        throw eee;
                }
                cnt50++;
            } while (true);

            if ( state.backtracking==1 ) {
              _channel=HIDDEN;
            }

            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "WS"

    // $ANTLR start "LEADING_WS"
    public final void mLEADING_WS() throws RecognitionException {
        try {
            int _type = LEADING_WS;
            int _channel = DEFAULT_TOKEN_CHANNEL;

                        int spaces = 0;
                    
            // PythonLexer.g:335:9: ({...}? => ({...}? ( ' ' | '\\t' )+ | ( ' ' | '\\t' )+ ( ( '\\r' )? '\\n' )* ) )
            // PythonLexer.g:335:11: {...}? => ({...}? ( ' ' | '\\t' )+ | ( ' ' | '\\t' )+ ( ( '\\r' )? '\\n' )* )
            {
            if ( !((startPos==0)) ) {
                if (state.backtracking>0) {state.failed=true; return ;}
                throw new FailedPredicateException(input, "LEADING_WS", "startPos==0");
            }
            // PythonLexer.g:336:9: ({...}? ( ' ' | '\\t' )+ | ( ' ' | '\\t' )+ ( ( '\\r' )? '\\n' )* )
            int alt55=2;
            int LA55_0 = input.LA(1);

            if ( (LA55_0==' ') ) {
                int LA55_1 = input.LA(2);

                if ( ((synpred97_PythonLexer()&&(implicitLineJoiningLevel>0))) ) {
                    alt55=1;
                }
                else if ( (true) ) {
                    alt55=2;
                }
                else {
                    if (state.backtracking>0) {state.failed=true; return ;}
                    NoViableAltException nvae =
                        new NoViableAltException("", 55, 1, input);

                    throw nvae;
                }
            }
            else if ( (LA55_0=='\t') ) {
                int LA55_2 = input.LA(2);

                if ( ((synpred97_PythonLexer()&&(implicitLineJoiningLevel>0))) ) {
                    alt55=1;
                }
                else if ( (true) ) {
                    alt55=2;
                }
                else {
                    if (state.backtracking>0) {state.failed=true; return ;}
                    NoViableAltException nvae =
                        new NoViableAltException("", 55, 2, input);

                    throw nvae;
                }
            }
            else {
                if (state.backtracking>0) {state.failed=true; return ;}
                NoViableAltException nvae =
                    new NoViableAltException("", 55, 0, input);

                throw nvae;
            }
            switch (alt55) {
                case 1 :
                    // PythonLexer.g:336:11: {...}? ( ' ' | '\\t' )+
                    {
                    if ( !((implicitLineJoiningLevel>0)) ) {
                        if (state.backtracking>0) {state.failed=true; return ;}
                        throw new FailedPredicateException(input, "LEADING_WS", "implicitLineJoiningLevel>0");
                    }
                    // PythonLexer.g:336:41: ( ' ' | '\\t' )+
                    int cnt51=0;
                    loop51:
                    do {
                        int alt51=2;
                        int LA51_0 = input.LA(1);

                        if ( (LA51_0=='\t'||LA51_0==' ') ) {
                            alt51=1;
                        }


                        switch (alt51) {
                    	case 1 :
                    	    // PythonLexer.g:
                    	    {
                    	    if ( input.LA(1)=='\t'||input.LA(1)==' ' ) {
                    	        input.consume();
                    	    state.failed=false;
                    	    }
                    	    else {
                    	        if (state.backtracking>0) {state.failed=true; return ;}
                    	        MismatchedSetException mse = new MismatchedSetException(null,input);
                    	        recover(mse);
                    	        throw mse;}


                    	    }
                    	    break;

                    	default :
                    	    if ( cnt51 >= 1 ) break loop51;
                    	    if (state.backtracking>0) {state.failed=true; return ;}
                                EarlyExitException eee =
                                    new EarlyExitException(51, input);
                                throw eee;
                        }
                        cnt51++;
                    } while (true);

                    if ( state.backtracking==1 ) {
                      _channel=HIDDEN;
                    }

                    }
                    break;
                case 2 :
                    // PythonLexer.g:337:13: ( ' ' | '\\t' )+ ( ( '\\r' )? '\\n' )*
                    {
                    // PythonLexer.g:337:13: ( ' ' | '\\t' )+
                    int cnt52=0;
                    loop52:
                    do {
                        int alt52=3;
                        int LA52_0 = input.LA(1);

                        if ( (LA52_0==' ') ) {
                            alt52=1;
                        }
                        else if ( (LA52_0=='\t') ) {
                            alt52=2;
                        }


                        switch (alt52) {
                    	case 1 :
                    	    // PythonLexer.g:337:14: ' '
                    	    {
                    	    match(' '); if (state.failed) return ;
                    	    if ( state.backtracking==1 ) {
                    	       spaces++; 
                    	    }

                    	    }
                    	    break;
                    	case 2 :
                    	    // PythonLexer.g:338:17: '\\t'
                    	    {
                    	    match('\t'); if (state.failed) return ;
                    	    if ( state.backtracking==1 ) {
                    	       spaces += 8; spaces -= (spaces % 8); 
                    	    }

                    	    }
                    	    break;

                    	default :
                    	    if ( cnt52 >= 1 ) break loop52;
                    	    if (state.backtracking>0) {state.failed=true; return ;}
                                EarlyExitException eee =
                                    new EarlyExitException(52, input);
                                throw eee;
                        }
                        cnt52++;
                    } while (true);

                    if ( state.backtracking==1 ) {

                                      // make a string of n spaces where n is column number - 1
                                      char[] indentation = new char[spaces];
                                      for (int i=0; i<spaces; i++) {
                                          indentation[i] = ' ';
                                      }
                                      String s = new String(indentation);
                                      emit(new ClassicToken(LEADING_WS, new String(indentation)));
                                  
                    }
                    // PythonLexer.g:351:13: ( ( '\\r' )? '\\n' )*
                    loop54:
                    do {
                        int alt54=2;
                        int LA54_0 = input.LA(1);

                        if ( (LA54_0=='\n'||LA54_0=='\r') ) {
                            alt54=1;
                        }


                        switch (alt54) {
                    	case 1 :
                    	    // PythonLexer.g:352:17: ( '\\r' )? '\\n'
                    	    {
                    	    // PythonLexer.g:352:17: ( '\\r' )?
                    	    int alt53=2;
                    	    int LA53_0 = input.LA(1);

                    	    if ( (LA53_0=='\r') ) {
                    	        alt53=1;
                    	    }
                    	    switch (alt53) {
                    	        case 1 :
                    	            // PythonLexer.g:352:18: '\\r'
                    	            {
                    	            match('\r'); if (state.failed) return ;

                    	            }
                    	            break;

                    	    }

                    	    match('\n'); if (state.failed) return ;
                    	    if ( state.backtracking==1 ) {

                    	                          if (state.token!=null)
                    	                              state.token.setChannel(HIDDEN);
                    	                          else
                    	                              _channel=HIDDEN;
                    	                      
                    	    }

                    	    }
                    	    break;

                    	default :
                    	    break loop54;
                        }
                    } while (true);


                    }
                    break;

            }


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "LEADING_WS"

    // $ANTLR start "COMMENT"
    public final void mCOMMENT() throws RecognitionException {
        try {
            int _type = COMMENT;
            int _channel = DEFAULT_TOKEN_CHANNEL;

                    _channel=HIDDEN;
                
            // PythonLexer.g:373:5: ({...}? => ( ' ' | '\\t' )* '#' (~ '\\n' )* ( '\\n' )+ | '#' (~ '\\n' )* )
            int alt60=2;
            alt60 = dfa60.predict(input);
            switch (alt60) {
                case 1 :
                    // PythonLexer.g:373:7: {...}? => ( ' ' | '\\t' )* '#' (~ '\\n' )* ( '\\n' )+
                    {
                    if ( !((startPos==0)) ) {
                        if (state.backtracking>0) {state.failed=true; return ;}
                        throw new FailedPredicateException(input, "COMMENT", "startPos==0");
                    }
                    // PythonLexer.g:373:24: ( ' ' | '\\t' )*
                    loop56:
                    do {
                        int alt56=2;
                        int LA56_0 = input.LA(1);

                        if ( (LA56_0=='\t'||LA56_0==' ') ) {
                            alt56=1;
                        }


                        switch (alt56) {
                    	case 1 :
                    	    // PythonLexer.g:
                    	    {
                    	    if ( input.LA(1)=='\t'||input.LA(1)==' ' ) {
                    	        input.consume();
                    	    state.failed=false;
                    	    }
                    	    else {
                    	        if (state.backtracking>0) {state.failed=true; return ;}
                    	        MismatchedSetException mse = new MismatchedSetException(null,input);
                    	        recover(mse);
                    	        throw mse;}


                    	    }
                    	    break;

                    	default :
                    	    break loop56;
                        }
                    } while (true);

                    match('#'); if (state.failed) return ;
                    // PythonLexer.g:373:44: (~ '\\n' )*
                    loop57:
                    do {
                        int alt57=2;
                        int LA57_0 = input.LA(1);

                        if ( ((LA57_0>='\u0000' && LA57_0<='\t')||(LA57_0>='\u000B' && LA57_0<='\uFFFF')) ) {
                            alt57=1;
                        }


                        switch (alt57) {
                    	case 1 :
                    	    // PythonLexer.g:373:46: ~ '\\n'
                    	    {
                    	    if ( (input.LA(1)>='\u0000' && input.LA(1)<='\t')||(input.LA(1)>='\u000B' && input.LA(1)<='\uFFFF') ) {
                    	        input.consume();
                    	    state.failed=false;
                    	    }
                    	    else {
                    	        if (state.backtracking>0) {state.failed=true; return ;}
                    	        MismatchedSetException mse = new MismatchedSetException(null,input);
                    	        recover(mse);
                    	        throw mse;}


                    	    }
                    	    break;

                    	default :
                    	    break loop57;
                        }
                    } while (true);

                    // PythonLexer.g:373:55: ( '\\n' )+
                    int cnt58=0;
                    loop58:
                    do {
                        int alt58=2;
                        int LA58_0 = input.LA(1);

                        if ( (LA58_0=='\n') ) {
                            alt58=1;
                        }


                        switch (alt58) {
                    	case 1 :
                    	    // PythonLexer.g:0:0: '\\n'
                    	    {
                    	    match('\n'); if (state.failed) return ;

                    	    }
                    	    break;

                    	default :
                    	    if ( cnt58 >= 1 ) break loop58;
                    	    if (state.backtracking>0) {state.failed=true; return ;}
                                EarlyExitException eee =
                                    new EarlyExitException(58, input);
                                throw eee;
                        }
                        cnt58++;
                    } while (true);


                    }
                    break;
                case 2 :
                    // PythonLexer.g:374:7: '#' (~ '\\n' )*
                    {
                    match('#'); if (state.failed) return ;
                    // PythonLexer.g:374:11: (~ '\\n' )*
                    loop59:
                    do {
                        int alt59=2;
                        int LA59_0 = input.LA(1);

                        if ( ((LA59_0>='\u0000' && LA59_0<='\t')||(LA59_0>='\u000B' && LA59_0<='\uFFFF')) ) {
                            alt59=1;
                        }


                        switch (alt59) {
                    	case 1 :
                    	    // PythonLexer.g:374:13: ~ '\\n'
                    	    {
                    	    if ( (input.LA(1)>='\u0000' && input.LA(1)<='\t')||(input.LA(1)>='\u000B' && input.LA(1)<='\uFFFF') ) {
                    	        input.consume();
                    	    state.failed=false;
                    	    }
                    	    else {
                    	        if (state.backtracking>0) {state.failed=true; return ;}
                    	        MismatchedSetException mse = new MismatchedSetException(null,input);
                    	        recover(mse);
                    	        throw mse;}


                    	    }
                    	    break;

                    	default :
                    	    break loop59;
                        }
                    } while (true);


                    }
                    break;

            }
            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "COMMENT"

    // $ANTLR start "DEDENT"
    public final void mDEDENT() throws RecognitionException {
        try {
            int _type = DEDENT;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:381:7: ({...}? => ( '\\n' ) )
            // PythonLexer.g:381:9: {...}? => ( '\\n' )
            {
            if ( !((0==1)) ) {
                if (state.backtracking>0) {state.failed=true; return ;}
                throw new FailedPredicateException(input, "DEDENT", "0==1");
            }
            // PythonLexer.g:381:19: ( '\\n' )
            // PythonLexer.g:381:20: '\\n'
            {
            match('\n'); if (state.failed) return ;

            }


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "DEDENT"

    // $ANTLR start "INDENT"
    public final void mINDENT() throws RecognitionException {
        try {
            int _type = INDENT;
            int _channel = DEFAULT_TOKEN_CHANNEL;
            // PythonLexer.g:382:7: ({...}? => ( '\\n' ) )
            // PythonLexer.g:382:9: {...}? => ( '\\n' )
            {
            if ( !((0==1)) ) {
                if (state.backtracking>0) {state.failed=true; return ;}
                throw new FailedPredicateException(input, "INDENT", "0==1");
            }
            // PythonLexer.g:382:19: ( '\\n' )
            // PythonLexer.g:382:20: '\\n'
            {
            match('\n'); if (state.failed) return ;

            }


            }

            state.type = _type;
            state.channel = _channel;
        }
        finally {
        }
    }
    // $ANTLR end "INDENT"

    public void mTokens() throws RecognitionException {
        // PythonLexer.g:1:39: ( STRINGLITERAL | BYTESLITERAL | FALSE | NONE | TRUE | AND | AS | ASSERT | FOR | BREAK | CLASS | CONTINUE | DEF | DEL | ELIF | ELSE | EXCEPT | FINALLY | FROM | GLOBAL | IF | IMPORT | IN | IS | LAMBDA | NONLOCAL | NOT | OR | PASS | RAISE | RETURN | TRY | WHILE | WITH | YIELD | INTEGER | FLOATNUMBER | IMAGNUMBER | IDENTIFIER | PLUS | MINUS | STAR | DOUBLESTAR | SLASH | DOUBLESLASH | PERCENT | LEFTSHIFT | RIGHTSHIFT | AMPERSAND | VBAR | CIRCUMFLEX | TILDE | LESS | GREATER | LESSEQUAL | GREATEREQUAL | EQUAL | NOTEQUAL | LPAREN | RPAREN | LBRACK | RBRACK | LCURLY | RCURLY | COMMA | COLON | DOT | SEMI | AT | ASSIGN | PLUSEQUAL | MINUSEQUAL | STAREQUAL | SLASHEQUAL | DOUBLESLASHEQUAL | PERCENTEQUAL | AMPERSANDEQUAL | VBAREQUAL | CIRCUMFLEXEQUAL | LEFTSHIFTEQUAL | RIGHTSHIFTEQUAL | DOUBLESTAREQUAL | CONTINUED_LINE | NEWLINE | WS | LEADING_WS | COMMENT | DEDENT | INDENT )
        int alt61=89;
        alt61 = dfa61.predict(input);
        switch (alt61) {
            case 1 :
                // PythonLexer.g:1:41: STRINGLITERAL
                {
                mSTRINGLITERAL(); if (state.failed) return ;

                }
                break;
            case 2 :
                // PythonLexer.g:1:55: BYTESLITERAL
                {
                mBYTESLITERAL(); if (state.failed) return ;

                }
                break;
            case 3 :
                // PythonLexer.g:1:68: FALSE
                {
                mFALSE(); if (state.failed) return ;

                }
                break;
            case 4 :
                // PythonLexer.g:1:74: NONE
                {
                mNONE(); if (state.failed) return ;

                }
                break;
            case 5 :
                // PythonLexer.g:1:79: TRUE
                {
                mTRUE(); if (state.failed) return ;

                }
                break;
            case 6 :
                // PythonLexer.g:1:84: AND
                {
                mAND(); if (state.failed) return ;

                }
                break;
            case 7 :
                // PythonLexer.g:1:88: AS
                {
                mAS(); if (state.failed) return ;

                }
                break;
            case 8 :
                // PythonLexer.g:1:91: ASSERT
                {
                mASSERT(); if (state.failed) return ;

                }
                break;
            case 9 :
                // PythonLexer.g:1:98: FOR
                {
                mFOR(); if (state.failed) return ;

                }
                break;
            case 10 :
                // PythonLexer.g:1:102: BREAK
                {
                mBREAK(); if (state.failed) return ;

                }
                break;
            case 11 :
                // PythonLexer.g:1:108: CLASS
                {
                mCLASS(); if (state.failed) return ;

                }
                break;
            case 12 :
                // PythonLexer.g:1:114: CONTINUE
                {
                mCONTINUE(); if (state.failed) return ;

                }
                break;
            case 13 :
                // PythonLexer.g:1:123: DEF
                {
                mDEF(); if (state.failed) return ;

                }
                break;
            case 14 :
                // PythonLexer.g:1:127: DEL
                {
                mDEL(); if (state.failed) return ;

                }
                break;
            case 15 :
                // PythonLexer.g:1:131: ELIF
                {
                mELIF(); if (state.failed) return ;

                }
                break;
            case 16 :
                // PythonLexer.g:1:136: ELSE
                {
                mELSE(); if (state.failed) return ;

                }
                break;
            case 17 :
                // PythonLexer.g:1:141: EXCEPT
                {
                mEXCEPT(); if (state.failed) return ;

                }
                break;
            case 18 :
                // PythonLexer.g:1:148: FINALLY
                {
                mFINALLY(); if (state.failed) return ;

                }
                break;
            case 19 :
                // PythonLexer.g:1:156: FROM
                {
                mFROM(); if (state.failed) return ;

                }
                break;
            case 20 :
                // PythonLexer.g:1:161: GLOBAL
                {
                mGLOBAL(); if (state.failed) return ;

                }
                break;
            case 21 :
                // PythonLexer.g:1:168: IF
                {
                mIF(); if (state.failed) return ;

                }
                break;
            case 22 :
                // PythonLexer.g:1:171: IMPORT
                {
                mIMPORT(); if (state.failed) return ;

                }
                break;
            case 23 :
                // PythonLexer.g:1:178: IN
                {
                mIN(); if (state.failed) return ;

                }
                break;
            case 24 :
                // PythonLexer.g:1:181: IS
                {
                mIS(); if (state.failed) return ;

                }
                break;
            case 25 :
                // PythonLexer.g:1:184: LAMBDA
                {
                mLAMBDA(); if (state.failed) return ;

                }
                break;
            case 26 :
                // PythonLexer.g:1:191: NONLOCAL
                {
                mNONLOCAL(); if (state.failed) return ;

                }
                break;
            case 27 :
                // PythonLexer.g:1:200: NOT
                {
                mNOT(); if (state.failed) return ;

                }
                break;
            case 28 :
                // PythonLexer.g:1:204: OR
                {
                mOR(); if (state.failed) return ;

                }
                break;
            case 29 :
                // PythonLexer.g:1:207: PASS
                {
                mPASS(); if (state.failed) return ;

                }
                break;
            case 30 :
                // PythonLexer.g:1:212: RAISE
                {
                mRAISE(); if (state.failed) return ;

                }
                break;
            case 31 :
                // PythonLexer.g:1:218: RETURN
                {
                mRETURN(); if (state.failed) return ;

                }
                break;
            case 32 :
                // PythonLexer.g:1:225: TRY
                {
                mTRY(); if (state.failed) return ;

                }
                break;
            case 33 :
                // PythonLexer.g:1:229: WHILE
                {
                mWHILE(); if (state.failed) return ;

                }
                break;
            case 34 :
                // PythonLexer.g:1:235: WITH
                {
                mWITH(); if (state.failed) return ;

                }
                break;
            case 35 :
                // PythonLexer.g:1:240: YIELD
                {
                mYIELD(); if (state.failed) return ;

                }
                break;
            case 36 :
                // PythonLexer.g:1:246: INTEGER
                {
                mINTEGER(); if (state.failed) return ;

                }
                break;
            case 37 :
                // PythonLexer.g:1:254: FLOATNUMBER
                {
                mFLOATNUMBER(); if (state.failed) return ;

                }
                break;
            case 38 :
                // PythonLexer.g:1:266: IMAGNUMBER
                {
                mIMAGNUMBER(); if (state.failed) return ;

                }
                break;
            case 39 :
                // PythonLexer.g:1:277: IDENTIFIER
                {
                mIDENTIFIER(); if (state.failed) return ;

                }
                break;
            case 40 :
                // PythonLexer.g:1:288: PLUS
                {
                mPLUS(); if (state.failed) return ;

                }
                break;
            case 41 :
                // PythonLexer.g:1:293: MINUS
                {
                mMINUS(); if (state.failed) return ;

                }
                break;
            case 42 :
                // PythonLexer.g:1:299: STAR
                {
                mSTAR(); if (state.failed) return ;

                }
                break;
            case 43 :
                // PythonLexer.g:1:304: DOUBLESTAR
                {
                mDOUBLESTAR(); if (state.failed) return ;

                }
                break;
            case 44 :
                // PythonLexer.g:1:315: SLASH
                {
                mSLASH(); if (state.failed) return ;

                }
                break;
            case 45 :
                // PythonLexer.g:1:321: DOUBLESLASH
                {
                mDOUBLESLASH(); if (state.failed) return ;

                }
                break;
            case 46 :
                // PythonLexer.g:1:333: PERCENT
                {
                mPERCENT(); if (state.failed) return ;

                }
                break;
            case 47 :
                // PythonLexer.g:1:341: LEFTSHIFT
                {
                mLEFTSHIFT(); if (state.failed) return ;

                }
                break;
            case 48 :
                // PythonLexer.g:1:351: RIGHTSHIFT
                {
                mRIGHTSHIFT(); if (state.failed) return ;

                }
                break;
            case 49 :
                // PythonLexer.g:1:362: AMPERSAND
                {
                mAMPERSAND(); if (state.failed) return ;

                }
                break;
            case 50 :
                // PythonLexer.g:1:372: VBAR
                {
                mVBAR(); if (state.failed) return ;

                }
                break;
            case 51 :
                // PythonLexer.g:1:377: CIRCUMFLEX
                {
                mCIRCUMFLEX(); if (state.failed) return ;

                }
                break;
            case 52 :
                // PythonLexer.g:1:388: TILDE
                {
                mTILDE(); if (state.failed) return ;

                }
                break;
            case 53 :
                // PythonLexer.g:1:394: LESS
                {
                mLESS(); if (state.failed) return ;

                }
                break;
            case 54 :
                // PythonLexer.g:1:399: GREATER
                {
                mGREATER(); if (state.failed) return ;

                }
                break;
            case 55 :
                // PythonLexer.g:1:407: LESSEQUAL
                {
                mLESSEQUAL(); if (state.failed) return ;

                }
                break;
            case 56 :
                // PythonLexer.g:1:417: GREATEREQUAL
                {
                mGREATEREQUAL(); if (state.failed) return ;

                }
                break;
            case 57 :
                // PythonLexer.g:1:430: EQUAL
                {
                mEQUAL(); if (state.failed) return ;

                }
                break;
            case 58 :
                // PythonLexer.g:1:436: NOTEQUAL
                {
                mNOTEQUAL(); if (state.failed) return ;

                }
                break;
            case 59 :
                // PythonLexer.g:1:445: LPAREN
                {
                mLPAREN(); if (state.failed) return ;

                }
                break;
            case 60 :
                // PythonLexer.g:1:452: RPAREN
                {
                mRPAREN(); if (state.failed) return ;

                }
                break;
            case 61 :
                // PythonLexer.g:1:459: LBRACK
                {
                mLBRACK(); if (state.failed) return ;

                }
                break;
            case 62 :
                // PythonLexer.g:1:466: RBRACK
                {
                mRBRACK(); if (state.failed) return ;

                }
                break;
            case 63 :
                // PythonLexer.g:1:473: LCURLY
                {
                mLCURLY(); if (state.failed) return ;

                }
                break;
            case 64 :
                // PythonLexer.g:1:480: RCURLY
                {
                mRCURLY(); if (state.failed) return ;

                }
                break;
            case 65 :
                // PythonLexer.g:1:487: COMMA
                {
                mCOMMA(); if (state.failed) return ;

                }
                break;
            case 66 :
                // PythonLexer.g:1:493: COLON
                {
                mCOLON(); if (state.failed) return ;

                }
                break;
            case 67 :
                // PythonLexer.g:1:499: DOT
                {
                mDOT(); if (state.failed) return ;

                }
                break;
            case 68 :
                // PythonLexer.g:1:503: SEMI
                {
                mSEMI(); if (state.failed) return ;

                }
                break;
            case 69 :
                // PythonLexer.g:1:508: AT
                {
                mAT(); if (state.failed) return ;

                }
                break;
            case 70 :
                // PythonLexer.g:1:511: ASSIGN
                {
                mASSIGN(); if (state.failed) return ;

                }
                break;
            case 71 :
                // PythonLexer.g:1:518: PLUSEQUAL
                {
                mPLUSEQUAL(); if (state.failed) return ;

                }
                break;
            case 72 :
                // PythonLexer.g:1:528: MINUSEQUAL
                {
                mMINUSEQUAL(); if (state.failed) return ;

                }
                break;
            case 73 :
                // PythonLexer.g:1:539: STAREQUAL
                {
                mSTAREQUAL(); if (state.failed) return ;

                }
                break;
            case 74 :
                // PythonLexer.g:1:549: SLASHEQUAL
                {
                mSLASHEQUAL(); if (state.failed) return ;

                }
                break;
            case 75 :
                // PythonLexer.g:1:560: DOUBLESLASHEQUAL
                {
                mDOUBLESLASHEQUAL(); if (state.failed) return ;

                }
                break;
            case 76 :
                // PythonLexer.g:1:577: PERCENTEQUAL
                {
                mPERCENTEQUAL(); if (state.failed) return ;

                }
                break;
            case 77 :
                // PythonLexer.g:1:590: AMPERSANDEQUAL
                {
                mAMPERSANDEQUAL(); if (state.failed) return ;

                }
                break;
            case 78 :
                // PythonLexer.g:1:605: VBAREQUAL
                {
                mVBAREQUAL(); if (state.failed) return ;

                }
                break;
            case 79 :
                // PythonLexer.g:1:615: CIRCUMFLEXEQUAL
                {
                mCIRCUMFLEXEQUAL(); if (state.failed) return ;

                }
                break;
            case 80 :
                // PythonLexer.g:1:631: LEFTSHIFTEQUAL
                {
                mLEFTSHIFTEQUAL(); if (state.failed) return ;

                }
                break;
            case 81 :
                // PythonLexer.g:1:646: RIGHTSHIFTEQUAL
                {
                mRIGHTSHIFTEQUAL(); if (state.failed) return ;

                }
                break;
            case 82 :
                // PythonLexer.g:1:662: DOUBLESTAREQUAL
                {
                mDOUBLESTAREQUAL(); if (state.failed) return ;

                }
                break;
            case 83 :
                // PythonLexer.g:1:678: CONTINUED_LINE
                {
                mCONTINUED_LINE(); if (state.failed) return ;

                }
                break;
            case 84 :
                // PythonLexer.g:1:693: NEWLINE
                {
                mNEWLINE(); if (state.failed) return ;

                }
                break;
            case 85 :
                // PythonLexer.g:1:701: WS
                {
                mWS(); if (state.failed) return ;

                }
                break;
            case 86 :
                // PythonLexer.g:1:704: LEADING_WS
                {
                mLEADING_WS(); if (state.failed) return ;

                }
                break;
            case 87 :
                // PythonLexer.g:1:715: COMMENT
                {
                mCOMMENT(); if (state.failed) return ;

                }
                break;
            case 88 :
                // PythonLexer.g:1:723: DEDENT
                {
                mDEDENT(); if (state.failed) return ;

                }
                break;
            case 89 :
                // PythonLexer.g:1:730: INDENT
                {
                mINDENT(); if (state.failed) return ;

                }
                break;

        }

    }

    // $ANTLR start synpred97_PythonLexer
    public final void synpred97_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:336:11: ({...}? ( ' ' | '\\t' )+ )
        // PythonLexer.g:336:11: {...}? ( ' ' | '\\t' )+
        {
        if ( !((implicitLineJoiningLevel>0)) ) {
            if (state.backtracking>0) {state.failed=true; return ;}
            throw new FailedPredicateException(input, "synpred97_PythonLexer", "implicitLineJoiningLevel>0");
        }
        // PythonLexer.g:336:41: ( ' ' | '\\t' )+
        int cnt71=0;
        loop71:
        do {
            int alt71=2;
            int LA71_0 = input.LA(1);

            if ( (LA71_0=='\t'||LA71_0==' ') ) {
                alt71=1;
            }


            switch (alt71) {
        	case 1 :
        	    // PythonLexer.g:
        	    {
        	    if ( input.LA(1)=='\t'||input.LA(1)==' ' ) {
        	        input.consume();
        	    state.failed=false;
        	    }
        	    else {
        	        if (state.backtracking>0) {state.failed=true; return ;}
        	        MismatchedSetException mse = new MismatchedSetException(null,input);
        	        recover(mse);
        	        throw mse;}


        	    }
        	    break;

        	default :
        	    if ( cnt71 >= 1 ) break loop71;
        	    if (state.backtracking>0) {state.failed=true; return ;}
                    EarlyExitException eee =
                        new EarlyExitException(71, input);
                    throw eee;
            }
            cnt71++;
        } while (true);


        }
    }
    // $ANTLR end synpred97_PythonLexer

    // $ANTLR start synpred106_PythonLexer
    public final void synpred106_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:373:7: ({...}? => ( ' ' | '\\t' )* '#' (~ '\\n' )* ( '\\n' )+ )
        // PythonLexer.g:373:7: {...}? => ( ' ' | '\\t' )* '#' (~ '\\n' )* ( '\\n' )+
        {
        if ( !((startPos==0)) ) {
            if (state.backtracking>0) {state.failed=true; return ;}
            throw new FailedPredicateException(input, "synpred106_PythonLexer", "startPos==0");
        }
        // PythonLexer.g:373:24: ( ' ' | '\\t' )*
        loop73:
        do {
            int alt73=2;
            int LA73_0 = input.LA(1);

            if ( (LA73_0=='\t'||LA73_0==' ') ) {
                alt73=1;
            }


            switch (alt73) {
        	case 1 :
        	    // PythonLexer.g:
        	    {
        	    if ( input.LA(1)=='\t'||input.LA(1)==' ' ) {
        	        input.consume();
        	    state.failed=false;
        	    }
        	    else {
        	        if (state.backtracking>0) {state.failed=true; return ;}
        	        MismatchedSetException mse = new MismatchedSetException(null,input);
        	        recover(mse);
        	        throw mse;}


        	    }
        	    break;

        	default :
        	    break loop73;
            }
        } while (true);

        match('#'); if (state.failed) return ;
        // PythonLexer.g:373:44: (~ '\\n' )*
        loop74:
        do {
            int alt74=2;
            int LA74_0 = input.LA(1);

            if ( ((LA74_0>='\u0000' && LA74_0<='\t')||(LA74_0>='\u000B' && LA74_0<='\uFFFF')) ) {
                alt74=1;
            }


            switch (alt74) {
        	case 1 :
        	    // PythonLexer.g:373:46: ~ '\\n'
        	    {
        	    if ( (input.LA(1)>='\u0000' && input.LA(1)<='\t')||(input.LA(1)>='\u000B' && input.LA(1)<='\uFFFF') ) {
        	        input.consume();
        	    state.failed=false;
        	    }
        	    else {
        	        if (state.backtracking>0) {state.failed=true; return ;}
        	        MismatchedSetException mse = new MismatchedSetException(null,input);
        	        recover(mse);
        	        throw mse;}


        	    }
        	    break;

        	default :
        	    break loop74;
            }
        } while (true);

        // PythonLexer.g:373:55: ( '\\n' )+
        int cnt75=0;
        loop75:
        do {
            int alt75=2;
            int LA75_0 = input.LA(1);

            if ( (LA75_0=='\n') ) {
                alt75=1;
            }


            switch (alt75) {
        	case 1 :
        	    // PythonLexer.g:0:0: '\\n'
        	    {
        	    match('\n'); if (state.failed) return ;

        	    }
        	    break;

        	default :
        	    if ( cnt75 >= 1 ) break loop75;
        	    if (state.backtracking>0) {state.failed=true; return ;}
                    EarlyExitException eee =
                        new EarlyExitException(75, input);
                    throw eee;
            }
            cnt75++;
        } while (true);


        }
    }
    // $ANTLR end synpred106_PythonLexer

    // $ANTLR start synpred108_PythonLexer
    public final void synpred108_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:41: ( STRINGLITERAL )
        // PythonLexer.g:1:41: STRINGLITERAL
        {
        mSTRINGLITERAL(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred108_PythonLexer

    // $ANTLR start synpred109_PythonLexer
    public final void synpred109_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:55: ( BYTESLITERAL )
        // PythonLexer.g:1:55: BYTESLITERAL
        {
        mBYTESLITERAL(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred109_PythonLexer

    // $ANTLR start synpred110_PythonLexer
    public final void synpred110_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:68: ( FALSE )
        // PythonLexer.g:1:68: FALSE
        {
        mFALSE(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred110_PythonLexer

    // $ANTLR start synpred111_PythonLexer
    public final void synpred111_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:74: ( NONE )
        // PythonLexer.g:1:74: NONE
        {
        mNONE(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred111_PythonLexer

    // $ANTLR start synpred112_PythonLexer
    public final void synpred112_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:79: ( TRUE )
        // PythonLexer.g:1:79: TRUE
        {
        mTRUE(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred112_PythonLexer

    // $ANTLR start synpred113_PythonLexer
    public final void synpred113_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:84: ( AND )
        // PythonLexer.g:1:84: AND
        {
        mAND(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred113_PythonLexer

    // $ANTLR start synpred114_PythonLexer
    public final void synpred114_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:88: ( AS )
        // PythonLexer.g:1:88: AS
        {
        mAS(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred114_PythonLexer

    // $ANTLR start synpred115_PythonLexer
    public final void synpred115_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:91: ( ASSERT )
        // PythonLexer.g:1:91: ASSERT
        {
        mASSERT(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred115_PythonLexer

    // $ANTLR start synpred116_PythonLexer
    public final void synpred116_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:98: ( FOR )
        // PythonLexer.g:1:98: FOR
        {
        mFOR(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred116_PythonLexer

    // $ANTLR start synpred117_PythonLexer
    public final void synpred117_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:102: ( BREAK )
        // PythonLexer.g:1:102: BREAK
        {
        mBREAK(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred117_PythonLexer

    // $ANTLR start synpred118_PythonLexer
    public final void synpred118_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:108: ( CLASS )
        // PythonLexer.g:1:108: CLASS
        {
        mCLASS(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred118_PythonLexer

    // $ANTLR start synpred119_PythonLexer
    public final void synpred119_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:114: ( CONTINUE )
        // PythonLexer.g:1:114: CONTINUE
        {
        mCONTINUE(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred119_PythonLexer

    // $ANTLR start synpred120_PythonLexer
    public final void synpred120_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:123: ( DEF )
        // PythonLexer.g:1:123: DEF
        {
        mDEF(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred120_PythonLexer

    // $ANTLR start synpred121_PythonLexer
    public final void synpred121_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:127: ( DEL )
        // PythonLexer.g:1:127: DEL
        {
        mDEL(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred121_PythonLexer

    // $ANTLR start synpred122_PythonLexer
    public final void synpred122_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:131: ( ELIF )
        // PythonLexer.g:1:131: ELIF
        {
        mELIF(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred122_PythonLexer

    // $ANTLR start synpred123_PythonLexer
    public final void synpred123_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:136: ( ELSE )
        // PythonLexer.g:1:136: ELSE
        {
        mELSE(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred123_PythonLexer

    // $ANTLR start synpred124_PythonLexer
    public final void synpred124_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:141: ( EXCEPT )
        // PythonLexer.g:1:141: EXCEPT
        {
        mEXCEPT(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred124_PythonLexer

    // $ANTLR start synpred125_PythonLexer
    public final void synpred125_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:148: ( FINALLY )
        // PythonLexer.g:1:148: FINALLY
        {
        mFINALLY(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred125_PythonLexer

    // $ANTLR start synpred126_PythonLexer
    public final void synpred126_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:156: ( FROM )
        // PythonLexer.g:1:156: FROM
        {
        mFROM(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred126_PythonLexer

    // $ANTLR start synpred127_PythonLexer
    public final void synpred127_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:161: ( GLOBAL )
        // PythonLexer.g:1:161: GLOBAL
        {
        mGLOBAL(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred127_PythonLexer

    // $ANTLR start synpred128_PythonLexer
    public final void synpred128_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:168: ( IF )
        // PythonLexer.g:1:168: IF
        {
        mIF(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred128_PythonLexer

    // $ANTLR start synpred129_PythonLexer
    public final void synpred129_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:171: ( IMPORT )
        // PythonLexer.g:1:171: IMPORT
        {
        mIMPORT(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred129_PythonLexer

    // $ANTLR start synpred130_PythonLexer
    public final void synpred130_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:178: ( IN )
        // PythonLexer.g:1:178: IN
        {
        mIN(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred130_PythonLexer

    // $ANTLR start synpred131_PythonLexer
    public final void synpred131_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:181: ( IS )
        // PythonLexer.g:1:181: IS
        {
        mIS(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred131_PythonLexer

    // $ANTLR start synpred132_PythonLexer
    public final void synpred132_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:184: ( LAMBDA )
        // PythonLexer.g:1:184: LAMBDA
        {
        mLAMBDA(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred132_PythonLexer

    // $ANTLR start synpred133_PythonLexer
    public final void synpred133_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:191: ( NONLOCAL )
        // PythonLexer.g:1:191: NONLOCAL
        {
        mNONLOCAL(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred133_PythonLexer

    // $ANTLR start synpred134_PythonLexer
    public final void synpred134_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:200: ( NOT )
        // PythonLexer.g:1:200: NOT
        {
        mNOT(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred134_PythonLexer

    // $ANTLR start synpred135_PythonLexer
    public final void synpred135_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:204: ( OR )
        // PythonLexer.g:1:204: OR
        {
        mOR(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred135_PythonLexer

    // $ANTLR start synpred136_PythonLexer
    public final void synpred136_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:207: ( PASS )
        // PythonLexer.g:1:207: PASS
        {
        mPASS(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred136_PythonLexer

    // $ANTLR start synpred137_PythonLexer
    public final void synpred137_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:212: ( RAISE )
        // PythonLexer.g:1:212: RAISE
        {
        mRAISE(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred137_PythonLexer

    // $ANTLR start synpred138_PythonLexer
    public final void synpred138_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:218: ( RETURN )
        // PythonLexer.g:1:218: RETURN
        {
        mRETURN(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred138_PythonLexer

    // $ANTLR start synpred139_PythonLexer
    public final void synpred139_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:225: ( TRY )
        // PythonLexer.g:1:225: TRY
        {
        mTRY(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred139_PythonLexer

    // $ANTLR start synpred140_PythonLexer
    public final void synpred140_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:229: ( WHILE )
        // PythonLexer.g:1:229: WHILE
        {
        mWHILE(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred140_PythonLexer

    // $ANTLR start synpred141_PythonLexer
    public final void synpred141_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:235: ( WITH )
        // PythonLexer.g:1:235: WITH
        {
        mWITH(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred141_PythonLexer

    // $ANTLR start synpred142_PythonLexer
    public final void synpred142_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:240: ( YIELD )
        // PythonLexer.g:1:240: YIELD
        {
        mYIELD(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred142_PythonLexer

    // $ANTLR start synpred143_PythonLexer
    public final void synpred143_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:246: ( INTEGER )
        // PythonLexer.g:1:246: INTEGER
        {
        mINTEGER(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred143_PythonLexer

    // $ANTLR start synpred144_PythonLexer
    public final void synpred144_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:254: ( FLOATNUMBER )
        // PythonLexer.g:1:254: FLOATNUMBER
        {
        mFLOATNUMBER(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred144_PythonLexer

    // $ANTLR start synpred145_PythonLexer
    public final void synpred145_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:266: ( IMAGNUMBER )
        // PythonLexer.g:1:266: IMAGNUMBER
        {
        mIMAGNUMBER(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred145_PythonLexer

    // $ANTLR start synpred146_PythonLexer
    public final void synpred146_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:277: ( IDENTIFIER )
        // PythonLexer.g:1:277: IDENTIFIER
        {
        mIDENTIFIER(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred146_PythonLexer

    // $ANTLR start synpred147_PythonLexer
    public final void synpred147_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:288: ( PLUS )
        // PythonLexer.g:1:288: PLUS
        {
        mPLUS(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred147_PythonLexer

    // $ANTLR start synpred148_PythonLexer
    public final void synpred148_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:293: ( MINUS )
        // PythonLexer.g:1:293: MINUS
        {
        mMINUS(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred148_PythonLexer

    // $ANTLR start synpred149_PythonLexer
    public final void synpred149_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:299: ( STAR )
        // PythonLexer.g:1:299: STAR
        {
        mSTAR(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred149_PythonLexer

    // $ANTLR start synpred150_PythonLexer
    public final void synpred150_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:304: ( DOUBLESTAR )
        // PythonLexer.g:1:304: DOUBLESTAR
        {
        mDOUBLESTAR(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred150_PythonLexer

    // $ANTLR start synpred151_PythonLexer
    public final void synpred151_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:315: ( SLASH )
        // PythonLexer.g:1:315: SLASH
        {
        mSLASH(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred151_PythonLexer

    // $ANTLR start synpred152_PythonLexer
    public final void synpred152_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:321: ( DOUBLESLASH )
        // PythonLexer.g:1:321: DOUBLESLASH
        {
        mDOUBLESLASH(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred152_PythonLexer

    // $ANTLR start synpred153_PythonLexer
    public final void synpred153_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:333: ( PERCENT )
        // PythonLexer.g:1:333: PERCENT
        {
        mPERCENT(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred153_PythonLexer

    // $ANTLR start synpred154_PythonLexer
    public final void synpred154_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:341: ( LEFTSHIFT )
        // PythonLexer.g:1:341: LEFTSHIFT
        {
        mLEFTSHIFT(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred154_PythonLexer

    // $ANTLR start synpred155_PythonLexer
    public final void synpred155_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:351: ( RIGHTSHIFT )
        // PythonLexer.g:1:351: RIGHTSHIFT
        {
        mRIGHTSHIFT(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred155_PythonLexer

    // $ANTLR start synpred156_PythonLexer
    public final void synpred156_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:362: ( AMPERSAND )
        // PythonLexer.g:1:362: AMPERSAND
        {
        mAMPERSAND(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred156_PythonLexer

    // $ANTLR start synpred157_PythonLexer
    public final void synpred157_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:372: ( VBAR )
        // PythonLexer.g:1:372: VBAR
        {
        mVBAR(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred157_PythonLexer

    // $ANTLR start synpred158_PythonLexer
    public final void synpred158_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:377: ( CIRCUMFLEX )
        // PythonLexer.g:1:377: CIRCUMFLEX
        {
        mCIRCUMFLEX(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred158_PythonLexer

    // $ANTLR start synpred160_PythonLexer
    public final void synpred160_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:394: ( LESS )
        // PythonLexer.g:1:394: LESS
        {
        mLESS(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred160_PythonLexer

    // $ANTLR start synpred161_PythonLexer
    public final void synpred161_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:399: ( GREATER )
        // PythonLexer.g:1:399: GREATER
        {
        mGREATER(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred161_PythonLexer

    // $ANTLR start synpred162_PythonLexer
    public final void synpred162_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:407: ( LESSEQUAL )
        // PythonLexer.g:1:407: LESSEQUAL
        {
        mLESSEQUAL(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred162_PythonLexer

    // $ANTLR start synpred163_PythonLexer
    public final void synpred163_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:417: ( GREATEREQUAL )
        // PythonLexer.g:1:417: GREATEREQUAL
        {
        mGREATEREQUAL(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred163_PythonLexer

    // $ANTLR start synpred164_PythonLexer
    public final void synpred164_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:430: ( EQUAL )
        // PythonLexer.g:1:430: EQUAL
        {
        mEQUAL(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred164_PythonLexer

    // $ANTLR start synpred174_PythonLexer
    public final void synpred174_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:499: ( DOT )
        // PythonLexer.g:1:499: DOT
        {
        mDOT(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred174_PythonLexer

    // $ANTLR start synpred177_PythonLexer
    public final void synpred177_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:511: ( ASSIGN )
        // PythonLexer.g:1:511: ASSIGN
        {
        mASSIGN(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred177_PythonLexer

    // $ANTLR start synpred178_PythonLexer
    public final void synpred178_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:518: ( PLUSEQUAL )
        // PythonLexer.g:1:518: PLUSEQUAL
        {
        mPLUSEQUAL(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred178_PythonLexer

    // $ANTLR start synpred179_PythonLexer
    public final void synpred179_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:528: ( MINUSEQUAL )
        // PythonLexer.g:1:528: MINUSEQUAL
        {
        mMINUSEQUAL(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred179_PythonLexer

    // $ANTLR start synpred180_PythonLexer
    public final void synpred180_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:539: ( STAREQUAL )
        // PythonLexer.g:1:539: STAREQUAL
        {
        mSTAREQUAL(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred180_PythonLexer

    // $ANTLR start synpred181_PythonLexer
    public final void synpred181_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:549: ( SLASHEQUAL )
        // PythonLexer.g:1:549: SLASHEQUAL
        {
        mSLASHEQUAL(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred181_PythonLexer

    // $ANTLR start synpred182_PythonLexer
    public final void synpred182_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:560: ( DOUBLESLASHEQUAL )
        // PythonLexer.g:1:560: DOUBLESLASHEQUAL
        {
        mDOUBLESLASHEQUAL(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred182_PythonLexer

    // $ANTLR start synpred183_PythonLexer
    public final void synpred183_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:577: ( PERCENTEQUAL )
        // PythonLexer.g:1:577: PERCENTEQUAL
        {
        mPERCENTEQUAL(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred183_PythonLexer

    // $ANTLR start synpred184_PythonLexer
    public final void synpred184_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:590: ( AMPERSANDEQUAL )
        // PythonLexer.g:1:590: AMPERSANDEQUAL
        {
        mAMPERSANDEQUAL(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred184_PythonLexer

    // $ANTLR start synpred185_PythonLexer
    public final void synpred185_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:605: ( VBAREQUAL )
        // PythonLexer.g:1:605: VBAREQUAL
        {
        mVBAREQUAL(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred185_PythonLexer

    // $ANTLR start synpred186_PythonLexer
    public final void synpred186_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:615: ( CIRCUMFLEXEQUAL )
        // PythonLexer.g:1:615: CIRCUMFLEXEQUAL
        {
        mCIRCUMFLEXEQUAL(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred186_PythonLexer

    // $ANTLR start synpred187_PythonLexer
    public final void synpred187_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:631: ( LEFTSHIFTEQUAL )
        // PythonLexer.g:1:631: LEFTSHIFTEQUAL
        {
        mLEFTSHIFTEQUAL(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred187_PythonLexer

    // $ANTLR start synpred188_PythonLexer
    public final void synpred188_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:646: ( RIGHTSHIFTEQUAL )
        // PythonLexer.g:1:646: RIGHTSHIFTEQUAL
        {
        mRIGHTSHIFTEQUAL(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred188_PythonLexer

    // $ANTLR start synpred189_PythonLexer
    public final void synpred189_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:662: ( DOUBLESTAREQUAL )
        // PythonLexer.g:1:662: DOUBLESTAREQUAL
        {
        mDOUBLESTAREQUAL(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred189_PythonLexer

    // $ANTLR start synpred191_PythonLexer
    public final void synpred191_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:693: ( NEWLINE )
        // PythonLexer.g:1:693: NEWLINE
        {
        mNEWLINE(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred191_PythonLexer

    // $ANTLR start synpred192_PythonLexer
    public final void synpred192_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:701: ( WS )
        // PythonLexer.g:1:701: WS
        {
        mWS(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred192_PythonLexer

    // $ANTLR start synpred193_PythonLexer
    public final void synpred193_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:704: ( LEADING_WS )
        // PythonLexer.g:1:704: LEADING_WS
        {
        mLEADING_WS(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred193_PythonLexer

    // $ANTLR start synpred194_PythonLexer
    public final void synpred194_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:715: ( COMMENT )
        // PythonLexer.g:1:715: COMMENT
        {
        mCOMMENT(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred194_PythonLexer

    // $ANTLR start synpred195_PythonLexer
    public final void synpred195_PythonLexer_fragment() throws RecognitionException {   
        // PythonLexer.g:1:723: ( DEDENT )
        // PythonLexer.g:1:723: DEDENT
        {
        mDEDENT(); if (state.failed) return ;

        }
    }
    // $ANTLR end synpred195_PythonLexer

    public final boolean synpred184_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred184_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred127_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred127_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred123_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred123_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred108_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred108_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred139_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred139_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred142_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred142_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred156_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred156_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred138_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred138_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred146_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred146_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred143_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred143_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred180_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred180_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred124_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred124_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred121_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred121_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred179_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred179_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred177_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred177_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred186_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred186_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred126_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred126_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred136_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred136_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred195_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred195_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred148_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred148_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred147_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred147_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred161_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred161_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred106_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred106_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred157_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred157_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred149_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred149_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred128_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred128_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred174_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred174_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred125_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred125_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred160_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred160_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred131_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred131_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred181_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred181_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred118_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred118_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred135_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred135_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred120_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred120_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred163_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred163_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred113_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred113_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred97_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred97_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred187_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred187_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred117_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred117_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred152_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred152_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred182_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred182_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred188_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred188_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred112_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred112_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred137_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred137_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred110_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred110_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred132_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred132_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred114_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred114_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred116_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred116_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred129_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred129_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred109_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred109_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred111_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred111_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred145_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred145_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred150_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred150_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred151_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred151_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred185_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred185_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred144_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred144_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred193_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred193_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred194_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred194_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred192_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred192_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred178_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred178_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred183_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred183_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred191_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred191_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred119_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred119_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred155_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred155_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred189_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred189_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred158_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred158_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred164_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred164_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred140_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred140_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred154_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred154_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred133_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred133_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred153_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred153_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred134_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred134_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred122_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred122_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred115_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred115_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred130_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred130_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred162_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred162_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }
    public final boolean synpred141_PythonLexer() {
        state.backtracking++;
        int start = input.mark();
        try {
            synpred141_PythonLexer_fragment(); // can never throw exception
        } catch (RecognitionException re) {
            System.err.println("impossible: "+re);
        }
        boolean success = !state.failed;
        input.rewind(start);
        state.backtracking--;
        state.failed=false;
        return success;
    }


    protected DFA35 dfa35 = new DFA35(this);
    protected DFA37 dfa37 = new DFA37(this);
    protected DFA38 dfa38 = new DFA38(this);
    protected DFA43 dfa43 = new DFA43(this);
    protected DFA60 dfa60 = new DFA60(this);
    protected DFA61 dfa61 = new DFA61(this);
    static final String DFA35_eotS =
        "\4\uffff\2\6\1\uffff";
    static final String DFA35_eofS =
        "\7\uffff";
    static final String DFA35_minS =
        "\2\56\1\60\1\uffff\2\60\1\uffff";
    static final String DFA35_maxS =
        "\1\71\1\145\1\71\1\uffff\2\145\1\uffff";
    static final String DFA35_acceptS =
        "\3\uffff\1\2\2\uffff\1\1";
    static final String DFA35_specialS =
        "\7\uffff}>";
    static final String[] DFA35_transitionS = {
            "\1\2\1\uffff\12\1",
            "\1\4\1\uffff\12\1\13\uffff\1\3\37\uffff\1\3",
            "\12\5",
            "",
            "\12\5\13\uffff\1\3\37\uffff\1\3",
            "\12\5\13\uffff\1\3\37\uffff\1\3",
            ""
    };

    static final short[] DFA35_eot = DFA.unpackEncodedString(DFA35_eotS);
    static final short[] DFA35_eof = DFA.unpackEncodedString(DFA35_eofS);
    static final char[] DFA35_min = DFA.unpackEncodedStringToUnsignedChars(DFA35_minS);
    static final char[] DFA35_max = DFA.unpackEncodedStringToUnsignedChars(DFA35_maxS);
    static final short[] DFA35_accept = DFA.unpackEncodedString(DFA35_acceptS);
    static final short[] DFA35_special = DFA.unpackEncodedString(DFA35_specialS);
    static final short[][] DFA35_transition;

    static {
        int numStates = DFA35_transitionS.length;
        DFA35_transition = new short[numStates][];
        for (int i=0; i<numStates; i++) {
            DFA35_transition[i] = DFA.unpackEncodedString(DFA35_transitionS[i]);
        }
    }

    class DFA35 extends DFA {

        public DFA35(BaseRecognizer recognizer) {
            this.recognizer = recognizer;
            this.decisionNumber = 35;
            this.eot = DFA35_eot;
            this.eof = DFA35_eof;
            this.min = DFA35_min;
            this.max = DFA35_max;
            this.accept = DFA35_accept;
            this.special = DFA35_special;
            this.transition = DFA35_transition;
        }
        public String getDescription() {
            return "186:1: FLOATNUMBER : ( POINTFLOAT | EXPONENTFLOAT );";
        }
    }
    static final String DFA37_eotS =
        "\3\uffff\1\4\1\uffff";
    static final String DFA37_eofS =
        "\5\uffff";
    static final String DFA37_minS =
        "\2\56\1\uffff\1\60\1\uffff";
    static final String DFA37_maxS =
        "\2\71\1\uffff\1\71\1\uffff";
    static final String DFA37_acceptS =
        "\2\uffff\1\1\1\uffff\1\2";
    static final String DFA37_specialS =
        "\5\uffff}>";
    static final String[] DFA37_transitionS = {
            "\1\2\1\uffff\12\1",
            "\1\3\1\uffff\12\1",
            "",
            "\12\2",
            ""
    };

    static final short[] DFA37_eot = DFA.unpackEncodedString(DFA37_eotS);
    static final short[] DFA37_eof = DFA.unpackEncodedString(DFA37_eofS);
    static final char[] DFA37_min = DFA.unpackEncodedStringToUnsignedChars(DFA37_minS);
    static final char[] DFA37_max = DFA.unpackEncodedStringToUnsignedChars(DFA37_maxS);
    static final short[] DFA37_accept = DFA.unpackEncodedString(DFA37_acceptS);
    static final short[] DFA37_special = DFA.unpackEncodedString(DFA37_specialS);
    static final short[][] DFA37_transition;

    static {
        int numStates = DFA37_transitionS.length;
        DFA37_transition = new short[numStates][];
        for (int i=0; i<numStates; i++) {
            DFA37_transition[i] = DFA.unpackEncodedString(DFA37_transitionS[i]);
        }
    }

    class DFA37 extends DFA {

        public DFA37(BaseRecognizer recognizer) {
            this.recognizer = recognizer;
            this.decisionNumber = 37;
            this.eot = DFA37_eot;
            this.eof = DFA37_eof;
            this.min = DFA37_min;
            this.max = DFA37_max;
            this.accept = DFA37_accept;
            this.special = DFA37_special;
            this.transition = DFA37_transition;
        }
        public String getDescription() {
            return "187:10: fragment POINTFLOAT : ( ( ( INTPART )? FRACTION ) | ( INTPART '.' ) );";
        }
    }
    static final String DFA38_eotS =
        "\4\uffff";
    static final String DFA38_eofS =
        "\4\uffff";
    static final String DFA38_minS =
        "\2\56\2\uffff";
    static final String DFA38_maxS =
        "\1\71\1\145\2\uffff";
    static final String DFA38_acceptS =
        "\2\uffff\1\2\1\1";
    static final String DFA38_specialS =
        "\4\uffff}>";
    static final String[] DFA38_transitionS = {
            "\1\2\1\uffff\12\1",
            "\1\2\1\uffff\12\1\13\uffff\1\3\37\uffff\1\3",
            "",
            ""
    };

    static final short[] DFA38_eot = DFA.unpackEncodedString(DFA38_eotS);
    static final short[] DFA38_eof = DFA.unpackEncodedString(DFA38_eofS);
    static final char[] DFA38_min = DFA.unpackEncodedStringToUnsignedChars(DFA38_minS);
    static final char[] DFA38_max = DFA.unpackEncodedStringToUnsignedChars(DFA38_maxS);
    static final short[] DFA38_accept = DFA.unpackEncodedString(DFA38_acceptS);
    static final short[] DFA38_special = DFA.unpackEncodedString(DFA38_specialS);
    static final short[][] DFA38_transition;

    static {
        int numStates = DFA38_transitionS.length;
        DFA38_transition = new short[numStates][];
        for (int i=0; i<numStates; i++) {
            DFA38_transition[i] = DFA.unpackEncodedString(DFA38_transitionS[i]);
        }
    }

    class DFA38 extends DFA {

        public DFA38(BaseRecognizer recognizer) {
            this.recognizer = recognizer;
            this.decisionNumber = 38;
            this.eot = DFA38_eot;
            this.eof = DFA38_eof;
            this.min = DFA38_min;
            this.max = DFA38_max;
            this.accept = DFA38_accept;
            this.special = DFA38_special;
            this.transition = DFA38_transition;
        }
        public String getDescription() {
            return "192:11: ( INTPART | POINTFLOAT )";
        }
    }
    static final String DFA43_eotS =
        "\4\uffff";
    static final String DFA43_eofS =
        "\4\uffff";
    static final String DFA43_minS =
        "\2\56\2\uffff";
    static final String DFA43_maxS =
        "\1\71\1\152\2\uffff";
    static final String DFA43_acceptS =
        "\2\uffff\1\1\1\2";
    static final String DFA43_specialS =
        "\4\uffff}>";
    static final String[] DFA43_transitionS = {
            "\1\2\1\uffff\12\1",
            "\1\2\1\uffff\12\1\13\uffff\1\2\4\uffff\1\3\32\uffff\1\2\4\uffff"+
            "\1\3",
            "",
            ""
    };

    static final short[] DFA43_eot = DFA.unpackEncodedString(DFA43_eotS);
    static final short[] DFA43_eof = DFA.unpackEncodedString(DFA43_eofS);
    static final char[] DFA43_min = DFA.unpackEncodedStringToUnsignedChars(DFA43_minS);
    static final char[] DFA43_max = DFA.unpackEncodedStringToUnsignedChars(DFA43_maxS);
    static final short[] DFA43_accept = DFA.unpackEncodedString(DFA43_acceptS);
    static final short[] DFA43_special = DFA.unpackEncodedString(DFA43_specialS);
    static final short[][] DFA43_transition;

    static {
        int numStates = DFA43_transitionS.length;
        DFA43_transition = new short[numStates][];
        for (int i=0; i<numStates; i++) {
            DFA43_transition[i] = DFA.unpackEncodedString(DFA43_transitionS[i]);
        }
    }

    class DFA43 extends DFA {

        public DFA43(BaseRecognizer recognizer) {
            this.recognizer = recognizer;
            this.decisionNumber = 43;
            this.eot = DFA43_eot;
            this.eof = DFA43_eof;
            this.min = DFA43_min;
            this.max = DFA43_max;
            this.accept = DFA43_accept;
            this.special = DFA43_special;
            this.transition = DFA43_transition;
        }
        public String getDescription() {
            return "205:15: ( FLOATNUMBER | INTPART )";
        }
    }
    static final String DFA60_eotS =
        "\2\uffff\2\5\2\uffff";
    static final String DFA60_eofS =
        "\6\uffff";
    static final String DFA60_minS =
        "\1\11\1\uffff\2\0\2\uffff";
    static final String DFA60_maxS =
        "\1\43\1\uffff\2\uffff\2\uffff";
    static final String DFA60_acceptS =
        "\1\uffff\1\1\2\uffff\1\1\1\2";
    static final String DFA60_specialS =
        "\1\1\1\uffff\1\2\1\0\2\uffff}>";
    static final String[] DFA60_transitionS = {
            "\1\1\26\uffff\1\1\2\uffff\1\2",
            "",
            "\12\3\1\4\ufff5\3",
            "\12\3\1\4\ufff5\3",
            "",
            ""
    };

    static final short[] DFA60_eot = DFA.unpackEncodedString(DFA60_eotS);
    static final short[] DFA60_eof = DFA.unpackEncodedString(DFA60_eofS);
    static final char[] DFA60_min = DFA.unpackEncodedStringToUnsignedChars(DFA60_minS);
    static final char[] DFA60_max = DFA.unpackEncodedStringToUnsignedChars(DFA60_maxS);
    static final short[] DFA60_accept = DFA.unpackEncodedString(DFA60_acceptS);
    static final short[] DFA60_special = DFA.unpackEncodedString(DFA60_specialS);
    static final short[][] DFA60_transition;

    static {
        int numStates = DFA60_transitionS.length;
        DFA60_transition = new short[numStates][];
        for (int i=0; i<numStates; i++) {
            DFA60_transition[i] = DFA.unpackEncodedString(DFA60_transitionS[i]);
        }
    }

    class DFA60 extends DFA {

        public DFA60(BaseRecognizer recognizer) {
            this.recognizer = recognizer;
            this.decisionNumber = 60;
            this.eot = DFA60_eot;
            this.eof = DFA60_eof;
            this.min = DFA60_min;
            this.max = DFA60_max;
            this.accept = DFA60_accept;
            this.special = DFA60_special;
            this.transition = DFA60_transition;
        }
        public String getDescription() {
            return "368:1: COMMENT : ({...}? => ( ' ' | '\\t' )* '#' (~ '\\n' )* ( '\\n' )+ | '#' (~ '\\n' )* );";
        }
        public int specialStateTransition(int s, IntStream _input) throws NoViableAltException {
            IntStream input = _input;
        	int _s = s;
            switch ( s ) {
                    case 0 : 
                        int LA60_3 = input.LA(1);

                         
                        int index60_3 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (LA60_3=='\n') && ((startPos==0))) {s = 4;}

                        else if ( ((LA60_3>='\u0000' && LA60_3<='\t')||(LA60_3>='\u000B' && LA60_3<='\uFFFF')) ) {s = 3;}

                        else s = 5;

                         
                        input.seek(index60_3);
                        if ( s>=0 ) return s;
                        break;
                    case 1 : 
                        int LA60_0 = input.LA(1);

                         
                        int index60_0 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (LA60_0=='\t'||LA60_0==' ') && ((startPos==0))) {s = 1;}

                        else if ( (LA60_0=='#') ) {s = 2;}

                         
                        input.seek(index60_0);
                        if ( s>=0 ) return s;
                        break;
                    case 2 : 
                        int LA60_2 = input.LA(1);

                         
                        int index60_2 = input.index();
                        input.rewind();
                        s = -1;
                        if ( ((LA60_2>='\u0000' && LA60_2<='\t')||(LA60_2>='\u000B' && LA60_2<='\uFFFF')) ) {s = 3;}

                        else if ( (LA60_2=='\n') && ((startPos==0))) {s = 4;}

                        else s = 5;

                         
                        input.seek(index60_2);
                        if ( s>=0 ) return s;
                        break;
            }
            if (state.backtracking>0) {state.failed=true; return -1;}
            NoViableAltException nvae =
                new NoViableAltException(getDescription(), 60, _s, input);
            error(nvae);
            throw nvae;
        }
    }
    static final String DFA61_eotS =
        "\u0082\uffff";
    static final String DFA61_eofS =
        "\u0082\uffff";
    static final String DFA61_minS =
        "\1\11\1\0\2\uffff\27\0\1\uffff\12\0\1\uffff\1\0\16\uffff\3\0\111"+
        "\uffff";
    static final String DFA61_maxS =
        "\1\176\1\0\2\uffff\27\0\1\uffff\12\0\1\uffff\1\0\16\uffff\3\0\111"+
        "\uffff";
    static final String DFA61_acceptS =
        "\2\uffff\1\1\30\uffff\1\47\12\uffff\1\64\1\uffff\1\72\1\73\1\74"+
        "\1\75\1\76\1\77\1\100\1\101\1\102\1\104\1\105\1\123\1\124\4\uffff"+
        "\1\127\1\36\1\37\1\2\1\12\1\3\1\4\1\5\1\6\1\7\1\10\1\11\1\22\1\23"+
        "\1\13\1\14\1\15\1\16\1\17\1\20\1\21\1\24\1\25\1\26\1\27\1\30\1\31"+
        "\1\32\1\33\1\34\1\35\1\40\1\41\1\42\1\43\1\44\1\45\1\46\1\103\1"+
        "\50\1\107\1\51\1\110\1\52\1\53\1\111\1\122\1\54\1\55\1\112\1\113"+
        "\1\56\1\114\1\57\1\65\1\67\1\120\1\60\1\66\1\70\1\121\1\61\1\115"+
        "\1\62\1\116\1\63\1\117\1\71\1\106\1\130\1\131\1\125\1\126";
    static final String DFA61_specialS =
        "\1\uffff\1\0\2\uffff\1\1\1\2\1\3\1\4\1\5\1\6\1\7\1\10\1\11\1\12"+
        "\1\13\1\14\1\15\1\16\1\17\1\20\1\21\1\22\1\23\1\24\1\25\1\26\1\27"+
        "\1\uffff\1\30\1\31\1\32\1\33\1\34\1\35\1\36\1\37\1\40\1\41\1\uffff"+
        "\1\42\16\uffff\1\43\1\44\1\45\111\uffff}>";
    static final String[] DFA61_transitionS = {
            "\1\70\1\66\1\uffff\2\64\22\uffff\1\67\1\50\1\2\1\71\1\uffff"+
            "\1\40\1\43\1\2\1\51\1\52\1\36\1\34\1\57\1\35\1\32\1\37\1\31"+
            "\11\30\1\60\1\61\1\41\1\47\1\42\1\uffff\1\62\1\33\1\12\3\33"+
            "\1\5\7\33\1\6\3\33\1\24\1\33\1\7\1\24\5\33\1\53\1\63\1\54\1"+
            "\45\1\33\1\uffff\1\10\1\4\1\13\1\14\1\15\1\11\1\16\1\33\1\17"+
            "\2\33\1\20\1\33\1\21\1\22\1\23\1\33\1\1\1\33\1\25\1\24\1\33"+
            "\1\26\1\33\1\27\1\33\1\55\1\44\1\56\1\46",
            "\1\uffff",
            "",
            "",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "",
            "\1\uffff",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "\1\uffff",
            "\1\uffff",
            "\1\uffff",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            ""
    };

    static final short[] DFA61_eot = DFA.unpackEncodedString(DFA61_eotS);
    static final short[] DFA61_eof = DFA.unpackEncodedString(DFA61_eofS);
    static final char[] DFA61_min = DFA.unpackEncodedStringToUnsignedChars(DFA61_minS);
    static final char[] DFA61_max = DFA.unpackEncodedStringToUnsignedChars(DFA61_maxS);
    static final short[] DFA61_accept = DFA.unpackEncodedString(DFA61_acceptS);
    static final short[] DFA61_special = DFA.unpackEncodedString(DFA61_specialS);
    static final short[][] DFA61_transition;

    static {
        int numStates = DFA61_transitionS.length;
        DFA61_transition = new short[numStates][];
        for (int i=0; i<numStates; i++) {
            DFA61_transition[i] = DFA.unpackEncodedString(DFA61_transitionS[i]);
        }
    }

    class DFA61 extends DFA {

        public DFA61(BaseRecognizer recognizer) {
            this.recognizer = recognizer;
            this.decisionNumber = 61;
            this.eot = DFA61_eot;
            this.eof = DFA61_eof;
            this.min = DFA61_min;
            this.max = DFA61_max;
            this.accept = DFA61_accept;
            this.special = DFA61_special;
            this.transition = DFA61_transition;
        }
        public String getDescription() {
            return "1:1: Tokens options {k=1; backtrack=true; } : ( STRINGLITERAL | BYTESLITERAL | FALSE | NONE | TRUE | AND | AS | ASSERT | FOR | BREAK | CLASS | CONTINUE | DEF | DEL | ELIF | ELSE | EXCEPT | FINALLY | FROM | GLOBAL | IF | IMPORT | IN | IS | LAMBDA | NONLOCAL | NOT | OR | PASS | RAISE | RETURN | TRY | WHILE | WITH | YIELD | INTEGER | FLOATNUMBER | IMAGNUMBER | IDENTIFIER | PLUS | MINUS | STAR | DOUBLESTAR | SLASH | DOUBLESLASH | PERCENT | LEFTSHIFT | RIGHTSHIFT | AMPERSAND | VBAR | CIRCUMFLEX | TILDE | LESS | GREATER | LESSEQUAL | GREATEREQUAL | EQUAL | NOTEQUAL | LPAREN | RPAREN | LBRACK | RBRACK | LCURLY | RCURLY | COMMA | COLON | DOT | SEMI | AT | ASSIGN | PLUSEQUAL | MINUSEQUAL | STAREQUAL | SLASHEQUAL | DOUBLESLASHEQUAL | PERCENTEQUAL | AMPERSANDEQUAL | VBAREQUAL | CIRCUMFLEXEQUAL | LEFTSHIFTEQUAL | RIGHTSHIFTEQUAL | DOUBLESTAREQUAL | CONTINUED_LINE | NEWLINE | WS | LEADING_WS | COMMENT | DEDENT | INDENT );";
        }
        public int specialStateTransition(int s, IntStream _input) throws NoViableAltException {
            IntStream input = _input;
        	int _s = s;
            switch ( s ) {
                    case 0 : 
                        int LA61_1 = input.LA(1);

                         
                        int index61_1 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred108_PythonLexer()) ) {s = 2;}

                        else if ( (synpred137_PythonLexer()) ) {s = 58;}

                        else if ( (synpred138_PythonLexer()) ) {s = 59;}

                        else if ( (synpred146_PythonLexer()) ) {s = 27;}

                         
                        input.seek(index61_1);
                        if ( s>=0 ) return s;
                        break;
                    case 1 : 
                        int LA61_4 = input.LA(1);

                         
                        int index61_4 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred109_PythonLexer()) ) {s = 60;}

                        else if ( (synpred117_PythonLexer()) ) {s = 61;}

                        else if ( (synpred146_PythonLexer()) ) {s = 27;}

                         
                        input.seek(index61_4);
                        if ( s>=0 ) return s;
                        break;
                    case 2 : 
                        int LA61_5 = input.LA(1);

                         
                        int index61_5 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred110_PythonLexer()) ) {s = 62;}

                        else if ( (synpred146_PythonLexer()) ) {s = 27;}

                         
                        input.seek(index61_5);
                        if ( s>=0 ) return s;
                        break;
                    case 3 : 
                        int LA61_6 = input.LA(1);

                         
                        int index61_6 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred111_PythonLexer()) ) {s = 63;}

                        else if ( (synpred146_PythonLexer()) ) {s = 27;}

                         
                        input.seek(index61_6);
                        if ( s>=0 ) return s;
                        break;
                    case 4 : 
                        int LA61_7 = input.LA(1);

                         
                        int index61_7 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred112_PythonLexer()) ) {s = 64;}

                        else if ( (synpred146_PythonLexer()) ) {s = 27;}

                         
                        input.seek(index61_7);
                        if ( s>=0 ) return s;
                        break;
                    case 5 : 
                        int LA61_8 = input.LA(1);

                         
                        int index61_8 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred113_PythonLexer()) ) {s = 65;}

                        else if ( (synpred114_PythonLexer()) ) {s = 66;}

                        else if ( (synpred115_PythonLexer()) ) {s = 67;}

                        else if ( (synpred146_PythonLexer()) ) {s = 27;}

                         
                        input.seek(index61_8);
                        if ( s>=0 ) return s;
                        break;
                    case 6 : 
                        int LA61_9 = input.LA(1);

                         
                        int index61_9 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred116_PythonLexer()) ) {s = 68;}

                        else if ( (synpred125_PythonLexer()) ) {s = 69;}

                        else if ( (synpred126_PythonLexer()) ) {s = 70;}

                        else if ( (synpred146_PythonLexer()) ) {s = 27;}

                         
                        input.seek(index61_9);
                        if ( s>=0 ) return s;
                        break;
                    case 7 : 
                        int LA61_10 = input.LA(1);

                         
                        int index61_10 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred109_PythonLexer()) ) {s = 60;}

                        else if ( (synpred146_PythonLexer()) ) {s = 27;}

                         
                        input.seek(index61_10);
                        if ( s>=0 ) return s;
                        break;
                    case 8 : 
                        int LA61_11 = input.LA(1);

                         
                        int index61_11 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred118_PythonLexer()) ) {s = 71;}

                        else if ( (synpred119_PythonLexer()) ) {s = 72;}

                        else if ( (synpred146_PythonLexer()) ) {s = 27;}

                         
                        input.seek(index61_11);
                        if ( s>=0 ) return s;
                        break;
                    case 9 : 
                        int LA61_12 = input.LA(1);

                         
                        int index61_12 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred120_PythonLexer()) ) {s = 73;}

                        else if ( (synpred121_PythonLexer()) ) {s = 74;}

                        else if ( (synpred146_PythonLexer()) ) {s = 27;}

                         
                        input.seek(index61_12);
                        if ( s>=0 ) return s;
                        break;
                    case 10 : 
                        int LA61_13 = input.LA(1);

                         
                        int index61_13 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred122_PythonLexer()) ) {s = 75;}

                        else if ( (synpred123_PythonLexer()) ) {s = 76;}

                        else if ( (synpred124_PythonLexer()) ) {s = 77;}

                        else if ( (synpred146_PythonLexer()) ) {s = 27;}

                         
                        input.seek(index61_13);
                        if ( s>=0 ) return s;
                        break;
                    case 11 : 
                        int LA61_14 = input.LA(1);

                         
                        int index61_14 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred127_PythonLexer()) ) {s = 78;}

                        else if ( (synpred146_PythonLexer()) ) {s = 27;}

                         
                        input.seek(index61_14);
                        if ( s>=0 ) return s;
                        break;
                    case 12 : 
                        int LA61_15 = input.LA(1);

                         
                        int index61_15 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred128_PythonLexer()) ) {s = 79;}

                        else if ( (synpred129_PythonLexer()) ) {s = 80;}

                        else if ( (synpred130_PythonLexer()) ) {s = 81;}

                        else if ( (synpred131_PythonLexer()) ) {s = 82;}

                        else if ( (synpred146_PythonLexer()) ) {s = 27;}

                         
                        input.seek(index61_15);
                        if ( s>=0 ) return s;
                        break;
                    case 13 : 
                        int LA61_16 = input.LA(1);

                         
                        int index61_16 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred132_PythonLexer()) ) {s = 83;}

                        else if ( (synpred146_PythonLexer()) ) {s = 27;}

                         
                        input.seek(index61_16);
                        if ( s>=0 ) return s;
                        break;
                    case 14 : 
                        int LA61_17 = input.LA(1);

                         
                        int index61_17 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred133_PythonLexer()) ) {s = 84;}

                        else if ( (synpred134_PythonLexer()) ) {s = 85;}

                        else if ( (synpred146_PythonLexer()) ) {s = 27;}

                         
                        input.seek(index61_17);
                        if ( s>=0 ) return s;
                        break;
                    case 15 : 
                        int LA61_18 = input.LA(1);

                         
                        int index61_18 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred135_PythonLexer()) ) {s = 86;}

                        else if ( (synpred146_PythonLexer()) ) {s = 27;}

                         
                        input.seek(index61_18);
                        if ( s>=0 ) return s;
                        break;
                    case 16 : 
                        int LA61_19 = input.LA(1);

                         
                        int index61_19 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred136_PythonLexer()) ) {s = 87;}

                        else if ( (synpred146_PythonLexer()) ) {s = 27;}

                         
                        input.seek(index61_19);
                        if ( s>=0 ) return s;
                        break;
                    case 17 : 
                        int LA61_20 = input.LA(1);

                         
                        int index61_20 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred108_PythonLexer()) ) {s = 2;}

                        else if ( (synpred146_PythonLexer()) ) {s = 27;}

                         
                        input.seek(index61_20);
                        if ( s>=0 ) return s;
                        break;
                    case 18 : 
                        int LA61_21 = input.LA(1);

                         
                        int index61_21 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred139_PythonLexer()) ) {s = 88;}

                        else if ( (synpred146_PythonLexer()) ) {s = 27;}

                         
                        input.seek(index61_21);
                        if ( s>=0 ) return s;
                        break;
                    case 19 : 
                        int LA61_22 = input.LA(1);

                         
                        int index61_22 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred140_PythonLexer()) ) {s = 89;}

                        else if ( (synpred141_PythonLexer()) ) {s = 90;}

                        else if ( (synpred146_PythonLexer()) ) {s = 27;}

                         
                        input.seek(index61_22);
                        if ( s>=0 ) return s;
                        break;
                    case 20 : 
                        int LA61_23 = input.LA(1);

                         
                        int index61_23 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred142_PythonLexer()) ) {s = 91;}

                        else if ( (synpred146_PythonLexer()) ) {s = 27;}

                         
                        input.seek(index61_23);
                        if ( s>=0 ) return s;
                        break;
                    case 21 : 
                        int LA61_24 = input.LA(1);

                         
                        int index61_24 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred143_PythonLexer()) ) {s = 92;}

                        else if ( (synpred144_PythonLexer()) ) {s = 93;}

                        else if ( (synpred145_PythonLexer()) ) {s = 94;}

                         
                        input.seek(index61_24);
                        if ( s>=0 ) return s;
                        break;
                    case 22 : 
                        int LA61_25 = input.LA(1);

                         
                        int index61_25 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred143_PythonLexer()) ) {s = 92;}

                        else if ( (synpred144_PythonLexer()) ) {s = 93;}

                        else if ( (synpred145_PythonLexer()) ) {s = 94;}

                         
                        input.seek(index61_25);
                        if ( s>=0 ) return s;
                        break;
                    case 23 : 
                        int LA61_26 = input.LA(1);

                         
                        int index61_26 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred144_PythonLexer()) ) {s = 93;}

                        else if ( (synpred145_PythonLexer()) ) {s = 94;}

                        else if ( (synpred174_PythonLexer()) ) {s = 95;}

                         
                        input.seek(index61_26);
                        if ( s>=0 ) return s;
                        break;
                    case 24 : 
                        int LA61_28 = input.LA(1);

                         
                        int index61_28 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred147_PythonLexer()) ) {s = 96;}

                        else if ( (synpred178_PythonLexer()) ) {s = 97;}

                         
                        input.seek(index61_28);
                        if ( s>=0 ) return s;
                        break;
                    case 25 : 
                        int LA61_29 = input.LA(1);

                         
                        int index61_29 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred148_PythonLexer()) ) {s = 98;}

                        else if ( (synpred179_PythonLexer()) ) {s = 99;}

                         
                        input.seek(index61_29);
                        if ( s>=0 ) return s;
                        break;
                    case 26 : 
                        int LA61_30 = input.LA(1);

                         
                        int index61_30 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred149_PythonLexer()) ) {s = 100;}

                        else if ( (synpred150_PythonLexer()) ) {s = 101;}

                        else if ( (synpred180_PythonLexer()) ) {s = 102;}

                        else if ( (synpred189_PythonLexer()) ) {s = 103;}

                         
                        input.seek(index61_30);
                        if ( s>=0 ) return s;
                        break;
                    case 27 : 
                        int LA61_31 = input.LA(1);

                         
                        int index61_31 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred151_PythonLexer()) ) {s = 104;}

                        else if ( (synpred152_PythonLexer()) ) {s = 105;}

                        else if ( (synpred181_PythonLexer()) ) {s = 106;}

                        else if ( (synpred182_PythonLexer()) ) {s = 107;}

                         
                        input.seek(index61_31);
                        if ( s>=0 ) return s;
                        break;
                    case 28 : 
                        int LA61_32 = input.LA(1);

                         
                        int index61_32 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred153_PythonLexer()) ) {s = 108;}

                        else if ( (synpred183_PythonLexer()) ) {s = 109;}

                         
                        input.seek(index61_32);
                        if ( s>=0 ) return s;
                        break;
                    case 29 : 
                        int LA61_33 = input.LA(1);

                         
                        int index61_33 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred154_PythonLexer()) ) {s = 110;}

                        else if ( (synpred160_PythonLexer()) ) {s = 111;}

                        else if ( (synpred162_PythonLexer()) ) {s = 112;}

                        else if ( (synpred187_PythonLexer()) ) {s = 113;}

                         
                        input.seek(index61_33);
                        if ( s>=0 ) return s;
                        break;
                    case 30 : 
                        int LA61_34 = input.LA(1);

                         
                        int index61_34 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred155_PythonLexer()) ) {s = 114;}

                        else if ( (synpred161_PythonLexer()) ) {s = 115;}

                        else if ( (synpred163_PythonLexer()) ) {s = 116;}

                        else if ( (synpred188_PythonLexer()) ) {s = 117;}

                         
                        input.seek(index61_34);
                        if ( s>=0 ) return s;
                        break;
                    case 31 : 
                        int LA61_35 = input.LA(1);

                         
                        int index61_35 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred156_PythonLexer()) ) {s = 118;}

                        else if ( (synpred184_PythonLexer()) ) {s = 119;}

                         
                        input.seek(index61_35);
                        if ( s>=0 ) return s;
                        break;
                    case 32 : 
                        int LA61_36 = input.LA(1);

                         
                        int index61_36 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred157_PythonLexer()) ) {s = 120;}

                        else if ( (synpred185_PythonLexer()) ) {s = 121;}

                         
                        input.seek(index61_36);
                        if ( s>=0 ) return s;
                        break;
                    case 33 : 
                        int LA61_37 = input.LA(1);

                         
                        int index61_37 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred158_PythonLexer()) ) {s = 122;}

                        else if ( (synpred186_PythonLexer()) ) {s = 123;}

                         
                        input.seek(index61_37);
                        if ( s>=0 ) return s;
                        break;
                    case 34 : 
                        int LA61_39 = input.LA(1);

                         
                        int index61_39 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred164_PythonLexer()) ) {s = 124;}

                        else if ( (synpred177_PythonLexer()) ) {s = 125;}

                         
                        input.seek(index61_39);
                        if ( s>=0 ) return s;
                        break;
                    case 35 : 
                        int LA61_54 = input.LA(1);

                         
                        int index61_54 = input.index();
                        input.rewind();
                        s = -1;
                        if ( (synpred191_PythonLexer()) ) {s = 52;}

                        else if ( ((synpred195_PythonLexer()&&(0==1))) ) {s = 126;}

                        else if ( ((0==1)) ) {s = 127;}

                         
                        input.seek(index61_54);
                        if ( s>=0 ) return s;
                        break;
                    case 36 : 
                        int LA61_55 = input.LA(1);

                         
                        int index61_55 = input.index();
                        input.rewind();
                        s = -1;
                        if ( ((synpred192_PythonLexer()&&(startPos>0))) ) {s = 128;}

                        else if ( (((synpred193_PythonLexer()&&(startPos==0))||((synpred193_PythonLexer()&&(startPos==0))&&(implicitLineJoiningLevel>0)))) ) {s = 129;}

                        else if ( ((synpred194_PythonLexer()&&(startPos==0))) ) {s = 57;}

                         
                        input.seek(index61_55);
                        if ( s>=0 ) return s;
                        break;
                    case 37 : 
                        int LA61_56 = input.LA(1);

                         
                        int index61_56 = input.index();
                        input.rewind();
                        s = -1;
                        if ( ((synpred192_PythonLexer()&&(startPos>0))) ) {s = 128;}

                        else if ( (((synpred193_PythonLexer()&&(startPos==0))||((synpred193_PythonLexer()&&(startPos==0))&&(implicitLineJoiningLevel>0)))) ) {s = 129;}

                        else if ( ((synpred194_PythonLexer()&&(startPos==0))) ) {s = 57;}

                         
                        input.seek(index61_56);
                        if ( s>=0 ) return s;
                        break;
            }
            if (state.backtracking>0) {state.failed=true; return -1;}
            NoViableAltException nvae =
                new NoViableAltException(getDescription(), 61, _s, input);
            error(nvae);
            throw nvae;
        }
    }
 

}
