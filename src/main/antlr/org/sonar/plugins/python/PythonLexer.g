/*
 [The 'BSD licence']
 Copyright (c) 2009 Ales Teska
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions
 are met:
 1. Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
 3. The name of the author may not be used to endorse or promote products
    derived from this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/*
* Python 3.1 Grammar, C language target, ANTLR 3.2
*
* Ales Teska
* October 2009
*
* Requires python3lexerutils.c to be compiled and linked together with ANTLR output of this file.
*
* Tested on whole Python 3 source code set of Python 3.1 (more than 1600 python files, one test cycle took 26 (!) seconds on my laptop).
*
* Parts (mainly lexer and approach to parser) are based on Python 2.5 Grammar created by Terence Parr, Loring Craymer and Aaron Maxwell
* Copyright (c) 2004 Terence Parr and Loring Craymer
*
*/

//TODO: PEP 3131 - Supporting Non-ASCII Identifiers -> this effectivelly means switch to UTF16/USC-2 encoding, lexer part is in python3_pep3131.g (not used anyhow today)
//TODO: Encoding declarations - [[ http://docs.python.org/3.1/reference/lexical_analysis.html#encoding-declarations ]] )

lexer grammar PythonLexer;

options
{
    backtrack=true; // Yeh ... it will be hopefully fixed but for now and for C target is more than enought
    filter=true;
}

tokens
{
    INDENT;
    DEDENT;
}

@lexer::members {
    /** Handles context-sensitive lexing of implicit line joining such as
     *  the case where newline is ignored in cases like this:
     *  a = [3,
     *       4]
     */
    private int implicitLineJoiningLevel = 0;
    public int startPos=-1;
}


//////////////////////////////////// LEXER /////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// $<String and Bytes literals
// [[ http://docs.python.org/3.1/reference/lexical_analysis.html#string-and-bytes-literals ]]

STRINGLITERAL   : STRINGPREFIX? ( SHORTSTRING | LONGSTRING );

fragment STRINGPREFIX
        : ( 'r' | 'R'  | 'u' | 'U');

fragment SHORTSTRING
        : '"' ( ESCAPESEQ | ~( '\\'|'\n'|'"' ) )* '"'
        | '\'' ( ESCAPESEQ | ~( '\\'|'\n'|'\'' ) )* '\''
        ;

fragment LONGSTRING
        : '\'\'\'' ( options {greedy=false;}:TRIAPOS )* '\'\'\''
        | '"""' ( options {greedy=false;}:TRIQUOTE )* '"""'
        ;

BYTESLITERAL    : BYTESPREFIX ( SHORTBYTES | LONGBYTES );

fragment BYTESPREFIX
        : ( 'b' | 'B' ) ( 'r' | 'R' )?;

fragment SHORTBYTES
        : '"' ( ESCAPESEQ | ~( '\\' | '\n' | '"' ) )* '"' 
        | '\'' ( ESCAPESEQ | ~( '\\' | '\n' | '\'' ) )* '\'' 
        ;

fragment LONGBYTES 
        : '\'\'\'' ( options {greedy=false;}:TRIAPOS )* '\'\'\''
        | '"""' ( options {greedy=false;}:TRIQUOTE )* '"""'
        ;

fragment TRIAPOS
        : ( '\'' '\'' | '\''? ) ( ESCAPESEQ | ~( '\\' | '\'' ) )+;

fragment TRIQUOTE
        : ( '"' '"' | '"'? ) ( ESCAPESEQ | ~( '\\' | '"' ) )+;
    
fragment ESCAPESEQ
        : '\\' .;


////////////////////////////////////////////////////////////////////////////////
// $<Keywords
// [[ http://docs.python.org/3.1/reference/lexical_analysis.html#keywords ]]

FALSE       : 'False';
NONE        : 'None';
TRUE        : 'True';
AND         : 'and';
AS          : 'as';
ASSERT      : 'assert';
FOR         : 'for';
BREAK       : 'break';
CLASS       : 'class';
CONTINUE    : 'continue';
DEF         : 'def';
DEL         : 'del';
ELIF        : 'elif';
ELSE        : 'else';
EXCEPT      : 'except';
FINALLY     : 'finally';
FROM        : 'from';
GLOBAL      : 'global';
IF          : 'if';
IMPORT      : 'import';
IN          : 'in';
IS          : 'is';
LAMBDA      : 'lambda';
NONLOCAL    : 'nonlocal';
NOT         : 'not';
OR          : 'or';
PASS        : 'pass';
RAISE       : 'raise';
RETURN      : 'return';
TRY         : 'try';
WHILE       : 'while';
WITH        : 'with';
YIELD       : 'yield';


////////////////////////////////////////////////////////////////////////////////
// $<Integer literals
// [[ http://docs.python.org/3.1/reference/lexical_analysis.html#integer-literals ]]

INTEGER : DECIMALINTEGER | OCTINTEGER | HEXINTEGER | BININTEGER;
fragment DECIMALINTEGER
        : NONZERODIGIT DIGIT* ( 'l' | 'L' )? | '0'+ ( 'l' | 'L' )?;
fragment NONZERODIGIT
        : '1' .. '9';
fragment DIGIT
        : '0' .. '9';
fragment OCTINTEGER
        : '0' ( 'o' | 'O' ) OCTDIGIT+ ( 'l' | 'L' )?;
fragment HEXINTEGER
        : '0' ( 'x' | 'X' ) HEXDIGIT+ ( 'l' | 'L' )?;
fragment BININTEGER
        : '0' ( 'b' | 'B' ) BINDIGIT+ ( 'l' | 'L' )?;
fragment OCTDIGIT
        : '0' .. '7';
fragment HEXDIGIT
        : DIGIT | 'a' .. 'f' | 'A' .. 'F';
fragment BINDIGIT
        : '0' | '1';


////////////////////////////////////////////////////////////////////////////////
// $<Floating point literals
// [[ http://docs.python.org/3.1/reference/lexical_analysis.html#floating-point-literals ]]

FLOATNUMBER : POINTFLOAT | EXPONENTFLOAT;
fragment POINTFLOAT
        : ( INTPART? FRACTION )
        | ( INTPART '.' )
        ;
fragment EXPONENTFLOAT
        : ( INTPART | POINTFLOAT ) EXPONENT;
fragment INTPART
        : DIGIT+;
fragment FRACTION
        : '.' DIGIT+;
fragment EXPONENT
        : ( 'e' | 'E' ) ( '+' | '-' )? DIGIT+;


////////////////////////////////////////////////////////////////////////////////
// $<Imaginary literals
// [[ http://docs.python.org/3.1/reference/lexical_analysis.html#imaginary-literals ]]

IMAGNUMBER  : ( FLOATNUMBER | INTPART ) ( 'j' | 'J' );


////////////////////////////////////////////////////////////////////////////////
// $<Identifiers
// [[ http://docs.python.org/3.1/reference/lexical_analysis.html#identifiers ]]

IDENTIFIER  : ID_START ID_CONTINUE*;

//TODO: <all characters in general categories Lu, Ll, Lt, Lm, Lo, Nl, the underscore, and characters with the Other_ID_Start property> - see python3_pep3131.g
fragment ID_START
        : '_'
        | 'A'.. 'Z'
        | 'a' .. 'z'
        ;

//TODO: <all characters in id_start, plus characters in the categories Mn, Mc, Nd, Pc and others with the Other_ID_Continue property> - see python3_pep3131.g
fragment ID_CONTINUE
        : '_'
        | 'A'.. 'Z'
        | 'a' .. 'z'
        | '0' .. '9'
        ;
 

////////////////////////////////////////////////////////////////////////////////
// $<Operators
// [[ http://docs.python.org/3.1/reference/lexical_analysis.html#operators ]]

PLUS            : '+';
MINUS           : '-';
STAR            : '*';
DOUBLESTAR      : '**';
SLASH           : '/';
DOUBLESLASH     : '//';
PERCENT         : '%';
LEFTSHIFT       : '<<';
RIGHTSHIFT      : '>>';
AMPERSAND       : '&';
VBAR            : '|';
CIRCUMFLEX      : '^';
TILDE           : '~';
LESS            : '<';
GREATER         : '>';
LESSEQUAL       : '<=';
GREATEREQUAL    : '>=';
EQUAL           : '==';
NOTEQUAL        : '!=';


//////////////////////////////////////////////
// $<Delimiters
// [[ http://docs.python.org/3.1/reference/lexical_analysis.html#delimiters ]]

//  Implicit line joining - [[ http://docs.python.org/3.1/reference/lexical_analysis.html#implicit-line-joining ]]
LPAREN      : '(' {implicitLineJoiningLevel += 1;};
RPAREN      : ')' {implicitLineJoiningLevel -= 1;};
LBRACK      : '[' {implicitLineJoiningLevel += 1;};
RBRACK      : ']' {implicitLineJoiningLevel -= 1;};
LCURLY      : '{' {implicitLineJoiningLevel += 1;};
RCURLY      : '}' {implicitLineJoiningLevel -= 1;};

COMMA       : ',';
COLON       : ':';
DOT         : '.';
SEMI        : ';';
AT          : '@';
ASSIGN      : '=';

// Augmented assignment operators
PLUSEQUAL        : '+=';
MINUSEQUAL       : '-=';
STAREQUAL        : '*=';
SLASHEQUAL       : '/=';
DOUBLESLASHEQUAL : '//=';
PERCENTEQUAL     : '%=';
AMPERSANDEQUAL   : '&=';
VBAREQUAL        : '|=';
CIRCUMFLEXEQUAL  : '^=';
LEFTSHIFTEQUAL   : '<<=';
RIGHTSHIFTEQUAL  : '>>=';
DOUBLESTAREQUAL  : '**=';


//////////////////////////////////////////////
// $<Line structure
// [[ http://docs.python.org/3.1/reference/lexical_analysis.html#line-structure ]]

/** Consume a newline and any whitespace at start of next line
 *  unless the next line contains only white space, in that case
 *  emit a newline.
 */
CONTINUED_LINE
        : '\\' ('\r')? '\n' ( ' ' | '\t' )* { $channel=HIDDEN; }
        // Dont need? ( NEWLINE { static char __newlinebuf[] = "\n"; EMITNEW(python3Lexer_createLexerToken(LEXER, TOKTEXT(NEWLINE, __newlinebuf))); } )?
        ;

/** Treat a sequence of blank lines as a single blank line.  If
 *  nested within a (..), {..}, or [..], then ignore newlines.
 *  If the first newline starts in column one, they are to be ignored.
 *
 *  Frank Wierzbicki added: Also ignore FORMFEEDS (\u000C).
 */
NEWLINE : ( '\u000C'? '\r'? '\n' )+
        {
            if (startPos==0 || implicitLineJoiningLevel>0)
            {
                $channel=HIDDEN;
            }
        }
        ;


//////////////////////////////////////////////
// $<Whitespace
// [[ http://docs.python.org/3.1/reference/lexical_analysis.html#whitespace-between-tokens ]]

WS      : {startPos>0}?=> ( ' ' | '\t' )+ {$channel=HIDDEN;};

// [[ http://docs.python.org/3.1/reference/lexical_analysis.html#indentation ]]
/** Grab everything before a real symbol.  Then if newline, kill it
 *  as this is a blank line.  If whitespace followed by comment, kill it
 *  as it's a comment on a line by itself.
 *
 *  Ignore leading whitespace when nested in [..], (..), {..}.
 */
LEADING_WS
        @init {
            int spaces = 0;
        }
        : {startPos==0}?=>
        ( {implicitLineJoiningLevel>0}? ( ' ' | '\t' )+ {$channel=HIDDEN;}
          | (' ' { spaces++; }
              | '\t' { spaces += 8; spaces -= (spaces \% 8); }
            )+
            {
                // make a string of n spaces where n is column number - 1
                char[] indentation = new char[spaces];
                for (int i=0; i<spaces; i++) {
                    indentation[i] = ' ';
                }
                String s = new String(indentation);
                emit(new ClassicToken(LEADING_WS, new String(indentation)));
            }
    
            // kill trailing newline if present and then ignore
            (
                ('\r')? '\n'
                {
                    if (state.token!=null)
                        state.token.setChannel(HIDDEN);
                    else
                        $channel=HIDDEN;
                }
            )*
        )
        ;


//////////////////////////////////////////////
// $<Comments
// [[ http://docs.python.org/3.1/reference/lexical_analysis.html#comments ]]

COMMENT
    @init
    {
        $channel=HIDDEN;
    }
    : {startPos==0}?=> ( ' ' | '\t' )* '#' ( ~'\n' )* '\n'+ 
    | '#' ( ~'\n' )* // let NEWLINE handle \n unless char pos==0 for '#'
    //| '#' ( ~'\n' )* '\n'+
    ;


// Following two lexer rules are imaginary, condition is never meet ...
// they are here just to suppress warnings
DEDENT: {0==1}?=> ('\n');
INDENT: {0==1}?=> ('\n');
