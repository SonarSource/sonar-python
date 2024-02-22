grammar PyTypeTypeGrammar;

@header{
package org.sonar.python.types.pytype_grammar;
}

outer_type : type EOF;

type :  union_type
        | builtin_type
        | class_type
        | anything_type
        | generic_callable_type
        | qualified_type;

union_type : UNION_PREFIX type_list ')';

builtin_type: builtin | BUILTINS_PREFIX '.' builtin;

class_type : CLASS_PREFIX (builtin_type | qualified_type) ')';

generic_callable_type : GENERIC_CALLABLE_PREFIX ',' PARAMETERS type_list ')';

anything_type: ANYTHING_TYPE;

qualified_type: STRING ('.' STRING)*;

type_list : '(' type (',' type)+ ')' | '(' type ',' ')';

builtin: NONE_TYPE
         | BOOL
         | STR
         | INT
         | FLOAT
         | COMPLEX
         | TUPLE
         | LIST
         | SET
         | DICT;

// UNION TYPES
UNION_PREFIX: 'UnionType(type_list=';

// CLASS TYPES
CLASS_PREFIX: 'ClassType(';

// GENERIC CALLABLE TYPES
GENERIC_CALLABLE_PREFIX: 'GenericType(base_type=ClassType(typing.Callable)';

// ANYTHING TYPE
ANYTHING_TYPE : 'AnythingType()';

// BUILTIN TYPES
BUILTINS_PREFIX : 'builtins';

NONE_TYPE : 'NoneType';
STR : 'str';
BOOL : 'bool';
INT : 'int';
FLOAT : 'float';
COMPLEX : 'complex';
TUPLE : 'tuple';
LIST : 'list';
SET : 'set';
DICT : 'dict';

PARAMETERS : 'parameters=';

STRING : [A-Za-z]+;
//WS : ' ';
SKIPS : [ \t\r\n]+ -> channel(HIDDEN) ; // skip spaces, tabs,
