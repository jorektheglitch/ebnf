from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from typing import Any, Iterable, Optional, Type, Union
from typing import Dict, List
from typing import TypeVar

from typing_extensions import Literal


letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
digits = "0123456789"
symbols = """[]{}()<>'"=|.,;"""
characters = letters + digits + symbols + "_"

TOKEN = Enum(
    "token",
    "EQUALS SEMICOLON "  # = ;
    "COMMA VBAR "  # , |
    "OPTIONAL_START OPTIONAL_END "  # [ ]
    "REPETITION_START REPETITION_END "  # { }
    "GROUP_START GROUP_END "  # ( )
)
OP = Enum(
    "op",
    "ASSIGN "
    "ALTER CONCAT "
    "GROUPING OPTIONAL REPETITION"
)

T = TypeVar("T")


@dataclass
class Rule:
    name: str
    expr: Expression


@dataclass
class Grammar(Iterable[Rule]):
    rules: List[Rule]

    def __iter__(self):
        yield from self.rules


@dataclass
class Identifier:
    name: str


@dataclass
class Terminal:
    string: str


@dataclass
class Alternation:
    left: Expression
    right: Expression


@dataclass
class Concatenation:
    left: Expression
    right: Expression


@dataclass
class Group:
    expr: Expression


@dataclass
class EBNFOptional:
    expr: Expression


@dataclass
class Repetition:
    expr: Expression


Expression = Union[Identifier, Terminal, Alternation, Concatenation, Group, EBNFOptional, Repetition]
BinOpType = Union[Type[Alternation], Type[Concatenation]]
UnOpType = Union[Type[Group], Type[EBNFOptional], Type[Repetition]]
OpNodeType = Union[BinOpType, UnOpType]
OpNode = Union[Alternation, Concatenation, Group, EBNFOptional, Repetition]


def get_last(list: List[T]) -> Optional[T]:
    return list[-1] if list else None


PRIORITIES = {OP.ASSIGN: 0, OP.ALTER: 1, OP.CONCAT: 2}
TOKEN_OPS = {
    TOKEN.EQUALS: OP.ASSIGN,
    TOKEN.COMMA: OP.CONCAT,
    TOKEN.VBAR: OP.ALTER,
    TOKEN.GROUP_END: OP.GROUPING,
    TOKEN.OPTIONAL_END: OP.OPTIONAL,
    TOKEN.REPETITION_END: OP.REPETITION}
OPEN_PARENS = {TOKEN.GROUP_START, TOKEN.OPTIONAL_START, TOKEN.REPETITION_START}
CLOSE_PARENS = {TOKEN.GROUP_END, TOKEN.OPTIONAL_END, TOKEN.REPETITION_END}
SPECIAL = {TOKEN.COMMA, TOKEN.VBAR, TOKEN.EQUALS, TOKEN.SEMICOLON, *CLOSE_PARENS}
PARENS_PAIRS = {
    TOKEN.GROUP_START: TOKEN.GROUP_END,
    TOKEN.GROUP_END: TOKEN.GROUP_START,
    TOKEN.OPTIONAL_START: TOKEN.OPTIONAL_END,
    TOKEN.OPTIONAL_END: TOKEN.OPTIONAL_START,
    TOKEN.REPETITION_START: TOKEN.REPETITION_END,
    TOKEN.REPETITION_END: TOKEN.REPETITION_START
}
BINARY_OPS = {OP.ALTER, OP.CONCAT}
UNARY_OPS = {OP.GROUPING, OP.OPTIONAL, OP.REPETITION}
OP_NODES: Dict[OP, Any] = {
    OP.ALTER: Alternation,
    OP.CONCAT: Concatenation,
    OP.GROUPING: Group,
    OP.OPTIONAL: Optional,
    OP.REPETITION: Repetition
}


def tokenize(string: str) -> List[Union[TOKEN, str]]:
    special_chars: Dict[str, TOKEN] = {
        "=": TOKEN.EQUALS,
        ";": TOKEN.SEMICOLON,
        "|": TOKEN.VBAR,
        ",": TOKEN.COMMA,
        "[": TOKEN.OPTIONAL_START,
        "]": TOKEN.OPTIONAL_END,
        "{": TOKEN.REPETITION_START,
        "}": TOKEN.REPETITION_END,
        "(": TOKEN.GROUP_START,
        ")": TOKEN.GROUP_END
    }
    tokens: List[Union[TOKEN, str]] = []
    Ctx = Literal["[", "{", "(", "|", ",",  "'", "\""]
    ctx_stack: List[Ctx] = []
    string = string.strip()
    usuals = f"{letters}{digits}_ "
    buffer = ""
    wipe_buffer = False
    for char in string:
        ctx = ctx_stack[-1] if ctx_stack else None
        if ctx in ("'", '"'):
            if ctx == char:  # end of quoted terminal
                tokens.append(f"{ctx}{buffer}{ctx}")
                ctx_stack.pop()
                wipe_buffer = True
            else:  # chars inside quoted teminal
                buffer += char
        elif char in usuals:
            buffer += char
        elif char in ("'", '"'):  # start of quoted terminal
            ctx_stack.append(char)
            wipe_buffer = True  # clear whitespaces between terminals
        elif char in special_chars:
            buffer = buffer.strip()
            token = special_chars[char]
            if buffer:
                if token in SPECIAL:  # got name before | or , or =
                    tokens.append(buffer)
                else:
                    raise SyntaxError(f"Unexpected '{char}' operator")  # noqa
            if token in CLOSE_PARENS:
                target = PARENS_PAIRS[token]
                for checked in reversed(tokens):  # type: ignore
                    if checked == target:
                        break
                    elif checked in OPEN_PARENS:
                        raise SyntaxError(f"Bracket mismatch '{char}'")
            tokens.append(token)
            wipe_buffer = True
        if wipe_buffer:
            buffer = ""
            wipe_buffer = False
    buffer = buffer.strip()
    if buffer:
        tokens.append(buffer)
    if ctx_stack:
        pass  # TODO: raise TokenizingError
    return tokens


def build_stack(
    tokens: List[Union[TOKEN, str]]
) -> List[Union[OP, Identifier, Terminal]]:
    stack: List[Union[OP, Identifier, Terminal]] = []
    op_stack: List[OP] = []

    for token in tokens:
        if isinstance(token, str):
            operand: Union[Identifier, Terminal]
            if token[0] in ("'", '"'):
                operand = Terminal(token[1:-1])
            else:
                operand = Identifier(token)
            stack.append(operand)
        elif token in {TOKEN.COMMA, TOKEN.EQUALS, TOKEN.VBAR}:
            op = TOKEN_OPS[token]
            last_op = get_last(op_stack)
            curr_priority = PRIORITIES[op]
            while last_op and (
                last_op in UNARY_OPS or
                (last_op in PRIORITIES and curr_priority <= PRIORITIES[last_op])
            ):
                stack.append(last_op)
                op_stack.pop()
                last_op = get_last(op_stack)
            op_stack.append(op)
        elif token in OPEN_PARENS:
            op_stack.append(token)  # type: ignore
        elif token in CLOSE_PARENS:
            starter = PARENS_PAIRS[token]
            last_op = get_last(op_stack)
            while last_op and (last_op is not starter):
                if last_op in OPEN_PARENS:
                    raise SyntaxError("Brackets mismatch")
                op_stack.pop()
                stack.append(last_op)
                last_op = get_last(op_stack)
            op_stack.pop()  # remove open parenthesis token from op_stack
            op = TOKEN_OPS[token]
            op_stack.append(op)  # previous was just stack.append
        elif token is TOKEN.SEMICOLON:
            for op in op_stack:
                if isinstance(op, TOKEN):
                    None
            stack.extend(reversed(op_stack))
            op_stack.clear()
    if op_stack:
        stack.extend(reversed(op_stack))
        op_stack.clear()
    return stack


def build_ast(stack: List[Union[OP, Identifier, Terminal]]):
    stack = stack.copy()
    operand_stack: List[Expression] = []
    rules: List[Rule] = []
    for element in stack:
        node: OpNode
        if not isinstance(element, OP):
            operand_stack.append(element)
        elif element is OP.ASSIGN:
            expr = operand_stack.pop()
            identifier: Identifier = operand_stack.pop()  # type: ignore
            rule = Rule(identifier.name, expr)
            rules.append(rule)
        elif element in BINARY_OPS:
            right = operand_stack.pop()
            left = operand_stack.pop()
            node_type: BinOpType = OP_NODES[element]
            node = node_type(left, right)
            operand_stack.append(node)
        elif element in UNARY_OPS:
            expr = operand_stack.pop()
            node_type: UnOpType = OP_NODES[element]
            node = node_type(expr)
            operand_stack.append(node)
    grammar = Grammar(rules)
    return grammar


def parse(string: str):
    tokens = tokenize(string)
    stack = build_stack(tokens)
    ast = build_ast(stack)
    return ast


if __name__ == "__main__":
    source = "A, B | C"
    print("Source:")
    for line in source.split("\n"):
        print(f"  {line}")
    print()

    tokens: List[Union[TOKEN, str]] = tokenize(source)
    print("Tokens:")
    for t in tokens:
        print(f"  {t}")
    print()

    stack = build_stack(tokens)
    print("Stack:")
    for e in stack:
        print(f"  {e}")
    print()

    EBNF_GRAMMAR = """
        letter = "A" | "B" | "C" | "D" | "E" | "F" | "G"
            | "H" | "I" | "J" | "K" | "L" | "M" | "N"
            | "O" | "P" | "Q" | "R" | "S" | "T" | "U"
            | "V" | "W" | "X" | "Y" | "Z" | "a" | "b"
            | "c" | "d" | "e" | "f" | "g" | "h" | "i"
            | "j" | "k" | "l" | "m" | "n" | "o" | "p"
            | "q" | "r" | "s" | "t" | "u" | "v" | "w"
            | "x" | "y" | "z" ;
        digit = "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" ;
        symbol = "[" | "]" | "{" | "}" | "(" | ")" | "<" | ">"
            | "'" | '"' | "=" | "|" | "." | "," | ";" ;

        character = letter | digit | symbol | "_" ;

        identifier = letter , { letter | digit | "_" } ;

        terminal = "'" , character , { character } , "'"
                | '"' , character , { character } , '"' ;

        lhs = identifier ;

        rhs = identifier
            | terminal
            | "[" , rhs , "]"
            | "{" , rhs , "}"
            | "(" , rhs , ")"
            | rhs , "|" , rhs
            | rhs , "," , rhs ;

        rule = lhs , "=" , rhs , ";" ;

        grammar = { rule } ;
        """
    grammar = parse(EBNF_GRAMMAR)
    for rule in grammar:
        print(rule.name)
    print()
