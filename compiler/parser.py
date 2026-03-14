import os
from lark import Lark
from lark.indenter import Indenter
from lark.exceptions import UnexpectedToken, UnexpectedInput

class SonicIndenter(Indenter):
    NL_type = 'NEWLINE'
    OPEN_PAREN_types = []
    CLOSE_PAREN_types = []
    INDENT_type = 'INDENT'
    DEDENT_type = 'DEDENT'
    tab_len = 4

def get_parser():
    """
    Loads grammar.lark and returns a Lark instance with the indenter.
    """
    base_path = os.path.dirname(__file__)
    grammar_path = os.path.join(base_path, 'grammar.lark')
    
    with open(grammar_path, 'r') as f:
        grammar = f.read()
        
    return Lark(
        grammar,
        parser='lalr',
        postlex=SonicIndenter(),
        maybe_placeholders=False
    )

def parse_text(text):
    """
    Parses SonicScript text and returns the AST.
    """
    parser = get_parser()
    # Add a newline at the end to ensure the last statement is parsed in case it's missing
    if not text.endswith('\n'):
        text += '\n'
    return parser.parse(text)

def parse_file(file_path):
    """
    Reads a file and parses its content.
    """
    with open(file_path, 'r') as f:
        content = f.read()
    return parse_text(content)
