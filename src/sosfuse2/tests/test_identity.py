"""sosfuse2.identity: fixed samples of the three normalizers, the refused inputs, and package-wide checks.

CPU only. The case tables below are plain literals (source, expected identity text) so that they can be replayed under
another interpreter without pytest. The clang++ comparison is skipped when the compiler is not on PATH.
"""

import ast
import copy
import hashlib
import json
import subprocess

import pytest

from sosfuse2 import identity, paths, spectral, testing
from sosfuse2.components import canonical_state

# (source, identity text). Python 3.12 syntax only.
PYTHON_CASES = [
    (
        '#!/usr/bin/env python\n'
        '# -*- coding: utf-8 -*-\n'
        '"""Module docstring.\n'
        '\n'
        'Second paragraph  # not a comment\n'
        '"""\n'
        '\n'
        'import os  # trailing comment\n'
        '# comment-only line\n'
        '\n'
        'X = """kept # not a comment\n'
        '\n'
        '  trailing blanks stay   \n'
        '"""  # tail comment goes\n'
        '\n'
        '\n'
        'class A:\n'
        '    """Class docstring."""\n'
        '\n'
        '    y = 1   \n'
        '\n'
        '    def f(self):\n'
        "        '''Function docstring.'''\n"
        '        "a statement string that is not a docstring"\n'
        '        return f"""{self.y}\n'
        '  # inside an f-string\n'
        '\n'
        '"""\n'
        '\n'
        '    async def g(self):\n'
        '        "Async docstring."  # comment after a docstring\n'
        "        return 'é # still a string'  # comment after non-ASCII\n"
        '\n'
        '\n'
        'def h():\n'
        '    ("parenthesized "  # comment inside the docstring expression\n'
        '     "docstring")\n'
        '    return 1\n',
        'import os\n'
        'X = """kept # not a comment\n'
        '\n'
        '  trailing blanks stay   \n'
        '"""\n'
        'class A:\n'
        '    y = 1\n'
        '    def f(self):\n'
        '        "a statement string that is not a docstring"\n'
        '        return f"""{self.y}\n'
        '  # inside an f-string\n'
        '\n'
        '"""\n'
        '    async def g(self):\n'
        "        return 'é # still a string'\n"
        'def h():\n'
        '    return 1\n',
    ),
    # An empty or comment line after a continuation backslash ends the logical line: it stays as an empty line.
    ('x = 1 \\\n\n+ 2\n', 'x = 1 \\\n\n+ 2\n'),
    ('x = 1 \\\n# c\n+ 2\n', 'x = 1 \\\n\n+ 2\n'),
    ('x = 1 \\\n  \t\n\n   # c\n+ 2\n', 'x = 1 \\\n\n+ 2\n'),
    ('x = 1 \\\n+ 2\n', 'x = 1 \\\n+ 2\n'),
    ('x = 1 + \\\n    2  # c\n', 'x = 1 + \\\n    2\n'),
    ('x = (1 + \\\n\n2)\n', 'x = (1 + \\\n\n2)\n'),
    # Only the string span is kept as written; a comment after the closing quotes goes.
    ('v = """a\nb""" # note\n', 'v = """a\nb"""\n'),
    ('v = """a # kept\nb"""\n', 'v = """a # kept\nb"""\n'),
    ('a = """x\ny""" + """z   \nw"""  # c\n', 'a = """x\ny""" + """z   \nw"""\n'),
    ('x = [  # c\n    """a\n# s\n\n""",  # d\n]  # e\n', 'x = [\n    """a\n# s\n\n""",\n]\n'),
    ("s = 'a\\\n# b' # c\n", "s = 'a\\\n# b'\n"),
    # f-strings: one span from the start token to the matching end token; a comment in a replacement field stays.
    ('x = f"{\'#\' + f\'{1}\'} # in" # out\n', 'x = f"{\'#\' + f\'{1}\'} # in"\n'),
    ('y = f"{1 # kept\n}"  # gone\n', 'y = f"{1 # kept\n}"\n'),
    ('z = f"""{1}\n  {f"""{2}\n#"""}\n"""  # c\n', 'z = f"""{1}\n  {f"""{2}\n#"""}\n"""\n'),
    # Docstrings: the whole expression statement goes; only the first statement of a module, class or function.
    ('class E(Exception):\n    """doc"""\n\n\nx = 1\n', 'class E(Exception):\nx = 1\n'),
    ('class Ünï: "dóc"  # c\n', 'class Ünï:\n'),
    ('def f(): "doc"; return 1\n', 'def f(): ; return 1\n'),
    ('def f():\n    """a"""\n    """b"""\n', 'def f():\n    """b"""\n'),
    ('def f():\n    b"bytes"\n    return 1\n', 'def f():\n    b"bytes"\n    return 1\n'),
    ('def f():\n    f"formatted"\n', 'def f():\n    f"formatted"\n'),
    ('if x:\n    """not a docstring"""\n', 'if x:\n    """not a docstring"""\n'),
    ('"a" \\\n  "b"\nx = 1\n', 'x = 1\n'),
    # Line ends and blanks.
    ('x = 1\r\n# c\r\ny = """a\r\nb"""\r\n', 'x = 1\ny = """a\nb"""\n'),
    ('x = 1\t \n\x0c\n\t\ny = 2\n', 'x = 1\ny = 2\n'),
    ('x = 1  # c', 'x = 1\n'),
    ('', '\n'),
    ('# only a comment\n\n', '\n'),
    ('# coding: utf8\nx = 1\n', 'x = 1\n'),
]

# Pairs with different meaning: the identity texts must differ.
PYTHON_DISTINCT = [
    ('x = 1 \\\n\n+ 2\n', 'x = 1 \\\n+ 2\n'),
    ('x = 1 \\\n# c\n+ 2\n', 'x = 1 \\\n+ 2\n'),
    ('v = """a\nb"""\n', 'v = """a\nc"""\n'),
    ('v = """a\nb"""\n', 'v = """a \nb"""\n'),
    ('v = """a\n\nb"""\n', 'v = """a\nb"""\n'),
    ('def f():\n    x = "doc"\n', 'def f():\n    x = "other"\n'),
    ('def f():\n    """a"""\n    """b"""\n', 'def f():\n    """a"""\n    """c"""\n'),
    ('if x:\n    y = 1\nz = 2\n', 'if x:\n    y = 1\n    z = 2\n'),
    ('x = 1\ny = 2\n', 'y = 2\nx = 1\n'),
    ('x = "#"\n', 'x = ""\n'),
]

# Pairs that differ only in comments, docstrings, empty lines or trailing blanks: the identity texts must be equal.
PYTHON_EQUAL = [
    ('x = 1 \\\n\n+ 2\n', 'x = 1 \\\n# c\n+ 2\n'),
    ('v = """a\nb""" # note\n', 'v = """a\nb"""\n'),
    ('def f():\n    """one line"""\n    return 1\n', 'def f():\n    """three\n\n    lines"""\n    return 1\n'),
    ('def f():\n    """doc"""\n    return 1\n', 'def f():\n    return 1\n'),
    ('x = 1\n\n\n# a\ny = 2  # b\n', 'x = 1\ny = 2\n'),
    ('x = [\n    1,  # one\n    # two\n    2,\n]\n', 'x = [\n    1,\n    2,\n]\n'),
    ('x = 1\n', 'x = 1   \n'),
    ('x = 1\n', 'x = 1\r\n'),
]

PYTHON_REFUSED = [
    b'\xef\xbb\xbfx = 1\n',  # byte order mark
    b'x = "\xff"\n',  # not UTF-8
    b'# -*- coding: latin-1 -*-\nx = 1\n',  # another encoding declared
    b'# coding: no-such-codec\nx = 1\n',
    b'def f(:\n',  # syntax error
    b'x = 1\ry = 2\n',  # carriage return without line feed
    b'x = 1\x00\n',  # null byte
    b'x = 1 \\ \n+ 2\n',  # blank after a continuation backslash
]

C_CASES = [
    ('s = "a//b"; // c\n', 's = "a//b";\n'),
    ('int a; // c \\\n still a comment\nint b;\n', 'int a;\nint b;\n'),
    ('int v; /\\\n/ c\nint w;\n', 'int v;\nint w;\n'),
    ("#error don't // kept\nint c; // gone\n", "#error don't // kept\nint c;\n"),
    ('#warning "open // kept\nint c; // gone\n', '#warning "open // kept\nint c;\n'),
    ('int/* c */d;\n', 'int d;\n'),
    ('int e = 1 /* c */ + 2;\n', 'int e = 1   + 2;\n'),
    ('int f; /* a\n b\n */ int g;\n', 'int f;   int g;\n'),
    # A block comment is one space also when it spans lines: the macro stays one line.
    ('#define X 1 /*\n*/ + 2\nint n = X;\n', '#define X 1   + 2\nint n = X;\n'),
    ('#define Y(a) \\\n  ((a) + 1) // c\nint y = Y(2);\n', '#define Y(a)   ((a) + 1)\nint y = Y(2);\n'),
    ('long k = 1\'000\'000; char *s = "it\'s //x"; // c\n', 'long k = 1\'000\'000; char *s = "it\'s //x";\n'),
    ('int m = 0x1\'F; const char *t = "//y"; // c\n', 'int m = 0x1\'F; const char *t = "//y";\n'),
    ('double d = 1.e+5\'0; const char *u = "/*z"; /* c */\n', 'double d = 1.e+5\'0; const char *u = "/*z";\n'),
    ("char q = '\"'; char r = '\\''; int u = '//'; // c\n", "char q = '\"'; char r = '\\''; int u = '//';\n"),
    ('s = "a\\"/*b*/"; /* c */\n', 's = "a\\"/*b*/";\n'),
    ('s = "ab\\\ncd//e"; // c\n', 's = "abcd//e";\n'),
    ('#include <a//b.h> // c\n#include "p//q.h" // c\n', '#include <a//b.h>\n#include "p//q.h"\n'),
    ('  /* c */ # /* d */ include /* e */ <x/*y.h>\n', '    #   include   <x/*y.h>\n'),
    ('/* it\'s "x */ int a; // don\'t\nint b;\n', '  int a;\nint b;\n'),
    ('int a; // x /* y\nint b; /* // */ int c;\n', 'int a;\nint b;   int c;\n'),
    ('_Pragma("unroll") for(;;) x; // c\n', '_Pragma("unroll") for(;;) x;\n'),
    ('\n  \t\nint x;  \n\n', 'int x;\n'),
    ('int y; // c\r\nint z;\r\n', 'int y;\nint z;\n'),
    ('x = a / b; y = *p / *q;\n', 'x = a / b; y = *p / *q;\n'),
    ('', '\n'),
]

C_DISTINCT = [
    ('#define X 1 /*\n*/ + 2\n', '#define X 1\n+ 2\n'),
    ('s = "a//b";\n', 's = "a//c";\n'),
    ('int a;\n', 'int b;\n'),
    ('#include <a//b.h>\n', '#include <a//c.h>\n'),
    ("#error don't // x\n", "#error don't // y\n"),
    ('int ab;\n', 'int a/**/b;\n'),
    ('#define F(x) x\n', '#define F/**/(x) x\n'),
]

C_EQUAL = [
    ('int a; // one\n', 'int a; /* two\n three */\n'),
    ('int a;\n// c\nint b;\n', 'int a;\n\n\nint b;\n'),
    ('#define X 1 /* a */ + 2\n', '#define X 1 /*\nb\n*/ + 2\n'),
    ('#define Y 1 \\\n + 2\n', '#define Y 1  + 2\n'),
    ('int a;\n', 'int a;   \n'),
]

C_REFUSED = [
    b'const char *s = R"(a"//b)";\n',  # raw string literal
    b'auto s = u8R"x(a)x";\n',
    b'int a; ??/\nint b;\n',  # trigraph
    b'#define X 1 \\ \n + 2\n',  # blank between the backslash and the line end
    b'int a; /* open\n',  # unterminated block comment
    b'#include <open\n>\n',  # unterminated header name
    b'%:define X 1\n',  # directive spelled with a digraph
    b'#if __has_include(<a//b>)\n#endif\n',
    b'int a;\rint b;\n',  # carriage return without line feed
    b'\xef\xbb\xbfint a;\n',  # byte order mark
    b'int a = "\xff";\n',  # not UTF-8
]

# Keys json_identity() drops from each hashed JSON file ('[]' stands for every list element). A change of this list is
# a change of what the signatures see: review it, do not regenerate it.
STATE_KEYS_DROPPED = {'description', 'group'}
JSON_DROPPED = {
    'assets/states_v1.json': {'states[].description', 'states[].group'},
    'assets/states_v2.json': {'REPORT.note', 'states[].description', 'states[].group'},
    'assets/states_v1_usstd.json': {'note', 'states[].description', 'states[].group'},
    'gate/fixtures.json': {
        'note',
        'settings_note',
        'solve_parameters_note',
        'directions.note',
        'fixtures.FX-A.note',
        'fixtures.FX-AP.note',
        'state.description',
        'state.group',
        'state.holdout',
        'state.note_enable',
        'state.out_of_axis_range',
        'state.q2d_override',
        'state.role',
    },
    'configs/devices.json': {
        'note', 'profiles.eda.note', 'profiles.eda.memory_note', 'profiles.eda.requires_split_note',
        'profiles.rig.note', 'profiles.mac.note',
    },
    'configs/C3.json': {'notes'},
    'configs/C3_nostrato.json': {'notes'},
}
VERBATIM = ('settings', 'parameters', 'stratosphere', 'density')

JSON_REFUSED = [
    b'\xef\xbb\xbf{"a": 1}',  # byte order mark
    b'{"a": "\xff"}',  # not UTF-8
    b'{"a": 1,}',  # not JSON
    b'{"a": NaN}',
    b'{"a": -Infinity}',
    b'{"a": 1e999}',  # overflows to infinity
    b'{"note": 1e400}',
    b'{"description": {"x": 1e400}}',
    b'{"a": 1, "a": 2}',  # repeated key
    b'',
]

KERNEL_FLAGS = (
    ('-DFW=15', '-DK_FIRST=1', '-DK_PHASE=0', '-DK_BOUNDARY=0'),
    ('-DSTRATO', '-DFW=15', '-DK_FIRST=0', '-DK_PHASE=2', '-DK_BOUNDARY=0'),
    ('-DFW=1', '-DK_FIRST=0', '-DK_PHASE=0', '-DK_BOUNDARY=1'),
)


def python_sources():
    """Every Python file package_hashes() lists."""
    P = paths.PKG
    return [
        *paths.production_files(),
        *[P / 'accepted' / n for n in identity.ACCEPTED_SOURCES],
        *[P / 'host' / n for n in identity.HOST_MODULES],
    ]


def c_sources():
    return [paths.PKG / 'kernel.cu', paths.PKG / 'native/sh_cpu.cpp']


def without_docstrings(tree):
    """ast.dump of a tree after removing the docstring statement of every module, class and function."""
    only_docstring = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                del node.body[0]
                only_docstring |= not node.body and not isinstance(node, ast.Module)
    return ast.dump(tree), only_docstring


def sha(kind, text):
    return hashlib.sha256(f'sosfuse2-identity-2:{kind}\n'.encode() + text.encode()).hexdigest()


# --- Python ---


@pytest.mark.parametrize('index', range(len(PYTHON_CASES)))
def test_python_text_fixed_samples(index):
    source, expected = PYTHON_CASES[index]
    assert identity.python_text(source.encode()) == expected


def test_python_text_distinct_and_equal_pairs():
    for a, b in PYTHON_DISTINCT:
        ast.parse(a), ast.parse(b)  # both are valid programs
        assert identity.python_text(a.encode()) != identity.python_text(b.encode()), (a, b)
    for a, b in PYTHON_EQUAL:
        assert identity.python_text(a.encode()) == identity.python_text(b.encode()), (a, b)


@pytest.mark.parametrize('raw', PYTHON_REFUSED)
def test_python_text_refuses(raw):
    with pytest.raises(ValueError, match=r'^E_IDENTITY_SOURCE: probe\.py: '):
        identity.python_text(raw, 'probe.py')


def test_python_sources_comment_and_code_edits(tmp_path):
    """A comment appended to any hashed Python file keeps its file_sha; an appended statement changes it."""
    for i, path in enumerate(python_sources()):
        raw = path.read_bytes()
        copy_path = tmp_path / f'{i}_{path.name}'
        copy_path.write_bytes(raw)
        base = identity.file_sha(copy_path)
        assert base == identity.file_sha(path) == sha('python', identity.python_text(raw))
        copy_path.write_bytes(raw + b'\n# x\n')
        assert identity.file_sha(copy_path) == base, path.name
        copy_path.write_bytes(raw + b'\nIDENTITY_PROBE = 1\n')
        assert identity.file_sha(copy_path) != base, path.name


def test_python_sources_edits_inside_the_file():
    """Comment lines before every statement and rewritten docstrings (other text, other line count) keep the identity
    of every hashed Python file. The edits are made on the bytes with the ast offsets, and the docstring-free AST
    confirms that they changed nothing else."""
    edited = 0
    for path in python_sources():
        raw = path.read_bytes()
        tree = ast.parse(raw)
        reference, _ = without_docstrings(ast.parse(raw))
        lines = raw.split(b'\n')
        docstrings = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
                first = node.body[0]
                if (
                    isinstance(first, ast.Expr)
                    and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)
                ):
                    docstrings.append((first.lineno, first.col_offset, first.end_lineno, first.end_col_offset))
        for l0, c0, l1, c1 in sorted(docstrings, reverse=True):
            head, tail = lines[l0 - 1][:c0], lines[l1 - 1][c1:]
            lines[l0 - 1 : l1] = [head + b'"""rewritten', b'', b'  # not a comment', b'"""' + tail]
        rewritten = b'\n'.join(lines)
        assert without_docstrings(ast.parse(rewritten))[0] == reference, path.name
        assert identity.python_text(rewritten) == identity.python_text(raw), path.name

        lines = raw.split(b'\n')
        firsts = {node.lineno for node in ast.walk(tree) if isinstance(node, ast.stmt)}
        for lineno in sorted(firsts, reverse=True):
            lines.insert(lineno - 1, b'  # inserted comment')
        commented = b'\n'.join(lines)
        assert without_docstrings(ast.parse(commented))[0] == reference, path.name
        assert identity.python_text(commented) == identity.python_text(raw), path.name
        edited += len(docstrings) + len(firsts)
    assert edited > 5000


def test_python_text_keeps_the_ast():
    """The identity text of every hashed Python file parses to the AST of the file without its docstrings. The only
    texts that do not parse are those of files with a class or function whose body was a docstring alone."""
    unparsable, expected_unparsable = set(), set()
    for path in python_sources():
        raw = path.read_bytes()
        reference, only_docstring = without_docstrings(ast.parse(raw))
        key = paths.rel(path)
        if only_docstring:
            expected_unparsable.add(key)
        try:
            tree = ast.parse(identity.python_text(raw))
        except SyntaxError:
            unparsable.add(key)
            continue
        assert ast.dump(tree) == reference, key
    assert unparsable == expected_unparsable
    assert len(unparsable) < 5


def test_docstrings_are_not_read_at_run_time():
    """Auxiliary check, not a proof: a docstring may feed an argparse description and nothing else. It finds the
    spellings `__doc__` and `getdoc`; an indirect read is outside its reach."""
    message = 'docstrings are not identity inputs; move the text into a constant'
    for path in python_sources():
        tree = ast.parse(path.read_bytes())
        allowed = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == 'description':
                allowed.update(id(inner) for inner in ast.walk(node.value))
        for node in ast.walk(tree):
            name = node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else None
            assert name != 'getdoc', f'{path.name}:{node.lineno}: {message}'
            assert name != '__doc__' or id(node) in allowed, f'{path.name}:{node.lineno}: {message}'


# --- C and CUDA ---


@pytest.mark.parametrize('index', range(len(C_CASES)))
def test_c_text_fixed_samples(index):
    source, expected = C_CASES[index]
    assert identity.c_text(source.encode()) == expected


def test_c_text_distinct_and_equal_pairs():
    for a, b in C_DISTINCT:
        assert identity.c_text(a.encode()) != identity.c_text(b.encode()), (a, b)
    for a, b in C_EQUAL:
        assert identity.c_text(a.encode()) == identity.c_text(b.encode()), (a, b)


@pytest.mark.parametrize('raw', C_REFUSED)
def test_c_text_refuses(raw):
    with pytest.raises(ValueError, match=r'^E_IDENTITY_SOURCE: probe\.cu: '):
        identity.c_text(raw, 'probe.cu')


def with_comments(raw, comment=b'// x\n'):
    """C source with a comment before the first line and at the end of every line that is not continued by a
    backslash (a comment there would stop the continuation)."""
    lines = raw.split(b'\n')
    return comment + b'\n'.join(line if line.endswith(b'\\') or not line else line + b' // x' for line in lines)


def test_c_sources_comment_and_code_edits(tmp_path):
    """Comments added to the kernel sources keep their file_sha; an added declaration changes it."""
    for path in c_sources():
        raw = path.read_bytes()
        copy_path = tmp_path / path.name
        copy_path.write_bytes(raw)
        base = identity.file_sha(copy_path)
        assert base == identity.file_sha(path) == sha('c', identity.c_text(raw))
        for comment in (b'// x\n', b'/* it\'s "x\n */\n'):
            copy_path.write_bytes(raw + comment)
            assert identity.file_sha(copy_path) == base, path.name
            copy_path.write_bytes(with_comments(raw, comment))
            assert identity.file_sha(copy_path) == base, path.name
        copy_path.write_bytes(raw + b'int identity_probe;\n')
        assert identity.file_sha(copy_path) != base, path.name
        assert identity.raw_sha(copy_path) != identity.raw_sha(path)


def test_every_listed_source_is_accepted():
    """No file that enters a signature, and no C or CUDA file of the package, is refused by its normalizer."""
    assert len(identity.package_hashes()) == len(python_sources()) + len(c_sources()) + len(identity.DATA_ASSETS)
    others = sorted(p for p in paths.PKG.rglob('*') if p.suffix in ('.cu', '.cpp', '.h'))
    assert set(c_sources()) <= set(others)
    for path in others:
        assert identity.file_sha(path) != identity.raw_sha(path)
    for name in JSON_DROPPED:
        identity.file_sha(paths.PKG / name, 'json')


def preprocessed(clang, source, flags, tmp_path, name):
    path = tmp_path / name
    path.write_text(source)
    command = [clang, '-E', '-P', '-x', 'c++', *flags, str(path)]
    run = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr[-2000:]
    assert run.stdout.strip()
    return run.stdout


def test_c_text_preprocesses_like_the_source(tmp_path):
    """clang++ -E -P gives the same output for the identity text as for the source; every run must exit 0.
    The kernel (three flag sets) and the native kernel are compared byte for byte, also against their copies with
    added comments. The fixed samples are compared as token sequences: clang prints a token on the line it came from,
    and the identity text has fewer lines where a comment or a continuation spanned several."""
    clang = testing.require_tool('clang++')
    kernel = (paths.PKG / 'kernel.cu').read_bytes()
    native = (paths.PKG / 'native/sh_cpu.cpp').read_bytes()
    for i, (raw, flags) in enumerate([*[(kernel, flags) for flags in KERNEL_FLAGS], (native, ())]):
        source = preprocessed(clang, raw.decode(), flags, tmp_path, f'source_{i}.cpp')
        for name, text in (('identity', identity.c_text(raw)), ('commented', with_comments(raw).decode())):
            assert preprocessed(clang, text, flags, tmp_path, f'{name}_{i}.cpp') == source, (name, i, flags)
    compared = 0
    for i, (source, text) in enumerate(C_CASES):
        if not source.strip() or source.lstrip().startswith(('#error', '#warning', '#include', '/* c */ #')):
            continue  # nothing to print, a directive that stops the run, or a header that does not exist
        a = preprocessed(clang, source, (), tmp_path, f'case_source_{i}.cpp')
        b = preprocessed(clang, text, (), tmp_path, f'case_identity_{i}.cpp')
        assert a.split() == b.split(), source
        compared += 1
    assert compared >= 18


# --- states and JSON ---


def test_state_identity_and_canonical_state():
    state = dict(
        id='s',
        description='prose',
        note='n',
        notes=['n'],
        note_enable='n',
        solver_note='n',
        group='g',
        role='r',
        holdout=True,
        out_of_axis_range=False,
        q2d_override='n',
        runnable=True,
        future_key=dict(description='stays', group='stays'),
        parameters=dict(aod=0.1, note='stays', description='stays', group='stays'),
        stratosphere=dict(enabled=False, note_enable='stays'),
    )
    before = copy.deepcopy(state)
    got = identity.state_identity(state)
    assert state == before
    assert got == dict(
        id='s',
        runnable=True,
        future_key=dict(description='stays', group='stays'),
        parameters=dict(aod=0.1, note='stays', description='stays', group='stays'),
        stratosphere=dict(enabled=False, note_enable='stays'),
    )
    got['parameters']['aod'] = 9
    assert state['parameters']['aod'] == 0.1  # a deep copy
    assert identity.STATE_LABELS == ('group', 'role', 'holdout', 'out_of_axis_range', 'q2d_override')
    for key in ('description', 'note', 'notes', 'note_x', 'x_note'):
        assert identity.is_note(key), key
    for key in ('notebook', 'annotation', 'noted', 'descriptions', 'id', 'Note', 7):
        assert not identity.is_note(key), key

    real = json.loads((paths.ASSETS / 'states_v2.json').read_text())['states'][0]
    base = canonical_state(real)
    assert 'description' not in base and 'group' not in base and base['id'] == real['id']
    assert canonical_state(dict(real, description='other', note_enable='x', group='other')) == base
    changed = copy.deepcopy(real)
    first = next(iter(changed['parameters']))
    changed['parameters'][first] = 0.123456
    assert canonical_state(changed) != base
    assert canonical_state(dict(real, id='other')) != base


def dropped_paths(doc, kept, prefix=''):
    """Paths of the keys present in `doc` and absent from json_identity(doc) == `kept`."""
    out = set()
    if isinstance(doc, dict):
        for key, value in doc.items():
            path = f'{prefix}.{key}' if prefix else key
            if key not in kept:
                out.add(path)
            else:
                out |= dropped_paths(value, kept[key], path)
    elif isinstance(doc, list):
        assert len(doc) == len(kept)
        for value, kept_value in zip(doc, kept):
            out |= dropped_paths(value, kept_value, prefix + '[]')
    else:
        assert doc == kept
    return out


def test_json_identity_rule():
    state = dict(id='s', description='d', group='g', parameters=dict(note='stays', aod=1), unknown=dict(note='stays'))
    doc = dict(
        note='n',
        settings_note='n',
        version=1,
        REPORT=dict(note='n', rows=[dict(description='n', value=1)]),
        settings=dict(note='stays', tol=1e-3, inner=dict(description='stays')),
        density=dict(notes='stays'),
        state=state,
        states=[state, dict(state, id='t')],
        profiles=[dict(name='a', note='n'), dict(name='b')],
    )
    before = copy.deepcopy(doc)
    got = identity.json_identity(doc)
    assert doc == before
    assert got == dict(
        version=1,
        REPORT=dict(rows=[dict(value=1)]),
        settings=dict(note='stays', tol=1e-3, inner=dict(description='stays')),
        density=dict(notes='stays'),
        state=identity.state_identity(state),
        states=[identity.state_identity(state), identity.state_identity(dict(state, id='t'))],
        profiles=[dict(name='a'), dict(name='b')],
    )
    assert list(got) == ['version', 'REPORT', 'settings', 'density', 'state', 'states', 'profiles']
    # A state has one identity, whether it is read from a file or held in memory.
    assert identity.json_identity(dict(states=[state]))['states'][0] == identity.state_identity(state)
    assert identity.json_identity(dict(state=state))['state'] == identity.state_identity(state)


def test_json_dropped_keys_are_the_fixed_list():
    """The keys dropped from the hashed JSON files are exactly the listed ones, and nothing is dropped below a key that
    holds inputs of the computation."""
    for name, expected in JSON_DROPPED.items():
        doc = json.loads((paths.PKG / name).read_text())
        got = dropped_paths(doc, identity.json_identity(doc))
        assert got == expected, (name, got ^ expected)
        for path in got:
            assert not set(path.replace('[]', '').split('.')[:-1]) & set(VERBATIM), (name, path)
        states = doc['states'] if 'states' in doc else [doc['state']] if 'state' in doc else []
        for state in states:
            kept = identity.state_identity(state)
            assert all(kept[key] == state[key] for key in kept)
            assert set(state) - set(kept) <= {p.rsplit('.', 1)[-1] for p in expected}
    assert {p for p in JSON_DROPPED['assets/states_v1.json']} == {f'states[].{k}' for k in STATE_KEYS_DROPPED}


def test_json_file_sha(tmp_path):
    path = tmp_path / 'doc.json'

    def digest(text):
        path.write_bytes(text if isinstance(text, bytes) else text.encode())
        return identity.file_sha(path, 'json')

    base = digest('{"a": 1, "b": {"c": [1.5, "é"], "note": "x"}, "states": [{"id": "s", "group": "g"}]}')
    assert base == sha('json', '{"a":1,"b":{"c":[1.5,"\\u00e9"]},"states":[{"id":"s"}]}')
    # Whitespace and the spelling of a value are not identity; neither are note and label keys.
    assert digest('{\r\n\t"a":1,"b":{"c":[1.50,"\\u00e9"]},\n  "states":[{"id":"s"}]}\n') == base
    assert digest('{"a": 1, "description": "d", "b": {"c": [1.5, "é"], "x_note": 3}, "states": [{"id": "s"}]}') == base
    assert digest('{"a": 1, "b": {"c": [1.5, "é"]}, "states": [{"id": "s", "role": 1, "note": "n"}]}') == base
    assert digest('{"a": 2, "b": {"c": [1.5, "é"]}, "states": [{"id": "s"}]}') != base  # a value
    assert digest('{"b": {"c": [1.5, "é"]}, "a": 1, "states": [{"id": "s"}]}') != base  # key order is identity
    assert digest('{"a": 1, "b": {"c": ["é", 1.5]}, "states": [{"id": "s"}]}') != base
    assert digest('{"a": 1.0, "b": {"c": [1.5, "é"]}, "states": [{"id": "s"}]}') != base  # 1 and 1.0 are kept apart
    assert digest('{"a": 1, "b": {"c": [1.5, "é"]}, "states": [{"id": "s", "extra": null}]}') != base
    # Without kind='json' the suffix decides, and a .json file is raw bytes.
    assert identity.file_sha(path) == identity.raw_sha(path) == hashlib.sha256(path.read_bytes()).hexdigest()
    for raw in JSON_REFUSED:
        path.write_bytes(raw)
        with pytest.raises(ValueError, match='^E_IDENTITY_SOURCE: '):
            identity.file_sha(path, 'json')
    with pytest.raises(ValueError, match='^E_IDENTITY_KIND'):
        identity.file_sha(path, 'yaml')


def test_state_labels_are_not_read():
    """Auxiliary check, not a proof: no hashed Python file subscripts or .get()s a state label key. It finds literal
    keys only; a key built at run time is outside its reach. The listed uses are not state dicts."""
    allowed = {
        ('pipeline_execution.py', 'group', 'prepared'): 'the prepared launch bundle, not a state',
        ('plan_queue.py', 'role', 'r'): 'a plan row written by the planner',
        ('plan_queue.py', 'role', 'by_id[i]'): 'a plan row written by the planner',
    }
    found = set()
    for path in python_sources():
        for node in ast.walk(ast.parse(path.read_bytes())):
            key = holder = None
            if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
                key, holder = node.slice.value, node.value
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ('get', 'pop', 'setdefault')
                and node.args
                and isinstance(node.args[0], ast.Constant)
            ):
                key, holder = node.args[0].value, node.func.value
            if isinstance(key, str) and key in identity.STATE_LABELS:
                found.add((path.name, key, ast.unparse(holder)))
    assert found <= set(allowed), sorted(found - set(allowed))


# --- hashes, lists, version ---


def test_file_sha_kinds_and_memo(tmp_path):
    assert identity.RULES == 'sosfuse2-identity-2'
    a, b = tmp_path / 'a.py', tmp_path / 'b.txt'
    a.write_text('x = 1  # c\n')
    b.write_text('x = 1  # c\n')
    assert identity.file_sha(a) == sha('python', 'x = 1\n')
    assert identity.file_sha(b) == identity.raw_sha(b) == hashlib.sha256(b'x = 1  # c\n').hexdigest()
    assert identity.file_sha(b, 'python') == identity.file_sha(a)  # the content decides, not the path
    assert identity.file_sha(a, 'raw') == identity.raw_sha(a)
    a.write_text('x = 2  # c\n')  # same path, new content
    assert identity.file_sha(a) == sha('python', 'x = 2\n')
    for name in ('k.cu', 'k.cpp', 'k.h'):
        (tmp_path / name).write_text('int a; // c\n')
        assert identity.file_sha(tmp_path / name) == sha('c', 'int a;\n')
    assert sha('python', 'int a;\n') != sha('c', 'int a;\n')
    bad = tmp_path / 'bad.py'
    bad.write_bytes(b'def f(:\n')
    with pytest.raises(ValueError, match=r'^E_IDENTITY_SOURCE: .*bad\.py: SyntaxError'):
        identity.file_sha(bad)
    with pytest.raises(ValueError, match=r'^E_IDENTITY_SOURCE: '):
        identity.file_sha(bad)  # a refusal is not remembered as a hash


def test_package_hashes(monkeypatch):
    hashes = identity.package_hashes()
    expected = {
        'sosfuse2/__init__.py',
        'sosfuse2/air_wavelength.py',
        'sosfuse2/arithmetic.py',
        'sosfuse2/cache_store.py',
        'sosfuse2/column_cache.py',
        'sosfuse2/column_parallel.py',
        'sosfuse2/column_table.py',
        'sosfuse2/compile_cache.py',
        'sosfuse2/components.py',
        'sosfuse2/density.py',
        'sosfuse2/device_profile.py',
        'sosfuse2/fused.py',
        'sosfuse2/gate.py',
        'sosfuse2/gate_flags.py',
        'sosfuse2/generate_density_profile.py',
        'sosfuse2/generate_minor_absorbers.py',
        'sosfuse2/generate_strato_optics.py',
        'sosfuse2/generate_tropo_optics.py',
        'sosfuse2/geometry.py',
        'sosfuse2/geometry_contract.py',
        'sosfuse2/geometry_cpu.py',
        'sosfuse2/geometry_parallel.py',
        'sosfuse2/ground_parallel.py',
        'sosfuse2/ground_sh.py',
        'sosfuse2/host_cache.py',
        'sosfuse2/host_limits.py',
        'sosfuse2/identity.py',
        'sosfuse2/iprt.py',
        'sosfuse2/kernel_config.py',
        'sosfuse2/ksplit.py',
        'sosfuse2/mie.py',
        'sosfuse2/minor_absorbers.py',
        'sosfuse2/numerical_checks.py',
        'sosfuse2/ozone_temperature.py',
        'sosfuse2/paths.py',
        'sosfuse2/prepare_minor_absorbers.py',
        'sosfuse2/solver_columns.py',
        'sosfuse2/solver_optics.py',
        'sosfuse2/solver_sky.py',
        'sosfuse2/solver_legacy.py',
        'sosfuse2/solver_poly.py',
        'sosfuse2/solver_core.py',
        'sosfuse2/solver_render.py',
        'sosfuse2/pipeline.py',
        'sosfuse2/pipeline_execution.py',
        'sosfuse2/plan_queue.py',
        'sosfuse2/precompile.py',
        'sosfuse2/provenance.py',
        'sosfuse2/ray_columns.py',
        'sosfuse2/rayleigh.py',
        'sosfuse2/reference.py',
        'sosfuse2/run_fuse.py',
        'sosfuse2/run_gpu_state_v3.py',
        'sosfuse2/sh_layout.py',
        'sosfuse2/spectral.py',
        'sosfuse2/strato_parallel.py',
        'sosfuse2/strato_states.py',
        'sosfuse2/stratosphere.py',
        'sosfuse2/sun_parallel.py',
        'sosfuse2/table_fast.py',
        'sosfuse2/table_q.py',
        'sosfuse2/tables_from_sources.py',
        'sosfuse2/timers.py',
        'sosfuse2/tropo_mie.py',
        'sosfuse2/tropo_profile.py',
        'sosfuse2/validate.py',
        'sosfuse2/validate_strato.py',
        'sosfuse2/verify_tol.py',
        'sosfuse2/kernel.cu',
        'sosfuse2/accepted/__init__.py',
        'sosfuse2/accepted/engine.py',
        'sosfuse2/host/__init__.py',
        'sosfuse2/host/host_cores.py',
        'sosfuse2/host/pack_state.py',
        'sosfuse2/native/sh_cpu.cpp',
        'sosfuse2/assets/spectra.json',
        'sosfuse2/assets/spectra_fine.json',
        'sosfuse2/assets/spectra_fine_temperature.json',
        'sosfuse2/assets/lef_cie.json',
        'sosfuse2/assets/cmf_1nm.json',
        'sosfuse2/assets/states_v1.json',
        'sosfuse2/assets/states_v2.json',
        'sosfuse2/assets/states_v1_usstd.json',
        'sosfuse2/assets/rgb_weights.npy',
        'sosfuse2/assets/temperature_usstd76.json',
        'sosfuse2/assets/no2_afglus_profile.json',
        'sosfuse2/assets/no2_vandaele1998_fine.json',
    }
    assert set(hashes) == expected
    assert identity.ACCEPTED_SOURCES == ('__init__.py', 'engine.py')
    assert identity.HOST_MODULES == ('__init__.py', 'host_cores.py', 'pack_state.py')
    assert set(identity.STATE_ASSETS) < set(identity.DATA_ASSETS) and len(identity.DATA_ASSETS) == 12
    for name in identity.DATA_ASSETS:
        path = paths.ASSETS / name
        if name in identity.ASSET_NOTES:
            assert hashes[paths.rel(path)] == identity.asset_sha(path) != identity.raw_sha(path)
            continue
        kind = 'json' if name in identity.STATE_ASSETS else 'raw'
        assert hashes[paths.rel(path)] == identity.file_sha(path, kind)
        assert (hashes[paths.rel(path)] == identity.raw_sha(path)) == (kind == 'raw')
    for path in python_sources() + c_sources():
        assert hashes[paths.rel(path)] == identity.file_sha(path) != identity.raw_sha(path)
    info = identity.package_info()
    assert set(info) == {'package_version', 'package_commit'}
    with monkeypatch.context() as patch:
        patch.setattr(identity, 'DATA_ASSETS', identity.DATA_ASSETS + ('no_such_asset.json',))
        with pytest.raises(RuntimeError, match='^E_PACKAGE_FILE_MISSING.*no_such_asset'):
            identity.package_hashes()
    monkeypatch.setattr(paths, 'PRODUCTION_MODULES', paths.PRODUCTION_MODULES + ('no_such_module',))
    with pytest.raises(RuntimeError, match='^E_PACKAGE_FILE_MISSING'):
        identity.package_hashes()


def test_meta_contract_versions():
    for version in ('v3', 'v4', 'v5'):
        assert spectral.check_meta_contract(dict(signature='s', contract=version)) == version
    for version in ('v2', 'v6', 'V5', None, 5):
        with pytest.raises(ValueError, match='^E_PROVENANCE_CONTRACT'):
            spectral.check_meta_contract(dict(signature='s', contract=version))
    assert spectral.check_meta_contract(dict(contract='free text of a derived table')) == 'free text of a derived table'


def test_work_root_error(monkeypatch, tmp_path):
    blocker = tmp_path / 'file'
    blocker.write_text('x')
    monkeypatch.setenv('SOSFUSE2_WORK_ROOT', str(blocker / 'below'))
    with pytest.raises(RuntimeError, match=r'^E_WORK_ROOT: .*below: \w+Error: '):
        paths.work_root()
    monkeypatch.setenv('SOSFUSE2_WORK_ROOT', str(tmp_path / 'made' / 'here'))
    assert paths.work_root() == (tmp_path / 'made' / 'here').resolve()
