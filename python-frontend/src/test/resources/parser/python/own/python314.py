from string.templatelib import Template

name = "World"
template: Template = t"Hello {name}"

t''
t""
t''''''
t'a'
t"a"
t'''a'''
t'hello {var}!'
t'''hello '{var}'!'''
t'{{abc}}'
t'{{abc}}{xyz}'
t'{x} {+y}!'
t'hello {var}!'
t'''hello\n {var}'''
t'{x!a}'
t"{foo("!a")!a}"
t'{user=!s}'
t'{today:%B %d, %Y}'
t'{number:#0x}'
t'result: {value:{width}.{precision}}'
t'{delta.days=:,d}'
t"This is the playlist: {"\n".join(songs)}"
t"Foo {
    h
    }"
t"Current value: \"{value}\" (type: {type(value)}). "
t'\N{RIGHTWARDS ARROW}'
t" \\"
T"\\ \"{a}\":\\"
tr"""\s*\{{(.+)\}}"""
rt'^add_example\(\s*"[^"]*",\s*{foo()},\s*\d+,\s*async \(client, console\) => \{{\n(.*?)^(?:\}}| *\}},\n)\);$'
tr'\"foo\"\s*{42}'
tr'\\\\'
rT'\\'
