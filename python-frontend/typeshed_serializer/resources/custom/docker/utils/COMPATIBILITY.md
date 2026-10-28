These complete modules are copied from Typeshed revision
`c333d5d346a469c49b309ddbafe1d8173fae7a21` under its included Apache 2.0 license.

The analyzer does not select stubs by installed third-party dependency version.
Keep known historical signatures alongside the current signature as explicit
overloads, preserving positional order and named parameters. Do not add `**kwargs`
or replace entire modules with only the compatibility functions. Unknown arguments
must remain detectable for every supported signature. This deliberately accepts a
call when any retained release supports it; it does not diagnose dependency-version
incompatibility.

`utils.pyi` retains all of `stubs/docker/docker/utils/utils.pyi` and adds the
Docker 3.7.3 `kwargs_from_env(ssl_version, assert_hostname, environment)` signature.
Runtime source: https://github.com/docker/docker-py/blob/3.7.3/docker/utils/utils.py

Keep these modules synchronized when advancing Typeshed. No parent `__init__.pyi`
is added: upstream package exports must remain available. Existing upstream classes
retain their original bases rather than `CustomStubBase`, preserving inheritance
and member checks. Regenerate with the supported serializer runner.
