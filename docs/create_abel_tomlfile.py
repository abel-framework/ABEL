#!/usr/bin/env python

# This file exists to trigger the creation of the .abelconfig.toml
# before building sphinx docs on readthedocs.org

try:
    import abel
except ValueError as e:
    print("Created tomlfile - error message was: '"+str(e)+"'")
