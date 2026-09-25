"""Benchmark adapters for third-party Catan engines.

Only :mod:`catanbot.bench.catanatron_adapter` lives here for now.  It is
imported lazily on purpose: ``catanatron`` (and its ``networkx`` dependency)
are *not* required by the rest of catanbot, so importing ``catanbot.bench``
must stay free of third-party imports.
"""
