#!/usr/bin/env python3

"""
RECURSIVE SYMBOL TOPOLOGY V3

Typed recursive abstraction experiment.

LEVEL 0:
    bytes

LEVEL 1:
    repeated byte structures

LEVEL 2:
    patterns of patterns

LEVEL 3:
    higher structural relationships


No external libraries.
Exact reconstruction required.
"""

import sys
from collections import defaultdict


MIN_PATTERN = 4
MAX_PATTERN = 32
MIN_OCCURRENCES = 2


# ============================================================
# SYMBOL SYSTEM
# ============================================================

class Symbol:

    def __init__(self, kind, value):
        self.kind = kind
        self.value = value

    def __repr__(self):

        return f"{self.kind}:{self.value}"

    def __eq__(self, other):

        return (
            isinstance(other, Symbol)
            and self.kind == other.kind
            and self.value == other.value
        )

    def __hash__(self):

        return hash(
            (
                self.kind,
                self.value
            )
        )



# ============================================================
# PATTERN OBJECT
# ============================================================

class Pattern:

    def __init__(
        self,
        pid,
        level,
        sequence
    ):

        self.pid = pid
        self.level = level
        self.sequence = sequence
        self.locations = []


    def symbol(self):

        return Symbol(
            "PATTERN",
            self.pid
        )



# ============================================================
# FIND STRUCTURES
# ============================================================

def discover_patterns(
    stream,
    level,
    start_id
):

    found = defaultdict(list)


    for size in range(
        MAX_PATTERN,
        MIN_PATTERN - 1,
        -1
    ):

        for i in range(
            len(stream)-size+1
        ):

            chunk = tuple(
                stream[i:i+size]
            )

            found[chunk].append(i)



    patterns = {}

    next_id = start_id


    for sequence, locations in found.items():

        if len(locations) < MIN_OCCURRENCES:
            continue


        already_inside = False


        for existing in patterns.values():

            if (
                len(existing.sequence)
                >
                len(sequence)
                and
                sequence
                in existing.sequence
            ):

                already_inside = True
                break


        if already_inside:
            continue



        p = Pattern(
            next_id,
            level,
            sequence
        )

        p.locations = locations

        patterns[next_id] = p

        next_id += 1


    return patterns



# ============================================================
# ENCODE
# ============================================================

def encode(
    stream,
    patterns
):

    ordered = sorted(
        patterns.values(),
        key=lambda x:
            len(x.sequence),
        reverse=True
    )


    output = []

    i = 0


    while i < len(stream):

        found = False


        for pattern in ordered:

            length = len(
                pattern.sequence
            )


            if tuple(
                stream[i:i+length]
            ) == pattern.sequence:


                output.append(
                    pattern.symbol()
                )


                i += length

                found = True

                break



        if not found:

            output.append(
                stream[i]
            )

            i += 1



    return output



# ============================================================
# EXPAND
# ============================================================

def expand(
    stream,
    dictionaries
):

    lookup = {}


    for dictionary in dictionaries:

        for pattern in dictionary.values():

            lookup[
                pattern.symbol()
            ] = pattern.sequence



    output = []


    for symbol in stream:


        if isinstance(symbol, Symbol):


            if symbol.kind == "BYTE":

                output.append(
                    symbol.value
                )


            elif symbol.kind == "PATTERN":

                expanded = expand(
                    lookup[symbol],
                    dictionaries
                )

                output.extend(
                    expanded
                )

        else:

            output.append(symbol)



    return output



# ============================================================
# DISPLAY
# ============================================================

def print_patterns(
    level,
    patterns
):

    print()

    print("="*70)

    print(
        f"LEVEL {level} PATTERNS"
    )

    print("="*70)



    if not patterns:

        print("None")

        return



    for p in patterns.values():

        print()

        print(
            f"P{p.pid}"
        )

        print(
            " sequence:"
        )

        print(
            "   ",
            p.sequence
        )

        print(
            " locations:",
            p.locations[:10]
        )



# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv)<2:

        print(
            "usage:"
            "python recursive_symbol_topology_v3.py file"
        )

        return



    filename = sys.argv[1]


    with open(
        filename,
        "rb"
    ) as f:

        raw = list(
            f.read()
        )



    print("="*70)

    print(
        "RECURSIVE SYMBOL TOPOLOGY V3"
    )

    print("="*70)


    print()

    print(
        "FILE:",
        filename
    )

    print(
        "SIZE:",
        len(raw)
    )



    # ----------------------------
    # LEVEL 0
    # ----------------------------


    level0 = [
        Symbol(
            "BYTE",
            b
        )
        for b in raw
    ]



    dictionaries = []



    # ----------------------------
    # LEVEL 1
    # ----------------------------


    level1 = discover_patterns(
        level0,
        1,
        1000
    )


    dictionaries.append(
        level1
    )


    print_patterns(
        1,
        level1
    )



    stream1 = encode(
        level0,
        level1
    )



    print()

    print(
        "LEVEL 1 STREAM:"
    )

    print(
        stream1[:50]
    )



    rebuilt1 = expand(
        stream1,
        dictionaries
    )



    print()

    print(
        "LEVEL 1 EXACT:",
        rebuilt1 == raw
    )



    # ----------------------------
    # LEVEL 2
    # ----------------------------


    level2 = discover_patterns(
        stream1,
        2,
        2000
    )


    dictionaries.append(
        level2
    )


    print_patterns(
        2,
        level2
    )



    stream2 = encode(
        stream1,
        level2
    )



    print()

    print(
        "LEVEL 2 STREAM:"
    )

    print(
        stream2[:50]
    )



    rebuilt2 = expand(
        stream2,
        dictionaries
    )



    print()

    print(
        "LEVEL 2 EXACT:",
        rebuilt2 == raw
    )



    # ----------------------------
    # COST
    # ----------------------------


    print()

    print("="*70)

    print(
        "STRUCTURAL COST"
    )

    print("="*70)


    print(
        "RAW:",
        len(level0)
    )


    print(
        "LEVEL 1:",
        len(stream1)
    )


    print(
        "LEVEL 2:",
        len(stream2)
    )


    print()

    print(
        "COMPLETE"
    )



if __name__ == "__main__":

    main()
