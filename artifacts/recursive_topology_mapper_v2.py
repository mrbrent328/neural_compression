#!/usr/bin/env python3

"""
RECURSIVE TOPOLOGY MAPPER V1

Experimental structural representation system.

Tests:

RAW BYTES
    |
    v
LEVEL 1
    repeated byte regions
    |
    v
LEVEL 2
    patterns of patterns

No external libraries.
No statistical analysis.
Exact reconstruction required.
"""

import sys
from collections import defaultdict


# ============================================================
# SETTINGS
# ============================================================

MIN_PATTERN = 4
MAX_PATTERN = 32
MIN_OCCURRENCES = 2


# ============================================================
# PATTERN OBJECT
# ============================================================

class Pattern:
    def __init__(self, pid, sequence, level):
        self.id = pid
        self.sequence = sequence
        self.level = level
        self.locations = []

    def symbol(self):
        return f"L{self.level}:P{self.id}"



# ============================================================
# FIND REPEATED SEQUENCES
# ============================================================

def discover_patterns(stream, level):

    patterns = {}

    counter = 1000

    found = defaultdict(list)

    length_range = range(
        MAX_PATTERN,
        MIN_PATTERN - 1,
        -1
    )

    for size in length_range:

        for i in range(len(stream) - size + 1):

            chunk = tuple(stream[i:i+size])

            found[chunk].append(i)


    for seq, locations in found.items():

        if len(locations) >= MIN_OCCURRENCES:

            # avoid adding patterns inside bigger ones
            contained = False

            for existing in patterns.values():

                if (
                    len(existing.sequence) > len(seq)
                    and seq in existing.sequence
                ):
                    contained = True
                    break

            if contained:
                continue


            p = Pattern(
                counter,
                seq,
                level
            )

            p.locations = locations

            patterns[counter] = p

            counter += 1


    return patterns



# ============================================================
# ENCODE STREAM USING PATTERNS
# ============================================================

def encode(stream, patterns):

    result = []

    i = 0


    ordered = sorted(
        patterns.values(),
        key=lambda p: len(p.sequence),
        reverse=True
    )


    while i < len(stream):

        matched = False


        for pattern in ordered:

            size = len(pattern.sequence)

            if tuple(stream[i:i+size]) == pattern.sequence:

                result.append(
                    ("PATTERN", pattern.id)
                )

                i += size

                matched = True
                break


        if not matched:

            result.append(
                ("RAW", stream[i])
            )

            i += 1


    return result



# ============================================================
# EXPAND SYMBOLS
# ============================================================

def expand(encoded, patterns):

    output = []


    lookup = {}

    for p in patterns.values():

        lookup[p.id] = p.sequence


    for item in encoded:

        if item[0] == "RAW":

            output.append(item[1])


        elif item[0] == "PATTERN":

            seq = lookup[item[1]]

            output.extend(
                expand(
                    [("RAW", x) for x in seq],
                    {}
                )
                if False else seq
            )


    return list(output)



# ============================================================
# LEVEL 2
# ============================================================

def discover_symbol_patterns(encoded):

    symbols = encoded

    return discover_patterns(
        symbols,
        2
    )



# ============================================================
# PRINT HELPERS
# ============================================================

def show_patterns(title, patterns):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


    if not patterns:
        print("None")
        return


    for p in patterns.values():

        print()
        print(
            f"L{p.level}:P{p.id}"
        )

        print(
            " sequence:"
        )

        print(
            "   ",
            p.sequence
        )

        print(
            " occurrences:"
        )

        print(
            "   ",
            p.locations
        )



# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) < 2:

        print(
            "usage: python recursive_topology_mapper_v1.py file"
        )

        return


    filename = sys.argv[1]


    with open(filename, "rb") as f:
        data = list(f.read())


    print("=" * 70)
    print("RECURSIVE TOPOLOGY MAPPER V1")
    print("=" * 70)

    print()
    print("FILE:")
    print(filename)

    print()
    print("RAW SIZE:")
    print(len(data),"bytes")


    # ----------------------------
    # LEVEL 1
    # ----------------------------

    level1 = discover_patterns(
        data,
        1
    )


    show_patterns(
        "LEVEL 1 — BYTE STRUCTURES",
        level1
    )


    encoded1 = encode(
        data,
        level1
    )


    print()
    print("LEVEL 1 REPRESENTATION:")
    print(encoded1[:50])


    rebuilt1 = expand(
        encoded1,
        level1
    )


    print()
    print(
        "LEVEL 1 EXACT:",
        rebuilt1 == data
    )


    # ----------------------------
    # LEVEL 2
    # ----------------------------


    level2 = discover_symbol_patterns(
        encoded1
    )


    show_patterns(
        "LEVEL 2 — PATTERNS OF PATTERNS",
        level2
    )


    encoded2 = encode(
        encoded1,
        level2
    )


    print()
    print("LEVEL 2 REPRESENTATION:")
    print(encoded2[:50])


    print()
    print("STRUCTURAL COST")

    print(
        "Raw symbols:",
        len(data)
    )

    print(
        "Level 1 symbols:",
        len(encoded1)
    )

    print(
        "Level 2 symbols:",
        len(encoded2)
    )


    print()
    print("=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)



if __name__ == "__main__":
    main()
