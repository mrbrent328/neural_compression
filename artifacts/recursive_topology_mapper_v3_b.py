#!/usr/bin/env python3

"""
RECURSIVE TOPOLOGY EMERGENCE V4

Tests:

    bytes
      ↓
    level 1 byte patterns
      ↓
    level 1 symbols
      ↓
    level 2 patterns of patterns
      ↓
    recursive structural representation

No data analysis libraries.
Only Python standard library.
"""


from collections import defaultdict


# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

MIN_PATTERN = 3
MIN_OCCURRENCES = 2
MAX_LEVELS = 4


# ------------------------------------------------------------
# SYMBOL HELPERS
# ------------------------------------------------------------

def byte_symbol(b):
    return ("BYTE", b)


def pattern_symbol(level, ident):
    return (f"L{level}", ident)



# ------------------------------------------------------------
# PATTERN OBJECT
# ------------------------------------------------------------

class PatternDB:

    def __init__(self):
        self.patterns = {}
        self.next_id = defaultdict(int)


    def add(self, level, sequence):

        ident = self.next_id[level]
        self.next_id[level] += 1

        key = (f"L{level}", ident)

        self.patterns[key] = tuple(sequence)

        return key



    def expand(self, symbol):

        kind = symbol[0]

        if kind == "BYTE":
            return [symbol[1]]

        if kind.startswith("L"):
            result = []

            for child in self.patterns[symbol]:
                result.extend(self.expand(child))

            return result

        raise Exception("Unknown symbol")



# ------------------------------------------------------------
# FIND REPEATED SYMBOL PATHS
# ------------------------------------------------------------

def find_patterns(stream, db, level):

    candidates = {}

    length = len(stream)

    for size in range(MIN_PATTERN, length//2 + 1):

        counts = defaultdict(list)

        for i in range(length-size+1):

            chunk = tuple(stream[i:i+size])

            counts[chunk].append(i)


        for chunk, positions in counts.items():

            if len(positions) >= MIN_OCCURRENCES:

                saving = (
                    len(chunk) * len(positions)
                    -
                    (len(positions)+1)
                )

                if saving > 0:

                    candidates[chunk] = positions


    created = []

    for seq, positions in candidates.items():

        symbol = db.add(level, seq)

        created.append(
            (
                symbol,
                seq,
                positions
            )
        )


    return created



# ------------------------------------------------------------
# APPLY PATTERNS
# ------------------------------------------------------------

def replace_patterns(stream, patterns):

    if not patterns:
        return stream


    # largest first
    patterns = sorted(
        patterns,
        key=lambda x: len(x[1]),
        reverse=True
    )


    output = []

    i = 0

    while i < len(stream):

        replaced = False

        for symbol, seq, _ in patterns:

            size = len(seq)

            if tuple(stream[i:i+size]) == seq:

                output.append(symbol)

                i += size

                replaced = True
                break


        if not replaced:

            output.append(stream[i])
            i += 1


    return output



# ------------------------------------------------------------
# RUN ONE LEVEL
# ------------------------------------------------------------

def run_level(stream, db, level):

    print()
    print("="*70)
    print(f"LEVEL {level}")
    print("="*70)


    patterns = find_patterns(
        stream,
        db,
        level
    )


    if not patterns:

        print("No patterns discovered")
        return stream


    print()
    print("DISCOVERED:")

    for symbol, seq, pos in patterns:

        print()
        print(symbol)
        print("  contains:")

        for x in seq:
            print("   ", x)

        print("  occurrences:", pos)


    new_stream = replace_patterns(
        stream,
        patterns
    )


    print()
    print("NEW STREAM:")

    print(new_stream)


    return new_stream



# ------------------------------------------------------------
# VERIFICATION
# ------------------------------------------------------------

def verify(original, stream, db):

    rebuilt = []

    for symbol in stream:

        rebuilt.extend(
            db.expand(symbol)
        )


    return rebuilt == original



# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    print("="*70)
    print("RECURSIVE TOPOLOGY EMERGENCE V4")
    print("="*70)


    filename = input(
        "\nFile to test: "
    )


    with open(filename,"rb") as f:

        data = list(f.read())


    print()
    print("RAW BYTES:")
    print(len(data))


    stream = [
        byte_symbol(x)
        for x in data
    ]


    db = PatternDB()


    print()
    print("LEVEL 0")
    print(stream[:50])


    history = []


    for level in range(1,MAX_LEVELS+1):

        old_length = len(stream)


        stream = run_level(
            stream,
            db,
            level
        )


        history.append(
            len(stream)
        )


        print()
        print(
            "SYMBOL COUNT:",
            old_length,
            "->",
            len(stream)
        )


        if len(stream) >= old_length:

            print(
                "No further structural reduction"
            )

            break



    print()
    print("="*70)
    print("FINAL VERIFICATION")
    print("="*70)


    ok = verify(
        data,
        stream,
        db
    )


    print(
        "EXACT:",
        ok
    )


    print()
    print("COST EVOLUTION")

    print(
        "RAW:",
        len(data)
    )


    for i,x in enumerate(history):

        print(
            f"L{i+1}:",
            x
        )


    print()
    print("PATTERN DATABASE SIZE:")

    print(
        len(db.patterns)
    )



if __name__ == "__main__":
    main()
