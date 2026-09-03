from E40_adaptive_residual_codec import encode, decode

def test_exact_roundtrip():
    for data in (b"",bytes(range(256))*20,b"AAAAABBBBBCCCCCDDDD"*1000,bytes((i*37+11)%256 for i in range(10000))):
        blob=encode(data,region_size=257); assert decode(blob)==data

def test_trailing_bytes_rejected():
    blob=encode(b"persistent structure"*100)
    try: decode(blob+b"x")
    except ValueError as e: assert "trailing" in str(e)
    else: raise AssertionError("trailing bytes accepted")
