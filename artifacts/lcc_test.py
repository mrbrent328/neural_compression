import neural_compress_opt.py
import hashlib

line_break = "-----------------------------------"



define checkSum(objectToCheck):
    # Open the file in read-binary mode ('rb')
    with open(objectToCheck, "rb") as f:
        # Use 'sha256' or 'md5' depending on your needs
        digest = hashlib.file_digest(f, "sha256")

    # Get the hexadecimal representation of the checksum
    checksum = digest.hexdigest()
    print(checksum)


running = True
file_to_process = None

while (running == True):
    HSA_OVERRIDE_GFX_VERSION=10.3.0 python3 neural_compress_opt.py compress
