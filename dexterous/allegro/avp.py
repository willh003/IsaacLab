from avp_stream import VisionProStreamer
avp_ip = "192.168.0.50"   # example IP 
print(f"Connecting to AVP at {avp_ip}")
s = VisionProStreamer(ip = avp_ip, record = True)
print(f"Connected to AVP at {avp_ip}")

while True:
    r = s.latest
    print(r['head'], r['right_wrist'], r['right_fingers'])