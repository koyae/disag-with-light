# scripts/test_plugs.py
import requests
import time

HA_URL   = "http://homeassistant.local:8123"
HA_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI4YmZmNjJkNDA3ZDg0OGMzYWNlOWQwZDkwNDlmYmU3OCIsImlhdCI6MTc3NjgwMjkxNiwiZXhwIjoyMDkyMTYyOTE2fQ.deCyR50AtJKF2n1_rt_Dx0mcx4wtYoe5d5wLO_neTxM"

PLUGS = [
    "switch.lumi_lumi_plug_maus01",
    "switch.maus02_lumi_lumi_plug",
    "switch.maus03_lumi_lumi_plug",
    "switch.maus04_lumi_lumi_plug",
]

HEADERS = {
    "Authorization": f"Bearer {HA_TOKEN}",
    "Content-Type": "application/json",
}

def turn_on(entity_id):
    r = requests.post(
        f"{HA_URL}/api/services/switch/turn_on",
        headers=HEADERS,
        json={"entity_id": entity_id},
    )
    print(f"  ON  {entity_id} → {r.status_code}")

def turn_off(entity_id):
    r = requests.post(
        f"{HA_URL}/api/services/switch/turn_off",
        headers=HEADERS,
        json={"entity_id": entity_id},
    )
    print(f"  OFF {entity_id} → {r.status_code}")

print("Turning all plugs ON...")
for plug in PLUGS:
    turn_on(plug)
    time.sleep(1)

print("\nWaiting 3 seconds...")
time.sleep(3)

print("\nTurning all plugs OFF...")
for plug in PLUGS:
    turn_off(plug)
    time.sleep(1)

print("\nDone!")