#!/usr/bin/env python3
import random
from rainbow.devices.stm32 import rainbow_stm32f215 as rainbow_stm32
from rainbow import TraceConfig, HammingWeight
import numpy as np
from lascar import TraceBatchContainer, Session, NicvEngine
from rainbow.utils.plot import viewer
import pandas as pd


def containsPin(e, pin_attempt, stored_pin):
    """ Handle calling the pin comparison function using the emulator """
    e.reset()

    stor_pin = 0x08008110 + 0x189  # address of the storagePin->rom
    e[stor_pin] = bytes(stored_pin + "\x00", "ascii")

    input_pin_addr = 0xcafecafe
    e[input_pin_addr] = bytes(pin_attempt + "\x00", "ascii")

    e['r0'] = input_pin_addr
    e['lr'] = 0xaaaaaaaa

    e.start(e.functions['storage_containsPin'], 0xaaaaaaaa)




# N = 150000
N=1500
print("Setting up emulator")
e = rainbow_stm32(trace_config=TraceConfig(register=HammingWeight(), instruction=True))
e.load("trezor.elf")
e.setup()

print("Generating", N, "traces")

traces = []
for i in range(N):
    if i % 1000 == 0: print(i)
    input_pin = "".join(random.choice("123456789") for _ in range(4))
    STORED_PIN = "".join(random.choice("123456789") for _ in range(4))
    containsPin(e, input_pin, STORED_PIN)
    # print(([ord(x) for x in input_pin + STORED_PIN], dtype=np.uint8))
    # values.append(np.array([ord(x) for x in input_pin + STORED_PIN], dtype=np.uint8))
    register_values = np.array([event["register"] for event in e.trace if "register" in event])
    # traces.append(np.array([event["register"] for event in e.trace if "register" in event])) #46 values for each iteration
    # traces.append(np.array([event for event in e.trace]))
    traces.append(register_values.tolist()  + [input_pin, STORED_PIN])
    print(traces)
    # traces.append(input_pin)
    # traces.append(STORED_PIN)
    
# print(traces)
# df = pd.DataFrame(traces)
# df.to_csv("temptraces.csv")
# print("Saved traces to csv")