#!/usr/bin/env python3

import argparse

psr = argparse.ArgumentParser()
psr.add_argument("-o", dest="opt", type=str, help="output file")
# psr.add_argument("ipt", type=str, help="input file")
psr.add_argument("-i", dest="ipt", type=str, help="input file")
# psr.add_argument("-p", dest="parts", type=int, default=5, help="number of partitions to divide")
args = psr.parse_args()

import h5py as h5
import numpy as np
import uproot
from JPwaptool import JPwaptool

fipt = args.ipt
fopt = args.opt

with uproot.open(fipt) as ipt:
    eventIds = ipt["Readout/TriggerNo"].array(library='np')
    waveforms = ipt["Readout/Waveform"].array(library='np')
    channelIds = ipt["Readout/ChannelId"].array(library='np')
eids = []
chs = []
offsets = []
l_waves = []
segments = []
begin = 3005
end = 6005
for i in range(len(eventIds)):
    eid = eventIds[i]
    ch = channelIds[i][-1]
    w = waveforms[i]
    lw = end - begin
    eids.append(eid)
    chs.append(ch)
    offsets.append(begin)
    l_waves.append(lw)
    segments.append(w[begin:end])
print('finishh readout')    
waveform_processor_pool = {
    lw: JPwaptool(lw, min(100, lw), lw - min(100, lw))
    for lw in (x.item() for x in np.unique(l_waves))
}

opt_dtype = [
    ("eid", np.uint32),
    ("ch", np.uint16),
    ("offset", np.uint16),
    ("l_wave", np.uint16),
    ("baseline", np.float64),
    ("sig2w", np.float64),
    ("charge", np.float64),
    ("segment", (np.uint16, end-begin)),
]

opts = {"compression": "gzip", "shuffle": True}
with h5.File(fopt, "w") as opt:
    part = 0
    idx = np.arange(len(eids))
    l_part = np.take(l_waves, idx)
    opt_dtype[-1] = ("segment", (np.uint16, np.max(l_part)))
    opt_data = np.full(len(eids), 0xFFFF, dtype=opt_dtype)
    opt_data["eid"] = eids
    opt_data["ch"] = chs
    opt_data["offset"] = offsets
    opt_data["l_wave"] = l_part
    print("start to save")
    for i, (l_wave, j) in enumerate(zip(l_part, idx)):
        waveform_processor = waveform_processor_pool[l_wave]
        waveform_processor.FastCalculate(segments[j])
        opt_data[i]["baseline"] = waveform_processor.ChannelInfo.Ped
        opt_data[i]["sig2w"] = waveform_processor.ChannelInfo.PedStd**2
        opt_data[i]["charge"] = waveform_processor.ChannelInfo.Charge
        opt_data[i]["segment"][:l_wave] = segments[j]
    opt.create_dataset(f"{part}", data=opt_data, **opts)
    opt.attrs["parts"] = part
