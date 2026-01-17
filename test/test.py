import torch.nn as nn

import brain_modules.ANNs.GPT as v1
import brain_modules.ANNs.GPTv2Worse as v2
from brain_modules.ANNs.GRU import GRU


class GPT_GRU(nn.Module):
    def __init__(s, c: v1.GPTConf):
        super().__init__()
        s.emb = v1.TextEmb(c)
        s.gru = GRU(c.emb_dim, c.emb_dim, 128, 1)

    def forward(s, idx):
        return s.emb.out(s.gru(s.emb.emb(idx)))


# v1.test_GPT(v1.GPT)
# v1.test_GPT(v2.GPTv2)
v1.test_GPT(GPT_GRU)
# v2.test_correctness()

"""
v1.GPT
4900     loss: 0.2785    y: d for his treatment of federal government employees,[307][308][      yp:   tor tis treatment of federal government empooyee",[307][308][
4910     loss: 0.2535    y: rence in Vancouver, Musk stated that "the social cues were not       yp:   dce in Mancouver, Musk stated that hthe social cues were not
4920     loss: 0.2599    y:  small amount once every other week or something like that";[41      yp: nsaall ruount once ivery other week or something like that";[41
4930     loss: 0.2586    y: issed the accusations of Nazi sympathies, deriding them as "dir      yp: pcsia the accosations of Nazi sympathies, deriding them as "dir
4940     loss: 0.2789    y:  (equivalent to $2,300,000,000 in 2024) for Falcon 9-launched D      yp: et2quivalent to $2,600,000,000 in 2024).for Falcon 9-launched D
4950     loss: 0.2557    y: pts to become CEO were thwarted by the board.[62] Compaq acquir      yp: eae th be ome CEO were thwarted by the board.[62] Compaq acquir
4960     loss: 0.2539    y: he founded in 2001,[499][500] whose stated purpose is to provid      yp: te mornded in 2011, 499][500] whose stated purpose is to provid
4970     loss: 0.2646    y:  father berating him after he was discharged from the hospital.      yp:  aather oehating him after he was doscharged from the hospital.
4980     loss: 0.2586    y: eased on the platform after his takeover.[189][190][191][192] I      yp: est d an the tlatform after his takeover.[189][180][191] 182] I
4990     loss: 0.2318    y: 04] Trump said two days later that he had put Musk in charge of      yp: ]]] Ihump sait two days later that he had put Musk in charge of
"""
