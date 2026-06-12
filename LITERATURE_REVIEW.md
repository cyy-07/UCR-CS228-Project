# Literature Review: Where Does DPMixer Stand?

Comprehensive review of prior work for each component / extension of our
DPMixer family. **Read this before claiming novelty in the report.**

---

## DPMixer base — the 4 branches

| Branch | Closest prior art | Verdict |
|--------|-------------------|---------|
| 1. Persistence anchor | **NLinear** (Zeng et al., AAAI 2023; arXiv:2205.13504) — subtracts and re-adds `x[:, -1]` | OVERLAPS exactly |
| 2. Target DLinear decomposition | **DLinear** (Zeng et al., AAAI 2023; arXiv:2205.13504) | OVERLAPS, restricted to 1 channel |
| 3. Channel-mixer (cross-variate) | **TSMixer** feature-mixing layer (Chen et al., TMLR 2023; arXiv:2303.06053) | OVERLAPS, single-layer slice |
| 4. Time-mixer (cross-time) | **TSMixer** time-mixing layer (Chen et al., TMLR 2023) | OVERLAPS, single-layer slice |

**Closest *combined* prior work**: **TimeMixer** (Wang et al., ICLR 2024;
arXiv:2405.14616) combines decomposition + mixing, but uses *multi-scale*
PDM (Past-Decomposable-Mixing) blocks. DPMixer is intentionally a
*single-scale, additive minimum* of the same family.

**Honest framing**: "DPMixer is a deliberately minimal additive ensemble of
NLinear + DLinear + two single-layer TSMixer slices." Pitch as a
benchmark instrument, not as a novel architecture.

---

## Idea A — Time-Conditioned Branch Gating (TC-DPMixer)

**The damaging-but-survivable prior work**:
- **MoLE: Mixture-of-Linear-Experts for Long-term Time Series Forecasting**
  (Ni et al., AISTATS 2024; arXiv:2312.06786). A two-layer MLP router
  consumes a timestamp embedding and outputs per-expert weights over
  DLinear-style experts. Same structural construction as our TC-DPMixer.

**Defensible novelty angles** for TC-DPMixer:
1. **Heterogeneous branches.** MoLE routes over *homogeneous* DLinear
   experts. We route over *heterogeneous* inductive-bias branches
   (persistence, decomposition, channel-mixer, time-mixer). The gate
   values are therefore directly interpretable as *which inductive bias
   the data prefers*, not just *which expert was elected*.
2. **Forecast-step conditioning.** MoLE conditions on the input's
   *starting* timestamp; we condition on the time features at the
   forecast step (last input step). For short-horizon forecasting this is
   a more aligned signal.
3. **24-h schedule visualization.** We provide the first published
   interpretability artifact of this kind — plotting branch gates as a
   function of hour-of-day produces an autonomous schedule of when each
   inductive bias is most trusted.

**Other related**:
- **TFT** (Lim et al., 2021) — input-conditioned variable selection gates.
- **WaveNet** (van den Oord 2016) — gated activations per-unit (not
  per-branch).
- **GateTS** (arXiv:2508.17515, 2025), **FreqMoE** (arXiv:2501.15125,
  2025) — recent MoE-style TSF, frequency-band experts.

**Do-not-claim warning**: do *not* say "timestamp-conditioned routing over
forecasting sub-models is novel" — MoLE owns that. Say
"MoLE-style routing **extended to a heterogeneous additive mixer with
forecast-horizon conditioning**."

---

## Idea B — Quantile Output (Q-DPMixer)

**Verdict**: TRIVIAL EXTENSION (engineering, not research novelty).

Canonical prior work:
- **DeepAR** (Salinas et al., 2020; arXiv:1704.04110) — autoregressive
  RNN, parametric distributional outputs.
- **TFT** (Lim et al., 2021) — explicitly trained on quantile loss
  over q ∈ {0.1, 0.5, 0.9}, exactly our setup.
- **MQ-RNN/MQ-CNN** (Wen et al., 2017; arXiv:1711.11053) — direct
  multi-quantile MLP heads.
- **SQF-RNN** (Gasthaus et al., 2019), **Implicit Quantile Networks**
  (Gouttes et al., 2021; arXiv:2107.03743).

Existing libraries (Darts, NeuralForecast) already ship TSMixer+quantile
heads. We include Q-DPMixer **only as a documented negative result** —
quantile loss slightly raises MAE under right-skewed targets because the
P50 ≠ mean-optimal predictor.

---

## Idea C — Two-Stream Spike-Aware (SS-DPMixer)

**Prior literatures**:
1. **NILM (Non-Intrusive Load Monitoring)**:
   - **Sequence-to-Point NILM** (Zhang et al., AAAI 2018; arXiv:1612.09106)
     — same classify+magnitude head pattern, but for *disaggregation* (not
     forecasting).
   - Event-based NILM (arXiv:2009.02656, 2020).
2. **Electricity Price Forecasting (EPF) spike modeling**:
   - Lago / Marcjasz line — classical two-stage normal-vs-spike
     decomposition.
3. **Dual-stream / dual-decoder forecasters**:
   - **DSAT-HD** (arXiv:2509.24800, 2025) — seasonal vs trend dual stream.

**Defensible novelty**: applying NILM-style classify+magnitude auxiliary
heads to a *future spike indicator* on top of a base-load forecaster, on
the UCI Appliances dataset specifically. Frame as
"NILM-inspired forecasting auxiliary heads."

**Do-not-claim warning**: do *not* claim "dual-head classify+regression
on energy spikes" is novel without citing Zhang 2018 and the EPF
spike-modeling line.

---

## A + C combined (TC-SS-DPMixer)

This is our **defensible novel contribution** — combining time-conditioned
routing with NILM-inspired spike streams has not (to our knowledge) been
published. The two compose cleanly: the gate can learn to trust the
spike stream around meal-time hours and the base stream overnight.

---

## Recommended report citations (10 max)

1. **Candanedo, Feyt, Deramaix** (2017). Dataset paper. *Energy and
   Buildings* 140. DOI:10.1016/j.enbuild.2017.01.083.
2. **Zeng et al.** (2023). DLinear / NLinear. AAAI. arXiv:2205.13504.
3. **Chen et al.** (2023). TSMixer. TMLR. arXiv:2303.06053.
4. **Wang et al.** (2024). TimeMixer. ICLR. arXiv:2405.14616.
5. **Ni et al.** (2024). MoLE. AISTATS. arXiv:2312.06786. ← *closest TC prior*
6. **Lim et al.** (2021). TFT. *Int. J. Forecasting*.
7. **Oreshkin et al.** (2020). N-BEATS. ICLR. arXiv:1905.10437.
8. **Nie et al.** (2023). PatchTST. ICLR. arXiv:2211.14730.
9. **Liu et al.** (2024). iTransformer. ICLR. arXiv:2310.06625.
10. **Zhang et al.** (2018). Sequence-to-Point NILM. AAAI. arXiv:1612.09106.
   ← *closest SS prior*
