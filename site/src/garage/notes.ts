// Where the models and RoadSim are known to be weak, per vehicle. Numbers from shared/vehicle/roadsim/LIMITS.md and the
// three model cards (models/*/model.yaml, v2.0, 14 Nov 2026). Real-data numbers; the page itself compares with RoadSim.
export interface Note { who: 'RoadSim' | 'EnergyNet' | 'TripDetective' | 'HybridPilot'; text: string }

export const COMMON: string[] = [
  'On this page the truth is RoadSim, not the real car: every reading is compared with the simulator run that produced the log. The exams on the model pages compare with real data.',
  'The routes are real exam trips (VED, Ann Arbor 2017–18; ZTBus, Zurich). The vehicles driving them are simulated. The exams were already taken; nothing here chose anything.',
  'RoadSim runs precomputed in Python (physics v3.1, pinned) for 3 air temperatures × 2 loads. The three networks run in your browser on every change.',
];

export const NOTES: Record<string, Note[]> = {
  sedan: [
    { who: 'RoadSim', text: 'Conventional cars are its best case: held-out EPA FTP +0.4 %, highway −1.9 % (median). But one of ten benchmark maps stands in for each engine, so single cars scatter (FTP IQR 12 points).' },
    { who: 'EnergyNet', text: 'Petrol cars are a win on real trips (8.6 % per trip). EPA SC03 tests (hot, air conditioning on) are now a win too (8.6 %, still 6 % low; v2.0 read them 18 % low).' },
    { who: 'TripDetective', text: 'Mass within 127 kg per real car; CdA and rolling resistance are weakly identified (CdA × 1.5 moves its estimate by 3 %).' },
  ],
  prius: [
    { who: 'RoadSim', text: 'Toyota-type twins run the engine 4–10 points less than the real cars on VED roads, and older Toyota/Lexus hybrids are 15–20 % too frugal on the EPA city test (the TNGA-era engine map flatters them). Cold FTP −16 %.' },
    { who: 'EnergyNet', text: 'Hybrids win on real trips (13.4 % per trip), with a +3.8 % bias.' },
    { who: 'TripDetective', text: 'v2.0 called 10 of 86 real exam hybrids conventional, mostly small Prius-c-type cars. v2.1 rules out conventional when the engine stops while moving and ties the simple rule (348 of 350 cars). Per-car mass is worst on hybrids (181 kg).' },
    { who: 'HybridPilot', text: '+0.59 points of the DP optimum over A-ECMS on exam power-splits (physics v3.1). DP ignores the forced-on rules (engine on above 45 km/h, warm-up holds), so no causal controller reaches 100 %.' },
  ],
  p2: [
    { who: 'RoadSim', text: 'A generated car, not a real one. P2 physics is split on fresh data: Hyundai/Kia-type P2s burn 8–10 % too much, converter P2s 26–28 % too little in the city. Launch engine starts are not modelled at 1 Hz.' },
    { who: 'EnergyNet', text: 'The 247 real P2 trips got worse in v2 (18.3 % per trip in v2.1, v1.0 10.4 %).' },
    { who: 'TripDetective', text: 'Hybrids are where its type call and its mass are weakest (see the power-split).' },
    { who: 'HybridPilot', text: '+1.39 points over A-ECMS on exam P2s, but the interval is wide ([+0.31, +3.29]). Above 100 % of DP means the DP curve was extrapolated to that final charge.' },
  ],
  volt: [
    { who: 'RoadSim', text: 'Plug-ins are its weakest group: series plug-in twins run the engine 7 % of the time against 36 % on the dynamometer, and VED exam plug-ins come out 20–23 % low.' },
    { who: 'EnergyNet', text: 'Plug-ins win on real trips (16.8 %) only because it is told the start charge; plug-in car totals got worse in v2 (20.6 % in v2.1). The start charge here is low (30 %, near charge sustaining).' },
    { who: 'TripDetective', text: 'Reads the battery, fuel-rate and A/C channels a Volt logs; plug-in mass and type were not graded separately.' },
    { who: 'HybridPilot', text: 'Loses to A-ECMS on exam plug-ins (series −0.73 points [−1.78, −0.07], power-split −0.91 inside a wide interval). v2.2 keeps s at the hybrid level once a plug-in reaches its charge-sustaining band: charge held on 102 of 104 trips, fuel no better. Simulated plug-ins on real roads read 20–31 % low in fuel.' },
  ],
  leaf: [
    { who: 'RoadSim', text: 'Modern BEVs match the EPA tests (held-out UDDS −0.4 %), but 2012–15 BEVs like this Leaf used 10–17 % more on the dynamometer than their twins.' },
    { who: 'EnergyNet', text: 'Its loss: the two real exam Leafs read 20 % low and lose to a MOVES-like baseline. The one dev Leaf reads 5 % high, so dev data cannot show this bias.' },
    { who: 'TripDetective', text: 'The two exam Leafs read within 105–119 kg; the dev Leaf’s real battery power fits a lighter car than its twin (0.83×).' },
  ],
  bus: [
    { who: 'RoadSim', text: 'The trolleybus twin is −2.3 % on exam legs, but its 19 t curb mass is a convention: the logged power and traction force both regress to ~0.7 × (19 t + passengers). Heating saturates at stops (24 kW standing vs ~9 kW logged).' },
    { who: 'EnergyNet', text: 'Wins on full ZTBus missions (5.6 %) knowing only "bus, electric" and the published 19 t curb mass: the passengers you add never reach it, so its forecast does not move.' },
    { who: 'TripDetective', text: 'A separate bus network; the garage shows its per-leg reading. The released v2.1 estimate reads a whole day’s legs together and narrowly beats the dev-median guess (444 vs 453 kg per leg), but follows passengers weakly (slope 0.15). Shown, not trusted.' },
  ],
};
