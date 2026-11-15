// Where the models and RoadSim are known to be weak, per vehicle. Numbers from shared/vehicle/roadsim/LIMITS.md and the
// three model cards (models/*/model.yaml, v2.0, 14 Nov 2026). Real-data numbers; the page itself compares with RoadSim.
export interface Note { who: 'RoadSim' | 'EnergyNet' | 'TripDetective' | 'HybridPilot'; text: string }

export const COMMON: string[] = [
  'On this page the truth is RoadSim, not the real car: every reading is compared with the simulator run that produced the log. The exams on the model pages compare with real data.',
  'The routes are real exam trips (VED, Ann Arbor 2017–18; ZTBus, Zurich). The vehicles driving them are simulated. The exams were already taken; nothing here chose anything.',
  'RoadSim runs precomputed in Python (physics v2, pinned) for 3 air temperatures × 2 loads. The three networks run in your browser on every change.',
];

export const NOTES: Record<string, Note[]> = {
  sedan: [
    { who: 'RoadSim', text: 'Conventional cars are its best case: held-out EPA FTP +0.4 %, highway −1.9 % (median). But one of ten benchmark maps stands in for each engine, so single cars scatter (FTP IQR 12 points).' },
    { who: 'EnergyNet', text: 'Petrol cars are a win on real trips (8.7 % per trip). A/C is not: EPA SC03 tests (hot, air conditioning on) read 18 % low, so expect the 35 °C forecast to under-call the air conditioning.' },
    { who: 'TripDetective', text: 'Mass within 127 kg per real car; CdA and rolling resistance are weakly identified (CdA × 1.5 moves its estimate by 3 %).' },
  ],
  prius: [
    { who: 'RoadSim', text: 'Toyota-type twins run the engine 4–10 points less than the real cars on VED roads, and older Toyota/Lexus hybrids are 15–20 % too frugal on the EPA city test (the TNGA-era engine map flatters them). Cold FTP −16 %.' },
    { who: 'EnergyNet', text: 'Hybrids now win on real trips (13.5 % per trip), with a +3.9 % bias.' },
    { who: 'TripDetective', text: 'Its type call is its weak spot: it called 10 of 86 real exam hybrids conventional, mostly small Prius-c-type cars. Per-car mass is worst on hybrids (181 kg).' },
    { who: 'HybridPilot', text: '+0.89 points of the DP optimum over A-ECMS on exam power-splits. DP ignores the forced-on rules (engine on above 45 km/h, warm-up holds), so no causal controller reaches 100 %.' },
  ],
  p2: [
    { who: 'RoadSim', text: 'A generated car, not a real one. P2 physics is split on fresh data: Hyundai/Kia-type P2s burn 8–10 % too much, converter P2s 26–28 % too little in the city. Launch engine starts are not modelled at 1 Hz.' },
    { who: 'EnergyNet', text: 'The 247 real P2 trips got worse in v2 (17.2 % per trip, v1.0 10.4 %).' },
    { who: 'TripDetective', text: 'Hybrids are where its type call and its mass are weakest (see the power-split).' },
    { who: 'HybridPilot', text: '+1.32 points over A-ECMS on exam P2s, but the interval is wide ([+0.37, +2.85]). Above 100 % of DP means the DP curve was extrapolated to that final charge.' },
  ],
  volt: [
    { who: 'RoadSim', text: 'Plug-ins are its weakest group: series plug-in twins run the engine 7 % of the time against 36 % on the dynamometer, and VED exam plug-ins come out 20–23 % low.' },
    { who: 'EnergyNet', text: 'Plug-ins win on real trips (17.1 %) only because it is told the start charge; plug-in car totals got worse in v2 (20.1 %). The start charge here is low (30 %, near charge sustaining).' },
    { who: 'TripDetective', text: 'Reads the battery, fuel-rate and A/C channels a Volt logs; plug-in mass and type were not graded separately.' },
    { who: 'HybridPilot', text: 'Loses to A-ECMS on exam plug-ins (series −0.35 points, inside a wide interval). Simulated plug-ins on real roads read 22–31 % low in fuel.' },
  ],
  leaf: [
    { who: 'RoadSim', text: 'Modern BEVs match the EPA tests (held-out UDDS −0.4 %), but 2012–15 BEVs like this Leaf used 10–17 % more on the dynamometer than their twins.' },
    { who: 'EnergyNet', text: 'Its loss: the two real exam Leafs read 19.5 % low and lose to a MOVES-like baseline; only 43 % fall inside the 68 % band.' },
    { who: 'TripDetective', text: 'The two exam Leafs read within 105–119 kg; the dev Leaf’s real battery power fits a lighter car than its twin (0.83×).' },
  ],
  bus: [
    { who: 'RoadSim', text: 'The trolleybus twin is −2.3 % on exam legs, but its 19 t curb mass is a convention: the logged power and traction force both regress to ~0.7 × (19 t + passengers). Heating saturates at stops (24 kW standing vs ~9 kW logged).' },
    { who: 'EnergyNet', text: 'Wins on full ZTBus missions (9.5 %) knowing only "bus, electric": the passengers you add never reach it, so its forecast does not move.' },
    { who: 'TripDetective', text: 'A separate bus network that loses to the dev-median guess (1,204 vs 453 kg per leg) and follows passengers weakly (slope 0.70). Its reading is shown, not trusted.' },
  ],
};
