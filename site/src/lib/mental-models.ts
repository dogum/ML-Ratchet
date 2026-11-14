// Content for the "Mental models" page (site/src/pages/mental-models.astro).
// The wiring-diagram coordinates were tuned by hand; move nodes and edges together or not at all.
// Numbers that live in model.yaml / ratchet.yaml are filled in by the page at build time, not here.

export type Kind = 's' | 'r' | 'k' | '';

// ---- 02 · the wiring diagram -------------------------------------------------------------------
export interface FlowNode { x: number; w: number; y: number; t: string; s: string; k: Kind }
export interface FlowEdge { d: string; k: Kind; lx?: number; ly?: number; a?: 'start' | 'middle' | 'end' }

export const NODES: Record<string, FlowNode> = {
  phy: { x: 20, w: 130, y: 96, t: 'Physics', s: 'equations', k: 's' },
  sim: { x: 200, w: 140, y: 96, t: 'Simulator', s: 'forward model', k: 's' },
  sdat: { x: 390, w: 140, y: 96, t: 'Simulated data', s: 'labels are free', k: 's' },
  net: { x: 590, w: 140, y: 96, t: 'Network', s: 'learned weights', k: '' },
  out: { x: 800, w: 160, y: 96, t: 'Reading', s: 'answer + error bar', k: '' },
  dev: { x: 200, w: 140, y: 300, t: 'Dev set', s: 'a few real units', k: 'r' },
  noise: { x: 390, w: 140, y: 300, t: 'Unlabelled real', s: 'no answers attached', k: 'r' },
  train: { x: 590, w: 140, y: 300, t: 'Real labels', s: 'bulk, with answers', k: 'r' },
  exam: { x: 800, w: 160, y: 300, t: 'Exam set', s: 'held out', k: 'r' },
};

export const EDGES: Record<string, FlowEdge> = {
  phy_sim: { d: 'M150 124H196', k: 's' },
  sim_sdat: { d: 'M340 124H386', k: 's' },
  sdat_net: { d: 'M530 124H586', k: 's', lx: 560, ly: 86, a: 'middle' },
  net_out: { d: 'M730 124H796', k: 'k' },
  noise_sdat: { d: 'M460 300V158', k: 'r', lx: 468, ly: 180, a: 'start' },
  dev_sim: { d: 'M250 300V158', k: 'r', lx: 242, ly: 232, a: 'end' },
  // routed under the bottom row and up the gap between "Unlabelled real" and "Real labels", so it crosses no arrow
  dev_gate: { d: 'M300 356C300 394 560 394 560 330V214C560 186 620 190 620 158', k: 'r', lx: 552, ly: 262, a: 'end' },
  train_net: { d: 'M680 300V158', k: 'r', lx: 688, ly: 232, a: 'start' },
  exam_out: { d: 'M880 300V158', k: 'r', lx: 888, ly: 232, a: 'start' },
  phy_net: { d: 'M85 96C85 36 660 36 660 92', k: 's', lx: 372, ly: 44, a: 'middle' },
  sim_out: { d: 'M270 96C270 6 880 6 880 92', k: 's', lx: 575, ly: 20, a: 'middle' },
  noise_sim: { d: 'M420 300C420 220 300 230 296 158', k: 'r', lx: 356, ly: 214, a: 'middle' },
  noise_net: { d: 'M500 300C500 220 640 240 646 158', k: 'r', lx: 580, ly: 262, a: 'middle' },
};

/** Which nodes an edge lights up ("gate" lights the network). */
export const edgeNodes = (id: string) => id.split('_').map((n) => (n === 'gate' ? 'net' : n));

export interface Ref { short: string; full: string; url: string }
export interface Approach {
  key: string;
  chip: string;
  name: string;
  /** edge id -> label shown on it ('' = lit, no label) */
  edges: Record<string, string>;
  one: string;
  physics: string;
  real: string;
  proves: string;
  when: string;
  /** how it relates to this repository; {tokens} are filled by the page */
  repo: string;
}

export const APPROACHES: Approach[] = [
  {
    key: 'guess', chip: 'A common first guess', name: 'Pool simulated and real examples, train on both',
    edges: { phy_sim: '', sim_sdat: '', sdat_net: 'train', train_net: 'train too', net_out: '' },
    one: 'Simulated and real examples go into one training set. The literature calls this synthetic data augmentation or mixed training.',
    physics: 'The simulator that makes part of the training set.',
    real: 'Training rows, side by side with simulated ones.',
    proves: 'That the model fits data like its training data. If real test units resemble real training units, physics may add little, and the test cannot show it.',
    when: 'Real labels exist but are too few on their own, and losing the clean split between teacher and examiner is acceptable.',
    repo: 'Not what this repository does. It is close to stop 5 of the dial in section 07, without the guardrails.',
  },
  {
    key: 'repo', chip: 'This repository', name: 'Train on simulations, grade on reality',
    edges: { phy_sim: '', sim_sdat: '', noise_sdat: 'real noise', sdat_net: 'train', dev_sim: 'calibrate physics', dev_gate: 'gate each rung', net_out: '', exam_out: 'grade once' },
    one: 'Real data has four jobs here, and training labels is none of them. It supplies the noise, tunes the simulator, gates promotion and sets the final grade.',
    physics: 'The simulator and its dev-fitted constants. For some models, also the input features and the form of the output (section 03).',
    real: 'Backdrop noise, simulator calibration and priors, the promotion gate, error-bar width, the final exam.',
    proves: 'How far physics plus learning gets on data the network never saw in any form. A loss is as informative as a win.',
    when: 'Real labels are scarce, costly or untrustworthy, and a credible simulator exists or can be built.',
    repo: 'Research names for this pattern: sim-to-real transfer, simulation-based inference, amortised inversion.',
  },
  {
    key: 'blackbox', chip: 'Black box', name: 'Black-box supervised learning',
    edges: { train_net: 'train', net_out: '', exam_out: 'test' },
    one: 'Collect labelled real data, split it, train, test. No physics anywhere.',
    physics: 'Nowhere. Whatever structure exists, the network has to find it.',
    real: 'Everything: training, validation and test.',
    proves: 'That the model interpolates within the dataset. It can memorise the quirks of one rig, one fleet or one sensor.',
    when: 'Labelled data is plentiful, measured, and covers the conditions the model will be used in.',
    repo: '{blackbox}',
  },
  {
    key: 'finetune', chip: 'Sim pretrain + real fine-tune', name: 'Pretrain on simulations, fine-tune on real data',
    edges: { phy_sim: '', sim_sdat: '', sdat_net: 'pretrain', train_net: 'fine-tune', net_out: '', exam_out: 'test' },
    one: 'Learn the broad physics from unlimited simulated data, then adjust the weights with a smaller set of real labels.',
    physics: 'The pretraining data. Fine-tuning can overwrite it.',
    real: 'A second, smaller training stage, plus testing.',
    proves: 'That simulation plus real data beats either alone, if all three are reported. The test set must be split by unit from the fine-tuning set.',
    when: 'The simulator is good but biased in ways nobody can model, and measured real labels exist in the hundreds or more.',
    repo: 'The most direct way to bring real labels in. Stop 5 of the dial.',
  },
  {
    key: 'residual', chip: 'Physics + learned residual', name: 'Physics baseline plus a learned residual',
    edges: { phy_sim: '', sim_out: 'physics estimate', train_net: 'learn the leftover', net_out: 'correction', exam_out: 'test' },
    one: 'Run the physics model for a first answer. Train a small network on real data to predict only what the physics got wrong.',
    physics: 'The baseline at inference time. The network handles the residual.',
    real: 'The training target for the correction term.',
    proves: 'How much real data improves on physics alone. The physics-only number is the built-in baseline.',
    when: 'Physics gets within 10 to 20% and the remaining error is systematic. Also called hybrid modelling or delta learning.',
    repo: '{residual}',
  },
  {
    key: 'pinn', chip: 'PINN', name: 'Physics-informed neural network (PINN)',
    edges: { phy_net: 'equation residual in the loss', train_net: 'sparse measurements', net_out: '', exam_out: 'test' },
    one: 'The network represents the solution field itself. Its loss penalises breaking the governing equations at sampled points, alongside fitting a few measurements.',
    physics: 'The loss function. No simulator runs at all.',
    real: 'Sparse anchor points and boundary conditions.',
    proves: 'That one specific field was reconstructed. A classic PINN solves one instance and is retrained for the next.',
    when: 'There is a clean PDE, a few sensors, and the full field is wanted: flow, heat, stress.',
    repo: 'This is what "physics-informed machine learning" usually means. No model here uses it.',
  },
  {
    key: 'domrand', chip: 'Domain randomisation', name: 'Domain randomisation',
    edges: { phy_sim: '', sim_sdat: 'very wide parameter ranges', sdat_net: 'train', net_out: '', exam_out: 'test' },
    one: 'Skip calibration. Randomise the simulator’s nuisance parameters so widely that the real world looks like one more sample.',
    physics: 'The simulator, deliberately de-tuned.',
    real: 'Testing only. Nothing real informs training.',
    proves: 'Robustness. Accuracy usually trails a calibrated simulator, because the network hedges across worlds that do not exist.',
    when: 'Real data is nearly impossible to get before deployment. Standard in robot learning.',
    repo: 'When a calibrated simulator made real dev readings worse, SwellCast and StarQuake were released on a mix of old and calibrated simulation banks. That is a mild form of this.',
  },
  {
    key: 'surrogate', chip: 'Surrogate', name: 'Surrogate model, or emulator',
    edges: { phy_sim: '', sim_sdat: 'inputs and outputs', sdat_net: 'learn the forward map', net_out: 'fast prediction' },
    one: 'The network learns to copy the simulator, inputs to outputs, so it can stand in for it at a fraction of the run time.',
    physics: 'The simulator being imitated. The network inherits its errors.',
    real: 'None required. Validation is against the simulator.',
    proves: 'Fidelity to the simulator. That says nothing about reality.',
    when: 'The simulator is trusted but too slow for optimisation, Monte Carlo or embedded use.',
    repo: 'The mirror image of this repository. A surrogate runs the simulator’s direction, cause to signal. The networks here run backwards, signal to cause.',
  },
  {
    key: 'calib', chip: 'Calibration + discrepancy', name: 'Calibration with a discrepancy term',
    edges: { phy_sim: '', dev_sim: 'fit parameters + discrepancy', sim_out: 'prediction' },
    one: 'No network in the loop. Fit the simulator’s parameters to real data and add an explicit statistical term for what the physics cannot reach.',
    physics: 'Everything. The discrepancy term is a small learned patch on top.',
    real: 'Fits the parameters and the discrepancy model.',
    proves: 'The calibrated simulator’s predictive accuracy, with the uncertainty split between parameters and model form.',
    when: 'The simulator is the product. Known as Bayesian calibration, after Kennedy and O’Hagan.',
    repo: 'RoadSim’s v2 calibration is close to this, without the formal statistics. Its 700 W on-road accessory load for conventional cars (CALIBRATION.md, rung C6) lumps several unmodelled effects into one fitted number. Reading that as a hand-made discrepancy term is this page’s interpretation.',
  },
  {
    key: 'assim', chip: 'Data assimilation', name: 'Data assimilation: the digital twin',
    edges: { phy_sim: '', noise_sim: 'correct the state, live', sim_out: 'forecast' },
    one: 'The physics model runs continuously, and incoming measurements nudge its state toward reality. Kalman filters are the standard tool.',
    physics: 'The running model.',
    real: 'A live stream that corrects the state. Parameters and weights stay fixed.',
    proves: 'Forecast skill over the next interval.',
    when: 'One specific asset or system is tracked over time: the weather, a battery pack, a vehicle in service.',
    repo: 'Not used here. Each model reads one record cold, with no running state.',
  },
  {
    key: 'adapt', chip: 'Domain adaptation', name: 'Unsupervised domain adaptation',
    edges: { phy_sim: '', sim_sdat: '', sdat_net: 'train with labels', noise_net: 'align features, no labels', net_out: '', exam_out: 'test' },
    one: 'Train on labelled simulations while also showing the network unlabelled real inputs, and penalise it for telling the two apart.',
    physics: 'The simulator that provides every label.',
    real: 'Unlabelled inputs shape the features. No real answers are used.',
    proves: 'The same as this repository: a simulation-trained model graded on real answers it never saw.',
    when: 'Real recordings are plentiful and labels are the bottleneck.',
    repo: 'Stop 3 of the dial. RoadData already holds a train_noise split: 17 VED vehicles with no fuel signal, used so far only as real routes.',
  },
];

/** References per approach. Each was checked to exist (authors, year, venue, link) on 14 Nov 2026. */
export const REFS: Record<string, Ref[]> = {
  guess: [{ short: 'Richter et al. 2016', full: 'Richter, Vineet, Roth & Koltun. Playing for Data: Ground Truth from Computer Games. ECCV 2016, pp. 102–118.', url: 'https://doi.org/10.1007/978-3-319-46475-6_7' }],
  repo: [{ short: 'Cranmer, Brehmer & Louppe 2020', full: 'The frontier of simulation-based inference. PNAS 117(48): 30055–30062.', url: 'https://doi.org/10.1073/pnas.1912789117' }],
  blackbox: [{ short: 'Hastie, Tibshirani & Friedman 2009', full: 'The Elements of Statistical Learning, 2nd ed. Springer.', url: 'https://doi.org/10.1007/978-0-387-84858-7' }],
  finetune: [
    { short: 'Pan & Yang 2010', full: 'A Survey on Transfer Learning. IEEE TKDE 22(10): 1345–1359.', url: 'https://doi.org/10.1109/TKDE.2009.191' },
    { short: 'Yosinski et al. 2014', full: 'How transferable are features in deep neural networks? NeurIPS 27.', url: 'https://arxiv.org/abs/1411.1792' },
  ],
  residual: [
    { short: 'Psichogios & Ungar 1992', full: 'A hybrid neural network–first principles approach to process modeling. AIChE Journal 38(10): 1499–1511.', url: 'https://doi.org/10.1002/aic.690381003' },
    { short: 'Ramakrishnan et al. 2015', full: 'Big Data Meets Quantum Chemistry Approximations: The Δ-Machine Learning Approach. J. Chem. Theory Comput. 11(5): 2087–2096.', url: 'https://doi.org/10.1021/acs.jctc.5b00099' },
  ],
  pinn: [
    { short: 'Raissi, Perdikaris & Karniadakis 2019', full: 'Physics-informed neural networks. J. Comput. Phys. 378: 686–707.', url: 'https://doi.org/10.1016/j.jcp.2018.10.045' },
    { short: 'Karniadakis et al. 2021', full: 'Physics-informed machine learning. Nature Reviews Physics 3: 422–440.', url: 'https://doi.org/10.1038/s42254-021-00314-5' },
  ],
  domrand: [
    { short: 'Tobin et al. 2017', full: 'Domain randomization for transferring deep neural networks from simulation to the real world. IROS 2017.', url: 'https://doi.org/10.1109/IROS.2017.8202133' },
    { short: 'Peng et al. 2018', full: 'Sim-to-Real Transfer of Robotic Control with Dynamics Randomization. ICRA 2018.', url: 'https://doi.org/10.1109/ICRA.2018.8460528' },
  ],
  surrogate: [
    { short: 'Sacks et al. 1989', full: 'Design and Analysis of Computer Experiments. Statistical Science 4(4): 409–423.', url: 'https://doi.org/10.1214/ss/1177012413' },
    { short: 'Kasim et al. 2022', full: 'Building high accuracy emulators for scientific simulations with deep neural architecture search. Mach. Learn.: Sci. Technol. 3: 015013.', url: 'https://doi.org/10.1088/2632-2153/ac3ffa' },
  ],
  calib: [{ short: 'Kennedy & O’Hagan 2001', full: 'Bayesian calibration of computer models. J. R. Stat. Soc. B 63(3): 425–464.', url: 'https://doi.org/10.1111/1467-9868.00294' }],
  assim: [
    { short: 'Kalman 1960', full: 'A New Approach to Linear Filtering and Prediction Problems. J. Basic Eng. 82(1): 35–45.', url: 'https://doi.org/10.1115/1.3662552' },
    { short: 'Evensen 1994', full: 'Sequential data assimilation with a nonlinear quasi-geostrophic model using Monte Carlo methods to forecast error statistics. JGR Oceans 99(C5): 10143–10162.', url: 'https://doi.org/10.1029/94JC00572' },
  ],
  adapt: [
    { short: 'Ganin & Lempitsky 2015', full: 'Unsupervised Domain Adaptation by Backpropagation. ICML 2015, PMLR 37: 1180–1189.', url: 'https://proceedings.mlr.press/v37/ganin15.html' },
    { short: 'Ganin et al. 2016', full: 'Domain-Adversarial Training of Neural Networks. JMLR 17(59): 1–35.', url: 'https://jmlr.org/papers/v17/15-239.html' },
  ],
};

// ---- 01 · a common first guess, marked up --------------------------------------------------------
export type DiffPart = { k: 'ok' | 'ins' | 'del'; t: string };
export const GUESS_DIFF: DiffPart[] = [
  { k: 'ok', t: 'Build physics-based simulations of an area' },
  { k: 'ins', t: 'check them against an independent reference' },
  { k: 'ok', t: 'produce data' },
  { k: 'ins', t: 'laid into real instrument noise' },
  { k: 'ok', t: 'find real data' },
  { k: 'del', t: 'train a model on it' },
  { k: 'ins', t: 'train only on the simulations' },
  { k: 'ins', t: 'use a small real dev set to fix the simulator and gate each new version' },
  { k: 'ins', t: 'grade once on real data that chose nothing' },
];

// ---- 06 · eight lenses ---------------------------------------------------------------------------
export interface Lens { tag: string; name: string; one: string; see: string; blind: string; ask: string; evi: string }
export const LENSES: Lens[] = [
  {
    tag: 'The first guess', name: 'The pooled pile', one: 'Make simulated data, find real data, train on the lot.',
    see: 'That both kinds of data matter, and that physics comes first.',
    blind: 'Data has roles. A real record used for training can no longer be used for judging.',
    ask: 'If real data goes into training, what is left to tell whether the physics helped?',
    evi: 'README: {labels} real labelled examples were used for training.',
  },
  {
    tag: 'The method’s own', name: 'Teacher and examiner', one: 'Physics teaches. Reality examines. They never swap seats.',
    see: 'Why the exam is credible, and why a published loss is useful.',
    blind: 'It hides how much real data shapes the simulator. Reality is also the teacher’s teacher.',
    ask: 'Has this real record ever influenced a choice? If so, it cannot sit the exam.',
    evi: 'docs/method.md, step 3: the rule that makes the exam mean something.',
  },
  {
    tag: 'One step on from the first guess', name: 'Two learners, two loops', one: 'The simulator learns from reality. The network learns from the simulator.',
    see: 'Where the grey-box intuition belongs. The outer loop fits a few physical constants anyone can read. The inner loop fits a million weights nobody can.',
    blind: 'The outer loop is slow and manual. Each turn is a person or an agent plotting dev data beside simulations.',
    ask: 'Which loop does this mismatch belong to: a physical constant with a name, or the network?',
    evi: 'QuakeNet: building in 6.5% slower surface waves and a scattering coda lifted the frozen-set score from {q1} to {q3}.',
  },
  {
    tag: 'Inverse problems', name: 'A learned inverse of a forward model', one: 'The simulator maps causes to signals. The network is trained to run that map backwards.',
    see: 'Why training on simulations works at all: every simulated signal comes with its exact cause attached, so labels are free and unlimited. Also why the outputs carry error bars.',
    blind: 'The inverse is only as good as the forward model and the mix of cases chosen for simulation.',
    ask: 'Is the question an inversion (signal to cause) or a forecast (cause to effect)? A forecast might want a surrogate.',
    evi: 'ChirpNet’s posterior flow returns a full distribution over merger masses. TripDetective inverts a trip log back to mass and drag.',
  },
  {
    tag: 'Budgeting', name: 'Real data is spent once', one: 'Every real record has a budget of one look. Spend it as noise, calibration, gate or exam, and it is gone for the others.',
    see: 'Why splits are frozen with hashes, and why RoadSim needed a second exam.',
    blind: 'It makes real data feel scarcer than it is. Unlabelled data can be reused freely as backdrop.',
    ask: 'What has already been looked at, and what fresh data is left to judge the next change?',
    evi: 'RoadSim CALIBRATION.md, 14 Nov 2026: the v2 held-out results had already been seen, so they could judge nothing new. Exam-2 was frozen at commit b2f10c0 before any fitting.',
  },
  {
    tag: 'Error budget', name: 'Error has four addresses', one: 'Exam error = network gap + simulator gap + wrong mix of cases + noisy labels.',
    see: 'That each address has its own test and its own fix. The twin test separates the first two.',
    blind: 'The four interact. A calibrated simulator can still read real data worse, and the cause is not always found.',
    ask: 'Does the network read simulated twins of these real units correctly? If so, look at the simulator.',
    evi: 'Simulator: BearingNet read 100% of its twins right and 22% of real dev files. Mix: TripDetective’s dev mass bias fell from +106 kg to −9 kg by matching the fleet’s mass. Labels: VED fuel is estimated, not metered, for 351 of 384 vehicles.',
  },
  {
    tag: 'Experiment design', name: 'A pre-registered experiment', one: 'Each rung is a hypothesis with its pass mark fixed in advance and its result published either way.',
    see: 'Why {rejected} rejections are a feature. Also why rules may change only in the open, with both decisions kept in the log.',
    blind: 'Margins and gates are still human choices. A 0.01 margin is a convention.',
    ask: 'Was the test fixed before the result was seen?',
    evi: 'RoadSim v2.1 was frozen at commit 1896a7d and reported at 39f5e53: computed once after the freeze, nothing changed afterwards. A 171-run fingerprint holds v2 to 1e-9.',
  },
  {
    tag: 'Planning', name: 'The gap list is the roadmap', one: 'The simulator’s measured errors are the work queue. Each has a size and a done-when.',
    see: 'How the project always knows what to do next, and how work splits into independent passes.',
    blind: 'Some gaps will not close with physics anyone can specify. That is where learned terms earn their place.',
    ask: 'Which measured gap is largest, and is it physics that can be written down?',
    evi: 'shared/vehicle/roadsim/LIMITS.md: every known RoadSim error with its size, measured on held-out and exam-2 data, next to the fix it would need.',
  },
];

// ---- 05 · RoadSim v2 -> v2.1 exam-2 (shared/vehicle/roadsim/LIMITS.md, 14 Nov 2026) --------------
/** group, n with an FTP result, median FTP error % for v2 and v2.1 (+ = the simulation burns less) */
export const EXAM2: [string, number, number, number][] = [
  ['P2 TMED (Hyundai, Kia)', 53, -10.2, -9.8],
  ['P2 with converter', 19, 27.9, 26.1],
  ['P2 with a clutch', 9, 17.7, 15.4],
  ['Mild P0/P1', 38, 10.8, 8.9],
  ['Toyota / Lexus power-split', 166, 18.0, 17.2],
  ['Ford power-split', 10, 9.0, 8.9],
  ['Honda e:HEV', 4, -2.9, -3.6],
  ['Plug-in power-split, CS', 22, 0.7, 0.5],
  ['Plug-in series, CS', 15, 18.5, 19.4],
  ['Plug-in P2, CS', 67, 6.9, 7.3],
];

// ---- 07 · the dial -------------------------------------------------------------------------------
export type StopStatus = 'y' | 'p' | 'n';
export interface Stop { short: string; name: string; st: StopStatus; what: string; repo: string; gain: string; risk: string; guard: string; claim: number }
export const STOPS: Stop[] = [
  {
    short: 'Backdrop', name: 'Backdrop', st: 'y', what: 'Real recordings with no answers attached sit underneath the simulated signals.',
    repo: 'Quiet seismometer windows, healthy-machine vibration, quiet stars, real routes with lidar grade and airport weather.',
    gain: 'The network never learns to rely on idealised noise. The method calls it the cheapest realism there is.',
    risk: 'Very little. The backdrop pool must not overlap the exam.', guard: 'Keep noise sources out of dev and exam, as BearingNet does with its healthy files.', claim: 0,
  },
  {
    short: 'Calibrate', name: 'Calibrate the simulator', st: 'y', what: 'A few real dev units set physical constants and the mix of cases the simulator draws from.',
    repo: 'RoadSim v2 and v2.1: base accessory load 250 W, oil heat capacity doubled, belt A/C efficiency 0.25, engine run-on 2 s into braking. Each has a rung in CALIBRATION.md.',
    gain: 'The biggest wins in the collection came from here.',
    risk: 'Matching dev statistics without reading real data better. The ratchet rejected such simulators in SwellCast, StarQuake and BeatNet.',
    guard: 'Fit on dev only. Prefer measured values to fitted ones. Bump the physics version and re-take every decision.', claim: 0,
  },
  {
    short: 'Gate', name: 'Gate and error bars', st: 'y', what: 'Dev units decide whether a rung is kept, and set one widen-only factor per output for the error bars.',
    repo: 'mlratchet.ratchet (the gate) and mlratchet.calibrate (the widening).',
    gain: 'Protection against improvements that exist only in simulation.',
    risk: 'A small dev set can be too kind: BearingNet scored 0.78 on its Paderborn dev files and 0.45 on the exam.', guard: 'Draw dev units from several machines, conditions or strata.', claim: 0,
  },
  {
    short: 'Unlabelled', name: 'Unlabelled real inputs in training', st: 'p', what: 'The network sees real inputs with no answers: self-supervised pretraining, or a penalty for telling simulated from real.',
    repo: 'Partly. A train_noise split of 17 VED vehicles with no fuel signal exists and is used as real routes only.',
    gain: 'Features that match the real input distribution, including logger quirks the emulator misses.',
    risk: 'Alignment can erase real differences that carry signal.', guard: 'Split the unlabelled pool by unit, away from the exam. Judge with the same ratchet.', claim: 0,
  },
  {
    short: 'Discrepancy', name: 'A learned discrepancy term in the simulator', st: 'n', what: 'A small learned term, fitted on real data, for what the physics cannot reach. The network still trains on the simulator’s output.',
    repo: 'By hand so far: the 700 W on-road load and the OBD label model. No learned version.',
    gain: 'A route to structured gaps, such as per-engine spread, without touching the network’s training rule.',
    risk: 'The term absorbs errors that belong to physics. RoadSim rung C3b did exactly this and was rejected: per-map efficiency levels matched the EPA pool and made every metered D3 car worse.',
    guard: 'Fit on one data source and judge on another, as C3b was judged on metered D3 cars.', claim: 1,
  },
  {
    short: 'Fine-tune', name: 'Fine-tune on real labels', st: 'n', what: 'Start from the simulation-trained network and continue training on a labelled real split.',
    repo: 'Not done anywhere.',
    gain: 'Most where a regression trained on real data already wins, as on {petrolLoss}.',
    risk: 'The network can memorise one dataset’s quirks, and the zero-real-labels claim ends for that model.',
    guard: 'A third frozen split by vehicle. Publish simulation-only and simulation-plus-real side by side.', claim: 2,
  },
  {
    short: 'Real-first', name: 'Real-first, physics as features', st: 'n', what: 'Train mainly on real labels. Physics enters as input features or as the baseline being corrected.',
    repo: 'Only as the baselines the models are graded against.',
    gain: 'The best accuracy when measured labels are plentiful and cover the conditions of use.',
    risk: 'Poor behaviour outside the dataset’s envelope, where simulation-trained models keep working.',
    guard: 'Hold out whole vehicle families and operating regimes, and keep a simulation-only model as the reference.', claim: 3,
  },
];

/** Headline claim at each stop. Parts marked real are drawn in the real-data colour; struck parts are crossed out. */
export const CLAIMS: { t: string; real?: boolean; struck?: boolean }[][] = [
  [{ t: 'Trained on physics.' }, { t: ' Graded on reality.' }],
  [{ t: 'Trained on physics, ' }, { t: 'with a learned patch from dev data', real: true }, { t: '. Graded on reality.' }],
  [{ t: 'Trained on physics.', struck: true }, { t: ' ' }, { t: 'Pretrained on physics, tuned on reality.', real: true }, { t: ' Graded on unseen vehicles.' }],
  [{ t: 'Trained on reality, shaped by physics.', real: true }, { t: ' Graded on unseen vehicles.' }],
];

export const STATUS_LABEL: Record<StopStatus, string> = { y: 'In the repository', p: 'Partly there · possible direction', n: 'Possible direction' };
