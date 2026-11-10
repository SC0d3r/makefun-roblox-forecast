// Roblox Top Trending forecast: horizon-binned rank transitions, smoothed by
// a sticky-chart prior. Node 22, standard library only, deterministic output.

import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const MAX_OLD_RANK = 25;
const PLACES = 10;
const BUCKETS = PLACES + 1;
const HORIZONS = 24;
const MAX_GAP_HOURS = 30;
const SMOOTHING = 8;
const MIN_PROBABILITY = 1e-4;
const MAX_ROWS_PER_PLACE = 100;
const PLACE_CAP = 0.95;
// Riser mixture: games climbing the chart fast (14h rank velocity + player growth)
// absorb most next-read top10 entries, so they get a slice of every place.
const RISER_WINDOW_HOURS = 14;
const RISER_K = 4;
const RISER_W = 0.24;
const RISER_PROFILE = [0.02, 0.03, 0.05, 0.07, 0.1, 0.13, 0.15, 0.15, 0.15, 0.15];
const FLOOR_ALPHA = 0.02;

// With the house recorder's hourly reads, every one of the 24 scored slots has its own
// horizon kernel. One static forecast is scored against all of them, so the optimal
// static probability is the arithmetic mean of the per-slot marginals (a mixture); a
// geometric pool is sharper than that and loses log score. Keep the geometric pool
// when the history has no hourly pairs and the per-horizon counts are sparse.
const HOURLY_PAIRS_MIN = 40;


interface TrendingRow {
  rank: number;
  universe_id: number;
  playing?: number | null;
}

interface Read {
  at: string;
  atMs: number;
  source: unknown;
  top_trending: TrendingRow[];
}

interface KernelRead extends Read {
  byUniverse: Map<number, TrendingRow>;
}

interface Task {
  places: number;
  output: string;
}

interface ProbabilityRow {
  universe_id: number;
  place: number;
  p: number;
}

interface GameForecast {
  rank: number;
  places: number[];
}

function createRankBuckets(): number[][] {
  return Array.from({ length: MAX_OLD_RANK + 1 }, () => new Array<number>(BUCKETS).fill(0));
}

function parseHistory(text: string): Read[] {
  const reads: Read[] = [];
  for (const line of text.split('\n')) {
    const trimmed = line.trim();
    if (trimmed === '') continue;

    let value: unknown;
    try {
      value = JSON.parse(trimmed);
    } catch {
      continue;
    }
    if (typeof value !== 'object' || value === null) continue;

    const record = value as Record<string, unknown>;
    if (!Array.isArray(record.top_trending) || typeof record.at !== 'string') continue;
    const atMs = Date.parse(record.at);
    if (!Number.isFinite(atMs)) continue;

    reads.push({
      at: record.at,
      atMs,
      source: record.source,
      top_trending: record.top_trending as TrendingRow[],
    });
  }
  reads.sort((a, b) => a.atMs - b.atMs);
  return reads;
}

function hasMakefunSource(read: Read): boolean {
  return typeof read.source === 'string' && read.source.includes('makefun');
}

function selectKernelReads(reads: Read[]): KernelRead[] {
  const makefunReads = reads.filter(hasMakefunSource);
  const selected = makefunReads.length >= 20 ? makefunReads : reads;
  return selected.map((read) => ({
    ...read,
    byUniverse: new Map(read.top_trending.map((row) => [row.universe_id, row])),
  }));
}

function baselineForecast(task: Task, last: Read | undefined): { probabilities: ProbabilityRow[]; top10: number[] } {
  if (last === undefined) return { probabilities: [], top10: [] };

  const chart = [...last.top_trending]
    .sort((a, b) => a.rank - b.rank)
    .map((row) => row.universe_id);
  const probabilities: ProbabilityRow[] = [];
  for (let place = 1; place <= task.places && place <= chart.length; place++) {
    probabilities.push({ universe_id: chart[place - 1]!, place, p: 0.5 });
    if (place === 1) {
      if (chart[1] !== undefined) probabilities.push({ universe_id: chart[1], place, p: 0.3 });
      continue;
    }
    probabilities.push({ universe_id: chart[place - 2]!, place, p: 0.15 });
    if (chart[place] !== undefined) probabilities.push({ universe_id: chart[place], place, p: 0.15 });
  }
  return { probabilities, top10: chart.slice(0, task.places) };
}

function buildTransitionCounts(kernelReads: KernelRead[]): {
  counts: number[][][];
  countsAll: number[][];
  horizonTotals: number[];
} {
  const counts = Array.from({ length: HORIZONS + 1 }, createRankBuckets);
  const countsAll = createRankBuckets();
  const horizonTotals = new Array<number>(HORIZONS + 1).fill(0);

  for (let aIndex = 0; aIndex < kernelReads.length; aIndex++) {
    const earlier = kernelReads[aIndex]!;
    for (let bIndex = aIndex + 1; bIndex < kernelReads.length; bIndex++) {
      const later = kernelReads[bIndex]!;
      const gapHours = (later.atMs - earlier.atMs) / 3_600_000;
      if (gapHours <= 0 || gapHours > MAX_GAP_HOURS) continue;

      const h = Math.max(1, Math.min(HORIZONS, Math.round(gapHours)));
      const tolerance = h <= 8 ? 0.75 : 1.5;
      const inHorizon = Math.abs(gapHours - h) <= tolerance;

      for (const oldRow of earlier.top_trending) {
        const oldRank = oldRow.rank;
        if (!Number.isInteger(oldRank) || oldRank < 1 || oldRank > MAX_OLD_RANK) continue;

        const newRow = later.byUniverse.get(oldRow.universe_id);
        const newRank = newRow?.rank;
        const bucket =
          typeof newRank === 'number' && Number.isInteger(newRank) && newRank >= 1 && newRank <= PLACES
            ? newRank
            : 0;

        countsAll[oldRank]![bucket]! += 1;
        if (inHorizon) {
          counts[h]![oldRank]![bucket]! += 1;
          horizonTotals[h]! += 1;
        }
      }
    }
  }

  return { counts, countsAll, horizonTotals };
}

function buildPriors(): number[][][] {
  const priors = Array.from({ length: HORIZONS + 1 }, () => createRankBuckets());

  for (let h = 1; h <= HORIZONS; h++) {
    const scale = Math.sqrt(h / 12);
    const s = 1.3 * scale;
    const dropoutProbability = 0.02 * scale;
    const stayProbability = 1 - dropoutProbability;
    const pnScale = (1 - Math.exp(-1 / s)) / (1 + Math.exp(-1 / s));

    for (let oldRank = 1; oldRank <= MAX_OLD_RANK; oldRank++) {
      const prior = priors[h]![oldRank]!;
      const n = oldRank - 1;
      let binomial = 1;

      for (let k = 0; k <= n; k++) {
        const weight =
          binomial * Math.pow(dropoutProbability, k) * Math.pow(stayProbability, n - k);
        for (let d = -14; d <= 14; d++) {
          const place = oldRank - k + d;
          if (place < 1 || place > PLACES) continue;
          const pn = pnScale * Math.exp(-Math.abs(d) / s);
          prior[place]! += weight * pn;
        }
        if (k < n) binomial = (binomial * (n - k)) / (k + 1);
      }

      let namedTotal = 0;
      for (let place = 1; place <= PLACES; place++) namedTotal += prior[place]!;
      prior[0] = Math.max(0, 1 - namedTotal);
    }
  }

  return priors;
}

function averageKernels(
  counts: number[][][],
  countsAll: number[][],
  horizonTotals: number[],
  priors: number[][][],
): number[][] {
  const averages = createRankBuckets();
  const arithmetic = horizonTotals[1]! >= HOURLY_PAIRS_MIN;

  for (let oldRank = 1; oldRank <= MAX_OLD_RANK; oldRank++) {
    const totals = new Array<number>(PLACES + 1).fill(0);
    for (let h = 1; h <= HORIZONS; h++) {
      const source = horizonTotals[h]! >= 40 ? counts[h]! : countsAll;
      const row = source[oldRank]!;
      let observations = 0;
      for (let bucket = 0; bucket < BUCKETS; bucket++) observations += row[bucket]!;

      const denominator = observations + SMOOTHING;
      for (let place = 1; place <= PLACES; place++) {
        const kernel =
          (row[place]! + SMOOTHING * priors[h]![oldRank]![place]!) / denominator;
        if (arithmetic) {
          totals[place]! += kernel;
        } else {
          totals[place]! += Math.log(Math.max(kernel, 1e-12));
        }
      }
    }
    for (let place = 1; place <= PLACES; place++) {
      averages[oldRank]![place] = arithmetic
        ? totals[place]! / HORIZONS
        : Math.exp(totals[place]! / HORIZONS);
    }
  }

  return averages;
}

function scoreLastReadGames(last: KernelRead, kernelAverages: number[][]): Map<number, GameForecast> {
  const games = new Map<number, GameForecast>();

  for (const row of last.top_trending) {
    const oldRank = row.rank;
    if (!Number.isInteger(oldRank) || oldRank < 1 || oldRank > MAX_OLD_RANK) continue;

    let game = games.get(row.universe_id);
    if (game === undefined) {
      game = { rank: oldRank, places: new Array<number>(BUCKETS).fill(0) };
      games.set(row.universe_id, game);
    }
    for (let place = 1; place <= PLACES; place++) {
      game.places[place]! += kernelAverages[oldRank]![place]!;
    }
  }

  return games;
}

function linearSlope(points: Array<[number, number]>): number | null {
  if (points.length < 2) return null;
  let sumX = 0;
  let sumY = 0;
  for (const [x, y] of points) {
    sumX += x;
    sumY += y;
  }
  const meanX = sumX / points.length;
  const meanY = sumY / points.length;
  let denominator = 0;
  let numerator = 0;
  for (const [x, y] of points) {
    denominator += (x - meanX) ** 2;
    numerator += (x - meanX) * (y - meanY);
  }
  if (denominator <= 1e-6) return null;
  return numerator / denominator;
}

// Riser score: how fast a game is climbing (negative rank slope) plus how fast its
// player count is growing, each standardised across the last read's chart.
function computeRisers(reads: KernelRead[]): Map<number, number> {
  const scores = new Map<number, number>();
  const last = reads[reads.length - 1];
  if (last === undefined) return scores;

  const windowReads = reads.filter((read) => last.atMs - read.atMs <= RISER_WINDOW_HOURS * 3_600_000);
  const velocities = new Map<number, number | null>();
  const growths = new Map<number, number | null>();

  for (const row of last.top_trending) {
    const rankPoints: Array<[number, number]> = [];
    const growthPoints: Array<[number, number]> = [];
    for (const read of windowReads) {
      const seen = read.byUniverse.get(row.universe_id);
      if (seen === undefined) continue;
      const x = (read.atMs - last.atMs) / 3_600_000;
      rankPoints.push([x, seen.rank]);
      if (typeof seen.playing === 'number' && seen.playing > 0) {
        growthPoints.push([x, Math.log(seen.playing)]);
      }
    }
    velocities.set(row.universe_id, linearSlope(rankPoints));
    growths.set(row.universe_id, linearSlope(growthPoints));
  }

  const velocityValues = [...velocities.values()].filter((value): value is number => value !== null);
  const growthValues = [...growths.values()].filter((value): value is number => value !== null);
  const mean = (values: number[]): number =>
    values.length === 0 ? 0 : values.reduce((total, value) => total + value, 0) / values.length;
  const deviation = (values: number[]): number => {
    if (values.length < 2) return 1;
    const centre = mean(values);
    return Math.max(Math.sqrt(values.reduce((total, value) => total + (value - centre) ** 2, 0) / values.length), 1e-9);
  };
  const velocityMean = mean(velocityValues);
  const velocityDeviation = deviation(velocityValues);
  const growthMean = mean(growthValues);
  const growthDeviation = deviation(growthValues);

  for (const universeId of velocities.keys()) {
    let score = 0;
    const velocity = velocities.get(universeId);
    const growth = growths.get(universeId);
    if (velocity !== null && velocity !== undefined) score += -(velocity - velocityMean) / velocityDeviation;
    if (growth !== null && growth !== undefined) score += (growth - growthMean) / growthDeviation;
    scores.set(universeId, score);
  }

  return scores;
}

function applyRiserMixture(games: Map<number, GameForecast>, reads: KernelRead[]): void {
  const scores = computeRisers(reads);
  const chosen = [...scores.entries()].sort((a, b) => b[1] - a[1]).slice(0, RISER_K);
  if (chosen.length === 0) return;

  for (const [universeId] of chosen) {
    if (!games.has(universeId)) {
      games.set(universeId, { rank: 0, places: new Array<number>(BUCKETS).fill(0) });
    }
  }

  const profileTotal = RISER_PROFILE.reduce((total, value) => total + value, 0);
  const share = RISER_W / chosen.length;
  for (let place = 1; place <= PLACES; place++) {
    let total = 0;
    for (const game of games.values()) total += game.places[place]!;
    if (total <= 0) continue;
    for (const game of games.values()) {
      game.places[place] = ((1 - RISER_W) * game.places[place]!) / total;
    }
    for (const [universeId] of chosen) {
      const game = games.get(universeId)!;
      game.places[place]! += (share * RISER_PROFILE[place - 1]!) / profileTotal;
    }
  }
}

// Uniform floor over every game in the last chart, then renormalise each place.
function applyFloor(games: Map<number, GameForecast>, last: KernelRead): void {
  for (const row of last.top_trending) {
    if (!games.has(row.universe_id)) {
      games.set(row.universe_id, { rank: 0, places: new Array<number>(BUCKETS).fill(0) });
    }
  }
  const count = games.size;
  if (count === 0) return;
  for (let place = 1; place <= PLACES; place++) {
    let total = 0;
    for (const game of games.values()) total += game.places[place]!;
    for (const game of games.values()) {
      const base = total > 0 ? game.places[place]! / total : 0;
      game.places[place] = (1 - FLOOR_ALPHA) * base + FLOOR_ALPHA / count;
    }
  }
}

function capPlaces(games: Map<number, GameForecast>): void {  for (let place = 1; place <= PLACES; place++) {
    let total = 0;
    for (const game of games.values()) total += game.places[place]!;
    if (total <= PLACE_CAP) continue;

    const factor = PLACE_CAP / total;
    for (const game of games.values()) game.places[place]! *= factor;
  }
}

function compareProbabilityRows(a: ProbabilityRow, b: ProbabilityRow): number {
  return a.place - b.place || b.p - a.p || a.universe_id - b.universe_id;
}

function collectProbabilities(games: Map<number, GameForecast>): ProbabilityRow[] {
  const rows: ProbabilityRow[] = [];

  for (let place = 1; place <= PLACES; place++) {
    const atPlace: ProbabilityRow[] = [];
    for (const [universeId, game] of games) {
      const p = game.places[place]!;
      if (p >= MIN_PROBABILITY) atPlace.push({ universe_id: universeId, place, p });
    }
    atPlace.sort((a, b) => b.p - a.p || a.universe_id - b.universe_id);
    rows.push(...atPlace.slice(0, MAX_ROWS_PER_PLACE));
  }

  rows.sort(compareProbabilityRows);
  return rows;
}

function chooseTop10(
  games: Map<number, GameForecast>,
  probabilities: ProbabilityRow[],
): number[] {
  const retained = new Map<number, number[]>();
  for (const row of probabilities) {
    let places = retained.get(row.universe_id);
    if (places === undefined) {
      places = new Array<number>(BUCKETS).fill(0);
      retained.set(row.universe_id, places);
    }
    places[row.place] = row.p;
  }

  const scored = [...games].map(([universeId, game]) => {
    const places = retained.get(universeId) ?? new Array<number>(BUCKETS).fill(0);
    let score = 0;
    let bestPlace = PLACES + 1;
    let bestProbability = -1;
    for (let place = 1; place <= PLACES; place++) {
      const p = places[place]!;
      score += p;
      if (p > bestProbability) {
        bestProbability = p;
        bestPlace = place;
      }
    }
    return { universeId, rank: game.rank, score, bestPlace };
  });

  scored.sort((a, b) => b.score - a.score || a.rank - b.rank || a.universeId - b.universeId);
  const topGames = scored.slice(0, PLACES);
  topGames.sort(
    (a, b) => a.bestPlace - b.bestPlace || a.rank - b.rank || a.universeId - b.universeId,
  );
  return topGames.map((game) => game.universeId);
}

function main(): void {
  const task = JSON.parse(readFileSync(join(INPUT_DIR, 'task.json'), 'utf8')) as Task;
  const reads = parseHistory(readFileSync(join(INPUT_DIR, 'history.ndjson'), 'utf8'));

  let probabilities: ProbabilityRow[];
  let top10: number[];

  if (reads.length < 2) {
    ({ probabilities, top10 } = baselineForecast(task, reads[reads.length - 1]));
  } else {
    const kernelReads = selectKernelReads(reads);
    const { counts, countsAll, horizonTotals } = buildTransitionCounts(kernelReads);
    const priors = buildPriors();
    const kernelAverages = averageKernels(counts, countsAll, horizonTotals, priors);
    const games = scoreLastReadGames(kernelReads[kernelReads.length - 1]!, kernelAverages);

    applyRiserMixture(games, kernelReads);
    applyFloor(games, kernelReads[kernelReads.length - 1]!);
    capPlaces(games);
    probabilities = collectProbabilities(games);
    top10 = chooseTop10(games, probabilities);
  }

  const outputPath = join(OUTPUT_DIR, task.output);
  mkdirSync(dirname(outputPath), { recursive: true });
  writeFileSync(outputPath, `${JSON.stringify({ probabilities, top10 })}\n`);
}

main();
