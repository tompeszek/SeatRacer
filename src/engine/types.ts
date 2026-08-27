// Core data types for the SeatRacer engine. Pure data, no DOM.

/** One boat result row as loaded from a CSV. */
export interface RaceRow {
  /** Raw date string as it appears in the CSV, e.g. "1/5/2012". */
  dateRaw: string
  /** Piece number within the session (1-based). */
  pieceNumber: number
  km: number
  /** Slash-separated rigging, e.g. "c/p/s/p/s". */
  rigging: string
  /** Slash-separated athlete names, e.g. "Gennaro/Bertoldo/Banks/Monaghan". */
  personnel: string
  /** Result time string, MM:SS or MM:SS.f. */
  result: string
}

/** A prepped row: parsed, classified, suffixed, ready for the design matrix. */
export interface PreppedRow {
  dateRaw: string
  /** Date at local midnight. */
  date: Date
  pieceNumber: number
  /** Piece label combining raw date and piece number: "1/5/2012 #1". */
  piece: string
  km: number
  rigging: string
  /** Personnel with rigging superscripts appended (and Cox inserted if implied). */
  personnel: string[]
  shellClass: string
  timeSeconds: number
  /** Pace in seconds per 500m: timeSeconds / (km * 2). */
  timePer500m: number
  /** Absolute margin to the nearest boat in the same piece; null if alone. */
  closestMargin: number | null
  closenessFactor: number
  recencyFactor: number
  totalWeight: number
  /** Design-matrix fraction per athlete in this boat (sums to 1 per boat). */
  athleteFractions: Map<string, number>
}

export interface WeightSettings {
  /** Recency halflife in days, or null for off. */
  halflife: number | null
  /** Close-race weighting halflife in seconds of margin, or null for off. */
  weightClose: number | null
  /** Stern bias: extra weight for stern seats (0.1 / 0.5 / 1.0), or null for off. */
  weightStern: number | null
  includeCoxswains: boolean
  /** Shell classes to keep. */
  shellClasses: string[]
}

export type Loss =
  | { kind: 'squared' }
  | { kind: 'huber'; c: number }
  | { kind: 'lp'; p: number }

export type Shrinkage =
  | { kind: 'none' }
  | { kind: 'ridge'; lambda: number; center: 'zero' | 'erg' }

export interface ModelSpec {
  loss: Loss
  shrinkage: Shrinkage
  weights: WeightSettings
  /** Erg-implied pace per athlete (for ridge toward erg). */
  ergCenters?: Map<string, number>
}

/** Design matrix and metadata for one fit. */
export interface Design {
  /** Column names: athletes, then shell classes, then piece dummies. */
  columns: string[]
  athletes: string[]
  shellClasses: string[]
  pieces: string[]
  /** Row-major matrix, rows.length x columns.length. */
  x: Float64Array[]
  /** Response: pace per 500m. */
  y: Float64Array
  /** Observation weights (total_weight per row). */
  w: Float64Array
  rows: PreppedRow[]
}

export interface FitResult {
  columns: string[]
  params: Float64Array
  /** Standard errors; NaN where not defined for the loss. */
  bse: Float64Array
  ciLower: Float64Array
  ciUpper: Float64Array
  dfResid: number
  rank: number
  /** Fitted values on the training rows. */
  fitted: Float64Array
  paramMap: Map<string, number>
  /**
   * Square root of the coefficient covariance (k rows): multiply by an iid
   * standard normal vector to draw a correlated coefficient perturbation.
   * Absent for losses without a covariance (Lp).
   */
  covHalf?: Float64Array[]
}
