// Option maps for the model controls. Values mirror the old app's sidebar
// (seatracer/ng/models.py) so results carry over exactly.

export const RECENCY_OPTIONS: Record<string, { value: number | null; caption: string }> = {
  Off: { value: null, caption: 'All sessions weigh the same regardless of age' },
  Low: { value: 210.0, caption: 'Weight halves every 210 days' },
  Medium: { value: 56.0, caption: 'Weight halves every 56 days' },
  High: { value: 21.0, caption: 'Weight halves every 21 days' },
}

export const CLOSE_RACES_OPTIONS: Record<string, { value: number | null; caption: string }> = {
  Off: { value: null, caption: 'Margins do not affect race result weighting' },
  Low: { value: 12.0, caption: 'Races decided by 1 second weigh twice as much as 12 second margins' },
  Medium: { value: 8.0, caption: 'Races decided by 1 second weigh twice as much as 8 second margins' },
  High: { value: 5.0, caption: 'Races decided by 1 second weigh twice as much as 5 second margins' },
}

export const STERN_BIAS_OPTIONS: Record<string, { value: number; caption: string }> = {
  Off: { value: 0.0, caption: 'All seats get the same credit or blame for every result' },
  Low: { value: 0.1, caption: 'Stroke seat gets 10 percent more credit or blame than bow' },
  Medium: { value: 0.5, caption: 'Stroke seat gets 50 percent more credit or blame than bow' },
  High: { value: 1.0, caption: 'Stroke seat gets 100 percent more credit or blame than bow' },
}

export const LOSS_OPTIONS: Record<string, { caption: string }> = {
  Squared: { caption: 'Ordinary least squares: every second of miss counts, large misses dominate' },
  Huber: { caption: 'Squared for small misses, linear for large: outlier pieces get limited say' },
  Lp: { caption: 'Exponent below 2 discounts outlier pieces; below 1 discounts them hard' },
}

export const LP_P_OPTIONS = ['0.5', '1', '1.5'] as const

export const SHRINKAGE_OPTIONS: Record<string, { caption: string }> = {
  Off: { caption: 'No shrinkage: the defensible baseline, but confounded athletes get wild values' },
  'Toward Zero': { caption: 'Coefficients pulled toward average; thin data means average, not extreme' },
  'Toward Ergs': { caption: 'Coefficients pulled toward erg-implied values; ergs act as a tiebreaker' },
}

export const STRENGTH_OPTIONS: Record<string, { value: number; caption: string }> = {
  Low: { value: 0.1, caption: 'Barely a tiebreaker between equally good solutions' },
  Medium: { value: 1.0, caption: 'Noticeable pull for athletes with thin data' },
  High: { value: 10.0, caption: 'Strong pull; race data must argue hard to move a coefficient' },
}

export const BUNDLED_DATASETS = [
  'Olympic Selection - 2012.csv',
  'Olympic Selection - 2021.csv',
  'SDRC HOCR 2025 rate adjusted.csv',
  'SDRC HOCR 2025 raw.csv',
  'SDRC Masters Men HOCR Selection - 2024.csv',
  'fall_2025.csv',
]

export interface ControlState {
  recency: keyof typeof RECENCY_OPTIONS
  close: keyof typeof CLOSE_RACES_OPTIONS
  stern: keyof typeof STERN_BIAS_OPTIONS
  coxswains: boolean
  loss: keyof typeof LOSS_OPTIONS
  lpP: (typeof LP_P_OPTIONS)[number]
  shrinkage: keyof typeof SHRINKAGE_OPTIONS
  strength: keyof typeof STRENGTH_OPTIONS
  /** null = all shell classes in the file. */
  shells: string[] | null
}

export const DEFAULT_CONTROLS: ControlState = {
  recency: 'Off',
  close: 'Off',
  stern: 'Off',
  coxswains: false,
  loss: 'Squared',
  lpP: '1',
  shrinkage: 'Off',
  strength: 'Medium',
  shells: null,
}
