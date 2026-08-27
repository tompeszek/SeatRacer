// Validated data-viz palette (dataviz reference instance; validator run for
// both modes). Categorical slots are assigned in this fixed order, never
// cycled: more than 8 series folds or filters, never a generated hue.

export interface VizTheme {
  categorical: string[]
  divergingCool: string
  divergingWarm: string
  neutral: string
  surface: string
  primaryInk: string
  secondaryInk: string
  muted: string
  gridline: string
  baseline: string
}

export const VIZ_LIGHT: VizTheme = {
  categorical: ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948'],
  divergingCool: '#2a78d6',
  divergingWarm: '#e34948',
  neutral: '#898781',
  surface: '#fcfcfb',
  primaryInk: '#0b0b0b',
  secondaryInk: '#52514e',
  muted: '#898781',
  gridline: '#e1e0d9',
  baseline: '#c3c2b7',
}

export const VIZ_DARK: VizTheme = {
  categorical: ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767'],
  divergingCool: '#3987e5',
  divergingWarm: '#e66767',
  neutral: '#898781',
  surface: '#1a1a19',
  primaryInk: '#ffffff',
  secondaryInk: '#c3c2b7',
  muted: '#898781',
  gridline: '#2c2c2a',
  baseline: '#383835',
}

export function currentVizTheme(): VizTheme {
  return document.documentElement.getAttribute('data-theme') === 'dark' ? VIZ_DARK : VIZ_LIGHT
}
