import { useEffect, useState } from 'react'

export type Theme = 'light' | 'dark'

// Dark is the default; only an explicit toggle is stored. The key is new
// because the old 'theme' key was written on every visit, so it pinned
// returning visitors to light even though they never chose it.
// index.html reads the same key before first paint.
const STORAGE_KEY = 'themeChoice'

function readStoredTheme(): Theme {
  try {
    return localStorage.getItem(STORAGE_KEY) === 'light' ? 'light' : 'dark'
  } catch {
    return 'dark'
  }
}

export function useTheme(): [Theme, (t: Theme) => void] {
  const [theme, setThemeState] = useState<Theme>(readStoredTheme)
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
  }, [theme])
  const setTheme = (t: Theme) => {
    setThemeState(t)
    try {
      localStorage.setItem(STORAGE_KEY, t)
    } catch {
      // storage unavailable; theme stays session-only
    }
  }
  return [theme, setTheme]
}
