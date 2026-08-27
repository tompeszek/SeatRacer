import { useEffect, useState } from 'react'

export type Theme = 'light' | 'dark'

// Dark is the default; a stored light preference wins.
function readStoredTheme(): Theme {
  try {
    return localStorage.getItem('theme') === 'light' ? 'light' : 'dark'
  } catch {
    return 'dark'
  }
}

export function useTheme(): [Theme, (t: Theme) => void] {
  const [theme, setTheme] = useState<Theme>(readStoredTheme)
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme === 'dark' ? 'dark' : 'light')
    try {
      localStorage.setItem('theme', theme)
    } catch {
      // storage unavailable; theme stays session-only
    }
  }, [theme])
  return [theme, setTheme]
}
