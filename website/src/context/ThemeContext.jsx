import { createContext, useContext } from 'react'

const ThemeContext = createContext({ theme: 'dark', isDark: true, toggleTheme: () => {} })

export function ThemeProvider({ children }) {
  // LLMThinkBench site is dark-only for now; keep API-compatible with ported BB components
  const value = { theme: 'dark', isDark: true, toggleTheme: () => {} }
  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
}

export function useTheme() {
  return useContext(ThemeContext)
}
