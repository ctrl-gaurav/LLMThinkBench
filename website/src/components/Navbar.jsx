import React, { useState, useEffect } from 'react'
import { Brain, Github, Package, Quote, Menu, X, BookOpen, Sun, Moon } from 'lucide-react'
import { usePyPIVersion } from '../hooks/usePyPIVersion'
import { useTheme } from '../context/ThemeContext'

export default function Navbar({ scrollToSection }) {
  // Fallback: when rendered on /docs (no home-page sections available), jump to home and then scroll
  const handleSectionClick = (id) => {
    if (scrollToSection) return scrollToSection(id)
    window.location.hash = '#/'
    setTimeout(() => {
      const el = document.getElementById(id)
      if (el) el.scrollIntoView({ behavior: 'smooth' })
    }, 100)
  }
  const pypiVersion = usePyPIVersion()
  const { isDark, toggleTheme } = useTheme()
  const [scrolled, setScrolled] = useState(false)
  const [mobileOpen, setMobileOpen] = useState(false)
  const [activeSection, setActiveSection] = useState('')

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 50)

      // Scroll spy
      const sections = ['winners', 'leaderboard', 'charts']
      for (const id of sections.reverse()) {
        const el = document.getElementById(id)
        if (el && window.scrollY >= el.offsetTop - 200) {
          setActiveSection(id)
          break
        }
      }
    }
    window.addEventListener('scroll', handleScroll, { passive: true })
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  const navItems = [
    { label: 'Winners', id: 'winners' },
    { label: 'Leaderboard', id: 'leaderboard' },
    { label: 'Charts', id: 'charts' },
  ]

  return (
    <header
      className={`sticky top-0 z-[1000] transition-all duration-500 ${
        scrolled
          ? 'glass-strong shadow-lg shadow-neon-indigo/5 border-b border-neon-indigo/10'
          : 'bg-transparent border-b border-transparent'
      }`}
    >
      <div className="max-w-7xl mx-auto px-6 py-4">
        <div className="flex justify-between items-center">
          {/* Logo - always navigates to landing */}
          <a
            href="#/"
            className="flex items-center gap-3 cursor-pointer group no-underline"
            onClick={() => {
              // If already on home, also scroll to top
              if (window.location.hash === '' || window.location.hash === '#/' || window.location.hash === '#') {
                window.scrollTo({ top: 0, behavior: 'smooth' })
              }
            }}
          >
            <div className="relative">
              <Brain className="w-8 h-8 text-neon-indigo group-hover:text-neon-purple transition-colors duration-500" />
              <div className="absolute inset-0 w-8 h-8 bg-neon-indigo/30 rounded-full blur-xl group-hover:bg-neon-purple/40 transition-all duration-500 animate-pulse-soft" />
              {/* Orbit ring around brain */}
              <div className="absolute -inset-2 border border-neon-indigo/20 rounded-full animate-spin-slower opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
            </div>
            <h1 className="text-xl font-bold gradient-text tracking-wider">
              LLMThinkBench
            </h1>
          </a>

          {/* Desktop nav */}
          <nav className="hidden md:flex items-center gap-8">
            {navItems.map((item) => (
              <button
                key={item.id}
                onClick={() => handleSectionClick(item.id)}
                className={`text-sm font-medium transition-all duration-300 relative py-1.5 px-1 btn-press ${
                  activeSection === item.id
                    ? 'text-white nav-active-indicator'
                    : 'text-zinc-500 hover:text-zinc-200'
                }`}
              >
                <span className="relative z-10">{item.label}</span>
              </button>
            ))}

            <a
              href="#/docs"
              className="text-zinc-500 hover:text-neon-cyan transition-all duration-300 flex items-center gap-1.5 text-sm neon-hover group"
            >
              <BookOpen size={16} className="group-hover:scale-110 transition-transform" />
              Docs
            </a>

            <div className="h-5 w-px bg-gradient-to-b from-transparent via-zinc-700 to-transparent" />

            <a
              href="https://github.com/ctrl-gaurav/LLMThinkBench"
              target="_blank"
              rel="noopener noreferrer"
              className="text-zinc-500 hover:text-neon-indigo transition-all duration-300 flex items-center gap-1.5 text-sm neon-hover group"
            >
              <Github size={16} className="group-hover:scale-110 transition-transform" />
              GitHub
            </a>
            <a
              href="https://pypi.org/project/llmthinkbench/"
              target="_blank"
              rel="noopener noreferrer"
              className="text-zinc-500 hover:text-neon-purple transition-all duration-300 flex items-center gap-1.5 text-sm neon-hover group"
            >
              <Package size={16} className="group-hover:scale-110 transition-transform" />
              PyPI{pypiVersion ? ` v${pypiVersion}` : ''}
            </a>
            <button
              onClick={() => handleSectionClick("citation")}
              className="text-zinc-500 hover:text-neon-pink transition-all duration-300 flex items-center gap-1.5 text-sm neon-hover group"
            >
              <Quote size={16} className="group-hover:scale-110 transition-transform" />
              Cite
            </button>

            <button
              onClick={toggleTheme}
              aria-label={isDark ? 'Switch to light theme' : 'Switch to dark theme'}
              title={isDark ? 'Switch to light theme' : 'Switch to dark theme'}
              className="ml-1 p-2 rounded-lg text-zinc-500 hover:text-neon-cyan hover:bg-neon-cyan/5 transition-all duration-300 btn-press"
            >
              {isDark ? <Sun size={16} /> : <Moon size={16} />}
            </button>
          </nav>

          {/* Mobile theme toggle + menu */}
          <div className="md:hidden flex items-center gap-1">
            <button
              onClick={toggleTheme}
              aria-label={isDark ? 'Switch to light theme' : 'Switch to dark theme'}
              className="p-2 rounded-lg text-zinc-400 hover:text-neon-cyan transition-colors btn-press"
            >
              {isDark ? <Sun size={20} /> : <Moon size={20} />}
            </button>
          <button
            className="text-zinc-400 hover:text-white transition-colors btn-press"
            onClick={() => setMobileOpen(!mobileOpen)}
          >
            {mobileOpen ? <X size={24} /> : <Menu size={24} />}
          </button>
          </div>
        </div>

        {/* Mobile nav - slide in */}
        <div
          className={`md:hidden overflow-hidden transition-all duration-500 ease-out ${
            mobileOpen ? 'max-h-[400px] opacity-100 mt-4' : 'max-h-0 opacity-0 mt-0'
          }`}
        >
          <nav className="pb-2 border-t border-zinc-800/50 pt-4 space-y-1">
            {navItems.map((item, i) => (
              <button
                key={item.id}
                onClick={() => {
                  handleSectionClick(item.id)
                  setMobileOpen(false)
                }}
                className="block w-full text-left px-4 py-2.5 text-sm text-zinc-400 hover:text-neon-indigo rounded-xl hover:bg-neon-indigo/5 transition-all btn-press"
                style={{ animationDelay: `${i * 50}ms` }}
              >
                {item.label}
              </button>
            ))}
            <div className="border-t border-zinc-800/50 pt-3 space-y-1">
              <a
                href="#/docs"
                className="flex items-center gap-2 px-4 py-2.5 text-sm text-zinc-400 hover:text-neon-cyan rounded-xl hover:bg-neon-cyan/5 transition-all"
                onClick={() => setMobileOpen(false)}
              >
                <BookOpen size={16} /> Docs
              </a>
              <a
                href="https://github.com/ctrl-gaurav/LLMThinkBench"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 px-4 py-2.5 text-sm text-zinc-400 hover:text-neon-indigo rounded-xl hover:bg-neon-indigo/5 transition-all"
              >
                <Github size={16} /> GitHub
              </a>
              <a
                href="https://pypi.org/project/llmthinkbench/"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 px-4 py-2.5 text-sm text-zinc-400 hover:text-neon-purple rounded-xl hover:bg-neon-purple/5 transition-all"
              >
                <Package size={16} /> PyPI{pypiVersion ? ` v${pypiVersion}` : ''}
              </a>
            </div>
          </nav>
        </div>
      </div>
    </header>
  )
}
