import React, { useState, useMemo, useCallback, useEffect, lazy, Suspense } from 'react'
import { Routes, Route } from 'react-router-dom'
import { modelData, calculateRanks } from './data/modelData'
import Navbar from './components/Navbar'
import HeroSection from './components/HeroSection'
import CategoryWinners from './components/CategoryWinners'
import AdvancedFilters from './components/AdvancedFilters'
import LeaderboardTable from './components/LeaderboardTable'
import ComparisonModal from './components/ComparisonModal'
import FloatingActionMenu from './components/FloatingActionMenu'
import ScrollToTop from './components/ScrollToTop'
import ParticleBackground from './components/ParticleBackground'
import Footer from './components/Footer'
import Documentation from './pages/Documentation'
import { ThemeProvider } from './context/ThemeContext'

const PerformanceCharts = lazy(() => import('./components/PerformanceCharts'))

function HomePage() {
  const rankedData = useMemo(() => calculateRanks(modelData), [])

  // State
  const [searchTerm, setSearchTerm] = useState('')
  const [sortConfig, setSortConfig] = useState({ column: 'accuracy', order: 'desc' })
  const [selectedModels, setSelectedModels] = useState(new Set())
  const [showComparisonModal, setShowComparisonModal] = useState(false)
  const [detailModel, setDetailModel] = useState(null)
  const [filters, setFilters] = useState({
    accuracyMin: 0,
    accuracyMax: 100,
    family: '',
    size: '',
    efficiency: 0,
    quantization: '',
  })
  const [scrollProgress, setScrollProgress] = useState(0)

  // Mouse tracking for cursor glow
  useEffect(() => {
    const handleMouseMove = (e) => {
      document.documentElement.style.setProperty('--mouse-x', e.clientX + 'px')
      document.documentElement.style.setProperty('--mouse-y', e.clientY + 'px')
    }
    window.addEventListener('mousemove', handleMouseMove, { passive: true })
    return () => window.removeEventListener('mousemove', handleMouseMove)
  }, [])

  // Scroll progress
  useEffect(() => {
    const handleScroll = () => {
      const totalHeight = document.documentElement.scrollHeight - window.innerHeight
      const progress = (window.scrollY / totalHeight) * 100
      setScrollProgress(progress)
    }
    window.addEventListener('scroll', handleScroll, { passive: true })
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  // Intersection observer for reveals
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add('visible')
          }
        })
      },
      { threshold: 0.05, rootMargin: '0px 0px 50px 0px' }
    )

    const elements = document.querySelectorAll('.reveal, .reveal-left, .reveal-right, .reveal-scale')
    elements.forEach((el) => observer.observe(el))

    return () => observer.disconnect()
  })

  // Filtered + sorted data
  const processedData = useMemo(() => {
    let data = rankedData.filter((model) => {
      if (model.accuracy < filters.accuracyMin || model.accuracy > filters.accuracyMax) return false
      if (filters.family && !model.model.toLowerCase().includes(filters.family.toLowerCase())) return false
      if (filters.size) {
        const params = parseFloat(model.params.replace(/[^0-9.]/g, ''))
        if (isNaN(params)) {
          if (filters.size !== '') return false
        } else {
          if (filters.size === 'small' && params >= 3) return false
          if (filters.size === 'medium' && (params < 3 || params > 10)) return false
          if (filters.size === 'large' && params <= 10) return false
        }
      }
      if (model.efficiency < filters.efficiency) return false
      if (filters.quantization) {
        if (filters.quantization === 'normal' && model.quantization !== 'None') return false
        if (filters.quantization === 'quantized' && model.quantization === 'None') return false
        if (filters.quantization === 'GPTQ-8-Bit' && model.quantization !== 'GPTQ-8-Bit') return false
        if (filters.quantization === 'GPTQ-4-Bit' && model.quantization !== 'GPTQ-4-Bit') return false
      }
      if (searchTerm && !model.model.toLowerCase().includes(searchTerm.toLowerCase())) return false
      return true
    })

    data.sort((a, b) => {
      let aVal, bVal
      switch (sortConfig.column) {
        case 'model':
          aVal = a.model.toLowerCase()
          bVal = b.model.toLowerCase()
          break
        case 'params':
          aVal = parseFloat(a.params.replace(/[^0-9.]/g, '')) || 0
          bVal = parseFloat(b.params.replace(/[^0-9.]/g, '')) || 0
          break
        default:
          aVal = a[sortConfig.column]
          bVal = b[sortConfig.column]
      }
      if (sortConfig.order === 'asc') return aVal > bVal ? 1 : -1
      return aVal < bVal ? 1 : -1
    })

    return data
  }, [rankedData, searchTerm, sortConfig, filters])

  const handleSort = useCallback((column) => {
    setSortConfig((prev) => {
      if (prev.column === column) {
        return { column, order: prev.order === 'asc' ? 'desc' : 'asc' }
      }
      const defaultOrder = ['overthinking', 'tokens', 'words', 'chars'].includes(column) ? 'asc' : 'desc'
      return { column, order: defaultOrder }
    })
  }, [])

  const handleSortSelect = useCallback((column) => {
    const defaultOrder = ['overthinking', 'tokens', 'words', 'chars'].includes(column) ? 'asc' : 'desc'
    setSortConfig({ column, order: defaultOrder })
  }, [])

  const toggleModelSelection = useCallback((modelName) => {
    setSelectedModels((prev) => {
      const next = new Set(prev)
      if (next.has(modelName)) {
        next.delete(modelName)
      } else {
        if (next.size >= 5) {
          alert('You can compare up to 5 models at once.')
          return prev
        }
        next.add(modelName)
      }
      return next
    })
  }, [])

  const clearComparison = useCallback(() => {
    setSelectedModels(new Set())
  }, [])

  const openComparisonModal = useCallback(() => {
    if (selectedModels.size < 2) {
      alert('Please select at least 2 models to compare.')
      return
    }
    setDetailModel(null)
    setShowComparisonModal(true)
  }, [selectedModels])

  const showModelDetail = useCallback((model) => {
    setDetailModel(model)
    setShowComparisonModal(true)
  }, [])

  const exportData = useCallback(() => {
    const csvData = [
      ['Rank', 'Model', 'Parameters', 'Quantization', 'Accuracy', 'O-Score', 'Instruction Following', 'Verbose Ratio', 'Tokens', 'Words', 'Characters'],
      ...processedData.map((m) => [
        m.rank, m.model, m.params, m.quantization, m.accuracy, m.efficiency, m.instruction,
        m.overthinking, m.tokens, m.words, m.chars,
      ]),
    ]
      .map((row) => row.join(','))
      .join('\n')

    const blob = new Blob([csvData], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'llmthinkbench_leaderboard.csv'
    a.click()
    URL.revokeObjectURL(url)
  }, [processedData])

  const resetFilters = useCallback(() => {
    setFilters({
      accuracyMin: 0,
      accuracyMax: 100,
      family: '',
      size: '',
      efficiency: 0,
      quantization: '',
    })
    setSearchTerm('')
  }, [])

  const scrollToSection = useCallback((id) => {
    const el = document.getElementById(id)
    if (el) el.scrollIntoView({ behavior: 'smooth' })
  }, [])

  return (
    <div className="relative min-h-screen">
      {/* Noise overlay */}
      <div className="noise-overlay" />

      {/* Scroll progress bar */}
      <div className="scroll-progress" style={{ width: `${scrollProgress}%` }} />

      {/* Aurora / Gradient Mesh Background */}
      <div className="aurora-bg">
        <div
          className="aurora-blob"
          style={{
            width: '600px',
            height: '600px',
            top: '10%',
            left: '20%',
            background: 'radial-gradient(circle, rgba(102, 126, 234, 0.08) 0%, transparent 70%)',
            animationDuration: '25s',
          }}
        />
        <div
          className="aurora-blob"
          style={{
            width: '500px',
            height: '500px',
            top: '40%',
            right: '10%',
            background: 'radial-gradient(circle, rgba(124, 77, 255, 0.06) 0%, transparent 70%)',
            animationDuration: '30s',
            animationDelay: '5s',
          }}
        />
        <div
          className="aurora-blob"
          style={{
            width: '700px',
            height: '700px',
            bottom: '10%',
            left: '40%',
            background: 'radial-gradient(circle, rgba(240, 147, 251, 0.05) 0%, transparent 70%)',
            animationDuration: '35s',
            animationDelay: '10s',
          }}
        />
        <div
          className="aurora-blob"
          style={{
            width: '400px',
            height: '400px',
            top: '60%',
            left: '5%',
            background: 'radial-gradient(circle, rgba(0, 229, 255, 0.04) 0%, transparent 70%)',
            animationDuration: '28s',
            animationDelay: '8s',
          }}
        />
      </div>

      {/* Particle background */}
      <ParticleBackground />

      {/* Deep space base gradient */}
      <div className="fixed inset-0 z-[-2]">
        <div className="absolute inset-0 bg-gradient-to-br from-space-900 via-space-800 to-space-900" />
        <div className="absolute inset-0 dot-grid opacity-40" />
        <div
          className="absolute inset-0 opacity-20"
          style={{
            background:
              'radial-gradient(ellipse at 30% 20%, rgba(102, 126, 234, 0.1) 0%, transparent 50%), radial-gradient(ellipse at 70% 80%, rgba(240, 147, 251, 0.06) 0%, transparent 50%), radial-gradient(ellipse at 50% 50%, rgba(0, 229, 255, 0.04) 0%, transparent 60%)',
          }}
        />
      </div>

      <Navbar scrollToSection={scrollToSection} />

      <main>
        <HeroSection rankedData={rankedData} />

        {/* Section divider */}
        <div className="section-divider my-8" />

        <div id="winners">
          <CategoryWinners rankedData={rankedData} />
        </div>

        <div className="section-divider my-8" />

        <div id="filters">
          <AdvancedFilters filters={filters} setFilters={setFilters} resetFilters={resetFilters} />
        </div>

        <div id="leaderboard">
          <LeaderboardTable
            data={processedData}
            sortConfig={sortConfig}
            onSort={handleSort}
            onSortSelect={handleSortSelect}
            searchTerm={searchTerm}
            onSearchChange={setSearchTerm}
            selectedModels={selectedModels}
            onToggleModel={toggleModelSelection}
            onShowDetail={showModelDetail}
            onOpenComparison={openComparisonModal}
          />
        </div>

        <div className="section-divider my-8" />

        <div id="charts">
          <Suspense
            fallback={
              <div className="max-w-7xl mx-auto px-8 py-16">
                <div className="text-center">
                  <div className="futuristic-spinner mx-auto" />
                  <p className="mt-6 text-zinc-500 text-sm tracking-wider uppercase">Initializing performance analytics...</p>
                </div>
              </div>
            }
          >
            <PerformanceCharts rankedData={rankedData} />
          </Suspense>
        </div>
      </main>

      <Footer />

      <FloatingActionMenu
        onExport={exportData}
        onCompare={openComparisonModal}
        onFilter={() => scrollToSection('filters')}
      />

      <ScrollToTop />

      {/* Comparison / Detail Modal */}
      {showComparisonModal && (
        <ComparisonModal
          rankedData={rankedData}
          selectedModels={selectedModels}
          detailModel={detailModel}
          onClose={() => {
            setShowComparisonModal(false)
            setDetailModel(null)
          }}
          onToggleModel={toggleModelSelection}
        />
      )}

      {/* Compare panel at bottom */}
      {selectedModels.size > 0 && (
        <div className="fixed bottom-0 left-0 right-0 z-[1500] glass-strong border-t border-neon-indigo/30 p-4 transform transition-all duration-500 animate-slide-up-fade">
          <div className="max-w-7xl mx-auto">
            <div className="flex justify-between items-center mb-3">
              <h3 className="text-neon-indigo font-semibold flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-neon-indigo animate-pulse" />
                Selected Models ({selectedModels.size}/5)
              </h3>
              <div className="flex gap-2">
                <button
                  onClick={openComparisonModal}
                  className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-neon-indigo to-neon-purple text-white font-medium text-sm hover:shadow-lg hover:shadow-neon-indigo/30 transition-all btn-press"
                >
                  View Comparison
                </button>
                <button
                  onClick={clearComparison}
                  className="px-4 py-2 rounded-xl border border-red-500/30 text-red-400 font-medium text-sm hover:bg-red-500/10 hover:border-red-500/50 transition-all btn-press"
                >
                  Clear All
                </button>
              </div>
            </div>
            <div className="flex gap-2 overflow-x-auto pb-1">
              {Array.from(selectedModels).map((modelName) => {
                const model = rankedData.find((m) => m.model === modelName)
                return (
                  <div
                    key={modelName}
                    className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-neon-indigo/10 border border-neon-indigo/20 whitespace-nowrap text-sm hover:border-neon-indigo/40 transition-all"
                  >
                    <span className="text-zinc-300">{model?.model.split('/').pop()}</span>
                    <button
                      onClick={() => toggleModelSelection(modelName)}
                      className="text-red-400 hover:text-red-300 transition-colors hover:scale-110 transform"
                    >
                      x
                    </button>
                  </div>
                )
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function DocsPage() {
  // Ensure .reveal elements (footer, etc.) fade in on /docs
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) entry.target.classList.add('visible')
        })
      },
      { threshold: 0.05, rootMargin: '0px 0px 50px 0px' }
    )
    const elements = document.querySelectorAll('.reveal, .reveal-left, .reveal-right, .reveal-scale')
    elements.forEach((el) => observer.observe(el))
    return () => observer.disconnect()
  }, [])

  return (
    <div className="min-h-screen bg-space-900 relative overflow-hidden">
      {/* Ambient background orbs for visual consistency with landing */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        <div className="absolute top-0 -left-48 w-[500px] h-[500px] bg-neon-indigo/10 rounded-full blur-[120px]" />
        <div className="absolute top-1/3 -right-48 w-[500px] h-[500px] bg-neon-purple/10 rounded-full blur-[120px]" />
        <div className="absolute bottom-0 left-1/3 w-[400px] h-[400px] bg-neon-cyan/5 rounded-full blur-[120px]" />
      </div>
      <div className="relative z-10">
        <Navbar />
        <Documentation />
        <Footer />
      </div>
    </div>
  )
}

function App() {
  return (
    <ThemeProvider>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/docs" element={<DocsPage />} />
        <Route path="*" element={<HomePage />} />
      </Routes>
    </ThemeProvider>
  )
}

export default App
