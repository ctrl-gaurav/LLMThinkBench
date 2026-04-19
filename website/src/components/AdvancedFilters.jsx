import React, { useState } from 'react'
import { Filter, RotateCcw, ChevronDown, ChevronUp, Sliders } from 'lucide-react'

export default function AdvancedFilters({ filters, setFilters, resetFilters }) {
  const [expanded, setExpanded] = useState(true)

  const activeCount = [
    filters.accuracyMin > 0,
    filters.accuracyMax < 100,
    filters.family !== '',
    filters.size !== '',
    filters.efficiency > 0,
    filters.quantization !== '',
  ].filter(Boolean).length

  return (
    <section className="max-w-7xl mx-auto px-6 mb-8">
      <div className="glass-card rounded-2xl overflow-hidden transition-all duration-500 hover:border-neon-indigo/25 reveal group/panel">
        {/* Header - Control panel style */}
        <button
          className="w-full flex items-center justify-between p-5 md:p-6 text-left btn-press"
          onClick={() => setExpanded(!expanded)}
        >
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-neon-indigo/15 to-neon-purple/10 flex items-center justify-center border border-neon-indigo/10 group-hover/panel:border-neon-indigo/20 transition-all">
              <Sliders size={18} className="text-neon-indigo" />
            </div>
            <div>
              <h3 className="text-lg font-bold text-white tracking-wide">Advanced Filters</h3>
              <p className="text-[10px] text-zinc-600 uppercase tracking-[0.2em] mt-0.5">Control Panel</p>
            </div>
            {activeCount > 0 && (
              <span className="relative px-3 py-1 rounded-full bg-neon-indigo/15 text-neon-indigo text-xs font-bold border border-neon-indigo/20">
                {activeCount} active
                <span className="absolute inset-0 rounded-full border border-neon-indigo/30 animate-pulse-ring" />
              </span>
            )}
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={(e) => {
                e.stopPropagation()
                resetFilters()
              }}
              className="flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-xs font-semibold text-zinc-500 hover:text-white hover:bg-white/5 border border-transparent hover:border-white/10 transition-all btn-press"
            >
              <RotateCcw size={14} className="group-hover:rotate-[-180deg] transition-transform duration-500" />
              Reset
            </button>
            <div className={`w-8 h-8 rounded-lg bg-white/5 flex items-center justify-center transition-all duration-500 ${expanded ? 'rotate-0' : 'rotate-180'}`}>
              <ChevronUp size={18} className="text-zinc-400" />
            </div>
          </div>
        </button>

        {/* Filter grid */}
        <div
          className={`overflow-hidden transition-all duration-700 ease-[cubic-bezier(0.23,1,0.32,1)] ${
            expanded ? 'max-h-[500px] opacity-100' : 'max-h-0 opacity-0'
          }`}
        >
          <div className="px-5 md:px-6 pb-6 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-5 md:gap-6">
            {/* Accuracy Range */}
            <div className="space-y-3">
              <label className="text-[10px] font-bold text-neon-indigo uppercase tracking-[0.2em] flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-neon-indigo animate-pulse" />
                Accuracy Range
              </label>
              <div className="space-y-2">
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={filters.accuracyMin}
                  onChange={(e) => setFilters((f) => ({ ...f, accuracyMin: Number(e.target.value) }))}
                  className="w-full"
                />
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={filters.accuracyMax}
                  onChange={(e) => setFilters((f) => ({ ...f, accuracyMax: Number(e.target.value) }))}
                  className="w-full"
                />
              </div>
              <div className="flex justify-between">
                <span className="text-xs text-zinc-500 font-mono bg-white/[0.03] px-2 py-0.5 rounded">
                  {filters.accuracyMin}%
                </span>
                <span className="text-xs text-zinc-500 font-mono bg-white/[0.03] px-2 py-0.5 rounded">
                  {filters.accuracyMax}%
                </span>
              </div>
            </div>

            {/* Model Family */}
            <div className="space-y-3">
              <label className="text-[10px] font-bold text-neon-indigo uppercase tracking-[0.2em] flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-neon-purple animate-pulse" style={{ animationDelay: '0.5s' }} />
                Model Family
              </label>
              <select
                value={filters.family}
                onChange={(e) => setFilters((f) => ({ ...f, family: e.target.value }))}
                className="w-full px-4 py-3 rounded-xl bg-white/[0.03] border border-white/10 text-zinc-300 text-sm focus:border-neon-indigo/50 focus:outline-none focus:ring-2 focus:ring-neon-indigo/20 focus:shadow-neon-indigo transition-all appearance-none cursor-pointer hover:border-white/20"
              >
                <option value="">All Families</option>
                <option value="Qwen">Qwen</option>
                <option value="Phi">Microsoft Phi</option>
                <option value="Llama">Meta Llama</option>
                <option value="Mistral">Mistral</option>
                <option value="SmolLM">SmolLM</option>
                <option value="GPT">GPT</option>
                <option value="gemini">Gemini</option>
              </select>
            </div>

            {/* Parameter Size */}
            <div className="space-y-3">
              <label className="text-[10px] font-bold text-neon-indigo uppercase tracking-[0.2em] flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-neon-pink animate-pulse" style={{ animationDelay: '1s' }} />
                Parameter Size
              </label>
              <select
                value={filters.size}
                onChange={(e) => setFilters((f) => ({ ...f, size: e.target.value }))}
                className="w-full px-4 py-3 rounded-xl bg-white/[0.03] border border-white/10 text-zinc-300 text-sm focus:border-neon-indigo/50 focus:outline-none focus:ring-2 focus:ring-neon-indigo/20 focus:shadow-neon-indigo transition-all appearance-none cursor-pointer hover:border-white/20"
              >
                <option value="">All Sizes</option>
                <option value="small">{'Small (< 3B)'}</option>
                <option value="medium">Medium (3B - 10B)</option>
                <option value="large">{'Large (> 10B)'}</option>
              </select>
            </div>

            {/* Efficiency Threshold */}
            <div className="space-y-3">
              <label className="text-[10px] font-bold text-neon-indigo uppercase tracking-[0.2em] flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-neon-cyan animate-pulse" style={{ animationDelay: '1.5s' }} />
                Min Efficiency
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={filters.efficiency}
                onChange={(e) => setFilters((f) => ({ ...f, efficiency: Number(e.target.value) }))}
                className="w-full"
              />
              <span className="text-xs text-zinc-500 font-mono bg-white/[0.03] px-2 py-0.5 rounded inline-block">
                {filters.efficiency.toFixed(3)}
              </span>
            </div>

            {/* Quantization */}
            <div className="space-y-3">
              <label className="text-[10px] font-bold text-neon-indigo uppercase tracking-[0.2em] flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" style={{ animationDelay: '2s' }} />
                Quantization
              </label>
              <select
                value={filters.quantization}
                onChange={(e) => setFilters((f) => ({ ...f, quantization: e.target.value }))}
                className="w-full px-4 py-3 rounded-xl bg-white/[0.03] border border-white/10 text-zinc-300 text-sm focus:border-neon-indigo/50 focus:outline-none focus:ring-2 focus:ring-neon-indigo/20 focus:shadow-neon-indigo transition-all appearance-none cursor-pointer hover:border-white/20"
              >
                <option value="">All Models</option>
                <option value="normal">Normal Only</option>
                <option value="quantized">Quantized Only</option>
                <option value="GPTQ-8-Bit">GPTQ 8-Bit</option>
                <option value="GPTQ-4-Bit">GPTQ 4-Bit</option>
              </select>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
