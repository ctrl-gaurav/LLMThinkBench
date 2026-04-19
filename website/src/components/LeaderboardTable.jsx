import React, { memo } from 'react'
import { Search, ArrowUpDown, ArrowUp, ArrowDown, Plus, Check, BarChart3 } from 'lucide-react'
import { getMetricClass } from '../data/modelData'

const LeaderboardTable = memo(function LeaderboardTable({
  data,
  sortConfig,
  onSort,
  onSortSelect,
  searchTerm,
  onSearchChange,
  selectedModels,
  onToggleModel,
  onShowDetail,
  onOpenComparison,
}) {
  const columns = [
    { key: 'rank', label: 'Rank' },
    { key: 'model', label: 'Model' },
    { key: 'params', label: 'Params' },
    { key: 'accuracy', label: 'Accuracy' },
    { key: 'efficiency', label: 'Efficiency' },
    { key: 'instruction', label: 'Instr. Following' },
    { key: 'overthinking', label: 'Overthinking' },
    { key: 'tokens', label: 'Avg Tokens' },
    { key: 'words', label: 'Avg Words' },
    { key: 'chars', label: 'Avg Chars' },
  ]

  const getSortIcon = (key) => {
    if (sortConfig.column !== key) return <ArrowUpDown size={12} className="opacity-20 group-hover/th:opacity-50 transition-opacity" />
    return sortConfig.order === 'asc' ? (
      <ArrowUp size={12} className="text-neon-indigo drop-shadow-[0_0_4px_rgba(102,126,234,0.6)]" />
    ) : (
      <ArrowDown size={12} className="text-neon-indigo drop-shadow-[0_0_4px_rgba(102,126,234,0.6)]" />
    )
  }

  const getRankDisplay = (rank) => {
    if (rank === 1)
      return (
        <span className="inline-flex items-center">
          <span className="w-7 h-7 rounded-lg bg-gradient-to-br from-amber-400 to-yellow-600 flex items-center justify-center text-xs font-black text-white rank-gold">
            1
          </span>
        </span>
      )
    if (rank === 2)
      return (
        <span className="inline-flex items-center">
          <span className="w-7 h-7 rounded-lg bg-gradient-to-br from-zinc-300 to-zinc-500 flex items-center justify-center text-xs font-black text-space-900 rank-silver">
            2
          </span>
        </span>
      )
    if (rank === 3)
      return (
        <span className="inline-flex items-center">
          <span className="w-7 h-7 rounded-lg bg-gradient-to-br from-amber-600 to-amber-800 flex items-center justify-center text-xs font-black text-white rank-bronze">
            3
          </span>
        </span>
      )
    return <span className="text-zinc-500 font-mono text-sm tracking-tight">#{rank}</span>
  }

  return (
    <section className="max-w-7xl mx-auto px-6 mb-16">
      {/* Controls */}
      <div className="flex flex-col sm:flex-row gap-3 mb-6 reveal">
        {/* Search input - sci-fi feel */}
        <div className="relative flex-1 group">
          <Search size={18} className="absolute left-4 top-1/2 -translate-y-1/2 text-zinc-600 group-focus-within:text-neon-indigo transition-colors duration-300" />
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => onSearchChange(e.target.value)}
            placeholder="Search models... (Try 'Qwen', 'Phi', 'GPT', etc.)"
            className="w-full pl-12 pr-4 py-3.5 rounded-xl bg-white/[0.03] border border-white/10 text-zinc-200 text-sm placeholder-zinc-600 input-glow transition-all hover:border-white/15"
          />
          {/* Glowing bottom border on focus */}
          <div className="absolute bottom-0 left-4 right-4 h-px bg-gradient-to-r from-transparent via-neon-indigo/0 to-transparent group-focus-within:via-neon-indigo/50 transition-all duration-500" />
        </div>

        <select
          value={sortConfig.column}
          onChange={(e) => onSortSelect(e.target.value)}
          className="px-4 py-3.5 rounded-xl bg-white/[0.03] border border-white/10 text-zinc-300 text-sm input-glow transition-all appearance-none cursor-pointer min-w-[200px] hover:border-white/15"
        >
          <option value="rank">Sort by Rank</option>
          <option value="accuracy">Sort by Accuracy</option>
          <option value="efficiency">Sort by Efficiency</option>
          <option value="instruction">Sort by Instruction Following</option>
          <option value="overthinking">Sort by Overthinking Ratio</option>
          <option value="tokens">Sort by Tokens</option>
          <option value="words">Sort by Words</option>
          <option value="chars">Sort by Chars</option>
          <option value="params">Sort by Parameters</option>
        </select>

        {/* Compare button - animated gradient */}
        <button
          onClick={onOpenComparison}
          className="relative flex items-center gap-2.5 px-6 py-3.5 rounded-xl bg-gradient-to-r from-neon-indigo to-neon-purple text-white font-semibold text-sm hover:shadow-neon-indigo hover:-translate-y-0.5 transition-all whitespace-nowrap btn-press overflow-hidden group"
        >
          <div className="absolute inset-0 bg-gradient-to-r from-neon-purple to-neon-indigo opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
          <BarChart3 size={16} className="relative z-10" />
          <span className="relative z-10">Compare Models</span>
        </button>
      </div>

      {/* Results count */}
      <div className="mb-4 flex items-center gap-2">
        <span className="w-1.5 h-1.5 rounded-full bg-neon-indigo/50 animate-pulse" />
        <span className="text-sm text-zinc-500 tracking-wide">
          Showing <span className="text-zinc-300 font-semibold">{data.length}</span> model{data.length !== 1 ? 's' : ''}
        </span>
      </div>

      {/* Table */}
      <div className="reveal-scale overflow-hidden rounded-2xl border border-white/[0.06] glass-card">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="bg-gradient-to-r from-neon-indigo/[0.08] via-neon-purple/[0.05] to-neon-indigo/[0.08] border-b border-white/[0.06]">
                {columns.map((col) => (
                  <th
                    key={col.key}
                    onClick={() => onSort(col.key)}
                    className="group/th px-4 py-4 text-left text-[10px] font-bold text-neon-indigo uppercase tracking-[0.2em] cursor-pointer hover:bg-neon-indigo/[0.06] transition-all whitespace-nowrap select-none relative"
                  >
                    <div className="flex items-center gap-2">
                      {col.label}
                      {getSortIcon(col.key)}
                    </div>
                    {/* Hover underline */}
                    <div className="absolute bottom-0 left-2 right-2 h-px bg-neon-indigo/0 group-hover/th:bg-neon-indigo/30 transition-all duration-300" />
                  </th>
                ))}
                <th className="px-4 py-4 text-[10px] font-bold text-neon-indigo uppercase tracking-[0.2em]">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody>
              {data.map((model, index) => {
                const isSelected = selectedModels.has(model.model)
                return (
                  <tr
                    key={`${model.model}-${model.quantization}-${index}`}
                    onClick={() => onShowDetail(model)}
                    className={`border-b border-white/[0.03] cursor-pointer transition-all duration-300 group table-row-animate table-row-glow ${
                      index % 2 === 0 ? 'bg-white/[0.01]' : 'bg-transparent'
                    } ${isSelected ? 'bg-neon-indigo/[0.05]' : ''}`}
                    style={{ animationDelay: `${Math.min(index * 30, 600)}ms` }}
                  >
                    <td className="px-4 py-3.5">{getRankDisplay(model.rank)}</td>
                    <td className="px-4 py-3.5">
                      <div>
                        <span className="font-semibold text-zinc-200 text-sm group-hover:text-white transition-colors duration-300">
                          {model.model.split('/').pop()}
                        </span>
                        {model.model.includes('/') && (
                          <div className="text-[10px] text-zinc-600 mt-0.5 font-mono">
                            {model.model.split('/')[0]}
                          </div>
                        )}
                        {model.quantization !== 'None' && (
                          <span className="inline-block mt-1 px-2 py-0.5 text-[9px] font-semibold rounded-md bg-neon-purple/10 text-neon-purple border border-neon-purple/15 tracking-wider">
                            {model.quantization}
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="px-4 py-3.5 text-sm font-mono text-zinc-500">{model.params}</td>
                    <td className="px-4 py-3.5">
                      <div className="flex items-center gap-2.5">
                        <span className={`text-sm font-bold font-mono ${getMetricClass(model.accuracy, 'accuracy')}`}>
                          {model.accuracy.toFixed(2)}%
                        </span>
                        <div className="hidden lg:block w-20 h-1.5 rounded-full bg-white/5 overflow-hidden">
                          <div
                            className="h-full rounded-full bg-gradient-to-r from-neon-indigo via-neon-purple to-neon-pink progress-bar-fill"
                            style={{ width: `${model.accuracy}%` }}
                          />
                        </div>
                      </div>
                    </td>
                    <td className={`px-4 py-3.5 text-sm font-bold font-mono ${getMetricClass(model.efficiency * 100, 'efficiency')}`}>
                      {model.efficiency.toFixed(3)}
                    </td>
                    <td className={`px-4 py-3.5 text-sm font-bold font-mono ${getMetricClass(model.instruction, 'instruction')}`}>
                      {model.instruction.toFixed(2)}%
                    </td>
                    <td className={`px-4 py-3.5 text-sm font-mono ${getMetricClass(model.overthinking, 'overthinking')}`}>
                      {model.overthinking.toFixed(1)}
                    </td>
                    <td className={`px-4 py-3.5 text-sm font-mono ${getMetricClass(model.tokens, 'tokens')}`}>
                      {model.tokens.toFixed(1)}
                    </td>
                    <td className={`px-4 py-3.5 text-sm font-mono ${getMetricClass(model.words, 'words')}`}>
                      {model.words.toFixed(1)}
                    </td>
                    <td className={`px-4 py-3.5 text-sm font-mono ${getMetricClass(model.chars, 'chars')}`}>
                      {model.chars.toFixed(1)}
                    </td>
                    <td className="px-4 py-3.5">
                      <button
                        onClick={(e) => {
                          e.stopPropagation()
                          onToggleModel(model.model)
                        }}
                        className={`w-8 h-8 rounded-lg flex items-center justify-center transition-all duration-300 btn-press ${
                          isSelected
                            ? 'bg-gradient-to-r from-neon-indigo to-neon-purple text-white shadow-lg shadow-neon-indigo/30 scale-105'
                            : 'bg-white/[0.03] text-zinc-600 hover:bg-neon-indigo/15 hover:text-neon-indigo border border-white/10 hover:border-neon-indigo/30'
                        }`}
                      >
                        {isSelected ? <Check size={14} /> : <Plus size={14} />}
                      </button>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>

        {data.length === 0 && (
          <div className="py-20 text-center text-zinc-500">
            <Search size={48} className="mx-auto mb-4 opacity-20" />
            <p className="text-lg font-semibold text-zinc-400">No models match your filters</p>
            <p className="text-sm mt-1 text-zinc-600">Try adjusting your search or filter criteria</p>
          </div>
        )}
      </div>
    </section>
  )
})

export default LeaderboardTable
