import React, { useState } from 'react'
import { Plus, Download, BarChart3, Filter } from 'lucide-react'

export default function FloatingActionMenu({ onExport, onCompare, onFilter }) {
  const [open, setOpen] = useState(false)

  const actions = [
    {
      icon: Download,
      label: 'Export Data',
      onClick: () => { onExport(); setOpen(false) },
      color: 'hover:bg-emerald-500/15 hover:text-emerald-400 hover:border-emerald-500/30 hover:shadow-[0_0_15px_rgba(16,185,129,0.2)]',
      activeColor: 'bg-emerald-500/10 border-emerald-500/20',
    },
    {
      icon: BarChart3,
      label: 'Compare Models',
      onClick: () => { onCompare(); setOpen(false) },
      color: 'hover:bg-neon-indigo/15 hover:text-neon-indigo hover:border-neon-indigo/30 hover:shadow-[0_0_15px_rgba(102,126,234,0.2)]',
      activeColor: 'bg-neon-indigo/10 border-neon-indigo/20',
    },
    {
      icon: Filter,
      label: 'Filters',
      onClick: () => { onFilter(); setOpen(false) },
      color: 'hover:bg-neon-purple/15 hover:text-neon-purple hover:border-neon-purple/30 hover:shadow-[0_0_15px_rgba(124,77,255,0.2)]',
      activeColor: 'bg-neon-purple/10 border-neon-purple/20',
    },
  ]

  return (
    <div className="fixed bottom-8 right-8 z-[999] flex flex-col items-center gap-3">
      {/* Menu items */}
      <div
        className={`flex flex-col gap-3 transition-all duration-500 ${
          open ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-6 pointer-events-none'
        }`}
      >
        {actions.map((action, i) => {
          const Icon = action.icon
          return (
            <div
              key={i}
              className="relative group"
              style={{
                transition: `all 0.4s cubic-bezier(0.23, 1, 0.32, 1)`,
                transitionDelay: open ? `${i * 80}ms` : '0ms',
                opacity: open ? 1 : 0,
                transform: open ? 'translateY(0) scale(1)' : 'translateY(10px) scale(0.8)',
              }}
            >
              <button
                onClick={action.onClick}
                className={`w-12 h-12 rounded-xl glass border border-white/10 flex items-center justify-center text-zinc-500 transition-all duration-300 btn-press ${action.color}`}
              >
                <Icon size={18} />
              </button>
              {/* Tooltip */}
              <div className="absolute right-full mr-4 top-1/2 -translate-y-1/2 px-3.5 py-2 rounded-xl glass-strong text-xs font-medium text-zinc-300 whitespace-nowrap opacity-0 group-hover:opacity-100 transition-all duration-300 pointer-events-none border border-white/10 group-hover:translate-x-0 translate-x-2">
                {action.label}
              </div>
            </div>
          )
        })}
      </div>

      {/* Main FAB with glow ring */}
      <div className="relative">
        {/* Pulsing glow ring when closed */}
        {!open && (
          <div className="absolute -inset-2 rounded-full bg-gradient-to-r from-neon-indigo to-neon-purple opacity-20 blur-md animate-pulse-ring" />
        )}
        <button
          onClick={() => setOpen(!open)}
          className="relative w-14 h-14 rounded-full bg-gradient-to-r from-neon-indigo to-neon-purple text-white flex items-center justify-center shadow-lg shadow-neon-indigo/30 hover:shadow-neon-indigo/50 hover:scale-110 transition-all duration-500 btn-press"
        >
          <Plus
            size={24}
            className="transition-transform duration-500"
            style={{ transform: open ? 'rotate(45deg)' : 'rotate(0deg)' }}
          />
        </button>
      </div>
    </div>
  )
}
