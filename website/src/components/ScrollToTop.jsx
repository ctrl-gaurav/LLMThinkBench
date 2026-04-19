import React, { useState, useEffect } from 'react'
import { ArrowUp } from 'lucide-react'

export default function ScrollToTop() {
  const [visible, setVisible] = useState(false)

  useEffect(() => {
    const handleScroll = () => {
      setVisible(window.scrollY > 400)
    }
    window.addEventListener('scroll', handleScroll, { passive: true })
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  return (
    <div
      className={`fixed bottom-24 right-8 z-[999] transition-all duration-500 ${
        visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4 pointer-events-none'
      }`}
    >
      <div className="relative">
        {/* Glow ring */}
        <div className="absolute -inset-1.5 rounded-full bg-gradient-to-r from-neon-indigo to-neon-purple opacity-20 blur-md animate-pulse-ring" />
        <button
          onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
          className="relative w-12 h-12 rounded-full bg-gradient-to-r from-neon-indigo to-neon-purple text-white flex items-center justify-center shadow-lg shadow-neon-indigo/30 hover:scale-110 hover:shadow-neon-indigo/50 transition-all duration-300 btn-press"
          aria-label="Scroll to top"
        >
          <ArrowUp size={20} />
        </button>
      </div>
    </div>
  )
}
