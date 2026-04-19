import React, { useState } from 'react'
import { Github, Package, Bug, Copy, Check } from 'lucide-react'
import { usePyPIVersion } from '../hooks/usePyPIVersion'

export default function Footer() {
  const pypiVersion = usePyPIVersion()
  const [copied, setCopied] = useState(false)

  const citation = `@inproceedings{srivastava2026llmthinkbench,
  title     = {Do LLMs Overthink Basic Math Reasoning? Benchmarking the Accuracy-Efficiency Tradeoff in Language Models},
  author    = {Gaurav Srivastava and Aafiya Hussain and Sriram Srinivasan and Xuan Wang},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2507.04023}
}`

  const handleCopy = () => {
    navigator.clipboard.writeText(citation).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }

  return (
    <footer id="citation" className="relative mt-24">
      {/* Dramatic animated gradient separator */}
      <div className="relative h-px overflow-hidden">
        <div
          className="absolute inset-0 h-px"
          style={{
            background: 'linear-gradient(90deg, transparent, #667eea, #7C4DFF, #f093fb, #7C4DFF, #667eea, transparent)',
            backgroundSize: '200% 100%',
            animation: 'borderGradient 4s linear infinite',
          }}
        />
      </div>
      <div className="relative h-px overflow-hidden mt-px opacity-50">
        <div
          className="absolute inset-0 h-px"
          style={{
            background: 'linear-gradient(90deg, transparent, #7C4DFF, #f093fb, #00E5FF, #f093fb, #7C4DFF, transparent)',
            backgroundSize: '200% 100%',
            animation: 'borderGradient 6s linear infinite reverse',
          }}
        />
      </div>

      <div className="bg-gradient-to-b from-space-600/50 to-space-900/80 backdrop-blur-md py-14 md:py-16">
        <div className="max-w-5xl mx-auto px-6">
          {/* Citation - Terminal style */}
          <div className="reveal mb-12">
            <h3 className="text-lg font-bold gradient-text mb-6 text-center tracking-wide">Citation</h3>
            <div className="relative glass-card rounded-2xl p-6 md:p-8 group overflow-hidden">
              {/* Terminal header bar */}
              <div className="flex items-center gap-2 mb-4 pb-3 border-b border-white/[0.06]">
                <div className="w-3 h-3 rounded-full bg-red-500/60" />
                <div className="w-3 h-3 rounded-full bg-amber-500/60" />
                <div className="w-3 h-3 rounded-full bg-emerald-500/60" />
                <span className="ml-3 text-[10px] text-zinc-600 font-mono tracking-wider">bibtex</span>
              </div>

              <button
                onClick={handleCopy}
                className="absolute top-5 right-5 flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-xs font-semibold text-zinc-500 hover:text-neon-indigo hover:bg-neon-indigo/10 transition-all opacity-0 group-hover:opacity-100 btn-press border border-transparent hover:border-neon-indigo/15"
              >
                {copied ? (
                  <>
                    <Check size={14} className="text-emerald-400" />
                    <span className="text-emerald-400">Copied!</span>
                  </>
                ) : (
                  <>
                    <Copy size={14} />
                    Copy
                  </>
                )}
              </button>

              <pre className="font-mono text-sm text-zinc-400 whitespace-pre-wrap overflow-x-auto leading-relaxed">
                <span className="text-neon-indigo/70">$ </span>
                <span className="text-zinc-500">cat citation.bib</span>
                {'\n\n'}
                {citation}
                <span className="terminal-cursor" />
              </pre>
            </div>
          </div>

          {/* Links */}
          <div className="flex flex-wrap justify-center gap-6 md:gap-8 mb-10 reveal">
            <a
              href="https://github.com/ctrl-gaurav/LLMThinkBench"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2.5 text-sm text-zinc-500 hover:text-neon-indigo transition-all duration-300 group neon-hover btn-press"
            >
              <Github size={18} className="group-hover:scale-110 group-hover:drop-shadow-[0_0_10px_rgba(102,126,234,0.6)] transition-all duration-300" />
              <span className="tracking-wide">GitHub Repository</span>
            </a>
            <a
              href="https://pypi.org/project/llmthinkbench/"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2.5 text-sm text-zinc-500 hover:text-neon-purple transition-all duration-300 group neon-hover btn-press"
            >
              <Package size={18} className="group-hover:scale-110 group-hover:drop-shadow-[0_0_10px_rgba(124,77,255,0.6)] transition-all duration-300" />
              <span className="tracking-wide">PyPI{pypiVersion ? ` v${pypiVersion}` : ''}</span>
            </a>
            <a
              href="https://github.com/ctrl-gaurav/LLMThinkBench/issues"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2.5 text-sm text-zinc-500 hover:text-neon-pink transition-all duration-300 group neon-hover btn-press"
            >
              <Bug size={18} className="group-hover:scale-110 group-hover:drop-shadow-[0_0_10px_rgba(240,147,251,0.6)] transition-all duration-300" />
              <span className="tracking-wide">Report Issues</span>
            </a>
          </div>

          {/* Copyright */}
          <div className="text-center">
            <div className="inline-flex items-center gap-2 text-xs text-zinc-700">
              <span className="w-1 h-1 rounded-full bg-zinc-700" />
              <span className="tracking-wider">&copy; 2025 LLMThinkBench. Licensed under MIT.</span>
              <span className="w-1 h-1 rounded-full bg-zinc-700" />
            </div>
          </div>
        </div>
      </div>
    </footer>
  )
}
