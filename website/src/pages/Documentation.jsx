import { useState, useMemo } from 'react'
import { BookOpen, Terminal, Code, Settings, Layers, Cpu, Package, FileText, Zap, Database, Shield, BarChart3, Wrench, AlertTriangle, Lightbulb, Search, ChevronDown } from 'lucide-react'
import { useTheme } from '../context/ThemeContext'
import { usePyPIVersion } from '../hooks/usePyPIVersion'

/* ============ SYNTAX HIGHLIGHTING ============ */

const darkColors = {
  comment: '#6b7280', keyword: '#c084fc', string: '#34d399', number: '#f59e0b',
  function: '#60a5fa', operator: '#9ca3af', punctuation: '#9ca3af', key: '#60a5fa',
  boolean: '#f59e0b', decorator: '#c084fc', flag: '#60a5fa', text: '#d1d5db',
}
const lightColors = {
  comment: '#9ca3af', keyword: '#7c3aed', string: '#059669', number: '#d97706',
  function: '#2563eb', operator: '#64748b', punctuation: '#64748b', key: '#2563eb',
  boolean: '#d97706', decorator: '#7c3aed', flag: '#2563eb', text: '#374151',
}

function tokenizePython(code) {
  const tokens = []
  const keywords = new Set(['import','from','as','def','return','class','if','else','elif','for','while','with','try','except','raise','True','False','None','print','self','in','not','and','or','is','lambda','yield','async','await','pass','break','continue','del','global','nonlocal','assert','finally'])
  let i = 0
  while (i < code.length) {
    if (code[i] === '#') { let e = code.indexOf('\n', i); if (e === -1) e = code.length; tokens.push({ type: 'comment', value: code.slice(i, e) }); i = e; continue }
    if (code.slice(i, i+3) === '"""' || code.slice(i, i+3) === "'''") { const q = code.slice(i, i+3); let e = code.indexOf(q, i+3); if (e === -1) e = code.length-3; tokens.push({ type: 'string', value: code.slice(i, e+3) }); i = e+3; continue }
    if (code[i] === '"' || code[i] === "'") { const q = code[i]; let j = i+1; while (j < code.length && code[j] !== q) { if (code[j] === '\\') j++; j++ } tokens.push({ type: 'string', value: code.slice(i, j+1) }); i = j+1; continue }
    if (code[i] === '@' && (i === 0 || code[i-1] === '\n' || /\s/.test(code[i-1]))) { let j = i+1; while (j < code.length && /[\w.]/.test(code[j])) j++; tokens.push({ type: 'decorator', value: code.slice(i, j) }); i = j; continue }
    if (/\d/.test(code[i]) && (i === 0 || !/[\w.]/.test(code[i-1]))) { let j = i; while (j < code.length && /[\d.eE_xXa-fA-F]/.test(code[j])) j++; tokens.push({ type: 'number', value: code.slice(i, j) }); i = j; continue }
    if (/[a-zA-Z_]/.test(code[i])) { let j = i; while (j < code.length && /[\w]/.test(code[j])) j++; const w = code.slice(i, j); if (keywords.has(w)) tokens.push({ type: 'keyword', value: w }); else if (j < code.length && code[j] === '(') tokens.push({ type: 'function', value: w }); else tokens.push({ type: 'text', value: w }); i = j; continue }
    if (/[=+\-*/<>!&|^~%]/.test(code[i])) { tokens.push({ type: 'operator', value: code[i] }); i++; continue }
    if (/[()[\]{},;:.]/.test(code[i])) { tokens.push({ type: 'punctuation', value: code[i] }); i++; continue }
    tokens.push({ type: 'text', value: code[i] }); i++
  }
  return tokens
}

function tokenizeBash(code) {
  const tokens = []
  const keywords = new Set(['python','pip','git','cd','source','export','npm','pytest','llmthinkbench','CUDA_VISIBLE_DEVICES','rm','tail','mkdir','echo','cat','chmod','sudo','apt','brew'])
  let i = 0
  while (i < code.length) {
    if (code[i] === '#') { let e = code.indexOf('\n', i); if (e === -1) e = code.length; tokens.push({ type: 'comment', value: code.slice(i, e) }); i = e; continue }
    if (code[i] === '"' || code[i] === "'") { const q = code[i]; let j = i+1; while (j < code.length && code[j] !== q) { if (code[j] === '\\') j++; j++ } tokens.push({ type: 'string', value: code.slice(i, j+1) }); i = j+1; continue }
    if (code[i] === '-' && i+1 < code.length && /[a-zA-Z-]/.test(code[i+1])) { let j = i; while (j < code.length && /[\w-]/.test(code[j])) j++; tokens.push({ type: 'flag', value: code.slice(i, j) }); i = j; continue }
    if (/[a-zA-Z_]/.test(code[i])) { let j = i; while (j < code.length && /[\w.\-/]/.test(code[j])) j++; const w = code.slice(i, j); if (keywords.has(w)) tokens.push({ type: 'keyword', value: w }); else tokens.push({ type: 'text', value: w }); i = j; continue }
    if (/\d/.test(code[i])) { let j = i; while (j < code.length && /[\d.]/.test(code[j])) j++; tokens.push({ type: 'number', value: code.slice(i, j) }); i = j; continue }
    tokens.push({ type: 'text', value: code[i] }); i++
  }
  return tokens
}

function tokenizeJson(code) {
  const tokens = []
  let i = 0
  while (i < code.length) {
    if (code[i] === '"') {
      let j = i+1; while (j < code.length && code[j] !== '"') { if (code[j] === '\\') j++; j++ }
      const s = code.slice(i, j+1); let k = j+1; while (k < code.length && /\s/.test(code[k])) k++
      tokens.push({ type: code[k] === ':' ? 'key' : 'string', value: s }); i = j+1; continue
    }
    if (/\d/.test(code[i]) || (code[i] === '-' && i+1 < code.length && /\d/.test(code[i+1]))) { let j = i; if (code[j] === '-') j++; while (j < code.length && /[\d.eE+-]/.test(code[j])) j++; tokens.push({ type: 'number', value: code.slice(i, j) }); i = j; continue }
    if (/[a-zA-Z]/.test(code[i])) { let j = i; while (j < code.length && /[a-zA-Z]/.test(code[j])) j++; const w = code.slice(i, j); tokens.push({ type: (w === 'true' || w === 'false' || w === 'null') ? 'boolean' : 'text', value: w }); i = j; continue }
    if (/[{}[\]:,]/.test(code[i])) { tokens.push({ type: 'punctuation', value: code[i] }); i++; continue }
    tokens.push({ type: 'text', value: code[i] }); i++
  }
  return tokens
}

function tokenizeBibtex(code) {
  const tokens = []
  let i = 0
  while (i < code.length) {
    if (code[i] === '@') { let j = i+1; while (j < code.length && /\w/.test(code[j])) j++; tokens.push({ type: 'decorator', value: code.slice(i, j) }); i = j; continue }
    if (code[i] === '{' || code[i] === '}') { tokens.push({ type: 'punctuation', value: code[i] }); i++; continue }
    if (/[a-zA-Z]/.test(code[i])) { let j = i; while (j < code.length && /[\w]/.test(code[j])) j++; const w = code.slice(i, j); let k = j; while (k < code.length && code[k] === ' ') k++; tokens.push({ type: code[k] === '=' ? 'key' : 'string', value: w }); i = j; continue }
    tokens.push({ type: 'text', value: code[i] }); i++
  }
  return tokens
}

function tokenize(code, language) {
  switch (language) {
    case 'python': case 'py': return tokenizePython(code)
    case 'bash': case 'shell': case 'sh': return tokenizeBash(code)
    case 'json': return tokenizeJson(code)
    case 'bibtex': return tokenizeBibtex(code)
    default: return [{ type: 'text', value: code }]
  }
}

function CodeBlock({ code, language = 'bash' }) {
  const [copied, setCopied] = useState(false)
  const { isDark } = useTheme()
  const colors = isDark ? darkColors : lightColors

  const highlighted = useMemo(() => {
    return tokenize(code, language).map((token, i) => {
      if (token.type === 'text') return <span key={i}>{token.value}</span>
      return <span key={i} style={{ color: colors[token.type] }}>{token.value}</span>
    })
  }, [code, language, colors])

  function handleCopy() {
    navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className={`relative group my-3 rounded-xl overflow-hidden transition-all duration-300 ${
      isDark
        ? 'bg-[#0a0e18] border border-bb-dark-50/15 hover:border-bb-accent/20 shadow-lg shadow-black/20'
        : 'bg-[#f7f8fc] border border-gray-200/60 hover:border-bb-accent-dark/30 shadow-sm'
    }`}>
      <div className={`flex items-center justify-between px-4 py-2.5 border-b ${
        isDark ? 'border-bb-dark-50/10 bg-white/[0.02]' : 'border-gray-200/30 bg-gray-50/50'
      }`}>
        <div className="flex items-center gap-2.5">
          <div className="flex gap-1.5">
            <div className="w-2.5 h-2.5 rounded-full bg-[#ff5f57]/80" />
            <div className="w-2.5 h-2.5 rounded-full bg-[#febc2e]/80" />
            <div className="w-2.5 h-2.5 rounded-full bg-[#28c840]/80" />
          </div>
          <span className={`text-[10px] uppercase tracking-wider font-mono ${isDark ? 'text-gray-600' : 'text-gray-400'}`}>{language}</span>
        </div>
        <button
          onClick={handleCopy}
          className={`text-xs font-mono px-2.5 py-1 rounded-md transition-all duration-200 ${
            copied
              ? isDark ? 'text-green-400 bg-green-500/10' : 'text-green-600 bg-green-500/10'
              : isDark ? 'text-gray-600 hover:text-gray-300 hover:bg-white/5' : 'text-gray-400 hover:text-gray-600 hover:bg-gray-100'
          }`}
        >
          {copied ? 'Copied!' : 'Copy'}
        </button>
      </div>
      <pre className={`p-4 overflow-x-auto text-[13px] font-mono leading-relaxed ${
        isDark ? 'text-gray-300' : 'text-gray-700'
      }`}>
        <code>{highlighted}</code>
      </pre>
    </div>
  )
}

function Callout({ type = 'info', children }) {
  const { isDark } = useTheme()
  const styles = {
    info: isDark ? 'border-bb-accent/40 bg-bb-accent/5' : 'border-bb-accent-dark/40 bg-bb-accent-dark/5',
    warning: 'border-yellow-500/40 bg-yellow-500/5',
    tip: 'border-green-400/40 bg-green-400/5',
  }
  const icons = {
    info: <Shield className={`w-4 h-4 shrink-0 mt-0.5 ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`} />,
    warning: <AlertTriangle className="w-4 h-4 text-yellow-500 shrink-0 mt-0.5" />,
    tip: <Lightbulb className="w-4 h-4 text-green-400 shrink-0 mt-0.5" />,
  }
  const labels = { info: 'Note', warning: 'Warning', tip: 'Tip' }
  const labelColors = {
    info: isDark ? 'text-bb-accent' : 'text-bb-accent-dark',
    warning: 'text-yellow-500',
    tip: 'text-green-400',
  }
  return (
    <div className={`border-l-4 rounded-xl p-4 ${styles[type]}`}>
      <div className={`text-xs font-mono font-bold uppercase tracking-wider mb-1 flex items-center gap-1.5 ${labelColors[type]}`}>
        {icons[type]}
        {labels[type]}
      </div>
      <div className={`text-sm leading-relaxed ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>{children}</div>
    </div>
  )
}

const NAV_ITEMS = [
  { id: 'overview', label: 'Overview', iconPath: 'M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253' },
  { id: 'installation', label: 'Installation', iconPath: 'M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4' },
  { id: 'quickstart', label: 'Quick Start', iconPath: 'M13 10V3L4 14h7v7l9-11h-7z' },
  { id: 'cli', label: 'CLI Reference', iconPath: 'M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z' },
  { id: 'python-api', label: 'Python API', iconPath: 'M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4' },
  { id: 'tasks', label: 'Tasks', iconPath: 'M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10' },
  { id: 'overthinking', label: 'Overthinking Score', iconPath: 'M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z' },
  { id: 'configuration', label: 'Configuration', iconPath: 'M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.066 2.573c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.573 1.066c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.066-2.573c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z' },
  { id: 'output', label: 'Output Format', iconPath: 'M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4' },
  { id: 'troubleshooting', label: 'Troubleshooting', iconPath: 'M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.066 2.573c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.573 1.066c-.426 1.756-2.924 1.756-3.35 0' },
  { id: 'citation', label: 'Citation', iconPath: 'M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z' },
]

/* ============ SECTION CONTENT ============ */

function OverviewContent({ isDark, cardCls, headCls, textCls }) {
  return (
    <div className="space-y-6">
      <div className={cardCls}>
        <h2 className={headCls}>Overview</h2>
        <p className={textCls}>
          <strong className={isDark ? 'text-gray-200' : 'text-gray-800'}>LLMThinkBench</strong> is a framework for evaluating <strong className={isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}>basic math reasoning</strong> and <strong className={isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}>overthinking</strong> in Large Language Models. It combines standardized reasoning tasks with a novel Overthinking Score that balances accuracy against verbosity, surfacing models that reason efficiently rather than models that simply generate more tokens.
        </p>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {[
          { title: '14 Tasks', desc: 'Sorting, comparison, arithmetic, list aggregation, and statistical tasks with configurable difficulty.' },
          { title: 'Overthinking Score', desc: 'F1-harmonic mean of accuracy and token efficiency — penalizes both low accuracy and excessive verbosity.' },
          { title: 'vLLM Powered', desc: 'High-throughput batched inference via vLLM. Scales to multi-GPU with tensor parallelism.' },
          { title: 'HuggingFace Compatible', desc: 'Evaluate any HF model out of the box. Configurable quantization, temperature, token limits.' },
        ].map(item => (
          <div key={item.title} className={cardCls}>
            <div className={`text-sm font-semibold mb-1 ${isDark ? 'text-gray-200' : 'text-gray-800'}`}>{item.title}</div>
            <div className={`text-xs ${isDark ? 'text-gray-500' : 'text-gray-500'}`}>{item.desc}</div>
          </div>
        ))}
      </div>
      <div className={cardCls}>
        <p className={`text-xs ${isDark ? 'text-gray-500' : 'text-gray-500'}`}>
          Accepted at <strong className={isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}>ACL 2026</strong>. Source on <a href="https://github.com/ctrl-gaurav/LLMThinkBench" target="_blank" rel="noopener noreferrer" className={`${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'} hover:underline`}>GitHub</a>. Package on <a href="https://pypi.org/project/llmthinkbench/" target="_blank" rel="noopener noreferrer" className={`${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'} hover:underline`}>PyPI</a>. Paper: <a href="https://arxiv.org/abs/2507.04023" target="_blank" rel="noopener noreferrer" className={`${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'} hover:underline`}>arXiv:2507.04023</a>.
        </p>
      </div>
    </div>
  )
}

function InstallationContent({ isDark, cardCls, headCls, subheadCls, textCls, pypiVersion }) {
  return (
    <div className="space-y-6">
      <div className={cardCls}>
        <h2 className={headCls}>Installation</h2>
        <p className={textCls}>Install LLMThinkBench in a fresh Python environment.{pypiVersion && <span className="ml-2 font-mono text-xs opacity-60">Latest: v{pypiVersion}</span>}</p>
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>From PyPI (recommended)</h3>
        <CodeBlock code="pip install llmthinkbench" />
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>From Source</h3>
        <CodeBlock code={`git clone https://github.com/ctrl-gaurav/LLMThinkBench.git
cd LLMThinkBench
pip install -e .`} />
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>Verify</h3>
        <CodeBlock code={`python -c "from llmthinkbench import evaluate; print('OK')"
llmthinkbench --help`} />
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>Requirements</h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {[
            { label: 'Python', value: '3.8+' },
            { label: 'PyTorch', value: '2.0+' },
            { label: 'vLLM', value: 'For high-throughput inference' },
            { label: 'CUDA', value: 'Required for GPU inference' },
          ].map(r => (
            <div key={r.label} className={`flex items-center gap-3 px-4 py-2.5 rounded-lg ${isDark ? 'bg-bb-dark-400/30' : 'bg-gray-50'}`}>
              <span className={`text-xs font-mono font-bold ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`}>{r.label}</span>
              <span className={`text-sm ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>{r.value}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function QuickStartContent({ isDark, cardCls, headCls, subheadCls, textCls }) {
  return (
    <div className="space-y-6">
      <div className={cardCls}>
        <h2 className={headCls}>Quick Start</h2>
        <p className={textCls}>Run your first evaluation in under a minute.</p>
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>Command Line</h3>
        <CodeBlock code={`# Evaluate a model on two tasks
llmthinkbench --model_id "Qwen/Qwen3-4B" --tasks sorting comparison`} />
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>Python API</h3>
        <CodeBlock language="python" code={`from llmthinkbench import evaluate

results = evaluate(
    model_id="Qwen/Qwen3-4B",
    tasks=["sorting", "comparison", "sum"],
)
print(results)`} />
      </div>
      <Callout type="tip">Start with a small <code>--datapoints</code> value (e.g. 20) to sanity-check your setup before running the full suite.</Callout>
    </div>
  )
}

function CLIContent({ isDark, cardCls, headCls, subheadCls, textCls }) {
  return (
    <div className="space-y-6">
      <div className={cardCls}>
        <h2 className={headCls}>CLI Reference</h2>
        <p className={textCls}>The <code>llmthinkbench</code> CLI runs evaluations end-to-end and writes a full report to disk.</p>
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>Comprehensive evaluation</h3>
        <CodeBlock code={`llmthinkbench --model_id "Qwen/Qwen3-4B" \\
  --tasks "sorting comparison sum multiplication odd_count even_count absolute_difference division find_maximum find_minimum mean median mode subtraction" \\
  --datapoints 100 \\
  --folds 3 \\
  --range -1000 1000 \\
  --list_sizes "8 16 32 64" \\
  --cuda_device "cuda:0" \\
  --tensor_parallel_size 1 \\
  --gpu_memory_utilization 0.98 \\
  --temperature 0.1 \\
  --top_p 0.9 \\
  --max_tokens 1024 \\
  --trust_remote_code \\
  --store_details \\
  --output_dir "qwen3_4b_eval" \\
  --seed 42`} />
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>Key flags</h3>
        <div className={`text-sm space-y-2 ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
          {[
            ['--model_id', 'HuggingFace model identifier, e.g. Qwen/Qwen3-4B'],
            ['--tasks', 'Space-separated list of tasks to evaluate'],
            ['--datapoints', 'Samples per task per list size'],
            ['--folds', 'Number of folds for stability'],
            ['--list_sizes', 'Input list sizes for list-based tasks'],
            ['--tensor_parallel_size', 'Number of GPUs for tensor parallelism'],
            ['--store_details', 'Persist per-example outputs for inspection'],
            ['--output_dir', 'Directory for results and reports'],
          ].map(([flag, desc]) => (
            <div key={flag} className="flex gap-3">
              <code className={`font-mono text-xs shrink-0 ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`}>{flag}</code>
              <span className="text-xs">{desc}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function PythonAPIContent({ isDark, cardCls, headCls, subheadCls, textCls }) {
  return (
    <div className="space-y-6">
      <div className={cardCls}>
        <h2 className={headCls}>Python API</h2>
        <p className={textCls}>Two entrypoints: the high-level <code>evaluate()</code> helper and the lower-level task classes.</p>
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>High-level</h3>
        <CodeBlock language="python" code={`from llmthinkbench import evaluate

results = evaluate(
    model_id="Qwen/Qwen3-4B",
    tasks=["sorting", "comparison", "sum", "multiplication"],
    datapoints=500,
    list_sizes=[8, 16, 32],
    folds=3,
    range=[-1000, 1000],
    store_details=True,
    output_dir="./custom_results",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.98,
    temperature=0.1,
    top_p=0.9,
    max_tokens=1024,
)`} />
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>Low-level (per-task control)</h3>
        <CodeBlock language="python" code={`from llmthinkbench.models.model_handler import ModelHandler
from llmthinkbench.tasks.sorting_task import SortingTask
from llmthinkbench.tasks.comparison_task import ComparisonTask
from llmthinkbench.utils.reporting import generate_final_report

model_handler = ModelHandler(
    model_id="Qwen/Qwen3-4B",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.98,
)

output_dir = "qwen3_4b_eval_results"

sorting = SortingTask(
    model_handler=model_handler,
    output_dir=output_dir,
    min_val=-1000, max_val=1000,
    num_folds=3, num_samples=100,
    store_details=True,
    temperature=0.1, top_p=0.9, max_tokens=1024,
)
sorting_metrics = sorting.run_evaluation([8, 16, 32, 64])

comparison = ComparisonTask(
    model_handler=model_handler,
    output_dir=output_dir,
    min_val=-1000, max_val=1000,
    num_folds=3, num_samples=100,
    store_details=True,
    temperature=0.1, top_p=0.9, max_tokens=1024,
)
comparison_metrics = comparison.run_evaluation()

report = generate_final_report(
    sorting_metrics + comparison_metrics,
    [8, 16, 32, 64],
    output_dir,
)`} />
      </div>
    </div>
  )
}

function TasksContent({ isDark, cardCls, headCls, subheadCls }) {
  const groups = [
    { title: 'Basic Operations', items: [
      ['sorting', 'Sort a list of numbers'],
      ['comparison', 'Compare two numbers'],
      ['sum', 'Sum of a list of numbers'],
      ['subtraction', 'Subtract two numbers'],
      ['multiplication', 'Multiply two numbers'],
      ['division', 'Divide two numbers'],
    ]},
    { title: 'List Processing', items: [
      ['find_maximum', 'Largest value in a list'],
      ['find_minimum', 'Smallest value in a list'],
      ['odd_count', 'Count odd numbers in a list'],
      ['even_count', 'Count even numbers in a list'],
    ]},
    { title: 'Statistical', items: [
      ['mean', 'Arithmetic mean of a list'],
      ['median', 'Median value of a list'],
      ['mode', 'Most frequent value(s)'],
    ]},
    { title: 'Advanced', items: [
      ['absolute_difference', 'Absolute difference between two numbers'],
    ]},
  ]
  return (
    <div className="space-y-6">
      <div className={cardCls}>
        <h2 className={headCls}>Tasks</h2>
        <p className={`text-sm leading-relaxed ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
          LLMThinkBench ships 14 tasks grouped by reasoning style. Each task reports accuracy, instruction-following rate, token counts, and the Overthinking Score.
        </p>
      </div>
      {groups.map(group => (
        <div key={group.title} className={cardCls}>
          <h3 className={subheadCls}>{group.title}</h3>
          <div className={`text-sm space-y-2 ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
            {group.items.map(([k, v]) => (
              <div key={k} className="flex gap-3">
                <code className={`font-mono text-xs shrink-0 ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`}>{k}</code>
                <span className="text-xs">{v}</span>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  )
}

function OverthinkingContent({ isDark, cardCls, headCls, subheadCls, textCls }) {
  return (
    <div className="space-y-6">
      <div className={cardCls}>
        <h2 className={headCls}>Overthinking Score</h2>
        <p className={textCls}>
          Traditional benchmarks reward accuracy alone and miss a practical reality: a model at 95% accuracy with 50 tokens is often preferable to one at 98% with 500 tokens. The Overthinking Score is an F1-harmonic mean that balances both axes.
        </p>
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>Formula</h3>
        <CodeBlock language="python" code={`# F1-harmonic mean of accuracy and token efficiency
normalized_tokens  = (tokens - min_tokens) / (max_tokens - min_tokens)
token_efficiency   = 1 - normalized_tokens
overthinking_score = 2 * (accuracy * token_efficiency) / (accuracy + token_efficiency)`} />
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>Why harmonic mean</h3>
        <ul className={`text-sm space-y-1.5 list-disc pl-5 ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
          <li>Models can't game the score with just accuracy or just efficiency.</li>
          <li>Both dimensions must improve for the score to improve meaningfully.</li>
          <li>Extremely verbose or extremely inaccurate models are penalized hard.</li>
        </ul>
      </div>
      <Callout type="info">The full ranking live-updates on the <a href="/#/" className={`${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'} hover:underline`}>leaderboard</a>.</Callout>
    </div>
  )
}

function ConfigurationContent({ isDark, cardCls, headCls, subheadCls, textCls }) {
  return (
    <div className="space-y-6">
      <div className={cardCls}>
        <h2 className={headCls}>Configuration</h2>
        <p className={textCls}>Tune generation, GPU usage, and evaluation breadth with CLI flags or Python kwargs.</p>
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>Generation</h3>
        <CodeBlock code={`llmthinkbench --model_id "Qwen/Qwen3-4B" \\
  --tasks sorting \\
  --temperature 0.1 \\
  --top_p 0.9 \\
  --max_tokens 1024`} />
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>Multi-GPU</h3>
        <CodeBlock code={`llmthinkbench --model_id "meta-llama/Llama-3.1-70B-Instruct" \\
  --tensor_parallel_size 4 \\
  --gpu_memory_utilization 0.9 \\
  --tasks sorting comparison`} />
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>Reproducibility</h3>
        <CodeBlock code={`llmthinkbench --model_id "Qwen/Qwen3-4B" \\
  --tasks sorting \\
  --seed 42 \\
  --folds 3`} />
      </div>
    </div>
  )
}

function OutputContent({ isDark, cardCls, headCls, subheadCls, textCls }) {
  return (
    <div className="space-y-6">
      <div className={cardCls}>
        <h2 className={headCls}>Output Format</h2>
        <p className={textCls}>Each run produces a directory containing per-task metrics, per-example traces (if <code>--store_details</code>), and a final report.</p>
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>Directory layout</h3>
        <CodeBlock code={`output_dir/
├── report.json           # Aggregate metrics across tasks
├── report.md             # Human-readable summary
├── sorting/
│   ├── metrics.json      # Per-list-size accuracy, tokens, overthinking score
│   └── details.jsonl     # Per-example traces (if --store_details)
├── comparison/
│   ├── metrics.json
│   └── details.jsonl
└── ...`} />
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>Metric schema</h3>
        <CodeBlock language="json" code={`{
  "task": "sorting",
  "list_size": 16,
  "accuracy": 0.94,
  "instruction_following": 0.98,
  "avg_tokens": 142.3,
  "overthinking_score": 0.891
}`} />
      </div>
    </div>
  )
}

function TroubleshootingContent({ isDark, cardCls, headCls, subheadCls }) {
  return (
    <div className="space-y-6">
      <div className={cardCls}>
        <h2 className={headCls}>Troubleshooting</h2>
      </div>
      {[
        { title: 'CUDA out of memory', desc: 'Lower memory usage or shard across GPUs.', code: `# Reduce memory utilization
llmthinkbench --model_id MODEL --gpu_memory_utilization 0.8

# Shard across multiple GPUs
llmthinkbench --model_id MODEL --tensor_parallel_size 2` },
        { title: 'Slow inference', desc: 'Batch more aggressively by increasing datapoints per run.', code: `# vLLM batches implicitly — bigger --datapoints = better throughput
llmthinkbench --model_id MODEL --tasks sorting --datapoints 500` },
        { title: 'Model requires trust_remote_code', desc: 'Some HF models need code execution to load.', code: `llmthinkbench --model_id MODEL --trust_remote_code --tasks sorting` },
        { title: 'Import errors', desc: 'Reinstall in a clean environment.', code: `pip install --force-reinstall llmthinkbench
python -c "from llmthinkbench import evaluate; print('OK')"` },
      ].map(issue => (
        <div key={issue.title} className={cardCls}>
          <div className="text-xs font-semibold text-red-400 mb-1">{issue.title}</div>
          <p className={`text-xs mb-2 ${isDark ? 'text-gray-500' : 'text-gray-500'}`}>{issue.desc}</p>
          <CodeBlock code={issue.code} />
        </div>
      ))}
      <div className={cardCls}>
        <h3 className={subheadCls}>Getting help</h3>
        <div className={`text-xs space-y-2 ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
          <div>Issues: <a href="https://github.com/ctrl-gaurav/LLMThinkBench/issues" className={`${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'} hover:underline`}>GitHub Issues</a></div>
          <div>Source: <a href="https://github.com/ctrl-gaurav/LLMThinkBench" className={`${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'} hover:underline`}>github.com/ctrl-gaurav/LLMThinkBench</a></div>
        </div>
      </div>
    </div>
  )
}

function CitationContent({ isDark, cardCls, headCls, textCls }) {
  return (
    <div className="space-y-6">
      <div className={cardCls}>
        <h2 className={headCls}>Citation</h2>
        <p className={textCls}>Accepted at <strong className={isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}>ACL 2026</strong>. If LLMThinkBench supports your research, please cite:</p>
        <CodeBlock language="bibtex" code={`@inproceedings{srivastava2026llmthinkbench,
  title     = {Do LLMs Overthink Basic Math Reasoning? Benchmarking the Accuracy-Efficiency Tradeoff in Language Models},
  author    = {Gaurav Srivastava and Aafiya Hussain and Sriram Srinivasan and Xuan Wang},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2507.04023}
}`} />
      </div>
    </div>
  )
}

/* ============ MAIN ============ */

export default function Documentation() {
  const [activeSection, setActiveSection] = useState('overview')
  const [searchQuery, setSearchQuery] = useState('')
  const { isDark } = useTheme()
  const pypiVersion = usePyPIVersion()

  const cardCls = `p-6 sm:p-8 rounded-2xl ${isDark ? 'glass-card' : 'bg-white/70 backdrop-blur-xl border border-gray-200/60 shadow-sm'}`
  const headCls = `text-xl sm:text-2xl font-bold mb-4 ${isDark ? 'text-white' : 'text-gray-900'}`
  const subheadCls = `text-lg font-semibold mb-3 ${isDark ? 'text-white' : 'text-gray-900'}`
  const textCls = `text-sm leading-relaxed ${isDark ? 'text-gray-400' : 'text-gray-600'}`

  const contentProps = { isDark, cardCls, headCls, subheadCls, textCls, pypiVersion }

  const sectionContent = {
    'overview': <OverviewContent {...contentProps} />,
    'installation': <InstallationContent {...contentProps} />,
    'quickstart': <QuickStartContent {...contentProps} />,
    'cli': <CLIContent {...contentProps} />,
    'python-api': <PythonAPIContent {...contentProps} />,
    'tasks': <TasksContent {...contentProps} />,
    'overthinking': <OverthinkingContent {...contentProps} />,
    'configuration': <ConfigurationContent {...contentProps} />,
    'output': <OutputContent {...contentProps} />,
    'troubleshooting': <TroubleshootingContent {...contentProps} />,
    'citation': <CitationContent {...contentProps} />,
  }

  const filteredItems = useMemo(() => {
    if (!searchQuery.trim()) return NAV_ITEMS
    const q = searchQuery.toLowerCase()
    return NAV_ITEMS.filter(item => item.label.toLowerCase().includes(q) || item.id.toLowerCase().includes(q))
  }, [searchQuery])

  return (
    <section className="pt-24 pb-20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-8">
          <span className={`inline-block px-4 py-1.5 rounded-full text-xs font-semibold tracking-wider uppercase mb-4 ${
            isDark ? 'bg-bb-accent/10 text-bb-accent border border-bb-accent/20' : 'bg-bb-accent-dark/10 text-bb-accent-dark border border-bb-accent-dark/20'
          }`}>Documentation</span>
          <h1 className={`text-3xl sm:text-4xl lg:text-5xl font-bold mb-3 ${isDark ? 'text-white' : 'text-gray-900'}`}>
            LLMThinkBench Docs
          </h1>
          <p className={`text-lg max-w-2xl mx-auto ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
            Everything you need to benchmark reasoning and overthinking in language models.
          </p>
        </div>

        {/* Search */}
        <div className="max-w-md mx-auto mb-10">
          <div className={`relative rounded-xl transition-all duration-300 ${
            isDark
              ? 'bg-bb-dark-400/30 border border-bb-dark-50/20 focus-within:border-bb-accent/25'
              : 'bg-white border border-gray-200 focus-within:border-bb-accent-dark/40 shadow-sm'
          }`}>
            <Search className={`absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 ${isDark ? 'text-gray-500' : 'text-gray-400'}`} />
            <input
              type="text"
              placeholder="Search documentation..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className={`w-full pl-10 pr-4 py-2.5 rounded-xl text-sm bg-transparent outline-none ${
                isDark ? 'text-white placeholder:text-gray-600' : 'text-gray-900 placeholder:text-gray-400'
              }`}
            />
          </div>
        </div>

        {/* Mobile section selector */}
        <div className="lg:hidden mb-6">
          <div className="relative">
            <select
              value={activeSection}
              onChange={(e) => { setActiveSection(e.target.value); window.scrollTo({ top: 0, behavior: 'smooth' }) }}
              className={`w-full px-4 py-2.5 rounded-xl text-sm font-medium appearance-none ${
                isDark
                  ? 'bg-bb-dark-400/30 text-white border border-bb-dark-50/20'
                  : 'bg-white text-gray-900 border border-gray-200 shadow-sm'
              }`}
            >
              {NAV_ITEMS.map(item => (
                <option key={item.id} value={item.id}>{item.label}</option>
              ))}
            </select>
            <ChevronDown className={`absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 pointer-events-none ${isDark ? 'text-gray-500' : 'text-gray-400'}`} />
          </div>
        </div>

        {/* Sidebar + content */}
        <div className="flex gap-8">
          <nav className={`hidden lg:block w-56 shrink-0 sticky top-24 self-start rounded-xl p-2 max-h-[calc(100vh-7rem)] overflow-y-auto ${
            isDark
              ? 'bg-bb-dark-400/20 border border-bb-dark-50/10'
              : 'bg-white/60 border border-gray-200/30 shadow-sm'
          }`}>
            <div className="space-y-0.5">
              {filteredItems.map((item) => (
                <button
                  key={item.id}
                  onClick={() => { setActiveSection(item.id); window.scrollTo({ top: 0, behavior: 'smooth' }) }}
                  className={`w-full text-left px-3 py-2.5 rounded-lg text-sm transition-all duration-200 flex items-center gap-2.5 ${
                    activeSection === item.id
                      ? isDark
                        ? 'bg-bb-accent/10 text-bb-accent font-semibold'
                        : 'bg-bb-accent-dark/10 text-bb-accent-dark font-semibold'
                      : isDark
                        ? 'text-gray-500 hover:text-gray-300 hover:bg-bb-dark-300/30'
                        : 'text-gray-500 hover:text-gray-700 hover:bg-gray-50'
                  }`}
                >
                  <svg className="w-4 h-4 shrink-0" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" d={item.iconPath} />
                  </svg>
                  {item.label}
                </button>
              ))}
            </div>
          </nav>

          <div className="flex-1 min-w-0">
            <div key={activeSection} className="animate-fade-in">
              {sectionContent[activeSection]}
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
