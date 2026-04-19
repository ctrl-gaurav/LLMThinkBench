export const modelData = [
  // Qwen3 Series
  { model: "Qwen/Qwen3-0.6B", params: "0.6B", quantization: "None", accuracy: 49.99, instruction: 83.85, tokens: 3162.8, efficiency: 0.484, overthinking: 44636.6, words: 1620.9, chars: 8301.8 },
  { model: "Qwen/Qwen3-1.7B", params: "1.7B", quantization: "None", accuracy: 70.24, instruction: 86.54, tokens: 3157.2, efficiency: 0.555, overthinking: 31072.7, words: 1620.7, chars: 8445.9 },
  { model: "Qwen/Qwen3-4B", params: "4B", quantization: "None", accuracy: 81.90, instruction: 91.57, tokens: 3091.2, efficiency: 0.58, overthinking: 17399.7, words: 1623.1, chars: 8489.9 },
  { model: "Qwen/Qwen3-8B", params: "8B", quantization: "None", accuracy: 82.10, instruction: 91.58, tokens: 3027.8, efficiency: 0.615, overthinking: 23616, words: 1584.6, chars: 8260.3 },
  { model: "Qwen/Qwen3-14B", params: "14B", quantization: "None", accuracy: 86.52, instruction: 99.27, tokens: 3607.6, efficiency: 0.725, overthinking: 8139.6, words: 1941.2, chars: 10556.1 },
  { model: "Qwen/Qwen3-32B", params: "32B", quantization: "None", accuracy: 84.13, instruction: 93.05, tokens: 2845.9, efficiency: 0.627, overthinking: 10680.3, words: 1497.5, chars: 7790.1 },

  // Microsoft Phi-4 Series
  { model: "microsoft/phi-4", params: "14B", quantization: "None", accuracy: 78.92, instruction: 97.46, tokens: 378.6, efficiency: 0.593, overthinking: 1982.5, words: 194.6, chars: 989.9 },
  { model: "microsoft/Phi-4-mini-instruct", params: "3.8B", quantization: "None", accuracy: 54.55, instruction: 95.02, tokens: 292.1, efficiency: 0.567, overthinking: 3017.4, words: 146.6, chars: 684.9 },
  { model: "microsoft/Phi-4-reasoning-plus", params: "14B", quantization: "None", accuracy: 69.54, instruction: 88.89, tokens: 6780.7, efficiency: 0.288, overthinking: 132979, words: 3972, chars: 23893 },
  { model: "microsoft/Phi-4-reasoning", params: "14B", quantization: "None", accuracy: 72.23, instruction: 96.21, tokens: 6066.2, efficiency: 0.352, overthinking: 105156.7, words: 3710.8, chars: 23866.8 },
  { model: "microsoft/Phi-4-mini-reasoning", params: "3.8B", quantization: "None", accuracy: 70.16, instruction: 89.56, tokens: 3171.9, efficiency: 0.659, overthinking: 33533.3, words: 1571.7, chars: 8450.5 },
  { model: "microsoft/Phi-3-mini-128k-instruct", params: "3.8B", quantization: "None", accuracy: 35.82, instruction: 96.58, tokens: 89.4, efficiency: 0.413, overthinking: 1156.9, words: 40.6, chars: 208.9 },
  { model: "microsoft/Phi-3-medium-4k-instruct", params: "14B", quantization: "None", accuracy: 43.47, instruction: 89.87, tokens: 189.3, efficiency: 0.378, overthinking: 3134.7, words: 109.6, chars: 553.6 },
  { model: "microsoft/Phi-3-medium-128k-instruct", params: "14B", quantization: "None", accuracy: 40.76, instruction: 96.26, tokens: 140.0, efficiency: 0.390, overthinking: 1188.4, words: 74.8, chars: 367.3 },

  // Meta LLaMA Series
  { model: "meta-llama/Llama-3.2-1B-Instruct", params: "1B", quantization: "None", accuracy: 16.25, instruction: 47.15, tokens: 336.3, efficiency: 0.223, overthinking: 6659.1, words: 159.0, chars: 756.9 },
  { model: "meta-llama/Llama-3.2-3B-Instruct", params: "3B", quantization: "None", accuracy: 42.54, instruction: 89.88, tokens: 279.7, efficiency: 0.490, overthinking: 4737.8, words: 144.6, chars: 694.7 },
  { model: "meta-llama/Llama-3.1-8B-Instruct", params: "8B", quantization: "None", accuracy: 48.84, instruction: 85.66, tokens: 366.4, efficiency: 0.516, overthinking: 5667.7, words: 203.4, chars: 977.7 },
  { model: "meta-llama/Llama-3.1-70B-Instruct", params: "70B", quantization: "None", accuracy: 75.43, instruction: 98.12, tokens: 251.2, efficiency: 0.691, overthinking: 4519.7, words: 135.5, chars: 654.6 },
  { model: "meta-llama/Llama-3.3-70B-Instruct", params: "70B", quantization: "None", accuracy: 74.59, instruction: 97.40, tokens: 312.8, efficiency: 0.654, overthinking: 1641.0, words: 174.1, chars: 859.7 },

  // Qwen2.5 Series (Base Models)
  { model: "Qwen/Qwen2.5-0.5B-Instruct", params: "0.5B", quantization: "None", accuracy: 21.31, instruction: 77.57, tokens: 432.3, efficiency: 0.268, overthinking: 12885.7, words: 223.2, chars: 1144.5 },
  { model: "Qwen/Qwen2.5-1.5B-Instruct", params: "1.5B", quantization: "None", accuracy: 43.03, instruction: 85.45, tokens: 264.7, efficiency: 0.470, overthinking: 3348.5, words: 134.1, chars: 626.7 },
  { model: "Qwen/Qwen2.5-3B-Instruct", params: "3B", quantization: "None", accuracy: 45.75, instruction: 92.35, tokens: 331.3, efficiency: 0.463, overthinking: 2811.3, words: 176.5, chars: 861.4 },
  { model: "Qwen/Qwen2.5-7B-Instruct", params: "7B", quantization: "None", accuracy: 61.36, instruction: 96.47, tokens: 286.9, efficiency: 0.568, overthinking: 6106.2, words: 149.5, chars: 747.2 },
  { model: "Qwen/Qwen2.5-14B-Instruct", params: "14B", quantization: "None", accuracy: 63.74, instruction: 97.83, tokens: 260.2, efficiency: 0.578, overthinking: 1835.9, words: 137.1, chars: 685.7 },
  { model: "Qwen/Qwen2.5-32B-Instruct", params: "32B", quantization: "None", accuracy: 72.90, instruction: 99.26, tokens: 260.9, efficiency: 0.643, overthinking: 1425.9, words: 139.1, chars: 673.6 },

  // Other Models
  { model: "HuggingFaceTB/SmolLM2-1.7B-Instruct", params: "1.7B", quantization: "None", accuracy: 16.69, instruction: 68.98, tokens: 213.0, efficiency: 0.217, overthinking: 10923.8, words: 93.5, chars: 481.5 },
  { model: "mistralai/Mistral-7B-Instruct-v0.3", params: "7B", quantization: "None", accuracy: 27.66, instruction: 96.26, tokens: 207.1, efficiency: 0.289, overthinking: 4812.8, words: 113.7, chars: 585.9 },
  { model: "mistralai/Mistral-Nemo-Instruct-2407", params: "12B", quantization: "None", accuracy: 35.43, instruction: 82.95, tokens: 377.0, efficiency: 0.356, overthinking: 14208.7, words: 234.2, chars: 1123.7 },

  // Qwen2.5 Math Series
  { model: "Qwen/Qwen2.5-Math-1.5B-Instruct", params: "1.5B", quantization: "None", accuracy: 51.43, instruction: 94.04, tokens: 397.1, efficiency: 0.488, overthinking: 5465.9, words: 210.0, chars: 1076.9 },
  { model: "Qwen/Qwen2.5-Math-7B-Instruct", params: "7B", quantization: "None", accuracy: 60.68, instruction: 94.36, tokens: 411.7, efficiency: 0.538, overthinking: 3741.4, words: 221.5, chars: 1156.0 },

  // Qwen2.5 Quantized Models (GPTQ-8-Bit)
  { model: "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8", params: "0.5B", quantization: "GPTQ-8-Bit", accuracy: 21.29, instruction: 76.79, tokens: 431.5, efficiency: 0.268, overthinking: 5694.5, words: 223.3, chars: 1138.5 },
  { model: "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int8", params: "1.5B", quantization: "GPTQ-8-Bit", accuracy: 43.67, instruction: 86.64, tokens: 264.3, efficiency: 0.472, overthinking: 5268.4, words: 133.7, chars: 628.3 },
  { model: "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int8", params: "3B", quantization: "GPTQ-8-Bit", accuracy: 48.65, instruction: 91.99, tokens: 341.9, efficiency: 0.497, overthinking: 7244.8, words: 181.7, chars: 878.7 },
  { model: "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8", params: "7B", quantization: "GPTQ-8-Bit", accuracy: 60.61, instruction: 96.40, tokens: 287.5, efficiency: 0.564, overthinking: 6121.5, words: 149.4, chars: 745.5 },
  { model: "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8", params: "14B", quantization: "GPTQ-8-Bit", accuracy: 63.86, instruction: 97.89, tokens: 261.2, efficiency: 0.576, overthinking: 2070.4, words: 138.1, chars: 688.9 },
  { model: "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8", params: "32B", quantization: "GPTQ-8-Bit", accuracy: 73.08, instruction: 99.20, tokens: 261.9, efficiency: 0.645, overthinking: 1254.2, words: 139.6, chars: 675.7 },
  { model: "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8", params: "72B", quantization: "GPTQ-8-Bit", accuracy: 74.05, instruction: 96.28, tokens: 347.1, efficiency: 0.577, overthinking: 3424.8, words: 184.2, chars: 901.5 },

  // Qwen2.5 Quantized Models (GPTQ-4-Bit)
  { model: "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4", params: "0.5B", quantization: "GPTQ-4-Bit", accuracy: 12.77, instruction: 77.70, tokens: 478.0, efficiency: 0.176, overthinking: 22014.7, words: 260.9, chars: 1342.5 },
  { model: "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4", params: "1.5B", quantization: "GPTQ-4-Bit", accuracy: 39.42, instruction: 82.97, tokens: 292.1, efficiency: 0.434, overthinking: 6973.2, words: 142.3, chars: 685.0 },
  { model: "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4", params: "3B", quantization: "GPTQ-4-Bit", accuracy: 41.94, instruction: 90.97, tokens: 301.0, efficiency: 0.438, overthinking: 9158.4, words: 158.8, chars: 753.3 },
  { model: "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4", params: "7B", quantization: "GPTQ-4-Bit", accuracy: 58.03, instruction: 96.00, tokens: 291.9, efficiency: 0.550, overthinking: 3961.6, words: 152.2, chars: 758.9 },
  { model: "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4", params: "14B", quantization: "GPTQ-4-Bit", accuracy: 60.94, instruction: 96.69, tokens: 240.9, efficiency: 0.586, overthinking: 2501.9, words: 127.5, chars: 632.8 },
  { model: "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4", params: "32B", quantization: "GPTQ-4-Bit", accuracy: 72.67, instruction: 99.37, tokens: 260.5, efficiency: 0.640, overthinking: 1430.9, words: 139.2, chars: 663.7 },
  { model: "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4", params: "72B", quantization: "GPTQ-4-Bit", accuracy: 72.74, instruction: 94.85, tokens: 358.3, efficiency: 0.549, overthinking: 2606.3, words: 188.4, chars: 919.8 },

  // GPT Series
  { model: "GPT-4.1", params: "--", quantization: "None", accuracy: 89.88, instruction: 97.79, tokens: 338.8, efficiency: 0.752, overthinking: 801.3, words: 152.2, chars: 759.5 },
  { model: "GPT-4.1-mini", params: "--", quantization: "None", accuracy: 90.23, instruction: 98.14, tokens: 328.8, efficiency: 0.768, overthinking: 830.4, words: 145.1, chars: 740.7 },
  { model: "GPT-4.1-nano", params: "--", quantization: "None", accuracy: 75.35, instruction: 95.58, tokens: 338.8, efficiency: 0.713, overthinking: 493.9, words: 148.7, chars: 760.3 },
  { model: "GPT-4o", params: "--", quantization: "None", accuracy: 87.56, instruction: 99.42, tokens: 290.5, efficiency: 0.737, overthinking: 633.1, words: 154.8, chars: 749 },
  { model: "GPT-4o-mini", params: "--", quantization: "None", accuracy: 75.00, instruction: 97.67, tokens: 341.3, efficiency: 0.671, overthinking: 936.3, words: 172, chars: 848.7 },

  // Gemini Series
  { model: "gemini-2.0-flash-lite", params: "--", quantization: "None", accuracy: 73.33, instruction: 99.58, tokens: 215.5, efficiency: 0.613, overthinking: 398.3, words: 116.9, chars: 523 },
  { model: "gemini-2.0-flash", params: "--", quantization: "None", accuracy: 69.60, instruction: 94.44, tokens: 234.5, efficiency: 0.608, overthinking: 171.9, words: 118.9, chars: 517.9 },
  { model: "gemini-2.5-flash-lite-preview-06-17", params: "--", quantization: "None", accuracy: 66.21, instruction: 80.69, tokens: 528.7, efficiency: 0.431, overthinking: 1144.3, words: 285.8, chars: 1482.2 },
  { model: "gemini-2.5-flash", params: "--", quantization: "None", accuracy: 55.18, instruction: 63.49, tokens: 186.3, efficiency: 0.312, overthinking: 216.9, words: 104.8, chars: 539.5 },
];

// Calculate ranks dynamically based on accuracy
export function calculateRanks(data) {
  const sorted = [...data].sort((a, b) => b.accuracy - a.accuracy);
  sorted.forEach((model, index) => {
    model.rank = index + 1;
  });
  return data.map(model => {
    const rankedModel = sorted.find(m => m.model === model.model && m.accuracy === model.accuracy);
    return { ...model, rank: rankedModel.rank };
  });
}

export function getMetricClass(value, metric) {
  switch (metric) {
    case 'accuracy':
    case 'instruction':
    case 'efficiency':
      return value >= 70 ? 'text-emerald-400' : value >= 40 ? 'text-amber-400' : 'text-red-400';
    case 'overthinking':
    case 'tokens':
      return value <= 500 ? 'text-emerald-400' : value <= 1000 ? 'text-amber-400' : 'text-red-400';
    case 'words':
      return value <= 200 ? 'text-emerald-400' : value <= 400 ? 'text-amber-400' : 'text-red-400';
    case 'chars':
      return value <= 1000 ? 'text-emerald-400' : value <= 2000 ? 'text-amber-400' : 'text-red-400';
    default:
      return '';
  }
}

export function getMetricColor(value, metric) {
  switch (metric) {
    case 'accuracy':
    case 'instruction':
    case 'efficiency':
      return value >= 70 ? '#10b981' : value >= 40 ? '#f59e0b' : '#ef4444';
    case 'overthinking':
    case 'tokens':
      return value <= 500 ? '#10b981' : value <= 1000 ? '#f59e0b' : '#ef4444';
    case 'words':
      return value <= 200 ? '#10b981' : value <= 400 ? '#f59e0b' : '#ef4444';
    case 'chars':
      return value <= 1000 ? '#10b981' : value <= 2000 ? '#f59e0b' : '#ef4444';
    default:
      return '#e4e4e7';
  }
}

export function parseParams(params) {
  if (params === '--') return 0;
  return parseFloat(params.replace(/[^0-9.]/g, ''));
}
