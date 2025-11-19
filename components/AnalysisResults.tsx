import React, { useState } from 'react';
import { createPortal } from 'react-dom';
import { AnalysisResult } from '../types';
import TrajectoryChart from './TrajectoryChart';
import { AlertTriangle, CheckCircle, BrainCircuit, Activity, FileVideo, Info, X } from 'lucide-react';

interface AnalysisResultsProps {
  result: AnalysisResult;
}

const AnalysisResults: React.FC<AnalysisResultsProps> = ({ result }) => {
  const [activeInfo, setActiveInfo] = useState<string | null>(null);
  const isAi = result.isAi;
  const themeColor = isAi ? 'text-rose-500' : 'text-emerald-500';
  const themeBg = isAi ? 'bg-rose-500' : 'bg-emerald-500';
  const themeBorder = isAi ? 'border-rose-500' : 'border-emerald-500';

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Top Verdict Card */}
      <div className={`relative overflow-hidden rounded-xl border ${themeBorder} bg-slate-900/80 p-8 text-center shadow-2xl`}>
        <div className={`absolute inset-0 ${themeBg} opacity-5 blur-2xl`}></div>
        
        <div className="relative z-10 flex flex-col items-center">
            {isAi ? (
                <div className="w-20 h-20 rounded-full bg-rose-500/10 flex items-center justify-center mb-4 border border-rose-500/30 animate-pulse-slow">
                    <BrainCircuit className="w-10 h-10 text-rose-500" />
                </div>
            ) : (
                <div className="w-20 h-20 rounded-full bg-emerald-500/10 flex items-center justify-center mb-4 border border-emerald-500/30">
                    <CheckCircle className="w-10 h-10 text-emerald-500" />
                </div>
            )}
            
            <h2 className="text-3xl font-bold tracking-tighter text-white mb-1">
                {isAi ? "LIKELY AI-GENERATED" : "LIKELY REAL FOOTAGE"}
            </h2>
            <p className="text-slate-400 mb-6 text-sm uppercase tracking-widest">
                Confidence: <span className={`font-bold ${themeColor}`}>{result.confidence}%</span>
            </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Left Col: Metrics */}
        <div className="space-y-6">
            {/* Stats Grid */}
            <div className="grid grid-cols-2 gap-4">
                <div className="bg-slate-900 border border-slate-800 p-4 rounded-lg relative">
                    <div className="flex items-center justify-between mb-2 text-slate-400">
                        <div className="flex items-center gap-2">
                            <Activity className="w-4 h-4" />
                            <span className="text-xs uppercase font-semibold">Curvature</span>
                        </div>
                        <button onClick={() => setActiveInfo('curvature')} className="hover:text-white"><Info className="w-3 h-3" /></button>
                    </div>
                    <div className="text-2xl font-bold text-white">
                        {result.curvatureScore}<span className="text-xs text-slate-500 font-normal">/100</span>
                    </div>
                    <div className="w-full bg-slate-800 h-1 mt-3 rounded-full overflow-hidden">
                        <div className={`h-full ${result.curvatureScore > 50 ? 'bg-rose-500' : 'bg-emerald-500'}`} style={{ width: `${result.curvatureScore}%` }}></div>
                    </div>
                </div>
                <div className="bg-slate-900 border border-slate-800 p-4 rounded-lg relative">
                     <div className="flex items-center justify-between mb-2 text-slate-400">
                        <div className="flex items-center gap-2">
                            <FileVideo className="w-4 h-4" />
                            <span className="text-xs uppercase font-semibold">Physics</span>
                        </div>
                        <button onClick={() => setActiveInfo('physics')} className="hover:text-white"><Info className="w-3 h-3" /></button>
                    </div>
                     <div className="text-2xl font-bold text-white">
                        {result.distanceScore ? result.distanceScore : (100 - result.curvatureScore)}<span className="text-xs text-slate-500 font-normal">/100</span>
                    </div>
                    <p className="text-xs text-slate-500 mt-2">Coherence Score</p>
                </div>
            </div>

            {/* Trajectory Chart */}
            <div className="relative">
                <button 
                    onClick={() => setActiveInfo('trajectory')} 
                    className="absolute top-4 right-4 z-20 text-slate-500 hover:text-white"
                >
                    <Info className="w-4 h-4" />
                </button>
                <TrajectoryChart data={result.trajectoryData} isAi={isAi} />
            </div>
        </div>

        {/* Right Col: Reasoning */}
        <div className="bg-slate-900 border border-slate-800 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <AlertTriangle className="w-5 h-5 text-yellow-500" />
                Forensic Analysis
            </h3>
            <ul className="space-y-3">
                {result.reasoning.map((reason, idx) => (
                    <li key={idx} className="flex gap-3 items-start text-sm text-slate-300 p-3 bg-slate-950/50 rounded border border-slate-800/50">
                        <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-cyan-500 flex-shrink-0"></span>
                        <span className="leading-relaxed">{reason}</span>
                    </li>
                ))}
            </ul>
            {result.modelDetected && (
                <div className="mt-6 pt-6 border-t border-slate-800">
                    <span className="text-xs text-slate-500 uppercase">Suspected Generator</span>
                    <p className="text-lg font-mono text-cyan-400">{result.modelDetected}</p>
                </div>
            )}
        </div>
      </div>

      {/* Contextual Info Modal (Portalled to body to ensure proper viewport centering) */}
      {activeInfo && createPortal(
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in duration-200" onClick={() => setActiveInfo(null)}>
          <div className="bg-slate-900 border border-slate-700 rounded-xl max-w-md w-full shadow-2xl overflow-hidden relative animate-in zoom-in-95 duration-200" onClick={(e) => e.stopPropagation()}>
            <div className="flex justify-between items-center p-4 border-b border-slate-800">
              <h3 className="text-lg font-bold text-white flex items-center gap-2">
                <Info className="w-4 h-4 text-cyan-400" />
                {activeInfo === 'curvature' && 'Curvature Score'}
                {activeInfo === 'physics' && 'Physics Score'}
                {activeInfo === 'trajectory' && 'Latent Space Trajectory'}
              </h3>
              <button onClick={() => setActiveInfo(null)} className="text-slate-400 hover:text-white">
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="p-6 text-sm text-slate-300 leading-relaxed">
              {activeInfo === 'curvature' && (
                <>
                  <p className="mb-3">This measures how much the video's internal representation "curves" over time.</p>
                  <ul className="space-y-2 list-disc pl-4">
                    <li><strong className="text-emerald-400">Real Videos</strong> follow straight paths because real-world physics is continuous.</li>
                    <li><strong className="text-rose-400">AI Videos</strong> struggle with continuity, causing the path to curve or zigzag as it "hallucinates" new frames.</li>
                  </ul>
                  <p className="mt-3 text-xs text-slate-500">High Score (over 60) indicates likely AI generation.</p>
                </>
              )}
              
              {activeInfo === 'physics' && (
                <>
                  <p className="mb-3">Evaluates the video's adherence to physical laws like gravity, momentum, and object permanence.</p>
                  <ul className="space-y-2 list-disc pl-4">
                    <li><strong className="text-emerald-400">High Score:</strong> Objects move naturally and persist correctly.</li>
                    <li><strong className="text-rose-400">Low Score:</strong> Objects disappear, merge, or move impossible ways.</li>
                  </ul>
                </>
              )}

              {activeInfo === 'trajectory' && (
                <>
                  <p className="mb-3">A visual map of the video's stability.</p>
                  <ul className="space-y-2 list-disc pl-4">
                    <li><strong className="text-emerald-400">Straight Line:</strong> Stable, natural video.</li>
                    <li><strong className="text-rose-400">Chaotic/Jittery Line:</strong> The AI is losing coherence between frames.</li>
                  </ul>
                  <p className="mt-3">Use this chart to confirm if the Curvature Score matches visual evidence.</p>
                </>
              )}
            </div>
            <div className="p-3 bg-slate-950 border-t border-slate-800 text-center">
              <button 
                onClick={() => setActiveInfo(null)}
                className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-white rounded-lg transition-colors text-xs font-medium"
              >
                Close
              </button>
            </div>
          </div>
        </div>,
        document.body
      )}
    </div>
  );
};

export default AnalysisResults;
