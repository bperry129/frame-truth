import React from 'react';
import { ShieldCheck, ScanEye, MessageSquare } from 'lucide-react';

interface HeaderProps {
  onNavigate: (page: 'home' | 'methodology' | 'blog' | 'admin' | 'feedback' | 'privacy' | 'terms') => void;
  currentPage: 'home' | 'methodology' | 'blog' | 'admin' | 'feedback' | 'privacy' | 'terms';
}

const Header: React.FC<HeaderProps> = ({ onNavigate, currentPage }) => {
  return (
    <header className="w-full border-b border-slate-800 bg-slate-950/50 backdrop-blur-md sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        <div className="flex items-center gap-2 cursor-pointer" onClick={() => onNavigate('home')}>
          <div className="relative">
            <div className="absolute inset-0 bg-cyan-500 blur-md opacity-20 rounded-full"></div>
            <ScanEye className="w-8 h-8 text-cyan-400 relative z-10" />
          </div>
          <div>
            <h1 className="text-xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
              FrameTruth
            </h1>
            <p className="text-[10px] text-slate-400 tracking-widest uppercase">AI Video Forensics</p>
          </div>
        </div>
        
        <div className="hidden md:flex items-center gap-6 text-sm font-medium text-slate-400">
          <button 
            onClick={() => onNavigate('home')} 
            className={`transition-colors ${currentPage === 'home' ? 'text-cyan-400' : 'hover:text-cyan-400'}`}
          >
            Analyzer
          </button>
          <button 
            onClick={() => onNavigate('methodology')} 
            className={`transition-colors ${currentPage === 'methodology' ? 'text-cyan-400' : 'hover:text-cyan-400'}`}
          >
            Methodology
          </button>
          <a href="https://arxiv.org/abs/2501.16382" target="_blank" rel="noreferrer" className="hover:text-cyan-400 transition-colors">Paper</a>
          <button 
            onClick={() => onNavigate('feedback')} 
            className={`transition-colors flex items-center gap-1 ${currentPage === 'feedback' ? 'text-cyan-400' : 'hover:text-cyan-400'}`}
          >
            <MessageSquare className="w-4 h-4" />
            Feedback
          </button>
          <div className="flex items-center gap-1 px-3 py-1 rounded-full bg-slate-900 border border-slate-800">
            <ShieldCheck className="w-4 h-4 text-emerald-500" />
            <span className="text-slate-300">System Active</span>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
